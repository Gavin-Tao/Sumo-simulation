"""GATCoeffDQN: concat-state DQN + GAT-style dynamic reward coefficient.

Architecture:
  State input:  aug_state = concat(own_obs, nb_obs_up, nb_obs_down, nb_obs_left, nb_obs_right)
                            (same as CoeffDQN — 5x own_dim, zeros for missing neighbors)
  Q-network:    standard MLP on aug_state (same as CoeffDQN)
  Reward coord: eff_r = α · own_r + (1-α) · Σ_d β_d(state) · nb_r_d
                where:
                  α = 0.5 fixed (own protected, NEVER drowned)
                  β = softmax over GAT-style per-neighbor attention scores (sum=1)

Differences from CoeffDQN:
  • β is DYNAMIC (state-conditional via GAT layer), not 4 static learnable scalars
  • Convex combination (α + (1-α) = 1) → eff_r scale ≈ own_r 量级,不会爆
  • softmax-normalized β → cannot degenerate to all-zero (CoeffDQN's β did)
  • Same GAT-style scoring as CoLightOrig (W, a, LeakyReLU, softmax), but
    applied to reward weighting instead of state aggregation.

Position in the design space:
                    State 层协同              Reward 层协同
  static            concat (this agent's Q)   TPAMI α=0.5 / CoeffDQN β
  dynamic           GAT (CoLight)             ★ this agent's β (GAT-style)

This fills the "dynamic reward-layer coordination" gap in TSC literature.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import CoeffReplayBuffer
from sumo_rl.agents.noisy_linear import NoisyLinear


class Qnet(nn.Module):
    """Single-hidden-layer MLP on aug_state (identical to CoeffDQN.Qnet)."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 use_noisy: bool = False):
        super().__init__()
        linear = NoisyLinear if use_noisy else nn.Linear
        self.fc1 = linear(state_dim, hidden_dim)
        self.fc2 = linear(hidden_dim, action_dim)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class GATBetaLayer(nn.Module):
    """GAT-style attention layer applied to REWARD weighting (parallels CoLight GAT).

    Structure mirrors GATLayer in colight_orig/dqn_colight_orig.py:
      W : shared linear projection for own & neighbor states  [own_dim → hidden_dim]
      a : attention scoring vector                            [2*hidden_dim → 1]
      LeakyReLU on e, softmax over neighbors → β ∈ [0,1] with sum=1.

    Difference from CoLight's GAT:
      • Output is the attention weights (β) themselves, not the aggregated state.
      • These β are used downstream to weight neighbor REWARDS in the TD target.
    """

    def __init__(self, own_dim: int, hidden_dim: int, leaky_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(own_dim, hidden_dim, bias=False)
        self.a = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(leaky_slope)

    def forward(self, own: torch.Tensor, neighbors: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        own:       [B, own_dim]
        neighbors: [B, 4, own_dim]   zero rows = missing neighbor
        mask:      [B, 4, 1]         True = missing neighbor
        returns:
            β:    [B, 4]  softmax-normalized attention weights (sum=1 across active)
        """
        Wh_i = self.W(own)                                  # [B, hidden]
        Wh_j = self.W(neighbors)                            # [B, 4, hidden]

        # Attention score per neighbor: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        Wh_i_exp = Wh_i.unsqueeze(1).expand_as(Wh_j)        # [B, 4, hidden]
        e = self.leaky_relu(
            self.a(torch.cat([Wh_i_exp, Wh_j], dim=-1))     # [B, 4, 1]
        )
        e = e.masked_fill(mask, float('-inf'))              # missing → -inf
        beta = torch.softmax(e, dim=1).squeeze(-1)          # [B, 4], sum=1 across active
        beta = torch.nan_to_num(beta, nan=0.0)              # all-missing → 0 (safety)
        return beta


class GATCoeffDQN:
    """DQN with state-conditional GAT-style β on reward coordination.

    Interface mirrors CoeffDQN. Drop-in replacement in traincoeff_gat.py.
    """

    def __init__(self, aug_state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, capacity, mini_size, batch_size,
                 eps_start, eps_end, eps_decay, device,
                 # GATCoeffDQN-specific:
                 gat_hidden_dim: int = 64,
                 alpha: float = 0.5,
                 # Standard flags:
                 use_noisy: bool = False,
                 use_double: bool = False,
                 use_per: bool = False,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_end: float = 1.0,
                 per_beta_steps: int = 100_000,
                 per_eps: float = 1e-6):
        self.aug_state_dim = aug_state_dim
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.use_noisy     = use_noisy
        self.use_double    = use_double
        self.epsilon       = 0.0 if use_noisy else epsilon
        self.target_update = target_update
        self.mini_size     = mini_size
        self.batch_size    = batch_size
        self.eps_start     = 0.0 if use_noisy else eps_start
        self.eps_end       = 0.0 if use_noisy else eps_end
        self.eps_decay     = eps_decay
        self.device        = device
        self.count         = 0
        self.loss          = None
        self.start_train   = False

        # aug_state = 5 × own_dim (own + 4 neighbors), so own_dim = aug / 5
        assert aug_state_dim % 5 == 0, \
            f"aug_state_dim={aug_state_dim} must be divisible by 5 (own + 4 nb)"
        self.own_dim = aug_state_dim // 5

        # α (own weight) — fixed scalar, NOT learnable. Convex combo with (1-α) for neighbors.
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0,1], got {alpha}"
        self.alpha = alpha

        # Q-net: standard MLP on aug_state (same as CoeffDQN)
        self.q_net        = Qnet(aug_state_dim, hidden_dim, action_dim, use_noisy).to(device)
        self.target_q_net = Qnet(aug_state_dim, hidden_dim, action_dim, use_noisy).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # GAT-β layer: dynamic per-neighbor attention for reward weighting
        self.gat_beta = GATBetaLayer(self.own_dim, gat_hidden_dim).to(device)

        # Optimizer: Q-net + GAT-β jointly trained (gradient flows through eff_r)
        self.optimizer = torch.optim.Adam(
            list(self.q_net.parameters()) + list(self.gat_beta.parameters()),
            lr=learning_rate,
        )

        # PER setup
        self.use_per        = use_per
        self.per_alpha      = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end   = per_beta_end
        self.per_beta_steps = per_beta_steps
        self.per_eps        = per_eps

        if use_per:
            from sumo_rl.agents.prioritized_replay_buffer import PrioritizedReplayBuffer
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity, alpha=per_alpha, eps=per_eps,
            )
        else:
            self.replay_buffer = CoeffReplayBuffer(capacity)

        # Cached for wandb logging
        self._last_beta: torch.Tensor | None = None
        self._last_mask: torch.Tensor | None = None
        # Reward magnitude tracking (debugging:看邻居信号 vs 自己信号量级)
        self._last_own_r:          torch.Tensor | None = None
        self._last_nb_r:           torch.Tensor | None = None
        self._last_nb_r_weighted:  torch.Tensor | None = None
        self._last_eff_r:          torch.Tensor | None = None

    @property
    def current_beta(self) -> float:
        """PER importance-sampling β annealing."""
        if not self.use_per:
            return 1.0
        if self.count >= self.per_beta_steps:
            return self.per_beta_end
        frac = self.count / max(self.per_beta_steps, 1)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    def _split_aug(self, aug: torch.Tensor) -> tuple:
        """Split aug_state [B, 5*own_dim] → (own [B, own_dim], nb [B, 4, own_dim])."""
        own = aug[:, :self.own_dim]
        nb_flat = aug[:, self.own_dim:]
        nb = nb_flat.reshape(-1, 4, self.own_dim)
        return own, nb

    def take_action(self, aug_state: np.ndarray) -> int:
        """Same as CoeffDQN: Q-net forward on aug_state."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.tensor(aug_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_t).argmax().item()

    def attention_stats(self):
        """Return mean β per direction + entropy from last update (for wandb).

        Returns dict:  {beta_up, beta_down, beta_left, beta_right, beta_entropy}
        Notes:
          - For each direction, mean is taken ONLY over batch elements where that
            direction has an active neighbor (avoids 0-padding bias).
          - Entropy is over batch-averaged β distribution (informativeness indicator).
        """
        if self._last_beta is None or self._last_mask is None:
            return None
        b = self._last_beta              # [B, 4]
        m = self._last_mask.squeeze(-1)  # [B, 4]  True = missing
        active = (~m).float()            # [B, 4]  1 = active neighbor

        per_dir = []
        for d in range(4):
            mask_d = active[:, d]
            n_active = mask_d.sum().item()
            if n_active > 0:
                attn_d = (b[:, d] * mask_d).sum().item() / n_active
            else:
                attn_d = 0.0
            per_dir.append(attn_d)

        # Entropy of batch-averaged β
        avg_b = b.mean(dim=0) + 1e-12    # [4]
        entropy = -(avg_b * torch.log(avg_b)).sum().item()

        stats = {
            "beta_up":      per_dir[0],
            "beta_down":    per_dir[1],
            "beta_left":    per_dir[2],
            "beta_right":   per_dir[3],
            "beta_entropy": entropy,
        }

        # Reward magnitude debugging (看邻居信号 vs 自己信号的量级关系)
        if self._last_own_r is not None:
            own_r       = self._last_own_r.squeeze(-1)             # [B]
            nb_r        = self._last_nb_r                          # [B, 4]
            nb_weighted = self._last_nb_r_weighted.squeeze(-1)     # [B]  Σ β · nb_r
            eff_r       = self._last_eff_r.squeeze(-1)             # [B]

            # 原始 sum: nb_r 直接加和 (无权重) — 看"邻居信号原始量级"
            nb_sum_raw = nb_r.sum(dim=1)                           # [B]

            stats.update({
                "nb_reward_sum_raw":      nb_sum_raw.mean().item(),      # Σ nb_r (无加权)
                "nb_reward_weighted_sum": nb_weighted.mean().item(),     # Σ β · nb_r (GAT 加权)
                "own_reward_mean":        own_r.mean().item(),           # 自己 reward 量级
                "eff_reward_mean":        eff_r.mean().item(),           # 最终 eff_reward 量级
                # 比值:|邻居加权| / |自己|, 看 own 是否真的占主导 (应 ≤ (1-α)/α = 1 当 α=0.5)
                "nb_to_own_abs_ratio":    (nb_weighted.abs() / (own_r.abs() + 1e-6)).mean().item(),
            })

        return stats

    def update(self, transition_dict: dict):
        self.start_train = True

        states      = torch.tensor(transition_dict['states'],      dtype=torch.float32).to(self.device)
        actions     = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards     = torch.tensor(transition_dict['rewards'],     dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones       = torch.tensor(transition_dict['dones'],       dtype=torch.float32).view(-1, 1).to(self.device)
        nb_rewards  = torch.tensor(transition_dict['nb_rewards'],  dtype=torch.float32).to(self.device)  # [B, 4]

        # ── Compute dynamic β via GAT (from current states) ──────────────
        own_s, nb_s = self._split_aug(states)               # own [B, own_dim], nb [B, 4, own_dim]
        mask = (nb_s.abs().sum(dim=-1, keepdim=True) == 0)  # [B, 4, 1]  True = missing
        beta = self.gat_beta(own_s, nb_s, mask)             # [B, 4]  softmax sum=1

        # ── Convex-combo eff_reward: α · own + (1-α) · Σ β · nb_r ────────
        nb_weighted = (beta * nb_rewards).sum(dim=1, keepdim=True)   # [B, 1]
        eff_rewards = self.alpha * rewards + (1.0 - self.alpha) * nb_weighted

        # Cache for wandb logging (detach so it doesn't keep the graph alive)
        self._last_beta = beta.detach()
        self._last_mask = mask.detach()
        # Reward magnitude tracking
        self._last_own_r         = rewards.detach()              # [B, 1]
        self._last_nb_r          = nb_rewards.detach()           # [B, 4]  raw nb rewards
        self._last_nb_r_weighted = nb_weighted.detach()          # [B, 1]  Σ β · nb_r
        self._last_eff_r         = eff_rewards.detach()          # [B, 1]

        # ── Standard Q-learning update (with optional Double) ────────────
        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            if self.use_double:
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                max_next_q   = self.target_q_net(next_states).gather(1, next_actions)
            else:
                max_next_q = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_targets = eff_rewards + self.gamma * max_next_q * (1 - dones)

        # PER weighted loss / vanilla MSE
        if 'weights' in transition_dict:
            weights = torch.tensor(transition_dict['weights'],
                                   dtype=torch.float32).view(-1, 1).to(self.device)
            td_errors_for_per = (q_targets - q_values).detach()
            elementwise_sq    = (q_values - q_targets) ** 2
            loss              = (weights * elementwise_sq).mean()
        else:
            loss = F.mse_loss(q_values, q_targets)

        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_q_net.reset_noise()

        # PER priority update
        if 'weights' in transition_dict and 'indices' in transition_dict:
            td_np = td_errors_for_per.squeeze(-1).cpu().numpy()
            self.replay_buffer.update_priorities(  # type: ignore[attr-defined]
                transition_dict['indices'], td_np)
