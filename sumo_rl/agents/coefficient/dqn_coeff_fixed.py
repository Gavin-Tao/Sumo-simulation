"""FixedCoeffDQN: concat-state DQN + TPAMI-style static α uniform-mean reward coordination.

Reproduces the cooperative reward mechanism from Shi et al. TPAMI 2024 in DQN form:
    eff_r = α · own_r + (1 - α) · mean(active neighbor rewards)

with α = 0.5 fixed (TPAMI paper grid-search optimum).

Architecture:
  State input:  aug_state = concat(own_obs, nb_obs_up, nb_obs_down, nb_obs_left, nb_obs_right)
                            (same as CoeffDQN / GATCoeffDQN — 5x own_dim, zeros for missing)
  Q-network:    standard MLP on aug_state
  Reward coord: eff_r = α · own_r + (1 - α) · mean(active nb_r)
                  α  = 0.5 fixed (own protected)
                  mean is over ACTIVE neighbors only (mask-aware, uniform weighting)

Differences from GATCoeffDQN:
  • NO learnable β / GAT layer.
  • Neighbors aggregated by UNIFORM MEAN (no per-neighbor attention).
  • This is the static / non-state-conditional reward coordination baseline.

Differences from current CoeffDQN:
  • No additive Σ sigmoid(β)·nb_r (which empirically degenerates to β=0).
  • Convex combo α + (1-α) = 1 — eff_r scale bounded, own protected.

Position in design space:
                  State 层协同              Reward 层协同
  static          concat (this agent's Q)   ★ this agent (TPAMI baseline) / CoeffDQN β
  dynamic         GAT (CoLight)             GATCoeffDQN (proposed)

Direct comparison to GATCoeffDQN: same architecture, same state, same α,
only difference is "uniform mean" (TPAMI) vs "dynamic GAT softmax" (GATCoeffDQN).
→ Δ (GATCoeffDQN − FixedCoeffDQN) directly isolates the contribution of
   dynamic per-neighbor attention weighting.
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


class FixedCoeffDQN:
    """DQN with TPAMI-style static α uniform-mean reward coordination.

    Interface mirrors CoeffDQN / GATCoeffDQN. Drop-in replacement in traincoeff_tpami.py.
    """

    def __init__(self, aug_state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, capacity, mini_size, batch_size,
                 eps_start, eps_end, eps_decay, device,
                 # TPAMI-specific:
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
            f"aug_state_dim={aug_state_dim} must be divisible by 5"
        self.own_dim = aug_state_dim // 5

        # α (own weight) — fixed scalar, TPAMI paper optimum
        assert 0.0 <= alpha <= 1.0, f"alpha must be in [0,1], got {alpha}"
        self.alpha = alpha

        # Q-net: standard MLP on aug_state (same as CoeffDQN/GATCoeffDQN)
        self.q_net        = Qnet(aug_state_dim, hidden_dim, action_dim, use_noisy).to(device)
        self.target_q_net = Qnet(aug_state_dim, hidden_dim, action_dim, use_noisy).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # No learnable cooperation params — only Q-net parameters
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
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
        self._last_own_r:         torch.Tensor | None = None
        self._last_nb_r:          torch.Tensor | None = None
        self._last_nb_r_mean:     torch.Tensor | None = None
        self._last_eff_r:         torch.Tensor | None = None
        self._last_active:        torch.Tensor | None = None   # [B, 4] active mask

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
        """Return reward magnitude stats (for wandb).

        FixedCoeffDQN has no learnable β / GAT, so no attention to report.
        But we report:
          - alpha           : the fixed α (constant)
          - n_active_avg    : batch-averaged number of active neighbors
          - nb_reward_sum_raw / nb_reward_mean / own_reward_mean / eff_reward_mean
          - nb_to_own_abs_ratio: |mean(nb)| / |own| (debug)
        """
        if self._last_own_r is None:
            return None

        own_r       = self._last_own_r.squeeze(-1)             # [B]
        nb_r        = self._last_nb_r                          # [B, 4]
        nb_mean     = self._last_nb_r_mean.squeeze(-1)         # [B]  uniform mean over active
        eff_r       = self._last_eff_r.squeeze(-1)             # [B]
        active      = self._last_active                        # [B, 4]

        nb_sum_raw  = nb_r.sum(dim=1)                          # [B]

        return {
            "alpha":                  float(self.alpha),
            "n_active_avg":           active.sum(dim=1).mean().item(),
            "nb_reward_sum_raw":      nb_sum_raw.mean().item(),
            "nb_reward_mean":         nb_mean.mean().item(),
            "own_reward_mean":        own_r.mean().item(),
            "eff_reward_mean":        eff_r.mean().item(),
            "nb_to_own_abs_ratio":    (nb_mean.abs() / (own_r.abs() + 1e-6)).mean().item(),
        }

    def update(self, transition_dict: dict):
        self.start_train = True

        states      = torch.tensor(transition_dict['states'],      dtype=torch.float32).to(self.device)
        actions     = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards     = torch.tensor(transition_dict['rewards'],     dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones       = torch.tensor(transition_dict['dones'],       dtype=torch.float32).view(-1, 1).to(self.device)
        nb_rewards  = torch.tensor(transition_dict['nb_rewards'],  dtype=torch.float32).to(self.device)  # [B, 4]

        # ── Compute active mask from nb_obs ──────────────────────────────
        _, nb_s = self._split_aug(states)                       # [B, 4, own_dim]
        mask = (nb_s.abs().sum(dim=-1) == 0)                    # [B, 4]  True = missing
        active = (~mask).float()                                # [B, 4]  1 = active

        # ── Uniform mean over active neighbors only ──────────────────────
        n_active = active.sum(dim=1, keepdim=True).clamp(min=1.0)        # [B, 1]
        nb_mean = (nb_rewards * active).sum(dim=1, keepdim=True) / n_active  # [B, 1]

        # ── Convex-combo eff_reward: α · own + (1-α) · mean(active nb_r) ──
        eff_rewards = self.alpha * rewards + (1.0 - self.alpha) * nb_mean

        # Cache for wandb logging
        self._last_own_r     = rewards.detach()
        self._last_nb_r      = nb_rewards.detach()
        self._last_nb_r_mean = nb_mean.detach()
        self._last_eff_r     = eff_rewards.detach()
        self._last_active    = active.detach()

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
