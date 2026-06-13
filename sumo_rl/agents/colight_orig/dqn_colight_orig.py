"""Original CoLight DQN (Wei et al., KDD 2019).

Graph Attention Network formulation follows Veličković et al. (ICLR 2018):

    e_ij  = LeakyReLU( a^T · [W·h_i || W·h_j] )   # attention score
    α_ij  = softmax_j(e_ij)                          # normalised weight
    h'_i  = ELU( Σ_{j∈N(i)} α_ij · W·h_j )        # aggregated repr.

Multi-head: K independent heads, outputs concatenated before Q-head.

Q-network input: [own_feat || head1_agg || head2_agg || ...]
                  shape: (K+1) * hidden_dim → action_dim

Replay buffer is reused from the sibling colight/ package (same schema).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse identical replay buffer from colight/
from sumo_rl.agents.colight.replay_buffer import CoLightReplayBuffer
from sumo_rl.agents.noisy_linear import NoisyLinear


# ── GAT building blocks ───────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """Single-head GAT layer (Veličković et al., 2018).

    W  : shared linear projection for all nodes   [in_dim → out_dim]
    a  : attention scoring vector                  [2*out_dim → 1]
    """

    def __init__(self, in_dim: int, out_dim: int, leaky_slope: float = 0.2,
                 use_noisy: bool = False):
        super().__init__()
        self.W          = NoisyLinear(in_dim, out_dim) if use_noisy else nn.Linear(in_dim, out_dim, bias=False)
        self.a          = nn.Linear(2 * out_dim, 1, bias=False)   # attention stays deterministic
        self.leaky_relu = nn.LeakyReLU(leaky_slope)

    def forward(self, own: torch.Tensor, neighbors: torch.Tensor,
                mask: torch.Tensor):
        """
        own:       [B, in_dim]
        neighbors: [B, 4, in_dim]
        mask:      [B, 4, 1]  True = missing neighbor
        returns:
            agg:   [B, out_dim]  — ELU-activated aggregated representation
            Wh_i:  [B, out_dim]  — own projected feature (for Q-head concat)
        """
        Wh_i = self.W(own)                                # [B, out_dim]
        Wh_j = self.W(neighbors)                          # [B, 4, out_dim]

        # Attention scores: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        Wh_i_exp = Wh_i.unsqueeze(1).expand_as(Wh_j)     # [B, 4, out_dim]
        e = self.leaky_relu(
            self.a(torch.cat([Wh_i_exp, Wh_j], dim=-1))  # [B, 4, 1]
        )
        e = e.masked_fill(mask, float('-inf'))

        alpha = torch.softmax(e, dim=1)                   # [B, 4, 1]
        alpha = torch.nan_to_num(alpha, nan=0.0)          # all-missing → 0

        agg = F.elu((alpha * Wh_j).sum(dim=1))            # [B, out_dim]
        return agg, Wh_i, alpha


class CoLightOrigQNet(nn.Module):
    """Original CoLight Q-network: multi-head GAT + Q-head.

    n_heads independent GAT heads (each with its own W and a).
    Outputs are concatenated with own_feat then fed to the Q-head.

    Input  dim: obs_dim
    Hidden dim: hidden_dim  (per head)
    Output dim: action_dim
    Total concat before Q-head: hidden_dim * (n_heads + 1)
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int,
                 n_heads: int = 2, leaky_slope: float = 0.2, use_noisy: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.heads   = nn.ModuleList([
            GATLayer(obs_dim, hidden_dim, leaky_slope, use_noisy) for _ in range(n_heads)
        ])
        linear = NoisyLinear if use_noisy else nn.Linear
        self.own_enc = linear(obs_dim, hidden_dim)
        self.q_head  = linear(hidden_dim * (n_heads + 1), action_dim)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, own: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        own:       [B, obs_dim]
        neighbors: [B, 4, obs_dim]   zero rows = missing neighbor
        returns:   [B, action_dim]
        """
        mask = (neighbors.abs().sum(dim=-1, keepdim=True) == 0)  # [B, 4, 1]

        head_aggs = []
        head_alphas = []                                  # for attention logging
        for head in self.heads:
            agg, _, alpha = head(own, neighbors, mask)
            head_aggs.append(agg)                         # each [B, hidden_dim]
            head_alphas.append(alpha)                     # each [B, 4, 1]

        # Cache last forward's attention for wandb logging (no grad needed).
        # Shape: [n_heads, B, 4, 1] → for logging we usually reduce over batch & heads
        with torch.no_grad():
            self.last_attention = torch.stack(head_alphas, dim=0).detach()
            self.last_attention_mask = mask.detach()      # [B, 4, 1] same mask for all heads

        own_feat = F.relu(self.own_enc(own))              # [B, hidden_dim]
        cat_feat = torch.cat([own_feat] + head_aggs, dim=-1)  # [B, H*(n_heads+1)]
        return self.q_head(cat_feat)                      # [B, action_dim]


# ── DQN agent ─────────────────────────────────────────────────────────────────

class CoLightOrigDQN:
    """DQN using original CoLight (multi-head GAT) Q-network.

    Interface mirrors CoLightDQN in colight/dqn_colight.py.
    Extra constructor argument: n_heads (default 2).
    """

    def __init__(self, obs_dim, hidden_dim, action_dim, n_heads,
                 learning_rate, gamma, epsilon,
                 target_update, capacity, mini_size, batch_size,
                 eps_start, eps_end, eps_decay, device,
                 use_noisy: bool = False,
                 use_double: bool = False,
                 use_per: bool = False,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_end: float = 1.0,
                 per_beta_steps: int = 100_000,
                 per_eps: float = 1e-6,
                 grad_clip: float = None,
                 loss_fn: str = "mse",
                 target_clip_max=None):
        self.obs_dim       = obs_dim
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.use_noisy     = use_noisy
        self.use_double    = use_double  # 默认 False = vanilla DQN
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
        # grad_clip: max grad-norm before optimizer.step(); None = off
        # (exp186 diagnosis: GAT on 125-dim B obs destabilises without it —
        # same remedy as the transformer R2 clip, configurable per run)
        self.grad_clip     = grad_clip
        # loss_fn / target_clip_max (R1, forensics Part 7): same gated knobs
        # as dqn_agent_txw — defaults reproduce original behaviour exactly.
        assert loss_fn in ("mse", "huber"), loss_fn
        self.loss_fn = loss_fn
        self.target_clip_max = None if target_clip_max is None else float(target_clip_max)
        self.grad_norm     = None   # pre-clip total norm, set only when clipping
        self.start_train   = False

        self.q_net = CoLightOrigQNet(
            obs_dim, hidden_dim, action_dim, n_heads, use_noisy=use_noisy
        ).to(device)
        self.target_q_net = CoLightOrigQNet(
            obs_dim, hidden_dim, action_dim, n_heads, use_noisy=use_noisy
        ).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer     = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # ── PER 开关 + 超参 (默认 False = vanilla 行为) ──
        self.use_per        = use_per
        self.per_alpha      = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end   = per_beta_end
        self.per_beta_steps = per_beta_steps
        self.per_eps        = per_eps

        if use_per:
            from sumo_rl.agents.prioritized_replay_buffer import PrioritizedReplayBuffer
            self.replay_buffer = PrioritizedReplayBuffer(capacity, alpha=per_alpha, eps=per_eps)
        else:
            self.replay_buffer = CoLightReplayBuffer(capacity)

    @property
    def current_beta(self) -> float:
        """PER 模式下的 importance-sampling β,从 per_beta_start 线性退火到 per_beta_end。"""
        if not self.use_per:
            return 1.0
        if self.count >= self.per_beta_steps:
            return self.per_beta_end
        frac = self.count / max(self.per_beta_steps, 1)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    def attention_stats(self):
        # -> Optional[dict]  (Python 3.9 compat,不用 dict | None 语法)
        """Return mean GAT attention per direction across batch+heads from last forward pass.

        Useful for wandb logging. Returns None if no forward has run yet.
        Format: {"attn_up": float, "attn_down": float, "attn_left": float, "attn_right": float,
                 "attn_entropy": float}
        Notes:
          - Attention is masked by neighbor presence (missing neighbors = 0 attention).
          - To get "fair" attention per direction, we average ONLY over batch elements where
            that direction has an active neighbor.
          - Entropy: how peaked/spread the attention is (high = uniform, low = focused).
        """
        if not hasattr(self.q_net, 'last_attention'):
            return None
        a = self.q_net.last_attention            # [n_heads, B, 4, 1]
        m = self.q_net.last_attention_mask       # [B, 4, 1]  True = missing
        active = (~m).float().squeeze(-1)        # [B, 4]  1 = active neighbor
        # Mean attention per direction, averaged across heads and active samples
        a_mean_heads = a.mean(dim=0).squeeze(-1) # [B, 4]
        per_dir = []
        for d in range(4):
            mask_d = active[:, d]                # [B]  which batch items have this neighbor
            n_active = mask_d.sum().item()
            if n_active > 0:
                attn_d = (a_mean_heads[:, d] * mask_d).sum().item() / n_active
            else:
                attn_d = 0.0
            per_dir.append(attn_d)
        # Entropy of mean attention (across active dirs) — peaked = informative, uniform = unfocused
        avg_alpha = a_mean_heads.mean(dim=0)     # [4]  batch-averaged
        avg_alpha = avg_alpha + 1e-12
        entropy = -(avg_alpha * torch.log(avg_alpha)).sum().item()
        return {
            "attn_up":      per_dir[0],
            "attn_down":    per_dir[1],
            "attn_left":    per_dir[2],
            "attn_right":   per_dir[3],
            "attn_entropy": entropy,
        }

    def take_action(self, own_state: np.ndarray, nb_obs: np.ndarray,
                    mask=None) -> int:
        """
        own_state: [obs_dim]
        nb_obs:    [4, obs_dim]
        mask:      optional bool array over actions (M3 Dublin) — ε samples
                   only valid actions, greedy sets invalid Q to -inf.
        """
        if np.random.random() < self.epsilon:
            if mask is not None:
                return int(np.random.choice(np.flatnonzero(mask)))
            return np.random.randint(self.action_dim)
        own = torch.tensor(own_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        nbs = torch.tensor(nb_obs,    dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_net(own, nbs)
            if mask is not None:
                q = q.masked_fill(~torch.as_tensor(mask, dtype=torch.bool,
                                                   device=self.device), -1e9)
            return q.argmax().item()

    def update(self, transition_dict: dict):
        self.start_train = True

        states      = torch.tensor(transition_dict['states'],      dtype=torch.float32).to(self.device)
        actions     = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards     = torch.tensor(transition_dict['rewards'],     dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones       = torch.tensor(transition_dict['dones'],       dtype=torch.float32).view(-1, 1).to(self.device)
        nb_obs      = torch.tensor(transition_dict['nb_obs'],      dtype=torch.float32).to(self.device)  # [B,4,obs_dim]
        next_nb_obs = torch.tensor(transition_dict['next_nb_obs'], dtype=torch.float32).to(self.device)  # [B,4,obs_dim]

        q_values   = self.q_net(states, nb_obs).gather(1, actions)
        next_mask_t = None
        if 'next_masks' in transition_dict:
            next_mask_t = torch.as_tensor(np.asarray(transition_dict['next_masks']),
                                          dtype=torch.bool, device=self.device)
        with torch.no_grad():
            if self.use_double:
                # Double DQN: online net 选动作, target net 给值
                nq = self.q_net(next_states, next_nb_obs)
                if next_mask_t is not None:
                    nq = nq.masked_fill(~next_mask_t, -1e9)
                next_actions = nq.argmax(1, keepdim=True)
                max_next_q   = self.target_q_net(next_states, next_nb_obs).gather(1, next_actions)
            else:
                tq = self.target_q_net(next_states, next_nb_obs)
                if next_mask_t is not None:
                    tq = tq.masked_fill(~next_mask_t, -1e9)
                max_next_q = tq.max(1)[0].view(-1, 1)
        q_targets  = rewards + self.gamma * max_next_q * (1 - dones)
        if self.target_clip_max is not None:
            q_targets = q_targets.clamp(max=self.target_clip_max)   # R1 锚

        # ── Loss: PER 用 IS 权重加权,否则原 MSE ──
        if 'weights' in transition_dict:
            weights = torch.tensor(transition_dict['weights'],
                                   dtype=torch.float32).view(-1, 1).to(self.device)
            td_errors_for_per = (q_targets - q_values).detach()
            if self.loss_fn == "huber":
                elementwise = F.smooth_l1_loss(q_values, q_targets, reduction='none')
            else:
                elementwise = (q_values - q_targets) ** 2
            loss = (weights * elementwise).mean()
        else:
            if self.loss_fn == "huber":
                loss = F.smooth_l1_loss(q_values, q_targets)
            else:
                loss = F.mse_loss(q_values, q_targets)

        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            self.grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.grad_clip))
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_q_net.reset_noise()

        # ── PER: 用刚算的 TD-error 更新对应样本的优先级 ──
        if 'weights' in transition_dict and 'indices' in transition_dict:
            td_np = td_errors_for_per.squeeze(-1).cpu().numpy()
            self.replay_buffer.update_priorities(  # type: ignore[attr-defined]
                transition_dict['indices'], td_np)
