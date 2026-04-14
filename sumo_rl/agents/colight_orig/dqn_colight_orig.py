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


# ── GAT building blocks ───────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """Single-head GAT layer (Veličković et al., 2018).

    W  : shared linear projection for all nodes   [in_dim → out_dim]
    a  : attention scoring vector                  [2*out_dim → 1]
    """

    def __init__(self, in_dim: int, out_dim: int, leaky_slope: float = 0.2):
        super().__init__()
        self.W          = nn.Linear(in_dim, out_dim, bias=False)
        self.a          = nn.Linear(2 * out_dim, 1, bias=False)
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
        return agg, Wh_i


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
                 n_heads: int = 2, leaky_slope: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.heads   = nn.ModuleList([
            GATLayer(obs_dim, hidden_dim, leaky_slope) for _ in range(n_heads)
        ])
        self.own_enc = nn.Linear(obs_dim, hidden_dim)
        self.q_head  = nn.Linear(hidden_dim * (n_heads + 1), action_dim)

    def forward(self, own: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        own:       [B, obs_dim]
        neighbors: [B, 4, obs_dim]   zero rows = missing neighbor
        returns:   [B, action_dim]
        """
        mask = (neighbors.abs().sum(dim=-1, keepdim=True) == 0)  # [B, 4, 1]

        head_aggs = []
        for head in self.heads:
            agg, _ = head(own, neighbors, mask)
            head_aggs.append(agg)                         # each [B, hidden_dim]

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
                 eps_start, eps_end, eps_decay, device):
        self.obs_dim       = obs_dim
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.target_update = target_update
        self.mini_size     = mini_size
        self.batch_size    = batch_size
        self.eps_start     = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.device        = device
        self.count         = 0
        self.loss          = None
        self.start_train   = False

        self.q_net = CoLightOrigQNet(
            obs_dim, hidden_dim, action_dim, n_heads
        ).to(device)
        self.target_q_net = CoLightOrigQNet(
            obs_dim, hidden_dim, action_dim, n_heads
        ).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer     = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay_buffer = CoLightReplayBuffer(capacity)

    def take_action(self, own_state: np.ndarray, nb_obs: np.ndarray) -> int:
        """
        own_state: [obs_dim]
        nb_obs:    [4, obs_dim]
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        own = torch.tensor(own_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        nbs = torch.tensor(nb_obs,    dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(own, nbs).argmax().item()

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
        with torch.no_grad():
            max_next_q = self.target_q_net(next_states, next_nb_obs).max(1)[0].view(-1, 1)
        q_targets  = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
