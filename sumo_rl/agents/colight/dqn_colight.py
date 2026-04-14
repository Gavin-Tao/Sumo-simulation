"""CoLight DQN: Q-network with single-head graph attention over neighbors."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import CoLightReplayBuffer


class CoLightQNet(nn.Module):
    """Q-network with graph attention over up-to-4 neighbors.

    Architecture:
      1. Encode own obs  → own_feat  [B, H]
      2. Encode each neighbor obs → nb_feat  [B, 4, H]
      3. Attention score per neighbor: Linear(cat(own_feat, nb_feat_j)) → scalar
      4. Mask missing neighbors (all-zero input) → softmax over valid only
      5. Aggregate: agg = Σ weight_j * nb_feat_j  [B, H]
      6. Q-values: Linear(cat(own_feat, agg))  [B, action_dim]
    """

    def __init__(self, own_dim: int, nb_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.own_encoder = nn.Linear(own_dim, hidden_dim)
        self.nb_encoder  = nn.Linear(nb_dim,  hidden_dim)
        self.attn_score  = nn.Linear(hidden_dim * 2, 1)
        self.q_head      = nn.Linear(hidden_dim * 2, action_dim)

    def forward(self, own: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        own:       [B, own_dim]
        neighbors: [B, 4, nb_dim]  — zero rows = missing neighbor
        returns:   [B, action_dim]
        """
        own_feat = F.relu(self.own_encoder(own))          # [B, H]
        nb_feat  = F.relu(self.nb_encoder(neighbors))     # [B, 4, H]

        own_exp = own_feat.unsqueeze(1).expand(-1, 4, -1)               # [B, 4, H]
        scores  = self.attn_score(torch.cat([own_exp, nb_feat], dim=-1))  # [B, 4, 1]

        # Mask missing neighbors (entire row is zero → no real neighbor)
        missing = (neighbors.abs().sum(dim=-1, keepdim=True) == 0)      # [B, 4, 1]
        scores  = scores.masked_fill(missing, float('-inf'))

        weights = torch.softmax(scores, dim=1)            # [B, 4, 1]
        weights = torch.nan_to_num(weights, nan=0.0)      # all-missing → 0

        agg = (weights * nb_feat).sum(dim=1)              # [B, H]
        return self.q_head(torch.cat([own_feat, agg], dim=-1))


class CoLightDQN:
    """DQN using CoLightQNet. Interface mirrors dqn_agent_txw.DQN."""

    def __init__(self, own_dim, nb_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, capacity, mini_size, batch_size,
                 eps_start, eps_end, eps_decay, device):
        self.own_dim       = own_dim
        self.nb_dim        = nb_dim
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

        self.q_net        = CoLightQNet(own_dim, nb_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = CoLightQNet(own_dim, nb_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer     = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay_buffer = CoLightReplayBuffer(capacity)

    def take_action(self, own_state: np.ndarray, nb_obs: np.ndarray) -> int:
        """
        own_state: [own_dim]
        nb_obs:    [4, nb_dim]
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
        nb_obs      = torch.tensor(transition_dict['nb_obs'],      dtype=torch.float32).to(self.device)      # [B, 4, nb_dim]
        next_nb_obs = torch.tensor(transition_dict['next_nb_obs'], dtype=torch.float32).to(self.device)      # [B, 4, nb_dim]

        q_values   = self.q_net(states, nb_obs).gather(1, actions)
        max_next_q = self.target_q_net(next_states, next_nb_obs).max(1)[0].view(-1, 1)
        q_targets  = rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, q_targets.detach())
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
