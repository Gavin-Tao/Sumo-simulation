"""CoeffDQN: standard MLP Q-network + learnable cooperative reward weight β.

Observation: own_obs concatenated with [up, down, left, right] neighbor obs
             (zeros for missing neighbors).  Shape: own_dim + 4 * nb_dim.

Effective reward used in TD target:
    r_eff = r_local + sigmoid(β) * nb_reward_mean

β is an nn.Parameter jointly optimised with the Q-network via the same Adam
optimizer, so gradient flows through it automatically.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import CoeffReplayBuffer


class Qnet(nn.Module):
    """Standard single-hidden-layer MLP (identical to dqn_agent_txw.Qnet)."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class CoeffDQN:
    """DQN with learnable β. Interface mirrors dqn_agent_txw.DQN."""

    def __init__(self, aug_state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, capacity, mini_size, batch_size,
                 eps_start, eps_end, eps_decay, device):
        self.aug_state_dim = aug_state_dim
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

        self.q_net        = Qnet(aug_state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(aug_state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Learnable cooperative weight (unconstrained; sigmoid applied in update)
        self.beta = nn.Parameter(torch.tensor(0.0, device=device))

        # β is optimised together with Q-network weights
        self.optimizer = torch.optim.Adam(
            list(self.q_net.parameters()) + [self.beta],
            lr=learning_rate,
        )
        self.replay_buffer = CoeffReplayBuffer(capacity)

    @property
    def beta_value(self) -> float:
        """Effective β ∈ (0, 1) after sigmoid — use this for logging."""
        return torch.sigmoid(self.beta).item()

    def take_action(self, aug_state: np.ndarray) -> int:
        """aug_state: [own_dim + 4 * nb_dim]"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.tensor(aug_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_t).argmax().item()

    def update(self, transition_dict: dict):
        self.start_train = True

        states      = torch.tensor(transition_dict['states'],      dtype=torch.float32).to(self.device)
        actions     = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards     = torch.tensor(transition_dict['rewards'],     dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)
        dones       = torch.tensor(transition_dict['dones'],       dtype=torch.float32).view(-1, 1).to(self.device)
        nb_rewards  = torch.tensor(transition_dict['nb_rewards'],  dtype=torch.float32).view(-1, 1).to(self.device)

        # Effective reward: gradient flows through β
        eff_rewards = rewards + torch.sigmoid(self.beta) * nb_rewards

        q_values = self.q_net(states).gather(1, actions)

        # Detach only the target-network part so target_q_net params are not
        # updated, while eff_rewards stays in the graph so β receives gradients.
        with torch.no_grad():
            max_next_q = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_targets = eff_rewards + self.gamma * max_next_q * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
