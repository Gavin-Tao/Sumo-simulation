"""CoeffDQN: standard MLP Q-network + 4 learnable per-direction cooperative weights.

Observation: own_obs concatenated with [up, down, left, right] neighbor obs
             (zeros for missing neighbors).  Shape: own_dim + 4 * nb_dim.

Effective reward used in TD target:
    r_eff = r_local + sum(sigmoid(β_i) * r_nb_i)   for i in [up, down, left, right]

Each β_i is an independent learnable scalar:
  - sigmoid(β_i) → 1 : fully cooperative with direction i
  - sigmoid(β_i) → 0 : ignores direction i
  - missing neighbor   : r_nb_i = 0, contribution = 0 regardless of β_i

β is a 4-dim nn.Parameter jointly optimised with Q-network weights via Adam.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .replay_buffer import CoeffReplayBuffer
from sumo_rl.agents.noisy_linear import NoisyLinear


class Qnet(nn.Module):
    """Single-hidden-layer MLP; optionally uses NoisyLinear for parameter-space exploration."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, use_noisy: bool = False):
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


class CoeffDQN:
    """DQN with learnable β. Interface mirrors dqn_agent_txw.DQN."""

    def __init__(self, aug_state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon,
                 target_update, capacity, mini_size, batch_size,
                 eps_start, eps_end, eps_decay, device,
                 use_noisy: bool = False):
        self.aug_state_dim = aug_state_dim
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.use_noisy     = use_noisy
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

        self.q_net        = Qnet(aug_state_dim, hidden_dim, action_dim, use_noisy).to(device)
        self.target_q_net = Qnet(aug_state_dim, hidden_dim, action_dim, use_noisy).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Learnable per-direction cooperative weights [up, down, left, right]
        # (unconstrained; sigmoid applied in update to map to (0, 1))
        self.beta = nn.Parameter(torch.zeros(4, device=device))

        # β is optimised together with Q-network weights
        self.optimizer = torch.optim.Adam(
            list(self.q_net.parameters()) + [self.beta],
            lr=learning_rate,
        )
        self.replay_buffer = CoeffReplayBuffer(capacity)

    @property
    def beta_value(self) -> list:
        """Effective β values ∈ (0, 1) after sigmoid — [up, down, left, right]."""
        return torch.sigmoid(self.beta).tolist()

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
        nb_rewards  = torch.tensor(transition_dict['nb_rewards'],  dtype=torch.float32).to(self.device)  # [batch, 4]

        # Effective reward: weighted sum of per-direction neighbor rewards
        # sigmoid(beta) shape [4], nb_rewards shape [batch, 4]
        # gradient flows through beta automatically
        eff_rewards = rewards + (torch.sigmoid(self.beta) * nb_rewards).sum(dim=1, keepdim=True)

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

        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_q_net.reset_noise()
