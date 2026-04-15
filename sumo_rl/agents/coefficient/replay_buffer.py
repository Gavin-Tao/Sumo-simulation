"""Replay buffer for CoeffDQN: stores augmented obs + per-direction neighbor rewards."""
import random
import collections
import numpy as np


class CoeffReplayBuffer:
    """Stores (aug_state, action, reward, next_aug_state, done, nb_rewards).

    aug_state:  own_obs concatenated with [up, down, left, right] neighbor obs
                (zeros for missing neighbors). Shape: [own_dim + 4 * nb_dim].
    nb_rewards: per-direction neighbor rewards [r_up, r_down, r_left, r_right].
                0.0 for directions with no neighbor.
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, nb_rewards: np.ndarray):
        """nb_rewards: float32 array of shape [4] — [up, down, left, right]."""
        self.buffer.append((state, action, reward, next_state, done, nb_rewards))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, nb_rewards = zip(*transitions)
        return (
            np.array(state,      dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done,
            np.array(nb_rewards, dtype=np.float32),   # shape [batch, 4]
        )

    def size(self) -> int:
        return len(self.buffer)
