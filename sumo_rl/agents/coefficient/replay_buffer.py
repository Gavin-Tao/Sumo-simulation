"""Replay buffer for CoeffDQN: stores augmented obs + mean neighbor reward."""
import random
import collections
import numpy as np


class CoeffReplayBuffer:
    """Stores (aug_state, action, reward, next_aug_state, done, nb_reward_mean).

    aug_state: own_obs concatenated with [up, down, left, right] neighbor obs
               (zeros for missing neighbors). Shape: [own_dim + 4 * nb_dim].
    nb_reward_mean: mean reward of available neighbors at this step (float).
                    0.0 if the agent has no neighbors.
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, nb_reward_mean: float):
        self.buffer.append((state, action, reward, next_state, done, nb_reward_mean))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, nb_reward_mean = zip(*transitions)
        return (
            np.array(state,          dtype=np.float32),
            action,
            reward,
            np.array(next_state,     dtype=np.float32),
            done,
            np.array(nb_reward_mean, dtype=np.float32),
        )

    def size(self) -> int:
        return len(self.buffer)
