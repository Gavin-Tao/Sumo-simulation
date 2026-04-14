"""Replay buffer for CoLight: stores own obs + neighbor obs matrix."""
import random
import collections
import numpy as np


class CoLightReplayBuffer:
    """Stores (own_obs, action, reward, next_own_obs, done, nb_obs, next_nb_obs).

    nb_obs / next_nb_obs: np.array [4, nb_dim], zeros for missing neighbors.
    Neighbor order: [up, down, left, right].
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done,
            nb_obs: np.ndarray, next_nb_obs: np.ndarray):
        self.buffer.append((
            state, action, reward, next_state, done,
            nb_obs.copy(), next_nb_obs.copy(),
        ))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, nb_obs, next_nb_obs = zip(*transitions)
        return (
            np.array(state,       dtype=np.float32),
            action,
            reward,
            np.array(next_state,  dtype=np.float32),
            done,
            np.array(nb_obs,      dtype=np.float32),       # [B, 4, nb_dim]
            np.array(next_nb_obs, dtype=np.float32),       # [B, 4, nb_dim]
        )

    def size(self) -> int:
        return len(self.buffer)
