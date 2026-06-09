"""Prioritized Experience Replay buffer (Schaul et al. ICLR 2016).

Generic implementation: stores any (state, action, reward, next_state, done, ...) tuple,
samples proportional to |TD-error|^alpha, returns importance-sampling weights so the
gradient can be corrected for the non-uniform sampling bias.

Drop-in for sumo_rl/agents/replay_memory.py-style buffers used by
  - DQN          (5 fields)
  - CoeffDQN     (6 fields, +nb_rewards)
  - CoLightOrigDQN (7 fields, +nb_obs, +next_nb_obs)

Compared with the uniform ReplayBuffer:
  - sample() takes an extra `beta` arg and returns TWO extra arrays at the end:
        (..., weights[B] : float32, indices[B] : int64)
  - update_priorities(indices, td_errors) must be called after the agent computes
    TD errors, so priorities are refreshed for the just-trained transitions.

Sum-tree:
  - Tree storage is padded to the next power-of-2 of `capacity` for clean binary indexing.
  - Layout: array of size 2*tree_capacity, root at index 1,
            leaves at indices [tree_capacity .. 2*tree_capacity - 1].
  - Buffer slot i  ↔  tree leaf index  (i + tree_capacity).
  - sample: stratified — split [0, total_priority) into `batch_size` segments,
            uniformly sample one priority per segment, walk tree to find its leaf.
"""
import numpy as np


class PrioritizedReplayBuffer:
    """PER buffer; agnostic to the exact transition tuple shape."""

    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-6):
        """
        capacity: max number of transitions to store (FIFO eviction)
        alpha:    priority exponent. 0 → uniform sampling (= vanilla replay).
                  1 → strictly proportional to priority.  Default 0.6 (PER paper).
        eps:      tiny constant added to |TD-error| so priority never becomes exactly 0
                  (which would cause a transition to be permanently skipped).
        """
        assert capacity > 0
        self.capacity = int(capacity)
        self.alpha    = float(alpha)
        self.eps      = float(eps)

        # Transition storage
        self.buffer    = [None] * self.capacity
        self.size_     = 0
        self.next_idx  = 0

        # Sum-tree, padded to next power of 2
        self.tree_capacity = 1
        while self.tree_capacity < self.capacity:
            self.tree_capacity *= 2
        # tree[1] = total sum; leaves at [tree_capacity .. 2*tree_capacity - 1]
        self.tree = np.zeros(2 * self.tree_capacity, dtype=np.float64)

        # Initial priority for never-sampled transitions. Updated on every
        # update_priorities call so new transitions get current max priority.
        self.max_priority = 1.0

    # ── Public API ──────────────────────────────────────────────────────────
    def add(self, *transition):
        """Add a transition tuple. Variadic so it accepts whichever
        (s, a, r, s', d[, ...]) layout the wrapped buffer was using."""
        idx = self.next_idx
        self.buffer[idx] = tuple(transition)
        # New transition gets max-so-far priority → guaranteed to be sampled soon
        self._set_leaf(idx, self.max_priority ** self.alpha)
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size_    = min(self.size_ + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        """Stratified-prioritized sample.

        Returns a tuple in the SAME order as the wrapped buffer's sample(),
        with two extra arrays appended:
            (..., weights[batch] : float32,  indices[batch] : int64)
        """
        assert self.size_ >= batch_size, \
            f"PER buffer has only {self.size_} transitions, need {batch_size}"

        indices    = self._sample_indices(batch_size)
        priorities = np.array([self._leaf_value(i) for i in indices], dtype=np.float64)
        total      = float(self.tree[1])

        # Importance-sampling weights:  w_i = (1/N · 1/P(i))^beta, normalised
        sample_probs = priorities / total
        weights      = (self.size_ * sample_probs) ** (-beta)
        weights     /= weights.max()                 # normalise to [0,1]
        weights      = weights.astype(np.float32)

        # Unzip transitions component-wise (mirrors `zip(*transitions)` pattern in
        # the existing flat ReplayBuffer classes).
        transitions = [self.buffer[i] for i in indices]
        components  = list(zip(*transitions))
        out         = [np.asarray(c) for c in components]
        out.append(weights)
        out.append(np.asarray(indices, dtype=np.int64))
        return tuple(out)

    def update_priorities(self, indices, td_errors):
        """Call after each gradient step with the (per-sample) TD errors that
        came out of the network for the just-sampled batch."""
        for idx, err in zip(indices, td_errors):
            err = float(abs(err)) + self.eps
            self._set_leaf(int(idx), err ** self.alpha)
            if err > self.max_priority:
                self.max_priority = err

    def size(self) -> int:
        return self.size_

    # ── Sum-tree internals ──────────────────────────────────────────────────
    def _set_leaf(self, buffer_idx: int, value: float):
        """Update leaf value and propagate the delta to the root."""
        tree_idx = buffer_idx + self.tree_capacity
        change   = value - self.tree[tree_idx]
        self.tree[tree_idx] = value
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] += change
            tree_idx //= 2

    def _leaf_value(self, buffer_idx: int) -> float:
        return float(self.tree[buffer_idx + self.tree_capacity])

    def _sample_indices(self, batch_size: int):
        """Stratified prioritised sampling — one draw per equal-priority segment."""
        total   = float(self.tree[1])
        segment = total / batch_size
        out     = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = float(np.random.uniform(a, b))
            tree_idx   = self._find_leaf(s)
            buffer_idx = tree_idx - self.tree_capacity
            # Guard: numerical edge effects (e.g. all-zero priorities on first
            # samples before any add) could land in unfilled slots; clamp.
            if buffer_idx >= self.size_:
                buffer_idx = self.size_ - 1
            out.append(buffer_idx)
        return out

    def _find_leaf(self, s: float) -> int:
        """Walk sum-tree from root → leaf whose cumulative prefix sum spans s."""
        idx = 1
        while idx < self.tree_capacity:
            left  = 2 * idx
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s   -= self.tree[left]
                idx  = right
        return idx
