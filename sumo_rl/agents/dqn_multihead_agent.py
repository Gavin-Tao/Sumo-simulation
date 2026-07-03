"""Multi-head DQN with decision-time BNF weighting (B-line, 2026-07-03).

Motivation (EXP213-215 doc §6/§8 + KAN α readout): scalar 5-3-1 training
bakes the BNF weights into shared gradients where sparse classes get
diluted — exp211's effective weights came out l3=4.11 / l5=3.29 against
nominal 3 / 5. Fix: DISENTANGLE the value estimates (one head per
priority level, each trained ONLY on its level's reward channel) and
apply the BNF weights EXACTLY, at action-selection time:

    a* = argmax_a  Σ_l  w_l · Q_l(s, a)        (w = the BNF table itself)

Faithfulness of the ratio semantics is then by construction — the "5"
is multiplied in the argmax, never learned. Swapping the BNF table at
deployment changes behaviour immediately (approximately: head values
were learned under the old joint policy — small swaps ≈ zero-shot,
large swaps want a short fine-tune).

Correctness discipline (same class as the presence-mask D1 rule): the
TD target must be JOINT-POLICY CONSISTENT — every head evaluates the
SAME next action a*, chosen by the weighted score at s' (masked), NOT
its own per-head max. Otherwise each head learns the value of a
different policy and the decision-time sum is meaningless.

Anchor property: with w = the level values (default), the composite
Σ w_l Q_l estimates exactly the return of the scalar priority-avg-
waiting reward (the vector channels are its exact decomposition), so
this agent optimises the SAME objective as the scalar B-line — only
the estimation is disentangled.

Head space is FIXED at 5 levels (PRIORITY_LEVELS, zero-shot invariant):
absent levels receive identically-zero rewards, learn ≈ constant heads,
and cannot sway the weighted argmax — any BNF table works unchanged.

Interface mirrors sumo_rl.agents.dqn_agent_txw.DQN (take_action(state,
mask), update(transition_dict), replay_buffer/.mini_size/.epsilon/
.count/.q_net/.target_q_net/.optimizer, loss/grad_norm/q_mean/q_abs_max
diagnostics) so train.py's act/store/update/eval/checkpoint paths work
verbatim. PER and q_net_factory are intentionally unsupported.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .dqn_agent_txw import ReplayBuffer

N_LEVELS = 5


class MultiHeadQnet(torch.nn.Module):
    """Shared trunk (same shape as the B-line Qnet) + 5 per-level heads."""

    def __init__(self, state_dim, hidden_dim, action_dim, n_heads=N_LEVELS):
        super().__init__()
        self.n_heads = n_heads
        self.action_dim = action_dim
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, n_heads * action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).view(-1, self.n_heads, self.action_dim)  # (B,H,A)


class DQNMultiHead:
    """Per-level value heads + decision-time weighted argmax."""

    def __init__(self, starting_state, state_space, hidden_dim, action_space,
                 learning_rate, gamma, epsilon, target_update, capacity,
                 mini_size, batch_size, eps_start, eps_end, eps_decay, device,
                 weights=None, use_double=False, loss_fn="mse",
                 target_clip_max=None, grad_clip=None):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.q_net = MultiHeadQnet(state_space, hidden_dim,
                                   action_space).to(device)
        self.target_q_net = MultiHeadQnet(state_space, hidden_dim,
                                          action_space).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.capacity = capacity
        self.replay_buffer = ReplayBuffer(capacity)
        self.mini_size = mini_size
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.grad_clip = grad_clip
        self.target_clip_max = (None if target_clip_max is None
                                else float(target_clip_max))
        self.use_double = use_double
        self.use_per = False                     # train.py update-path switch
        assert loss_fn in ("mse", "huber"), loss_fn
        self.loss_fn = loss_fn
        # BNF weights, applied at DECISION time (default: bucket == weight)
        w = np.asarray(weights if weights is not None
                       else np.arange(1, N_LEVELS + 1), dtype=np.float64)
        assert w.shape == (N_LEVELS,), w.shape
        self.weights = w
        self._w_t = torch.tensor(w, dtype=torch.float, device=device)
        self.count = 0
        self.device = device
        self.loss = None
        self.grad_norm = None
        self.q_mean = None       # composite Σw·Q(s,a) mean (scalar-comparable)
        self.q_abs_max = None    # composite |Q| max — divergence sentinel
        self.start_train = False

    # ── act ────────────────────────────────────────────────────────────────
    def take_action(self, state, mask=None):
        """ε-greedy over the weighted composite score; mask semantics
        identical to DQN.take_action (invalid never chosen/stored)."""
        if np.random.random() < self.epsilon:
            if mask is not None:
                return int(np.random.choice(np.flatnonzero(mask)))
            return int(np.random.randint(self.action_space))
        state_t = torch.tensor([state], dtype=torch.float).to(self.device)
        with torch.no_grad():
            q = self.q_net(state_t)[0]                       # (H,A)
            score = torch.mv(q.t(), self._w_t)               # (A,)
        if mask is not None:
            score = score.masked_fill(
                ~torch.as_tensor(mask, dtype=torch.bool,
                                 device=self.device), -1e9)
        return int(score.argmax().item())

    # ── joint-policy-consistent targets (factored for hand-math tests) ────
    def _targets(self, rewards, next_states, dones, next_mask_t=None):
        """y (B,H) = r + γ·Q_target(s', a*)·(1-done), a* = weighted-score
        argmax at s' (double: online scores pick, target evaluates)."""
        with torch.no_grad():
            src = self.q_net if self.use_double else self.target_q_net
            nq = src(next_states)                            # (B,H,A)
            scores = torch.einsum("bha,h->ba", nq, self._w_t)
            if next_mask_t is not None:
                scores = scores.masked_fill(~next_mask_t, -1e9)
            a_star = scores.argmax(1)                        # (B,)
            tq = self.target_q_net(next_states)              # (B,H,A)
            idx = a_star.view(-1, 1, 1).expand(-1, tq.shape[1], 1)
            next_v = tq.gather(2, idx).squeeze(2)            # (B,H)
        y = rewards + self.gamma * next_v * (1.0 - dones)
        if self.target_clip_max is not None:
            y = y.clamp(max=self.target_clip_max)
        return y

    # ── learn ──────────────────────────────────────────────────────────────
    def update(self, transition_dict):
        states = torch.tensor(np.asarray(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(np.asarray(transition_dict['rewards']),
                               dtype=torch.float).to(self.device)   # (B,H)
        assert rewards.dim() == 2 and rewards.shape[1] == N_LEVELS, \
            "multihead update needs per-level reward vectors"
        next_states = torch.tensor(np.asarray(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        next_mask_t = None
        if 'next_masks' in transition_dict:
            next_mask_t = torch.as_tensor(
                np.asarray(transition_dict['next_masks']),
                dtype=torch.bool, device=self.device)

        q_all = self.q_net(states)                           # (B,H,A)
        idx = actions.view(-1, 1, 1).expand(-1, q_all.shape[1], 1)
        q_sa = q_all.gather(2, idx).squeeze(2)               # (B,H)
        y = self._targets(rewards, next_states, dones, next_mask_t)

        if self.loss_fn == "huber":
            dqn_loss = F.smooth_l1_loss(q_sa, y)
        else:
            dqn_loss = torch.mean((q_sa - y) ** 2)
        self.loss = dqn_loss.item()
        with torch.no_grad():
            comp = q_sa @ self._w_t                          # (B,) composite
            self.q_mean = comp.mean().item()
            self.q_abs_max = comp.abs().max().item()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        if self.grad_clip is not None:
            self.grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.grad_clip))
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
