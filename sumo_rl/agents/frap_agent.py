"""FRAP-enum agent: movement-duel network over enumerated protected phases.

Spec: experiments/analysis/FRAP_ENUM_DESIGN_2026-07-02.txt §2/§2.5.
    Q(p) = sum_{m in p} g(d_m)
         + sum_{n not in p} max_{m in p, rel(m,n)>=2} s_mn
(each suppressed movement priced exactly once; duels exist only on conflict
pairs rel in {2:merge, 3:crossing}; relation embedding = physical class).

Standalone module: nothing here is imported by the legacy DQN path.
"""
import collections
import random

import numpy as np
import torch
import torch.nn.functional as F

NEG = -1e9


class FRAPQNet(torch.nn.Module):
    """(B, header+12*slot_dim) obs -> (B, K_max) raw Q.

    Padded/invalid phase rows are garbage — the caller MUST mask with the
    junction's phase mask before argmax/max (same contract as train.py's
    masked-std mode)."""

    def __init__(self, header_dim, slot_dim, embed_dim=16, pair_dim=16, k_max=11):
        super().__init__()
        self.header_dim, self.slot_dim, self.k_max = header_dim, slot_dim, k_max
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(slot_dim, embed_dim), torch.nn.ReLU(),
            torch.nn.LayerNorm(embed_dim))
        self.g_head = torch.nn.Linear(embed_dim, 1)
        self.pair_fc = torch.nn.Linear(2 * embed_dim, pair_dim)
        self.rel_emb = torch.nn.Embedding(2, pair_dim)      # 0=merge, 1=crossing
        self.s_head = torch.nn.Linear(pair_dim, 1)

    def encode(self, x):
        B = x.shape[0]
        slots = x[:, self.header_dim:].reshape(B, 12, self.slot_dim)
        return self.enc(slots)                               # (B,12,E)

    def duel_scores(self, d, rel):
        """(B,12,E),(B,12,12)long -> (B,12,12); NEG on non-conflict pairs."""
        B, _, E = d.shape
        di = d.unsqueeze(2).expand(B, 12, 12, E)
        dj = d.unsqueeze(1).expand(B, 12, 12, E)
        h = self.pair_fc(torch.cat([di, dj], dim=-1))        # (B,12,12,P)
        conflict = rel >= 2                                   # (B,12,12)
        ridx = (rel.clamp(min=2) - 2).long()                  # {0,1}
        h = h * self.rel_emb(ridx)                            # relation modulation
        s = self.s_head(h).squeeze(-1)                        # (B,12,12)
        return s.masked_fill(~conflict, NEG)

    def forward(self, x, pm, rel, exist):
        """x (B,obs); pm (B,K,12) float; rel (B,12,12) long; exist (B,12) float."""
        d = self.encode(x)
        g = self.g_head(d).squeeze(-1)                        # (B,12)
        s = self.duel_scores(d, rel)                          # (B,12,12)
        q_self = (g.unsqueeze(1) * pm).sum(-1)                # (B,K)
        # cand[b,k,m,n] = 1 iff m in phase k AND (m,n) is a conflict pair
        cand = pm.unsqueeze(-1) * (rel >= 2).float().unsqueeze(1)     # (B,K,12,12)
        masked = s.unsqueeze(1).masked_fill(cand == 0, NEG)   # (B,K,12,12)
        duel_max = masked.max(dim=2).values                   # (B,K,12) over members m
        suppressed = (cand.max(dim=2).values > 0).float() \
            * (1.0 - pm) * exist.unsqueeze(1)                 # (B,K,12)
        q_sup = (torch.where(suppressed > 0, duel_max,
                             torch.zeros_like(duel_max)) * suppressed).sum(-1)
        return q_self + q_sup                                 # (B,K)


class FRAPReplayBuffer:
    """Uniform FIFO buffer; transitions carry the junction id so update can
    gather that junction's phase/relation tensors (PER deliberately absent —
    exp153 lesson, spec §7)."""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, tls_id):
        self.buffer.append((state, action, reward, next_state, done, tls_id))

    def sample(self, batch_size):
        s, a, r, ns, d, t = zip(*random.sample(self.buffer, batch_size))
        return np.array(s), a, r, np.array(ns), d, t

    def size(self):
        return len(self.buffer)


class FRAPAgent:
    """Drop-in agent for train.py's enum_frap branch. Exposes the same
    attribute surface the training loop reads from DQN (q_net/target_q_net/
    optimizer/epsilon/eps_*/count/loss/grad_norm/q_mean/q_abs_max/start_train/
    use_per/replay_buffer/mini_size/batch_size) so checkpointing, epsilon
    annealing and wandb diagnostics work unchanged."""

    def __init__(self, obs_dim, header_dim, slot_dim, tls_tensors, lr, gamma,
                 epsilon, target_update, capacity, mini_size, batch_size,
                 eps_start, eps_end, eps_decay, device, embed_dim=16,
                 pair_dim=16, k_max=11, use_double=True, loss_fn="huber",
                 grad_clip=1.0, target_clip_max=None):
        self.device = torch.device(device)
        self.q_net = FRAPQNet(header_dim, slot_dim, embed_dim, pair_dim,
                              k_max).to(self.device)
        self.target_q_net = FRAPQNet(header_dim, slot_dim, embed_dim, pair_dim,
                                     k_max).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma, self.epsilon = gamma, epsilon
        self.target_update, self.count = target_update, 0
        self.mini_size, self.batch_size = mini_size, batch_size
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.use_double, self.loss_fn, self.grad_clip = use_double, loss_fn, grad_clip
        # R1 stabilization parity with DQN (dqn_agent_txw.py): clamp TD target
        # at the domain bound (all-negative rewards -> true Q <= 0).
        self.target_clip_max = None if target_clip_max is None else float(target_clip_max)
        self.use_per, self.start_train = False, False
        self.loss = None
        self.grad_norm = None
        self.q_mean = None
        self.q_abs_max = None
        self.replay_buffer = FRAPReplayBuffer(capacity)
        self._ids = sorted(tls_tensors)
        self._idx = {t: k for k, t in enumerate(self._ids)}
        self.PM = torch.tensor(np.stack([tls_tensors[t]["pm"] for t in self._ids]),
                               dtype=torch.float, device=self.device)      # (T,K,12)
        self.REL = torch.tensor(np.stack([tls_tensors[t]["rel"] for t in self._ids]),
                                dtype=torch.long, device=self.device)      # (T,12,12)
        self.EXIST = torch.tensor(np.stack([tls_tensors[t]["exist"] for t in self._ids]),
                                  dtype=torch.float, device=self.device)   # (T,12)
        self.MASK = torch.tensor(np.stack([tls_tensors[t]["mask"] for t in self._ids]),
                                 dtype=torch.bool, device=self.device)     # (T,K)

    def _tensors(self, idx):
        return self.PM[idx], self.REL[idx], self.EXIST[idx], self.MASK[idx]

    def take_action(self, state, tls_id):
        """Returns a PHASE INDEX (dense green index) — always mask-valid."""
        i = self._idx[tls_id]
        if np.random.random() < self.epsilon:
            return int(np.random.choice(np.flatnonzero(self.MASK[i].cpu().numpy())))
        x = torch.tensor(np.asarray(state, dtype=np.float32),
                         device=self.device).unsqueeze(0)
        ii = torch.tensor([i], device=self.device)
        pm, rel, exist, mask = self._tensors(ii)
        with torch.no_grad():
            q = self.q_net(x, pm, rel, exist).masked_fill(~mask, NEG)
        return int(q.argmax().item())

    def learn_step(self):
        """Sample + one gradient step. No-op until the buffer holds both
        mini_size (warm-up convention) AND batch_size (sampling feasibility)."""
        if self.replay_buffer.size() <= max(self.mini_size, self.batch_size):
            return
        self.start_train = True
        s, a, r, ns, d, tids = self.replay_buffer.sample(self.batch_size)
        idx = torch.tensor([self._idx[t] for t in tids], device=self.device)
        pm, rel, exist, mask = self._tensors(idx)
        states = torch.tensor(s, dtype=torch.float, device=self.device)
        next_states = torch.tensor(ns, dtype=torch.float, device=self.device)
        actions = torch.tensor(a, device=self.device).view(-1, 1)
        rewards = torch.tensor(r, dtype=torch.float, device=self.device).view(-1, 1)
        dones = torch.tensor(d, dtype=torch.float, device=self.device).view(-1, 1)
        q = self.q_net(states, pm, rel, exist).gather(1, actions)
        with torch.no_grad():
            if self.use_double:
                nq = self.q_net(next_states, pm, rel, exist).masked_fill(~mask, NEG)
                na = nq.argmax(1, keepdim=True)
                mnq = self.target_q_net(next_states, pm, rel, exist).gather(1, na)
            else:
                tq = self.target_q_net(next_states, pm, rel, exist).masked_fill(~mask, NEG)
                mnq = tq.max(1)[0].view(-1, 1)
        tgt = rewards + self.gamma * mnq * (1 - dones)
        if self.target_clip_max is not None:
            tgt = tgt.clamp(max=self.target_clip_max)
        loss = F.smooth_l1_loss(q, tgt) if self.loss_fn == "huber" else F.mse_loss(q, tgt)
        self.loss = loss.item()
        with torch.no_grad():
            self.q_mean = q.mean().item()
            self.q_abs_max = q.abs().max().item()
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            self.grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.grad_clip))
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
