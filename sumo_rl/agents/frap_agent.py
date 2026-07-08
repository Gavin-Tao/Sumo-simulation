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


class MTTQNet(FRAPQNet):
    """Movement-Token Transformer variant (arch: mtt). Inserts a
    relation-biased self-attention refinement over the 12 movement tokens
    between encode() and the UNCHANGED FRAP phase-scoring aggregation, so the
    multi-hop movement context feeds both the self-demand (g) and duel (s)
    heads while the phase-competition structure and the forward signature are
    preserved (drop-in for FRAPAgent). Relation type is an additive per-head
    attention bias (the FRAP relation prior, made soft/learnable); absent
    movements are key-masked. FRAPQNet stays byte-for-byte unchanged; the only
    new parameters are the attention block.
    """

    def __init__(self, header_dim, slot_dim, embed_dim=16, pair_dim=16,
                 k_max=11, n_heads=4, n_layers=2):
        super().__init__(header_dim, slot_dim, embed_dim, pair_dim, k_max)
        self.n_heads = n_heads
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "attn": torch.nn.MultiheadAttention(
                    embed_dim, n_heads, batch_first=True),
                "ln1": torch.nn.LayerNorm(embed_dim),
                "ff": torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, 2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Linear(2 * embed_dim, embed_dim)),
                "ln2": torch.nn.LayerNorm(embed_dim),
            }) for _ in range(n_layers)])
        # rel in {-1..3} -> +1 -> {0..4}; per-head additive attention bias
        self.rel_bias = torch.nn.Embedding(5, n_heads)
        torch.nn.init.zeros_(self.rel_bias.weight)   # start = plain attention

    def refine(self, d, rel, exist):
        # Single float attn_mask carrying BOTH the per-head relation bias and
        # a -inf block on absent-movement KEY columns. Folding the padding into
        # the float mask (instead of a separate bool key_padding_mask) keeps the
        # two mask types identical — avoids torch's mismatched-mask deprecation
        # path and is future-proof; verified equivalent (batch/padded isolation
        # tests). A real query always sees >=1 present key (its own), so no row
        # is fully -inf; nan_to_num guards the degenerate all-absent case only.
        B = d.shape[0]
        bias = self.rel_bias(rel + 1).permute(0, 3, 1, 2)          # (B,H,12,12)
        key_block = torch.zeros_like(exist).masked_fill(
            exist < 0.5, float("-inf"))                            # (B,12)
        bias = bias + key_block.view(B, 1, 1, 12)                  # mask pad keys
        attn_mask = bias.reshape(B * self.n_heads, 12, 12)
        for L in self.layers:
            a, _ = L["attn"](d, d, d, attn_mask=attn_mask, need_weights=False)
            a = torch.nan_to_num(a)      # degenerate all-absent query -> 0;
            d = L["ln1"](d + a)          # ignored downstream by pm/exist anyway
            d = L["ln2"](d + L["ff"](d))
        return d

    def forward(self, x, pm, rel, exist):
        d = self.refine(self.encode(x), rel, exist)
        g = self.g_head(d).squeeze(-1)
        s = self.duel_scores(d, rel)
        q_self = (g.unsqueeze(1) * pm).sum(-1)
        cand = pm.unsqueeze(-1) * (rel >= 2).float().unsqueeze(1)
        masked = s.unsqueeze(1).masked_fill(cand == 0, NEG)
        duel_max = masked.max(dim=2).values
        suppressed = (cand.max(dim=2).values > 0).float() \
            * (1.0 - pm) * exist.unsqueeze(1)
        q_sup = (torch.where(suppressed > 0, duel_max,
                             torch.zeros_like(duel_max)) * suppressed).sum(-1)
        return q_self + q_sup


class MTTCoLightQNet(MTTQNet):
    """MTT + CoLight-style inter-junction coordination (arch: mtt_colight).
    Adds neighbour graph-attention on top of MTTQNet: the own junction is
    encoded by MTT into 12 movement tokens; up to 4 neighbour observations are
    each encoded to a summary, attention-weighted by their relevance to the own
    summary (missing neighbours masked, exactly as CoLightQNet), aggregated into
    a coordination context, and injected into every own movement token before
    the UNCHANGED FRAP phase-scoring aggregation. Neighbour dims equal own obs
    dims (junction-independent perphase obs). MTTQNet/FRAPQNet stay unchanged;
    the only new params are nb_encoder/attn_score/ctx_ln.
    forward(own_obs, neighbors, pm, rel, exist): neighbors (B,4,own_obs_dim),
    all-zero row = absent neighbour."""

    def __init__(self, header_dim, slot_dim, embed_dim=16, pair_dim=16,
                 k_max=11, n_heads=4, n_layers=2, n_neighbors=4):
        super().__init__(header_dim, slot_dim, embed_dim, pair_dim, k_max,
                         n_heads=n_heads, n_layers=n_layers)
        self.n_neighbors = n_neighbors
        own_dim = header_dim + 12 * slot_dim
        self.nb_encoder = torch.nn.Linear(own_dim, embed_dim)
        self.attn_score = torch.nn.Linear(2 * embed_dim, 1)   # own||nb -> scalar
        self.ctx_ln = torch.nn.LayerNorm(embed_dim)

    def _context(self, d, exist, neighbors):
        # own summary = mean over PRESENT movement tokens (masked pool)
        w = exist.unsqueeze(-1)                                     # (B,12,1)
        h_own = (d * w).sum(1) / w.sum(1).clamp(min=1.0)           # (B,E)
        nb_feat = torch.relu(self.nb_encoder(neighbors))          # (B,4,E)
        own_exp = h_own.unsqueeze(1).expand(-1, self.n_neighbors, -1)
        scores = self.attn_score(torch.cat([own_exp, nb_feat], -1))   # (B,4,1)
        missing = (neighbors.abs().sum(-1, keepdim=True) == 0)     # (B,4,1)
        scores = scores.masked_fill(missing, float("-inf"))
        wts = torch.nan_to_num(torch.softmax(scores, dim=1))      # all-absent -> 0
        return (wts * nb_feat).sum(1)                             # (B,E) context

    def forward(self, own_obs, neighbors, pm, rel, exist):
        d = self.refine(self.encode(own_obs), rel, exist)         # (B,12,E) own MTT
        c = self._context(d, exist, neighbors)                    # (B,E) coord ctx
        d = self.ctx_ln(d + c.unsqueeze(1))                       # inject into tokens
        g = self.g_head(d).squeeze(-1)
        s = self.duel_scores(d, rel)
        q_self = (g.unsqueeze(1) * pm).sum(-1)
        cand = pm.unsqueeze(-1) * (rel >= 2).float().unsqueeze(1)
        masked = s.unsqueeze(1).masked_fill(cand == 0, NEG)
        duel_max = masked.max(dim=2).values
        suppressed = (cand.max(dim=2).values > 0).float() \
            * (1.0 - pm) * exist.unsqueeze(1)
        q_sup = (torch.where(suppressed > 0, duel_max,
                             torch.zeros_like(duel_max)) * suppressed).sum(-1)
        return q_self + q_sup


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
                 grad_clip=1.0, target_clip_max=None, arch="frap",
                 mtt_heads=4, mtt_layers=2):
        self.device = torch.device(device)
        # arch: "frap" (default, unchanged) | "mtt" (movement-token transformer,
        # same forward contract). Absent key -> "frap" -> byte-identical legacy.
        def _mk():
            if arch == "mtt":
                return MTTQNet(header_dim, slot_dim, embed_dim, pair_dim, k_max,
                               n_heads=mtt_heads, n_layers=mtt_layers)
            if arch != "frap":
                raise ValueError(f"unknown frap arch: {arch!r}")
            return FRAPQNet(header_dim, slot_dim, embed_dim, pair_dim, k_max)
        self.q_net = _mk().to(self.device)
        self.target_q_net = _mk().to(self.device)
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


class MTTCoLightReplayBuffer:
    """Per-junction FIFO carrying each transition's own + neighbour snapshots
    (nb_obs / next_nb_obs, [4, own_dim]) so the coordination context can be
    recomputed at learn time WITHOUT a network-wide snapshot — same
    per-junction replay discipline as FRAPReplayBuffer, plus neighbours."""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, tls_id,
            nb_obs, next_nb_obs):
        self.buffer.append((state, action, reward, next_state, done, tls_id,
                            np.asarray(nb_obs, np.float32).copy(),
                            np.asarray(next_nb_obs, np.float32).copy()))

    def sample(self, batch_size):
        s, a, r, ns, d, t, nb, nnb = zip(*random.sample(self.buffer, batch_size))
        return (np.array(s), a, r, np.array(ns), d, t,
                np.array(nb), np.array(nnb))

    def size(self):
        return len(self.buffer)


class MTTCoLightAgent:
    """MTT + CoLight coordination agent. Mirrors FRAPAgent's attribute surface
    (q_net/target_q_net/optimizer/epsilon/eps_*/count/loss/grad_norm/q_mean/
    q_abs_max/start_train/use_per/replay_buffer/mini_size/batch_size) so
    train.py's epsilon/eval/checkpoint paths work verbatim; the ONLY interface
    differences are the neighbour args on take_action and the buffer."""

    def __init__(self, obs_dim, header_dim, slot_dim, tls_tensors, lr, gamma,
                 epsilon, target_update, capacity, mini_size, batch_size,
                 eps_start, eps_end, eps_decay, device, embed_dim=16,
                 pair_dim=16, k_max=11, use_double=True, loss_fn="huber",
                 grad_clip=1.0, target_clip_max=None, mtt_heads=4, mtt_layers=2,
                 n_neighbors=4):
        self.device = torch.device(device)
        self.n_neighbors = n_neighbors

        def _mk():
            return MTTCoLightQNet(header_dim, slot_dim, embed_dim, pair_dim,
                                  k_max, n_heads=mtt_heads, n_layers=mtt_layers,
                                  n_neighbors=n_neighbors)
        self.q_net = _mk().to(self.device)
        self.target_q_net = _mk().to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma, self.epsilon = gamma, epsilon
        self.target_update, self.count = target_update, 0
        self.mini_size, self.batch_size = mini_size, batch_size
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.use_double, self.loss_fn, self.grad_clip = use_double, loss_fn, grad_clip
        self.target_clip_max = None if target_clip_max is None else float(target_clip_max)
        self.use_per, self.start_train = False, False
        self.loss = self.grad_norm = self.q_mean = self.q_abs_max = None
        self.replay_buffer = MTTCoLightReplayBuffer(capacity)
        self._ids = sorted(tls_tensors)
        self._idx = {t: k for k, t in enumerate(self._ids)}
        self.obs_dim = obs_dim
        self.PM = torch.tensor(np.stack([tls_tensors[t]["pm"] for t in self._ids]),
                               dtype=torch.float, device=self.device)
        self.REL = torch.tensor(np.stack([tls_tensors[t]["rel"] for t in self._ids]),
                                dtype=torch.long, device=self.device)
        self.EXIST = torch.tensor(np.stack([tls_tensors[t]["exist"] for t in self._ids]),
                                  dtype=torch.float, device=self.device)
        self.MASK = torch.tensor(np.stack([tls_tensors[t]["mask"] for t in self._ids]),
                                 dtype=torch.bool, device=self.device)

    def _tensors(self, idx):
        return self.PM[idx], self.REL[idx], self.EXIST[idx], self.MASK[idx]

    def take_action(self, state, tls_id, nb_obs):
        """own state + nb_obs [n_neighbors, obs_dim] -> mask-valid phase index."""
        i = self._idx[tls_id]
        if np.random.random() < self.epsilon:
            return int(np.random.choice(np.flatnonzero(self.MASK[i].cpu().numpy())))
        own = torch.tensor(np.asarray(state, dtype=np.float32),
                           device=self.device).unsqueeze(0)
        nb = torch.tensor(np.asarray(nb_obs, dtype=np.float32),
                          device=self.device).unsqueeze(0)          # (1,4,obs)
        ii = torch.tensor([i], device=self.device)
        pm, rel, exist, mask = self._tensors(ii)
        with torch.no_grad():
            q = self.q_net(own, nb, pm, rel, exist).masked_fill(~mask, NEG)
        return int(q.argmax().item())

    def learn_step(self):
        if self.replay_buffer.size() <= max(self.mini_size, self.batch_size):
            return
        self.start_train = True
        s, a, r, ns, d, tids, nb, nnb = self.replay_buffer.sample(self.batch_size)
        idx = torch.tensor([self._idx[t] for t in tids], device=self.device)
        pm, rel, exist, mask = self._tensors(idx)
        states = torch.tensor(s, dtype=torch.float, device=self.device)
        next_states = torch.tensor(ns, dtype=torch.float, device=self.device)
        nbo = torch.tensor(nb, dtype=torch.float, device=self.device)
        nnbo = torch.tensor(nnb, dtype=torch.float, device=self.device)
        actions = torch.tensor(a, device=self.device).view(-1, 1)
        rewards = torch.tensor(r, dtype=torch.float, device=self.device).view(-1, 1)
        dones = torch.tensor(d, dtype=torch.float, device=self.device).view(-1, 1)
        q = self.q_net(states, nbo, pm, rel, exist).gather(1, actions)
        with torch.no_grad():
            if self.use_double:
                nq = self.q_net(next_states, nnbo, pm, rel, exist).masked_fill(~mask, NEG)
                na = nq.argmax(1, keepdim=True)
                mnq = self.target_q_net(next_states, nnbo, pm, rel, exist).gather(1, na)
            else:
                tq = self.target_q_net(next_states, nnbo, pm, rel, exist).masked_fill(~mask, NEG)
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
