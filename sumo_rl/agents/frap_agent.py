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
