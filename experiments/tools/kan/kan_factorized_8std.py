"""K2(8std) — the structure test: constrained (factorized) vs free KAN
distillation of a monolithic masked-8std DQN.

Hypothesis class (factorized):  Q_k ≈ Σ_m A_km · g(φ_m) + b·1[k == current]
  * g: ONE shared KAN over slot features (core16=15 φ dims here, no is_green
    in legacy layout — actually 15; or all 26) — the only learned function
  * A: fixed per-junction served-movement incidence (from the 8std meta)
  * b: single scalar hysteresis bias (the ONLY freedom outside the form)
Loss on VALID-ACTION-CENTERED Q values == pairwise Q-differences (removes
the monolithic net's action-independent V(s) baseline).

Modes:
  --mode factorized   the constrained hypothesis (structure test)
  --mode flat         free KAN baseline: full obs slots flattened -> 8 Q
                      (expected to be unreadable; fidelity reference only)
Metrics: centered-Q R² + masked action agreement. High factorized fidelity
=> the monolithic DQN implicitly learned a compositional structure; low =>
interpretability must be bought at architecture time (FRAP's case).

Usage: python kan_factorized_8std.py --data experiments/analysis/kan8_data \
         --mode factorized [--core15] [--steps 300] [--symbolic]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)

CORE15 = list(range(15))          # 5 levels × (cnt, que, awt) — legacy layout
NEG = -1e9


class FactorizedQ(torch.nn.Module):
    def __init__(self, in_dim, grid=5, k=3, width=3, seed=0):
        super().__init__()
        from kan import KAN
        self.g = KAN(width=[in_dim, width, 1], grid=grid, k=k, seed=seed)
        self.b_switch = torch.nn.Parameter(torch.zeros(1))

    def forward(self, slots, A, cur_onehot):
        """slots (B,12,d)  A (B,8,12)  cur_onehot (B,8) -> Q (B,8)."""
        B = slots.shape[0]
        g = self.g(slots.reshape(-1, slots.shape[-1])).reshape(B, 12)
        return torch.einsum("bkm,bm->bk", A, g) + self.b_switch * cur_onehot


class FlatQ(torch.nn.Module):
    def __init__(self, in_dim, grid=5, k=3, width=8, seed=0):
        super().__init__()
        from kan import KAN
        self.net = KAN(width=[in_dim, width, 8], grid=grid, k=k, seed=seed)

    def forward(self, slots, A, cur_onehot):
        return self.net(slots.reshape(slots.shape[0], -1))


def centered(q, mask):
    """center Q over VALID actions (== fitting pairwise differences)."""
    m = mask.float()
    mean = (q * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True)
    return (q - mean) * m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["factorized", "flat"], default="factorized")
    ap.add_argument("--core15", action="store_true")
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--amb-oversample", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=60_000)
    ap.add_argument("--symbolic", action="store_true",
                    help="factorized only: auto-symbolic the shared g")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data_dir = args.data if os.path.isabs(args.data) else os.path.join(_REPO, args.data)
    out = os.path.abspath(args.out or os.path.join(data_dir, f"fit8_{args.mode}"))
    os.makedirs(out, exist_ok=True)
    os.chdir(out)     # abs BEFORE chdir (same crash class as kan_distill K2)

    d = np.load(os.path.join(data_dir, "dqn8std_targets.npz"))
    manifest = json.load(open(os.path.join(data_dir, "manifest.json")))
    slots, q, mask = d["slots"], d["q"], d["mask"]
    cur, ts_idx, A_stack = d["cur"], d["ts_idx"], d["A"]
    if args.core15:
        slots = slots[:, :, CORE15]
    amb = (slots[:, :, 12] > 0).any(axis=1)

    rng = np.random.RandomState(0)
    idx = np.arange(len(q))
    if args.amb_oversample > 1 and amb.any():
        idx = np.concatenate([idx] + [idx[amb]] * (args.amb_oversample - 1))
    rng.shuffle(idx)
    idx = idx[: args.max_rows]
    split = int(0.9 * len(idx))

    def batchify(sel):
        return (torch.tensor(slots[sel]),
                torch.tensor(A_stack[ts_idx[sel]]),
                torch.nn.functional.one_hot(
                    torch.tensor(cur[sel], dtype=torch.long), 8).float(),
                torch.tensor(q[sel]), torch.tensor(mask[sel]))

    tr, te = idx[:split], idx[split:]
    model = (FactorizedQ(slots.shape[-1], args.grid, 3, args.width)
             if args.mode == "factorized"
             else FlatQ(12 * slots.shape[-1], args.grid, 3, max(args.width, 8)))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xtr = batchify(tr)
    bs = 4096
    for step in range(args.steps):
        sel = torch.randint(0, len(tr), (min(bs, len(tr)),))
        s, a, c, qt, m = (t[sel] for t in Xtr)
        opt.zero_grad()
        pred = centered(model(s, a, c), m)
        loss = torch.nn.functional.mse_loss(pred, centered(qt, m))
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.5f}")

    def evaluate(sel):
        s, a, c, qt, m = batchify(sel)
        with torch.no_grad():
            qp = model(s, a, c)
        cp, ct = centered(qp, m).numpy(), centered(qt, m).numpy()
        mm = m.numpy()
        ss_res = ((cp - ct) ** 2)[mm].sum()
        ss_tot = ((ct - ct[mm].mean()) ** 2)[mm].sum()
        agree = (np.where(mm, qp.numpy(), NEG).argmax(1)
                 == np.where(mm, qt.numpy(), NEG).argmax(1)).mean()
        return float(1 - ss_res / ss_tot), float(agree)

    r2_tr, ag_tr = evaluate(tr)
    r2_te, ag_te = evaluate(te)
    metrics = {"mode": args.mode, "core15": args.core15, "rows": len(idx),
               "amb_rows_raw": int(amb.sum()),
               "r2_centered_train": r2_tr, "r2_centered_test": r2_te,
               "action_agreement_train": ag_tr, "action_agreement_test": ag_te}
    if args.mode == "factorized":
        metrics["b_switch"] = float(model.b_switch.item())
        if args.symbolic:
            try:
                model.g.auto_symbolic()
                metrics["g_formula"] = str(model.g.symbolic_formula()[0][0])[:800]
            except Exception as e:
                print("symbolic failed:", e)
        try:
            model.g.plot()
            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(out, "g_splines.png"), dpi=150,
                        bbox_inches="tight")
        except Exception as e:
            print("plot skipped:", e)
    bundle = {"kind": f"fact8_{args.mode}", "in_dim": slots.shape[-1],
              "grid": args.grid, "width": args.width,
              "state_dict": model.state_dict(), "core15": args.core15,
              "feature_names": manifest["feature_names"]}
    torch.save(bundle, os.path.join(out, f"model_{args.mode}.pt"))
    json.dump(metrics, open(os.path.join(out, f"metrics_{args.mode}.json"), "w"),
              indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "g_formula"},
                     indent=1))


if __name__ == "__main__":
    main()
