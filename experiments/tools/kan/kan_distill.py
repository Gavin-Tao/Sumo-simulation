"""K2/K3 — fit KANs to the FRAP G/S targets, prune, symbolify.

Targets (from extract_frap_targets.py):
  g          : 27-dim slot features -> self-demand score
  s_merge    : 54-dim (x_m || x_n)  -> duel score, rel=merge
  s_crossing : 54-dim (x_m || x_n)  -> duel score, rel=crossing
Options:
  --core16          restrict to [is_green + 15 φ] dims (drop downstream/occ)
  --amb-oversample  duplicate rows with cnt_l5>0 (spline support in amb region)
  --shared-levels   structure test: G ≈ b + w_g·is_green + Σ_l α_l·φ(cnt,que,awt)
                    with ONE shared φ — if fidelity holds, α_l IS the policy's
                    internalized effective priority weight vector.
Outputs: model_<target>.pt (self-describing bundle), metrics.json, plots.

Read-only analysis tooling — trains no RL, touches no config.
Usage: python experiments/tools/kan/kan_distill.py --data <dir> --target g \
         [--steps 200] [--width 4] [--prune] [--symbolic] [--shared-levels]
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

CORE16 = list(range(16))                      # is_green + 5 levels × (cnt,que,awt)
AMB_CNT_IDX = 13                              # cnt_l5 within a 27-dim slot row


def load_bundle(path):
    """Rebuild a distilled model from its bundle (see save site)."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    if b["kind"] == "shared_g":
        model = SharedLevelG(grid=b["grid"])
    else:
        from kan import KAN
        model = KAN(width=b["width"], grid=b["grid"], k=b["k"], seed=0)
    model.load_state_dict(b["state_dict"])
    model.eval()
    return model, b


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


def load_xy(data_dir, target, core16):
    d = np.load(os.path.join(data_dir, "frap_targets.npz"))
    if target == "g":
        X, y = d["g_X"], d["g_y"]
        amb = X[:, AMB_CNT_IDX] > 0
        if core16:
            X = X[:, CORE16]
    else:
        rel = 2 if target == "s_merge" else 3
        sel = d["s_rel"] == rel
        Xm, Xn = d["s_Xm"][sel], d["s_Xn"][sel]
        amb = (Xm[:, AMB_CNT_IDX] > 0) | (Xn[:, AMB_CNT_IDX] > 0)
        if core16:
            Xm, Xn = Xm[:, CORE16], Xn[:, CORE16]
        X, y = np.concatenate([Xm, Xn], axis=1), d["s_y"][sel]
    return X.astype(np.float32), y.astype(np.float32), amb


class SharedLevelG(torch.nn.Module):
    """G ≈ b + w_g·is_green + Σ_l α_l · φ(cnt_l, que_l, awt_l), shared φ.
    α (5,) is the read-out of the policy's effective priority weights."""

    def __init__(self, grid=5, k=3, seed=0):
        super().__init__()
        from kan import KAN
        self.phi = KAN(width=[3, 3, 1], grid=grid, k=k, seed=seed)
        self.alpha = torch.nn.Parameter(torch.ones(5))
        self.w_g = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):                     # x: (B, 16) core dims
        is_green = x[:, :1]
        blocks = x[:, 1:16].reshape(-1, 5, 3)
        phi = self.phi(blocks.reshape(-1, 3)).reshape(-1, 5)
        return (self.b + self.w_g * is_green[:, 0]
                + (phi * self.alpha).sum(dim=1)).unsqueeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", choices=["g", "s_merge", "s_crossing"], default="g")
    ap.add_argument("--core16", action="store_true")
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lamb", type=float, default=1e-3)
    ap.add_argument("--amb-oversample", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=100_000)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--symbolic", action="store_true")
    ap.add_argument("--shared-levels", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # absolutize BEFORE chdir — a relative out re-resolves under itself at
    # save time (2026-07-03 K2 crash: all four fits trained fully, then
    # died at torch.save with "Parent directory does not exist")
    out = os.path.abspath(args.out or args.data)
    os.makedirs(out, exist_ok=True)
    os.chdir(out)                              # pykan writes ./model ckpts here

    X, y, amb = load_xy(args.data if os.path.isabs(args.data)
                        else os.path.join(_REPO, args.data), args.target,
                        args.core16 or args.shared_levels)
    manifest = json.load(open(os.path.join(
        args.data if os.path.isabs(args.data)
        else os.path.join(_REPO, args.data), "manifest.json")))
    names = manifest["feature_names"]
    if args.core16 or args.shared_levels:
        names = [names[i] for i in CORE16]
    if args.target != "g":
        names = [f"m:{n}" for n in names] + [f"n:{n}" for n in names]

    rng = np.random.RandomState(0)
    idx = np.arange(len(y))
    if args.amb_oversample > 1 and amb.any():
        idx = np.concatenate([idx] + [idx[amb]] * (args.amb_oversample - 1))
    rng.shuffle(idx)
    idx = idx[: args.max_rows]
    split = int(0.9 * len(idx))
    tr, te = idx[:split], idx[split:]
    ds = {"train_input": torch.tensor(X[tr]), "train_label": torch.tensor(y[tr, None]),
          "test_input": torch.tensor(X[te]), "test_label": torch.tensor(y[te, None])}
    print(f"target={args.target} rows={len(idx)} (amb rows raw={int(amb.sum())}) "
          f"dims={X.shape[1]}")

    metrics = {"target": args.target, "rows": len(idx),
               "amb_rows_raw": int(amb.sum()), "dims": int(X.shape[1])}
    if args.shared_levels:
        assert args.target == "g", "--shared-levels only applies to g"
        model = SharedLevelG(grid=args.grid, seed=0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for step in range(args.steps):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(ds["train_input"]),
                                                ds["train_label"])
            loss.backward(); opt.step()
            if step % 50 == 0:
                print(f"  step {step} mse {loss.item():.5f}")
        with torch.no_grad():
            yh_tr = model(ds["train_input"]).numpy()[:, 0]
            yh_te = model(ds["test_input"]).numpy()[:, 0]
        metrics["alpha"] = model.alpha.detach().tolist()
        metrics["alpha_normalized_l1"] = (model.alpha / model.alpha[0]
                                          ).detach().tolist()
        print("α (每等级有效权重, 归一到 l1):",
              [round(a, 3) for a in metrics["alpha_normalized_l1"]])
    else:
        from kan import KAN
        model = KAN(width=[X.shape[1], args.width, 1], grid=args.grid, k=3, seed=0)
        model.fit(ds, steps=args.steps, lamb=args.lamb)

        def _stage_r2(tag):
            # staged fidelity ledger (2026-07-08): keep a citable R² after
            # every destructive stage so a collapsed final formula can't
            # silently erase the earlier evidence (fit_g_core postmortem).
            with torch.no_grad():
                yh = model(ds["test_input"]).numpy()[:, 0]
            metrics[f"r2_test_{tag}"] = r2(ds["test_label"].numpy()[:, 0], yh)
            print(f"  [stage {tag}] r2_test = {metrics[f'r2_test_{tag}']:.4f}")

        _stage_r2("fit")
        if args.prune:
            try:
                model = model.prune()
                model.fit(ds, steps=max(20, args.steps // 4), lamb=args.lamb)
                _stage_r2("prune")
            except Exception as e:
                print("prune failed (pykan API):", e)
        if args.symbolic:
            try:
                model.auto_symbolic()
                model.fit(ds, steps=20, lamb=0.0)
                _stage_r2("symbolic")
                formula = model.symbolic_formula()
                metrics["formula"] = str(formula[0][0])
                print("symbolic:", metrics["formula"][:400])
            except Exception as e:
                print("symbolic failed (pykan API):", e)
        with torch.no_grad():
            yh_tr = model(ds["train_input"]).numpy()[:, 0]
            yh_te = model(ds["test_input"]).numpy()[:, 0]
        try:
            model.plot()
            import matplotlib.pyplot as plt
            plt.savefig(os.path.join(out, f"kan_{args.target}_splines.png"),
                        dpi=150, bbox_inches="tight")
        except Exception as e:
            print("plot skipped:", e)

    metrics["r2_train"] = r2(ds["train_label"].numpy()[:, 0], yh_tr)
    metrics["r2_test"] = r2(ds["test_label"].numpy()[:, 0], yh_te)
    # pykan modules hold lambdas -> not picklable; persist state_dict + the
    # constructor recipe (round-trip verified exact). Loader: load_bundle().
    if args.shared_levels:
        bundle = {"kind": "shared_g", "grid": args.grid,
                  "state_dict": model.state_dict()}
    else:
        # store the ACTUAL (post-prune) hidden width, not the args width —
        # prune changes topology and a stale width breaks load_bundle/K4
        # (2026-07-08). act_fun.0.coef is (in, hidden, n_coef).
        _hidden = int(model.state_dict()["act_fun.0.coef"].shape[1])
        bundle = {"kind": "kan", "width": [X.shape[1], _hidden, 1],
                  "grid": args.grid, "k": 3, "state_dict": model.state_dict()}
    bundle.update({"dims": (CORE16 if (args.core16 or args.shared_levels)
                            else list(range(X.shape[1] if args.target == "g"
                                            else X.shape[1] // 2))),
                   "target": args.target, "feature_names": names,
                   "shared_levels": args.shared_levels})
    torch.save(bundle, os.path.join(out, f"model_{args.target}.pt"))
    json.dump(metrics, open(os.path.join(out, f"metrics_{args.target}.json"), "w"),
              indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "formula"}, indent=1))


if __name__ == "__main__":
    main()
