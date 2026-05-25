"""Plot training-time metric trajectories (one curve per ckpt) for each
(weight, algo) combination — to diagnose training stability across weights.

Produces:
  figures/ckpt_trajectory/01_trajectory_system.png   (3 weight rows × 3 vtype cols, w/ co + w/o co per panel)
  figures/ckpt_trajectory/02_trajectory_t/e/J0.png   (same layout, per-ts)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(EXP_DIR)


WEIGHT_MAP = {
    "5-2-1": ("exp126", "exp127"),
    "5-3-1": ("exp134", "exp136"),
    "5-4-1": ("exp135", "exp137"),
}
VTYPES = ["car", "bus", "ambulance"]


def load_traj(exp_prefix: str) -> list[tuple[int, dict]]:
    files = sorted(glob.glob(f"./models/{exp_prefix}_*/*/trajectory/ep*.json"))
    out = []
    for f in files:
        try:
            d = json.load(open(f))
            out.append((int(d["_meta"]["episode"]), d))
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return out


def best_ckpt_episode(exp_prefix: str) -> Optional[int]:
    bm = glob.glob(f"./models/{exp_prefix}_*/*/best_metrics.json")
    if not bm: return None
    try:
        return int(json.load(open(bm[0])).get("_meta", {}).get("episode", -1))
    except Exception:
        return None


def trajectory_series(traj: list[tuple[int, dict]], scope: str, vtype: str,
                      metric: str) -> tuple[np.ndarray, np.ndarray]:
    eps, vals = [], []
    for ep, d in traj:
        try:
            v = d["metrics"][scope][vtype][metric]
            eps.append(ep)
            vals.append(float(v) if v is not None else np.nan)
        except (KeyError, TypeError):
            pass
    return np.array(eps), np.array(vals, dtype=float)


def plot_scope(scope: str, out_dir: str) -> None:
    fig, axes = plt.subplots(
        3, 3, figsize=(13, 9), squeeze=False, constrained_layout=True,
    )

    for i, (w, (co_exp, dq_exp)) in enumerate(WEIGHT_MAP.items()):
        co_traj = load_traj(co_exp)
        dq_traj = load_traj(dq_exp)
        co_best = best_ckpt_episode(co_exp)
        dq_best = best_ckpt_episode(dq_exp)

        for j, v in enumerate(VTYPES):
            ax = axes[i, j]
            co_ep, co_w = trajectory_series(co_traj, scope, v, "avg_wait")
            dq_ep, dq_w = trajectory_series(dq_traj, scope, v, "avg_wait")

            if len(co_ep) > 0:
                ax.plot(co_ep, co_w, "-o", markersize=3, linewidth=1.2,
                        color="C0", label=f"w/ co ({co_exp})")
            if len(dq_ep) > 0:
                ax.plot(dq_ep, dq_w, "-s", markersize=3, linewidth=1.2,
                        color="C1", label=f"w/o co ({dq_exp})")

            # Mark best-ckpt episodes
            if co_best is not None:
                ax.axvline(co_best, color="C0", linestyle="--", alpha=0.4, lw=1)
            if dq_best is not None:
                ax.axvline(dq_best, color="C1", linestyle="--", alpha=0.4, lw=1)

            ax.set_title(f"{w}  |  {v}_avg_wait", fontsize=10)
            ax.set_xlabel("training episode")
            ax.set_ylabel("avg_wait (s)")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")

    scope_label = ("system (all 3 intersections)"
                   if scope == "system" else f'intersection "{scope}"')
    fig.suptitle(
        f"Training-time metric trajectories (eval per ckpt, eval_seed=cfg)  @ {scope_label}\n"
        f"Dashed vertical lines = episode where best.pth was selected",
        fontsize=11,
    )
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/trajectory_{scope}.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures/ckpt_trajectory")
    args = parser.parse_args()

    for scope in ["system", "t", "e", "J0"]:
        plot_scope(scope, args.out_dir)

    print(f"\nDone — 4 figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
