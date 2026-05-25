"""Plot training-time trajectory of avg_stopped_time_per_visit (the fair
cross-vtype metric) to verify priority preservation throughout training.

Key thesis question:  is amb < bus < car maintained for the WHOLE training,
not just at the best ckpt?

Produces:
  figures/ckpt_trajectory/02_per_visit_priority_<algo>.png
      3 rows × 1 col panels (one per weight), 3 lines (car/bus/amb) each.
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

WEIGHT_TO_COLIGHT = {"5-2-1": "exp126", "5-3-1": "exp134", "5-4-1": "exp135"}
WEIGHT_TO_DQN     = {"5-2-1": "exp127", "5-3-1": "exp136", "5-4-1": "exp137"}
VTYPE_COLORS      = {"car": "#888888", "bus": "C0", "ambulance": "C3"}
VTYPE_LABELS      = {"car": "car (k_v varies)", "bus": "bus (k_v=1)", "ambulance": "amb (k_v=3)"}


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


def trajectory_per_visit(traj: list[tuple[int, dict]], vtype: str
                          ) -> tuple[np.ndarray, np.ndarray]:
    eps, vals = [], []
    for ep, d in traj:
        try:
            v = d["metrics"]["system"][vtype]["avg_stopped_time_per_visit"]
            eps.append(ep)
            vals.append(float(v) if v is not None else np.nan)
        except (KeyError, TypeError):
            pass
    return np.array(eps), np.array(vals, dtype=float)


def plot_algo(algo_label: str, weight_to_exp: dict, out_dir: str) -> None:
    weights = list(weight_to_exp.keys())
    fig, axes = plt.subplots(
        len(weights), 1, figsize=(11, 3.4 * len(weights)),
        squeeze=False, constrained_layout=True, sharex=True,
    )

    for i, w in enumerate(weights):
        exp = weight_to_exp[w]
        traj = load_traj(exp)
        best_ep = best_ckpt_episode(exp)

        ax = axes[i, 0]
        ax_log = ax  # plotting on log scale to see amb clearly

        for vtype in ["car", "bus", "ambulance"]:
            ep, val = trajectory_per_visit(traj, vtype)
            if len(ep):
                ax.plot(ep, val, "-o", markersize=3, linewidth=1.4,
                        color=VTYPE_COLORS[vtype], label=VTYPE_LABELS[vtype])

        if best_ep is not None:
            ax.axvline(best_ep, color="black", linestyle="--", alpha=0.4, lw=1)
            ax.text(best_ep, ax.get_ylim()[1] * 0.95, f"best (ep{best_ep})",
                    rotation=90, va="top", fontsize=8, alpha=0.6)

        ax.set_ylabel("avg_stopped_time_per_visit (s)")
        ax.set_title(f"{algo_label}  |  weights = {w}",
                     fontsize=11, loc="left")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=3)
        if i == len(weights) - 1:
            ax.set_xlabel("training episode")

    fig.suptitle(f"[Trajectory] Per-visit stopped time across training — {algo_label}\n"
                 f"(priority preserved iff amb line ≤ bus line ≤ car line throughout)",
                 fontsize=11)
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/02_per_visit_priority_{algo_label.lower().replace(' ','_')}.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_priority_margin(weight_to_co: dict, weight_to_dq: dict, out_dir: str) -> None:
    """Plot priority MARGIN = bus_per_visit - amb_per_visit across training.

    Positive margin = amb is below bus (priority preserved).
    Negative margin = priority inversion at that episode.

    One row per weight, two columns (CoLight, DQN).
    """
    weights = list(weight_to_co.keys())
    fig, axes = plt.subplots(
        len(weights), 2, figsize=(11, 3 * len(weights)),
        squeeze=False, constrained_layout=True, sharex=True,
    )

    for i, w in enumerate(weights):
        for j, (algo, weight_to_exp) in enumerate([("CoLight", weight_to_co), ("DQN", weight_to_dq)]):
            exp = weight_to_exp[w]
            traj = load_traj(exp)
            best_ep = best_ckpt_episode(exp)
            ax = axes[i, j]

            ep_bus, bus = trajectory_per_visit(traj, "bus")
            ep_amb, amb = trajectory_per_visit(traj, "ambulance")
            # Align (both should have same ep list since loaded together)
            if len(ep_bus) and len(ep_amb) and (ep_bus == ep_amb).all():
                margin = bus - amb
                ax.axhline(0, color="black", linewidth=0.8)
                ax.fill_between(ep_bus, 0, margin, where=margin >= 0,
                                alpha=0.3, color="C2", label="priority preserved")
                ax.fill_between(ep_bus, 0, margin, where=margin < 0,
                                alpha=0.3, color="C3", label="priority inverted")
                ax.plot(ep_bus, margin, "-", linewidth=1.5, color="black",
                        label="bus−amb")
                if best_ep is not None:
                    ax.axvline(best_ep, color="black", linestyle="--", alpha=0.4, lw=1)

            ax.set_ylabel("bus−amb per-visit (s)")
            ax.set_title(f"{algo}  |  {w}", fontsize=10, loc="left")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")
            if i == len(weights) - 1:
                ax.set_xlabel("training episode")

    fig.suptitle("Priority margin (bus−amb per-visit) over training episode\n"
                 "Green = priority preserved (amb < bus). Red = priority inverted.",
                 fontsize=11)
    save_path = f"{out_dir}/03_priority_margin.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures/ckpt_trajectory")
    args = parser.parse_args()

    plot_algo("CoLight",  WEIGHT_TO_COLIGHT, args.out_dir)
    plot_algo("DQN",      WEIGHT_TO_DQN,     args.out_dir)
    plot_priority_margin(WEIGHT_TO_COLIGHT, WEIGHT_TO_DQN, args.out_dir)
    print(f"\nDone — figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
