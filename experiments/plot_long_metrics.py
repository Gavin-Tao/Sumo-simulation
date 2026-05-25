"""Plot weight-sensitivity figures from long_metrics.json (best.pth + long sim).

Only weight axis is supported here (BUS_PCT scenarios have no long_eval yet —
they'd need their own long-eval yamls + run).

Saves to figures/long_eval_best/02_weight_long_<scope>.png

Usage:
    python experiments/plot_long_metrics.py
"""

from __future__ import annotations
import os
import json
import glob
import argparse
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(EXP_DIR)


# 5-3-1 / 5-4-1 only trained at NS 20% bus → weight sensitivity anchored at 20%
WEIGHT_MAP: dict = {
    "5-2-1": ("exp126", "exp127"),
    "5-3-1": ("exp134", "exp136"),
    "5-4-1": ("exp135", "exp137"),
}

METRICS = [
    "avg_speed",
    "avg_stopped_time",
    "avg_wait",                   # whole-trip wait — downstream-contaminated at per-ts
    "avg_waiting_inc",            # local Δ-acc-wait — clean per-ts (NEW)
    "avg_stopped_time_per_visit",
    "xts_avg_stopped_time",
    "xts_avg_speed",
    "xts_avg_waiting_inc",        # Metric B for local wait (NEW)
]
VTYPES = ["car", "bus", "ambulance"]


def load_long_metrics(exp_prefix: str) -> Optional[dict]:
    pattern = f"./models/{exp_prefix}_*/*/long_metrics.json"
    matches = glob.glob(pattern)
    if not matches:
        return None
    latest = max(matches, key=os.path.getmtime)
    try:
        with open(latest) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_metric(bm: Optional[dict], scope: str, vtype: str, metric: str) -> Optional[float]:
    if bm is None:
        return None
    try:
        v = bm["metrics"][scope][vtype][metric]
        return float(v) if v is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def get_n(bm: Optional[dict], scope: str, vtype: str) -> Optional[int]:
    return get_metric(bm, scope, vtype, "n_vehicles")


def plot_weight(scope: str, out_dir: str) -> None:
    n_rows = len(METRICS)
    n_cols = len(VTYPES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 2.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    x_values = list(WEIGHT_MAP.keys())

    n_amb_seen: list = []

    for i, m in enumerate(METRICS):
        for j, v in enumerate(VTYPES):
            ax = axes[i, j]
            co_vals, dq_vals = [], []
            for x in x_values:
                co_exp, dq_exp = WEIGHT_MAP[x]
                co_bm = load_long_metrics(co_exp)
                dq_bm = load_long_metrics(dq_exp)
                co_vals.append(get_metric(co_bm, scope, v, m))
                dq_vals.append(get_metric(dq_bm, scope, v, m))
                if i == 0 and j == 2 and v == "ambulance":
                    n_amb_seen += [get_n(co_bm, scope, v), get_n(dq_bm, scope, v)]

            x_plot = list(range(len(x_values)))
            co_arr = np.array([np.nan if v is None else v for v in co_vals], dtype=float)
            dq_arr = np.array([np.nan if v is None else v for v in dq_vals], dtype=float)
            ax.plot(x_plot, co_arr, marker="o", linewidth=1.8,
                    color="C0", label="w/ co")
            ax.plot(x_plot, dq_arr, marker="s", linewidth=1.8,
                    color="C1", label="w/o co")

            ax.set_xticks(x_plot)
            ax.set_xticklabels(x_values)
            ax.set_title(f"{m}  |  {v}", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("Priority weights  (ambulance / bus / car)")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")

    n_amb_clean = [int(n) for n in n_amb_seen if n is not None]
    n_amb_label = (f" — n_ambulance={min(n_amb_clean)}–{max(n_amb_clean)}/eval"
                   if n_amb_clean else "")

    scope_label = ("system (all 3 intersections aggregated)"
                   if scope == "system" else f'intersection "{scope}"')
    fig.suptitle(f"[LONG-EVAL best ckpt, num_seconds=21600] "
                 f"Sensitivity to Priority weight  @ {scope_label}{n_amb_label}",
                 fontsize=12)
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/02_weight_long_{scope}.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures/long_eval_best",
                        help="where to save PNG figures")
    args = parser.parse_args()

    for scope in ["system", "t", "e", "J0"]:
        plot_weight(scope, args.out_dir)

    print(f"\nDone — 4 figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
