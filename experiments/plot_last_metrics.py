"""Plot sensitivity figures from last_metrics.json (final ckpt) across exp122-137.

Mirror of plot_best_metrics.py but reads last_metrics.json (produced by
generate_last_metrics.py). Saves figures with `_last_` in filenames to keep
them separate from the best-ckpt figures.

  01_bus_pct_last_<scope>.png   X = NS bus%, scope ∈ {system, t, e, J0}
  02_weight_last_<scope>.png    X = priority weight (5-2-1 / 5-3-1 / 5-4-1)

Usage:
    python experiments/plot_last_metrics.py
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


# ── Experiment mapping (kept identical to plot_best_metrics.py) ──────────────
BUS_PCT_MAP: dict = {
    100: ("exp122", "exp123"),
     50: ("exp124", "exp125"),
     20: ("exp126", "exp127"),
     10: ("exp128", "exp129"),
}

# Priority weight sensitivity (NS 20% bus scenario — 5-3-1 / 5-4-1 only trained at 20%)
WEIGHT_MAP: dict = {
    "5-2-1": ("exp126", "exp127"),
    "5-3-1": ("exp134", "exp136"),
    "5-4-1": ("exp135", "exp137"),
}

BASELINE_EXP = "exp133"


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


_METRICS_SUFFIX = ""


def load_last_metrics(exp_prefix: str) -> Optional[dict]:
    """Find latest timestamp dir for this exp and load last_metrics{suffix}.json."""
    pattern = f"./models/{exp_prefix}_*/*/last_metrics{_METRICS_SUFFIX}.json"
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


def plot_sensitivity(
    axis_name: str,
    x_label: str,
    x_values: Sequence,
    exp_map: dict,
    save_path: str,
    scope: str = "system",
    include_baseline: bool = False,
) -> None:
    n_rows = len(METRICS)
    n_cols = len(VTYPES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 2.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    coverage = {"colight": 0, "dqn": 0, "baseline": 0}

    for i, m in enumerate(METRICS):
        for j, v in enumerate(VTYPES):
            ax = axes[i, j]
            co_vals, dq_vals = [], []
            for x in x_values:
                co_exp, dq_exp = exp_map[x]
                co_bm = load_last_metrics(co_exp)
                dq_bm = load_last_metrics(dq_exp)
                if i == 0 and j == 0:
                    if co_bm is not None: coverage["colight"] += 1
                    if dq_bm is not None: coverage["dqn"]     += 1
                co_vals.append(get_metric(co_bm, scope, v, m))
                dq_vals.append(get_metric(dq_bm, scope, v, m))

            x_plot = list(range(len(x_values)))
            co_arr = np.array([np.nan if v is None else v for v in co_vals], dtype=float)
            dq_arr = np.array([np.nan if v is None else v for v in dq_vals], dtype=float)
            ax.plot(x_plot, co_arr, marker="o", linewidth=1.8,
                    color="C0", label="w/ co")
            ax.plot(x_plot, dq_arr, marker="s", linewidth=1.8,
                    color="C1", label="w/o co")

            if include_baseline:
                base_val = get_metric(load_last_metrics(BASELINE_EXP), scope, v, m)
                if i == 0 and j == 0 and base_val is not None:
                    coverage["baseline"] = 1
                if base_val is not None:
                    ax.axhline(base_val, ls="--", color="gray", lw=1.2,
                               label="baseline")

            ax.set_xticks(x_plot)
            ax.set_xticklabels([str(x) for x in x_values])
            ax.set_title(f"{m}  |  {v}", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel(x_label)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")

    scope_label = ("system (all 3 intersections aggregated)"
                   if scope == "system" else f'intersection "{scope}"')
    fig.suptitle(f"[LAST ckpt] Sensitivity to {axis_name}  @ {scope_label}",
                 fontsize=13)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    print(f"      CoLight runs found: {coverage['colight']}/{len(x_values)}, "
          f"DQN runs: {coverage['dqn']}/{len(x_values)}"
          + (f", baseline: {coverage['baseline']}/1" if include_baseline else ""))


def main():
    global _METRICS_SUFFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures/last",
                        help="where to save PNG figures (relative to experiments/)")
    parser.add_argument("--suffix", default="",
                        help="metrics filename suffix, e.g. '_warmup20' to read "
                             "last_metrics_warmup20.json")
    args = parser.parse_args()
    _METRICS_SUFFIX = args.suffix

    SCOPES = ["system", "t", "e", "J0"]

    for scope in SCOPES:
        plot_sensitivity(
            axis_name="NS bus %",
            x_label="NS bus%  (rest is car, on NS lane 1)",
            x_values=sorted(BUS_PCT_MAP.keys()),
            exp_map=BUS_PCT_MAP,
            save_path=f"{args.out_dir}/01_bus_pct_last_{scope}.png",
            scope=scope,
            include_baseline=True,
        )

    for scope in SCOPES:
        plot_sensitivity(
            axis_name="Priority weight  (at NS 20% bus + 80% car scenario)",
            x_label="Priority weights  (ambulance / bus / car)",
            x_values=list(WEIGHT_MAP.keys()),
            exp_map=WEIGHT_MAP,
            save_path=f"{args.out_dir}/02_weight_last_{scope}.png",
            scope=scope,
            include_baseline=False,
        )

    print("\nDone — 8 figures saved (4 scopes × 2 axes):")
    print(f"  01_bus_pct_last_<scope>.png   (scope ∈ {SCOPES})")
    print(f"  02_weight_last_<scope>.png    (scope ∈ {SCOPES})")


if __name__ == "__main__":
    main()
