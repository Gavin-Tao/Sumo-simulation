"""Plot sensitivity figures from best_metrics.json across exp122-137.

Reads ./models/<exp_name>/<timestamp>/best_metrics.json for each experiment,
picks the *latest* timestamp (assumes most-recent training run is the one
you want to plot), and produces 2 figures saved to experiments/figures/:

  01_bus_pct_sensitivity.png
      X = NS bus%  (5, 10, 20, 50, 100)
      Lines = {CoLight + priority, DQN + priority, DQN baseline (horizontal)}
      Panels = 6 metrics × 3 vTypes = 18 subplots

  02_priority_weight_sensitivity.png
      X = priority weight ('5-2-1', '5-3-1', '5-4-1')
      Lines = {CoLight, DQN}  (both with priority, just different weights)
      Panels = 6 metrics × 3 vTypes = 18 subplots
      (No baseline overlay; baseline only exists at NS 100% bus, not 10%.)

Usage:
    python experiments/plot_best_metrics.py
"""

from __future__ import annotations
import os
import json
import glob
import argparse
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

# Always chdir to experiments/ so relative paths in this file resolve correctly:
#   ./models/<exp>/... → experiments/models/...   (train.py writes there)
#   ./figures/...      → experiments/figures/...  (where we save output)
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(EXP_DIR)


# ── Experiment mapping ────────────────────────────────────────────────────────
# Bus% sensitivity (priority 5-2-1 condition, vary bus%)
BUS_PCT_MAP: dict = {
    100: ("exp122", "exp123"),  # (CoLight, DQN)
     50: ("exp124", "exp125"),
     20: ("exp126", "exp127"),
     10: ("exp128", "exp129"),
    # NS 5% bus (exp130 / exp131) intentionally excluded — not in current run.
}

# Priority weight sensitivity (NS 10% bus scenario)
WEIGHT_MAP: dict = {
    "5-2-1": ("exp128", "exp129"),
    "5-3-1": ("exp134", "exp136"),
    "5-4-1": ("exp135", "exp137"),
}

BASELINE_EXP = "exp133"  # DQN + PressLight + plain avg-waiting-time, NS 100% bus


# ── Metrics to plot ───────────────────────────────────────────────────────────
# 3 base + 3 new variants (Metric A: per_visit;  Metric B: xts_)
METRICS = [
    "avg_speed",                  # base 1
    "avg_stopped_time",           # base 2
    "avg_wait",                   # base 3
    "avg_stopped_time_per_visit", # Metric A
    "xts_avg_stopped_time",       # Metric B (stopped)
    "xts_avg_speed",              # Metric B (speed)
]
VTYPES = ["car", "bus", "ambulance"]


# ── Loading ───────────────────────────────────────────────────────────────────
def load_best_metrics(exp_prefix: str) -> Optional[dict]:
    """Find latest timestamp dir for this exp and load best_metrics.json. None if missing."""
    pattern = f"./models/{exp_prefix}_*/*/best_metrics.json"
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Pick latest by mtime
    latest = max(matches, key=os.path.getmtime)
    try:
        with open(latest) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_metric(bm: Optional[dict], scope: str, vtype: str, metric: str) -> Optional[float]:
    """Read one metric from a best_metrics.json dict.

    Args:
        scope:  one of 'system', 't', 'e', 'J0' (or any ts_id in the network)
        vtype:  one of 'car', 'bus', 'ambulance', 'all'
        metric: e.g. 'avg_speed', 'avg_stopped_time', 'xts_avg_speed'

    Note: per-ts scopes ('t', 'e', 'J0') only carry the *base* metrics
    (avg_speed, avg_stopped_time, avg_wait, avg_stop_events, stopped_time,
     n_stop_events, n_vehicles, throughput, avg_stopped_time, ...). The
    A/B variants (avg_stopped_time_per_visit, xts_*) only exist at scope='system'.
    """
    if bm is None:
        return None
    try:
        v = bm["metrics"][scope][vtype][metric]
        return float(v) if v is not None else None
    except (KeyError, TypeError, ValueError):
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────
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

    # Aggregate "data availability" stats for printing
    coverage = {"colight": 0, "dqn": 0, "baseline": 0}

    for i, m in enumerate(METRICS):
        for j, v in enumerate(VTYPES):
            ax = axes[i, j]
            # CoLight priority series
            co_vals = []
            for x in x_values:
                co_exp, _ = exp_map[x]
                bm = load_best_metrics(co_exp)
                if bm is not None and j == 0 and i == 0:
                    coverage["colight"] += 1
                co_vals.append(get_metric(bm, scope, v, m))
            # DQN priority series
            dq_vals = []
            for x in x_values:
                _, dq_exp = exp_map[x]
                bm = load_best_metrics(dq_exp)
                if bm is not None and j == 0 and i == 0:
                    coverage["dqn"] += 1
                dq_vals.append(get_metric(bm, scope, v, m))

            # X axis: numeric or categorical?
            x_plot = list(range(len(x_values)))

            # Convert None → NaN so matplotlib draws a gap instead of erroring
            co_arr = np.array([np.nan if v is None else v for v in co_vals], dtype=float)
            dq_arr = np.array([np.nan if v is None else v for v in dq_vals], dtype=float)
            ax.plot(x_plot, co_arr, marker="o", linewidth=1.8,
                    color="C0", label="w/ co")
            ax.plot(x_plot, dq_arr, marker="s", linewidth=1.8,
                    color="C1", label="w/o co")

            if include_baseline:
                base_val = get_metric(load_best_metrics(BASELINE_EXP), scope, v, m)
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

    scope_label = "system (all 3 intersections aggregated)" if scope == "system" else f'intersection "{scope}"'
    fig.suptitle(f"Sensitivity to {axis_name}  @ {scope_label}",
                 fontsize=13)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
    print(f"      CoLight runs found: {coverage['colight']}/{len(x_values)}, "
          f"DQN runs: {coverage['dqn']}/{len(x_values)}, "
          f"baseline: {coverage['baseline']}/1" if include_baseline
          else f"      CoLight runs found: {coverage['colight']}/{len(x_values)}, "
               f"DQN runs: {coverage['dqn']}/{len(x_values)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures",
                        help="where to save PNG figures (relative to experiments/)")
    args = parser.parse_args()

    # Scopes to plot. For each axis we emit one figure per scope so user can
    # see system-level vs each individual intersection's behavior.
    SCOPES = ["system", "t", "e", "J0"]

    # Note: A/B metric variants (avg_stopped_time_per_visit, xts_*) only exist
    # at scope="system". For per-ts scopes, those panels will show NaN/empty.

    for scope in SCOPES:
        # Figure 1: NS bus% sensitivity (priority condition vs DQN baseline)
        plot_sensitivity(
            axis_name="NS bus %",
            x_label="NS bus%  (rest is car, on NS lane 1)",
            x_values=sorted(BUS_PCT_MAP.keys()),
            exp_map=BUS_PCT_MAP,
            save_path=f"{args.out_dir}/01_bus_pct_{scope}.png",
            scope=scope,
            include_baseline=True,
        )

    for scope in SCOPES:
        # Figure 2: priority weight sensitivity (NS 20% bus scenario)
        plot_sensitivity(
            axis_name="Priority weight  (at NS 20% bus + 80% car scenario)",
            x_label="Priority weights  (car / bus / ambulance)",
            x_values=list(WEIGHT_MAP.keys()),
            exp_map=WEIGHT_MAP,
            save_path=f"{args.out_dir}/02_weight_{scope}.png",
            scope=scope,
            include_baseline=False,
        )

    print("\nDone — 8 figures saved (4 scopes × 2 axes):")
    print(f"  01_bus_pct_<scope>.png   (scope ∈ {SCOPES})")
    print(f"  02_weight_<scope>.png    (scope ∈ {SCOPES})")


if __name__ == "__main__":
    main()
