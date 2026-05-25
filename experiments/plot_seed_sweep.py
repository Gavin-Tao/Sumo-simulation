"""Plot weight-sensitivity with mean ± std error bars across multiple seeds.

Reads <ckpt_dir>/seed_sweep/seed_<N>.json files produced by
generate_seed_sweep.py, then plots one figure per scope (system + per-ts).

Usage:
    python experiments/plot_seed_sweep.py
"""
from __future__ import annotations
import os
import json
import glob
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(EXP_DIR)


WEIGHT_MAP = {
    "5-2-1": ("exp126", "exp127"),  # (CoLight/orico, DQN)
    "5-3-1": ("exp134", "exp136"),
    "5-4-1": ("exp135", "exp137"),
}

METRICS = [
    "avg_speed",
    "avg_stopped_time",
    "avg_wait",
    "median_wait",      # NEW (robust to outliers)
    "max_wait",         # NEW (catches gridlock outliers)
    "avg_waiting_inc",  # NEW (local Δ-acc-wait, no downstream)
    "avg_stopped_time_per_visit",
    "xts_avg_stopped_time",
    "xts_avg_speed",
]
VTYPES = ["car", "bus", "ambulance"]
WEIGHTS = list(WEIGHT_MAP.keys())


def load_seed_metrics(exp_prefix: str) -> list[dict]:
    """Return list of (seed_metrics_dict, seed) from <run_dir>/seed_sweep/."""
    pattern = f"./models/{exp_prefix}_*/*/seed_sweep/seed_*.json"
    matches = sorted(glob.glob(pattern))
    out = []
    for m in matches:
        try:
            with open(m) as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return out


def get_metric(metric_data: dict, scope: str, vtype: str, metric: str) -> Optional[float]:
    try:
        v = metric_data["metrics"][scope][vtype][metric]
        return float(v) if v is not None else None
    except (KeyError, TypeError, ValueError):
        return None


def collect_stats(exp_prefix: str, scope: str, vtype: str, metric: str
                  ) -> tuple[Optional[float], Optional[float], int]:
    """Return (mean, std, n_valid_seeds) for this metric across seeds.

    **Paired filter**: a seed is INCLUDED only if `n_vehicles[ambulance] > 0`
    (i.e., at least one ambulance appeared during that seed's eval). This
    ensures cross-vtype comparisons use the SAME seed cohort and prevents
    Poisson-rare amb-empty seeds from biasing any vtype's statistics through
    zero-fallback values.

    Note: filter key is ambulance presence at the same `scope`, not the
    queried `vtype` — so bus/car stats are reported over the same 7 of 10
    seeds where amb is present, making amb vs bus paired comparison valid.
    """
    docs = load_seed_metrics(exp_prefix)
    vals = []
    for d in docs:
        # Paired filter: skip if ambulance absent (regardless of which vtype we query).
        n_amb = get_metric(d, scope, "ambulance", "n_vehicles")
        if n_amb is None or n_amb == 0:
            continue
        v = get_metric(d, scope, vtype, metric)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if not vals:
        return None, None, 0
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0), len(arr)


def plot_scope(scope: str, out_dir: str) -> None:
    n_rows = len(METRICS)
    n_cols = len(VTYPES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 2.4 * n_rows),
        squeeze=False, constrained_layout=True,
    )

    x_plot = list(range(len(WEIGHTS)))
    coverage = {"colight": 0, "dqn": 0}
    # Track per-amb n_valid for annotation (since amb gets filtered most often)
    amb_n_valid = {}    # (algo, weight) → n_valid_seeds

    for i, m in enumerate(METRICS):
        for j, v in enumerate(VTYPES):
            ax = axes[i, j]
            co_means, co_stds = [], []
            dq_means, dq_stds = [], []
            co_ns, dq_ns       = [], []
            for w in WEIGHTS:
                co_exp, dq_exp = WEIGHT_MAP[w]
                co_mean, co_std, co_n = collect_stats(co_exp, scope, v, m)
                dq_mean, dq_std, dq_n = collect_stats(dq_exp, scope, v, m)
                if i == 0 and j == 0:
                    coverage["colight"] += int(co_n > 0)
                    coverage["dqn"]     += int(dq_n > 0)
                if v == "ambulance" and i == 0:
                    amb_n_valid[("CoLight", w)] = co_n
                    amb_n_valid[("DQN", w)]     = dq_n
                co_means.append(co_mean); co_stds.append(co_std); co_ns.append(co_n)
                dq_means.append(dq_mean); dq_stds.append(dq_std); dq_ns.append(dq_n)

            co_m = np.array([np.nan if x is None else x for x in co_means], dtype=float)
            co_s = np.array([0.0    if x is None else x for x in co_stds],  dtype=float)
            dq_m = np.array([np.nan if x is None else x for x in dq_means], dtype=float)
            dq_s = np.array([0.0    if x is None else x for x in dq_stds],  dtype=float)

            ax.errorbar(x_plot, co_m, yerr=co_s, marker="o", linewidth=1.5,
                        capsize=4, color="C0", label="w/ co")
            ax.errorbar(x_plot, dq_m, yerr=dq_s, marker="s", linewidth=1.5,
                        capsize=4, color="C1", label="w/o co")

            ax.set_xticks(x_plot)
            ax.set_xticklabels(WEIGHTS)
            # Annotate n_valid_seeds for ambulance (most likely to be filtered)
            if v == "ambulance":
                tick_lbls = [f"{w}\n(n={co_ns[k]}/{dq_ns[k]})" for k, w in enumerate(WEIGHTS)]
                ax.set_xticklabels(tick_lbls, fontsize=7)
            ax.set_title(f"{m}  |  {v}", fontsize=9)
            ax.grid(alpha=0.3)
            if i == n_rows - 1 and v != "ambulance":
                ax.set_xlabel("Priority weights  (ambulance / bus / car)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")

    scope_label = ("system (all 3 intersections aggregated)"
                   if scope == "system" else f'intersection "{scope}"')
    fig.suptitle(
        f"[SEED SWEEP] Weight sensitivity (mean±std over 10 seeds, short eval) @ {scope_label}\n"
        f"Seeds with n_vehicles=0 for given vtype are EXCLUDED (e.g., amb-empty Poisson seeds)\n"
        f"Bottom row tick labels show (n_valid_CoLight/n_valid_DQN) for ambulance columns",
        fontsize=11,
    )
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/02_weight_seed_sweep_{scope}.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}   (CoLight cov: {coverage['colight']}/3, DQN cov: {coverage['dqn']}/3)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="figures/seed_sweep_best",
                        help="output folder (relative to experiments/)")
    args = parser.parse_args()

    for scope in ["system", "t", "e", "J0"]:
        plot_scope(scope, args.out_dir)

    print(f"\nDone — 4 figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
