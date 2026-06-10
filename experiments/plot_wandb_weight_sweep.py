"""Weight sensitivity plot (mirrors figures/best/02_weight_*.png format) but
data source = wandb logged eval history's last N evaluation rows.

For each (algo, weight) cell, mean±std is computed over the LAST N wandb-logged
eval rows (eval_interval=5 episodes → last N rows = last 5N training episodes).
This gives statistical confidence intervals analogous to multi-seed sweep,
but the variance source is "training-time policy drift" rather than seed.

Output: figures/wandb_weight_sweep/02_weight_wandb_<scope>.png
"""
from __future__ import annotations
import argparse
import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# ── Publication style (per scientific-figure-making skill) ──────────────────
PALETTE = {
    "blue_main":   "#0F4D92",   # CoLight (proposed / method of interest)
    "red_strong":  "#B64342",   # DQN (baseline / contrast)
    "neutral":     "#4D4D4D",
    "grid":        "#CFCECE",
}
COLOR_CO = PALETTE["blue_main"]
COLOR_DQ = PALETTE["red_strong"]

PUBLICATION_RCPARAMS = {
    "font.family":       ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size":         14,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "axes.linewidth":    1.6,
    "axes.labelsize":    13,
    "axes.titlesize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.frameon":    False,
    "legend.fontsize":   13,
    "svg.fonttype":      "none",
}

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(EXP_DIR)

WEIGHT_MAP = {
    "5-2-1": ("exp126_colightorig_1x3_521_avg_waiting_NS20bus_1amb_U_tuned",
              "exp127_1x3_521_avg_waiting_NS20bus_1amb_U_tuned"),
    "5-3-1": ("exp134_colightorig_1x3_531_avg_waiting_NS20bus_1amb_U_tuned",
              "exp136_1x3_531_avg_waiting_NS20bus_1amb_U_tuned"),
    "5-4-1": ("exp135_colightorig_1x3_541_avg_waiting_NS20bus_1amb_U_tuned",
              "exp137_1x3_541_avg_waiting_NS20bus_1amb_U_tuned"),
}

# Metrics wandb basic mode actually logs (no avg_wait)
# Note: dropped avg_stopped_time per user preference — keep the per_visit
# version (fair across vtypes with different route lengths)
METRICS = [
    "avg_stopped_time_per_visit",
    "xts_avg_stopped_time",
    "xts_avg_stop_events",
    "xts_avg_speed",
]
VTYPES = ["car", "bus", "ambulance"]
WEIGHTS = list(WEIGHT_MAP.keys())


def fetch_run_history(exp_name: str, project: str = "sumo-rl-1x3") -> pd.DataFrame:
    api = wandb.Api()
    runs = list(api.runs(project, filters={"config.name": exp_name}))
    if not runs:
        return pd.DataFrame()
    run = sorted(runs, key=lambda r: r.created_at)[-1]
    return run.history(samples=10000)


def collect_stats(hist: pd.DataFrame, scope: str, vtype: str, metric: str,
                  window: int) -> tuple:
    """Return (mean, std, n) of last `window` eval rows for given metric."""
    # Build column name based on scope
    if scope == "system":
        col = f"eval_system/{vtype}/{metric}"
    else:
        col = f"eval_{scope}/{vtype}/{metric}"

    if col not in hist.columns:
        return None, None, 0

    series = hist[col].dropna().reset_index(drop=True)
    series = series.tail(window)
    if len(series) < 2:
        return None, None, len(series)
    vals = series.values
    return float(vals.mean()), float(vals.std(ddof=1)), len(vals)


def plot_scope(histories: dict, scope: str, out_dir: str, window: int) -> None:
    """Publication-style figure with dedicated legend panel (no data overlap)."""
    # Apply publication rcParams locally for this figure
    with mpl.rc_context(PUBLICATION_RCPARAMS):
        n_rows = len(METRICS)
        n_cols = len(VTYPES)

        # GridSpec: 1 dedicated row for legend (top) + n_rows for data
        # height_ratios: legend row much smaller than data rows
        fig = plt.figure(figsize=(6.0 * n_cols, 3.6 * n_rows + 1.0))
        gs = GridSpec(n_rows + 1, n_cols, figure=fig,
                      height_ratios=[0.5] + [3.0] * n_rows,
                      hspace=0.45, wspace=0.30)

        # ── Legend panel (top row, spans all columns) ─────────────────────
        legend_ax = fig.add_subplot(gs[0, :])
        legend_ax.set_axis_off()
        legend_handles = [
            Line2D([0], [0], color=COLOR_CO, marker="o", lw=2.2, markersize=8,
                   label="w/ co  (CoLight, GAT-coordinated)"),
            Line2D([0], [0], color=COLOR_DQ, marker="s", lw=2.2, markersize=8,
                   label="w/o co  (DQN, independent)"),
        ]
        legend_ax.legend(handles=legend_handles, loc="center", ncol=2,
                         fontsize=14, frameon=False,
                         handletextpad=0.6, columnspacing=2.0)

        # ── Data panels ────────────────────────────────────────────────────
        x_plot = list(range(len(WEIGHTS)))
        for i, m in enumerate(METRICS):
            for j, v in enumerate(VTYPES):
                ax = fig.add_subplot(gs[i + 1, j])
                co_means, co_stds, co_ns = [], [], []
                dq_means, dq_stds, dq_ns = [], [], []
                for w in WEIGHTS:
                    co_exp, dq_exp = WEIGHT_MAP[w]
                    co_h = histories.get(co_exp, pd.DataFrame())
                    dq_h = histories.get(dq_exp, pd.DataFrame())
                    cm, cs, cn = collect_stats(co_h, scope, v, m, window)
                    dm, ds, dn = collect_stats(dq_h, scope, v, m, window)
                    co_means.append(cm); co_stds.append(cs); co_ns.append(cn)
                    dq_means.append(dm); dq_stds.append(ds); dq_ns.append(dn)

                co_m = np.array([np.nan if x is None else x for x in co_means], dtype=float)
                co_s = np.array([0.0 if x is None else x for x in co_stds], dtype=float)
                dq_m = np.array([np.nan if x is None else x for x in dq_means], dtype=float)
                dq_s = np.array([0.0 if x is None else x for x in dq_stds], dtype=float)

                # Data lines with error bars
                ax.errorbar(x_plot, co_m, yerr=co_s, marker="o", linewidth=2.0,
                            markersize=7, capsize=4, color=COLOR_CO, zorder=3)
                ax.errorbar(x_plot, dq_m, yerr=dq_s, marker="s", linewidth=2.0,
                            markersize=7, capsize=4, color=COLOR_DQ, zorder=3)

                # Horizontal padding so labels don't collide with y-axis
                ax.set_xlim(-0.6, len(WEIGHTS) - 1 + 0.6)

                # Expand y-axis to give label bands clear room
                ymin, ymax = ax.get_ylim()
                yrange = max(ymax - ymin, 1e-6)
                ax.set_ylim(ymin - 0.18 * yrange, ymax + 0.20 * yrange)
                yL, yH = ax.get_ylim()
                yR = yH - yL
                co_label_y = yH - 0.07 * yR
                dq_label_y = yL + 0.07 * yR

                # In-place numerical annotations (no bbox, color-matched)
                # CoLight values in top band, DQN values in bottom band
                for k in range(len(x_plot)):
                    if not np.isnan(co_m[k]):
                        s = co_s[k] if not np.isnan(co_s[k]) else 0.0
                        ax.text(x_plot[k], co_label_y,
                                f"{co_m[k]:.2f}\n±{s:.2f}",
                                ha="center", va="center",
                                fontsize=9.5, color=COLOR_CO,
                                fontweight="bold", linespacing=1.0)
                    if not np.isnan(dq_m[k]):
                        s = dq_s[k] if not np.isnan(dq_s[k]) else 0.0
                        ax.text(x_plot[k], dq_label_y,
                                f"{dq_m[k]:.2f}\n±{s:.2f}",
                                ha="center", va="center",
                                fontsize=9.5, color=COLOR_DQ,
                                fontweight="bold", linespacing=1.0)

                ax.set_xticks(x_plot)
                if v == "ambulance":
                    tick_lbls = [f"{w}\n(n={co_ns[k]}/{dq_ns[k]})"
                                 for k, w in enumerate(WEIGHTS)]
                    ax.set_xticklabels(tick_lbls, fontsize=9)
                else:
                    ax.set_xticklabels(WEIGHTS, fontsize=11)

                ax.set_title(f"{m}  |  {v}", fontsize=11, pad=8)
                ax.grid(alpha=0.25, color=PALETTE["grid"], zorder=0)
                if i == n_rows - 1:
                    ax.set_xlabel("Priority weights (amb / bus / car)",
                                  fontsize=11)

        # ── Figure title (placed above legend panel) ──────────────────────
        scope_label = ("system" if scope == "system"
                       else f'intersection "{scope}"')
        fig.suptitle(
            f"Weight Sensitivity @ {scope_label}   "
            f"(wandb last {window} evals, NS 20% bus, 1% amb U-route)",
            fontsize=15, fontweight="bold", y=0.995,
        )

        os.makedirs(out_dir, exist_ok=True)
        save_path = f"{out_dir}/02_weight_wandb_{scope}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"  ✓ Saved: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="figures/wandb_weight_sweep")
    p.add_argument("--window", type=int, default=50,
                   help="last N wandb eval rows to aggregate (default: 50)")
    args = p.parse_args()

    print(f"Fetching wandb histories for 6 exps...")
    histories = {}
    for w, (co_exp, dq_exp) in WEIGHT_MAP.items():
        for exp in (co_exp, dq_exp):
            if exp not in histories:
                h = fetch_run_history(exp)
                histories[exp] = h
                print(f"  ✓ {exp[:30]}...  {len(h)} rows")

    print(f"\nGenerating plots (window={args.window} evals = last {args.window*5} episodes)...")
    for scope in ["system", "t", "e", "J0"]:
        plot_scope(histories, scope, args.out_dir, args.window)

    print(f"\nDone — 4 figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
