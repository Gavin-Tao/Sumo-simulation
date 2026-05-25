"""Plot CoLight (w/ co) vs DQN (w/o co) training-time curves from wandb,
for each priority weight configuration (5-2-1, 5-3-1, 5-4-1).

Reads wandb logs (every 5 episodes) and plots:
  - amb_per_visit and bus_per_visit (key priority indicators)
  - smoothed with rolling window
  - one row per priority weight, two columns (amb / bus per-visit)

Output: figures/wandb_curves/01_cofg_vs_dqn_per_visit.png
"""
from __future__ import annotations
import argparse
import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(EXP_DIR)

# (weight_label, CoLight exp, DQN exp)
WEIGHT_CONFIGS = [
    ("5-2-1", "exp126_colightorig_1x3_521_avg_waiting_NS20bus_1amb_U_tuned",
              "exp127_1x3_521_avg_waiting_NS20bus_1amb_U_tuned"),
    ("5-3-1", "exp134_colightorig_1x3_531_avg_waiting_NS20bus_1amb_U_tuned",
              "exp136_1x3_531_avg_waiting_NS20bus_1amb_U_tuned"),
    ("5-4-1", "exp135_colightorig_1x3_541_avg_waiting_NS20bus_1amb_U_tuned",
              "exp137_1x3_541_avg_waiting_NS20bus_1amb_U_tuned"),
]


def fetch(exp_name: str, project: str = "sumo-rl-1x3") -> pd.DataFrame:
    api = wandb.Api()
    runs = list(api.runs(project, filters={"config.name": exp_name}))
    if not runs:
        return pd.DataFrame()
    run = sorted(runs, key=lambda r: r.created_at)[-1]
    hist = run.history(samples=10000)
    cols = {
        "amb_pv": "eval_system/ambulance/avg_stopped_time_per_visit",
        "bus_pv": "eval_system/bus/avg_stopped_time_per_visit",
        "car_pv": "eval_system/car/avg_stopped_time_per_visit",
    }
    avail = [c for c in cols.values() if c in hist.columns]
    if not avail:
        return pd.DataFrame()
    df = hist[avail].copy()
    df.columns = [k for k, v in cols.items() if v in avail]
    df = df.dropna(how="all").reset_index(drop=True)
    df["episode"] = (df.index + 1) * 5    # eval_interval=5
    return df


def smooth(series, window=10):
    return series.rolling(window, min_periods=1).mean().values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="figures/wandb_curves")
    p.add_argument("--smooth", type=int, default=10,
                   help="rolling window size for smoothing")
    args = p.parse_args()

    fig, axes = plt.subplots(3, 2, figsize=(13, 10),
                              squeeze=False, constrained_layout=True,
                              sharex=False)

    for i, (label, co_exp, dq_exp) in enumerate(WEIGHT_CONFIGS):
        co_df = fetch(co_exp)
        dq_df = fetch(dq_exp)
        if co_df.empty or dq_df.empty:
            continue

        for j, (metric, ylabel) in enumerate([
            ("amb_pv", "ambulance per-visit (s)"),
            ("bus_pv", "bus per-visit (s)")
        ]):
            ax = axes[i, j]
            # Raw (light) + smoothed (dark)
            co_ep = co_df["episode"].values
            dq_ep = dq_df["episode"].values
            ax.plot(co_ep, co_df[metric].values, color="C0", alpha=0.15, lw=0.6)
            ax.plot(co_ep, smooth(co_df[metric], args.smooth),
                    color="C0", lw=2.0, label="w/ co (CoLight)")
            ax.plot(dq_ep, dq_df[metric].values, color="C1", alpha=0.15, lw=0.6)
            ax.plot(dq_ep, smooth(dq_df[metric], args.smooth),
                    color="C1", lw=2.0, label="w/o co (DQN)")

            # For amb plot, also draw bus smoothed lines as reference (dashed)
            if metric == "amb_pv":
                ax.plot(co_ep, smooth(co_df["bus_pv"], args.smooth),
                        color="C0", lw=1.2, ls="--", alpha=0.6,
                        label="bus reference (CoLight)")
                ax.plot(dq_ep, smooth(dq_df["bus_pv"], args.smooth),
                        color="C1", lw=1.2, ls="--", alpha=0.6,
                        label="bus reference (DQN)")
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.1)
            else:
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.5)

            ax.set_xlabel("training episode")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.set_title(f"weights = {label}", fontsize=10, loc="left")
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Training-time priority enforcement from wandb logs (smoothed window=10)\n"
        "Left column: amb per-visit (lower=better, dashed=bus reference for inversion check)   "
        "Right column: bus per-visit (lower=better)",
        fontsize=11,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    save = f"{args.out_dir}/01_co_vs_dqn_per_visit.png"
    fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {save}")


if __name__ == "__main__":
    main()
