"""Pull eval-history time series from wandb and compute trailing-window stats.

This uses wandb's per-eval logs (every eval_interval=5 episodes), which is
much denser than the per-ckpt trajectory data (every 50 episodes). Gives
tighter mean±std estimates for "training-stable" performance.

Note: wandb basic mode does NOT log avg_wait, but DOES log:
  - eval_system/<vtype>/avg_stopped_time
  - eval_system/<vtype>/avg_stopped_time_per_visit
  - eval_<ts>/<vtype>/avg_stopped_time
  - eval_<ts>/<vtype>/avg_speed
  - eval_system/mean_reward, eval_system/best_reward
"""
from __future__ import annotations
import argparse
import wandb
import numpy as np
import pandas as pd
from scipy import stats

EXP_TO_TIMESTAMP_PREFIX = {
    "exp126": "exp126_colightorig_1x3_521_avg_waiting_NS20bus_1amb_U_tuned",
    "exp127": "exp127_1x3_521_avg_waiting_NS20bus_1amb_U_tuned",
    "exp134": "exp134_colightorig_1x3_531_avg_waiting_NS20bus_1amb_U_tuned",
    "exp135": "exp135_colightorig_1x3_541_avg_waiting_NS20bus_1amb_U_tuned",
    "exp136": "exp136_1x3_531_avg_waiting_NS20bus_1amb_U_tuned",
    "exp137": "exp137_1x3_541_avg_waiting_NS20bus_1amb_U_tuned",
}

WEIGHT_MAP = {
    "5-2-1": ("exp126", "exp127"),
    "5-3-1": ("exp134", "exp136"),
    "5-4-1": ("exp135", "exp137"),
}


def fetch_run_history(exp: str, entity_proj: str = "sumo-rl-1x3") -> pd.DataFrame | None:
    api = wandb.Api()
    name_prefix = EXP_TO_TIMESTAMP_PREFIX[exp]
    runs = list(api.runs(entity_proj, filters={"config.name": name_prefix}))
    if not runs:
        return None
    # Latest run (by created_at)
    run = sorted(runs, key=lambda r: r.created_at)[-1]
    hist = run.history(samples=10000)   # large enough to fetch all rows
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=50,
                   help="Number of most-recent eval rows to average over (default: 50)")
    p.add_argument("--project", default="sumo-rl-1x3")
    args = p.parse_args()

    print(f"=== Wandb trailing-window analysis (last {args.window} eval rows) ===\n")
    print(f"{'weight':7} {'algo':8}  "
          f"{'car_pv':>14} {'bus_pv':>14} {'amb_pv':>14}  "
          f"{'amb<bus p':>10}")
    print("-" * 90)

    for weight, (co_exp, dq_exp) in WEIGHT_MAP.items():
        for exp, algo in [(co_exp, "CoLight"), (dq_exp, "DQN")]:
            hist = fetch_run_history(exp, args.project)
            if hist is None:
                print(f"  {weight:5} {algo:8}  no wandb run found")
                continue

            # Find the per-visit columns (they exist if logging_mode=basic)
            metrics = {
                "car": "eval_system/car/avg_stopped_time_per_visit",
                "bus": "eval_system/bus/avg_stopped_time_per_visit",
                "amb": "eval_system/ambulance/avg_stopped_time_per_visit",
                "n_amb": "eval_system/ambulance/n_vehicles",
            }
            available = {k: c for k, c in metrics.items() if c in hist.columns}
            if "amb" not in available or "bus" not in available:
                print(f"  {weight:5} {algo:8}  missing per_visit columns "
                      f"(found: {list(available.keys())})")
                # Fallback: use avg_stopped_time (system total, not per-visit)
                metrics = {
                    "car": "eval_system/car/avg_stopped_time",
                    "bus": "eval_system/bus/avg_stopped_time",
                    "amb": "eval_system/ambulance/avg_stopped_time",
                    "n_amb": "eval_system/ambulance/n_vehicles",
                }
                available = {k: c for k, c in metrics.items() if c in hist.columns}

            df = hist[[c for c in available.values()]].copy()
            df.columns = list(available.keys())
            df = df.dropna(how="all").reset_index(drop=True)

            # Take last N eval rows
            window = df.tail(args.window).copy()
            # Filter rows where n_amb == 0 (paired filter)
            if "n_amb" in window.columns:
                window = window[window["n_amb"] > 0]
            n = len(window)
            if n < 2:
                print(f"  {weight:5} {algo:8}  too few valid rows (n={n})")
                continue

            car = window["car"].values
            bus = window["bus"].values
            amb = window["amb"].values
            t, p_two = stats.ttest_rel(amb, bus)
            p = p_two / 2 if t < 0 else 1 - p_two / 2

            def sig(p):
                return ("***" if p < 0.001 else "**" if p < 0.01 else
                        "*"   if p < 0.05  else "."  if p < 0.1  else "")

            print(f"  {weight:5} {algo:8} n={n:>3}  "
                  f"{car.mean():5.2f}±{car.std(ddof=1):4.2f}    "
                  f"{bus.mean():5.2f}±{bus.std(ddof=1):4.2f}    "
                  f"{amb.mean():5.2f}±{amb.std(ddof=1):4.2f}  "
                  f"{p:6.4f} {sig(p):>3}")

    print()
    print("  *** p<0.001    ** p<0.01    * p<0.05    . p<0.1")


if __name__ == "__main__":
    main()
