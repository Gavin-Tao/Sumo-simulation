"""Convert eval_summary.json (from eval_unified.py) to long_metrics.json.

Reads eval_summary.json (flat keys like "eval/system/ambulance/avg_wait")
produced by run_long_evals.sh and writes a nested long_metrics.json next to
it, matching the schema of best_metrics.json / last_metrics.json so the plot
script can reuse the same loader.

Usage:
  python experiments/generate_long_metrics.py             # all exp122..137
  python experiments/generate_long_metrics.py exp126 exp127
"""

from __future__ import annotations
import os
import sys
import json
import glob
import argparse
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
os.chdir(EXP_DIR)

DEFAULT_EXPS = [f"exp{i}" for i in (122, 123, 124, 125, 126, 127, 128, 129,
                                     133, 134, 135, 136, 137)]


def latest_run_dir(exp_prefix: str) -> Path | None:
    matches = sorted(glob.glob(f"./models/{exp_prefix}_*/*"))
    matches = [Path(m) for m in matches if Path(m).is_dir()]
    return matches[-1] if matches else None


def unflatten(flat: dict) -> dict:
    """Turn {"eval/system/ambulance/avg_wait": v, ...} into nested {system:{ambulance:{avg_wait: v}}}."""
    out: dict = {}
    for k, v in flat.items():
        parts = k.split("/")
        if parts[0] == "eval":
            parts = parts[1:]
        if len(parts) < 3:
            continue   # skip non-scope keys like "eval/episode_reward"
        scope, vtype, metric = parts[0], parts[1], "/".join(parts[2:])
        out.setdefault(scope, {}).setdefault(vtype, {})[metric] = v
    return out


def process_exp(exp_prefix: str) -> bool:
    run_dir = latest_run_dir(exp_prefix)
    if run_dir is None:
        print(f"[{exp_prefix}] no run dir — skipped")
        return False
    src = run_dir / "eval_summary.json"
    if not src.exists():
        print(f"[{exp_prefix}] eval_summary.json missing — skipped")
        return False

    with open(src) as f:
        data = json.load(f)

    # Prefer episodes[0] (a single eval episode produced by run_long_evals.sh
    # with --episodes 1).  Fallback: summary {"mean": ..., "std": ...} flat dict.
    episodes = data.get("episodes", [])
    if episodes:
        flat = episodes[0]
    else:
        flat = {k: v["mean"] for k, v in data.get("summary", {}).items()}

    metrics = unflatten(flat)
    payload = {
        "_meta": {
            "source":      "long_eval (eval_unified.py num_seconds=21600)",
            "agent_type":  data.get("agent_type", "?"),
            "n_episodes":  len(episodes) if episodes else 1,
        },
        "metrics": metrics,
    }
    out = run_dir / "long_metrics.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[{exp_prefix}] → {out}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exps", nargs="*", default=DEFAULT_EXPS)
    args = p.parse_args()

    n_ok = 0
    for exp in args.exps:
        try:
            if process_exp(exp):
                n_ok += 1
        except Exception as e:
            print(f"[{exp}] ERROR: {e}")
    print(f"\nDone — wrote long_metrics.json for {n_ok} experiments")


if __name__ == "__main__":
    main()
