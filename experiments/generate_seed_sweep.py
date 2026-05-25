"""Run multi-seed evaluation of best.pth for the weight-sensitivity 6 configs.

For each (exp, weight, algo) in WEIGHT_SWEEP and each seed in SEEDS:
  - Loads best.pth + training config (num_seconds=1000 from cfg)
  - Runs one greedy eval episode with eval_seed=seed
  - Saves metrics to <ckpt_dir>/seed_sweep/seed_<N>.json

The plot script (plot_seed_sweep.py) reads these and produces mean±std plots.

Usage:
  python experiments/generate_seed_sweep.py
  python experiments/generate_seed_sweep.py --seeds 0,1,2  --exps exp126 exp127
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import json
import time
from pathlib import Path

import torch
import yaml

os.environ["SUMO_RL_LIBSUMO"] = "0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(EXP_DIR)
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("SUMO_HOME not set")

from eval_unified import OBS_REGISTRY, load_agent, pick_action
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.metrics import EpisodeMetricsCollector

# (exp, weight_label, agent_type)
WEIGHT_SWEEP = [
    ("exp126", "5-2-1", "orico"),
    ("exp127", "5-2-1", "dqn"),
    ("exp134", "5-3-1", "orico"),
    ("exp136", "5-3-1", "dqn"),
    ("exp135", "5-4-1", "orico"),
    ("exp137", "5-4-1", "dqn"),
]
DEFAULT_SEEDS = list(range(1000, 1010))   # 10 seeds, kept separate from training seeds


def find_run_dir(exp_prefix: str) -> Path | None:
    matches = sorted(glob.glob(f"./models/{exp_prefix}_*/*"))
    matches = [Path(m) for m in matches if Path(m).is_dir()]
    return matches[-1] if matches else None


def run_single_seed(cfg: dict, ckpt_path: Path, agent_type: str,
                    seed: int, device: torch.device,
                    n_heads: int = 2) -> dict:
    obs_class    = OBS_REGISTRY[cfg["observation_class"]]
    neighbor_map = cfg.get("neighbor_map", {})

    env = SumoEnvironment(
        net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"],
        out_csv_name=None, use_gui=False,
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5), max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=cfg.get("single_agent", False),
        yellow_time=cfg.get("yellow_time", 2), delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"], observation_class=obs_class,
        sumo_seed=seed,
    )
    initial_states = env.reset(seed)
    obs_dim     = env.observation_space.shape[0]
    ts_lane_map = {ts: env.traffic_signals[ts].signal_controlled_lanes for ts in env.ts_ids}
    always_green = set().union(*(env.traffic_signals[ts].always_green_lanes for ts in env.ts_ids))

    agent = load_agent(agent_type, str(ckpt_path), cfg, obs_dim,
                       env.action_space.n, device, n_heads)
    mc = EpisodeMetricsCollector(
        ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green,
    )
    done = {"__all__": False}
    while not done["__all__"]:
        mc.collect_step(env.sumo)
        actions = {ts: pick_action(agent_type, agent, ts,
                                   initial_states, neighbor_map, obs_dim)
                   for ts in env.ts_ids}
        initial_states, _, done, _ = env.step(action=actions)
    mc.collect_step(env.sumo)
    mc.finalize(env.sumo)
    env.close()
    return mc.summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="comma-separated list of seeds")
    p.add_argument("--exps", nargs="*", default=None,
                   help="restrict to these exp prefixes (default: all 6)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    sweep = [c for c in WEIGHT_SWEEP if args.exps is None or c[0] in args.exps]

    device = torch.device(f"cuda:{args.gpu}"
                          if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sweep: {len(sweep)} configs × {len(seeds)} seeds = {len(sweep) * len(seeds)} evals")

    total = len(sweep) * len(seeds)
    done_n = 0
    t0 = time.time()
    for exp, weight, agent_type in sweep:
        run_dir = find_run_dir(exp)
        if run_dir is None:
            print(f"  [{exp}] no run dir — SKIP")
            done_n += len(seeds); continue
        cfg_path = run_dir / "config.yaml"
        ckpt = run_dir / "best.pth"
        if not cfg_path.exists() or not ckpt.exists():
            print(f"  [{exp}] missing config.yaml or best.pth — SKIP")
            done_n += len(seeds); continue

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        n_heads = cfg.get("n_heads", 2)

        out_dir = run_dir / "seed_sweep"
        out_dir.mkdir(exist_ok=True)

        for seed in seeds:
            out_path = out_dir / f"seed_{seed:04d}.json"
            if out_path.exists() and not args.overwrite:
                print(f"  [{exp} {weight} {agent_type} seed={seed}] exists — skip")
                done_n += 1; continue
            t_start = time.time()
            try:
                summary = run_single_seed(cfg, ckpt, agent_type, seed,
                                          device, n_heads=n_heads)
            except Exception as e:
                print(f"  [{exp} seed={seed}] ERROR: {e}")
                done_n += 1; continue
            payload = {
                "_meta": {
                    "exp":          exp,
                    "weight":       weight,
                    "agent_type":   agent_type,
                    "seed":         seed,
                    "num_seconds":  cfg.get("num_seconds", 1000),
                    "ckpt":         "best.pth",
                },
                "metrics": summary,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, default=float)
            done_n += 1
            elapsed = time.time() - t_start
            eta = (time.time() - t0) / done_n * (total - done_n)
            print(f"  [{done_n:3d}/{total}] {exp} {weight} {agent_type} seed={seed} "
                  f"done in {elapsed:.1f}s  (ETA {eta/60:.1f} min)")

    print(f"\nAll done in {(time.time() - t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
