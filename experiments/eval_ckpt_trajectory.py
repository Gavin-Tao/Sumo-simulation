"""Evaluate every ckpt of given experiments to reconstruct the training-time
metric trajectory. Saves <ckpt_dir>/trajectory/ckpt_epNNNNN.json.

Uses identical conditions to training-time eval (eval_seed from config, num_seconds
from config) so the trajectory matches what train.py was tracking.

Usage:
  python experiments/eval_ckpt_trajectory.py
  python experiments/eval_ckpt_trajectory.py --exps exp134
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

# (exp_prefix, agent_type)
WEIGHT_EXPS = [
    ("exp126", "orico"),
    ("exp127", "dqn"),
    ("exp134", "orico"),
    ("exp136", "dqn"),
    ("exp135", "orico"),
    ("exp137", "dqn"),
]


def find_run_dir(exp_prefix: str) -> Path | None:
    matches = sorted(glob.glob(f"./models/{exp_prefix}_*/*"))
    matches = [Path(m) for m in matches if Path(m).is_dir()]
    return matches[-1] if matches else None


def run_one_eval(cfg: dict, ckpt_path: Path, agent_type: str,
                 device: torch.device, n_heads: int = 2) -> dict:
    obs_class    = OBS_REGISTRY[cfg["observation_class"]]
    neighbor_map = cfg.get("neighbor_map", {})
    eval_seed    = int(cfg.get("eval_seed", cfg.get("seed", 0)))

    env = SumoEnvironment(
        net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"],
        out_csv_name=None, use_gui=False,
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5), max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=cfg.get("single_agent", False),
        yellow_time=cfg.get("yellow_time", 2), delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"], observation_class=obs_class,
        sumo_seed=eval_seed,
    )
    initial_states = env.reset(eval_seed)
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
    p.add_argument("--exps", nargs="*", default=None,
                   help="restrict to these exp prefixes (default: all 6)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    sweep = [c for c in WEIGHT_EXPS if args.exps is None or c[0] in args.exps]
    device = torch.device(f"cuda:{args.gpu}"
                          if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    total_done = 0
    for exp, agent_type in sweep:
        run_dir = find_run_dir(exp)
        if run_dir is None:
            print(f"  [{exp}] no run dir — SKIP")
            continue
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            print(f"  [{exp}] no config.yaml — SKIP")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        n_heads = cfg.get("n_heads", 2)
        ckpts = sorted(run_dir.glob("ckpt_ep*.pth"))
        traj_dir = run_dir / "trajectory"
        traj_dir.mkdir(exist_ok=True)
        print(f"\n=== {exp} ({agent_type}) — {len(ckpts)} ckpts ===")

        for ckpt in ckpts:
            ep = int(ckpt.stem.split("ep")[-1])
            out_path = traj_dir / f"ep{ep:05d}.json"
            if out_path.exists() and not args.overwrite:
                total_done += 1
                continue
            t_s = time.time()
            try:
                summary = run_one_eval(cfg, ckpt, agent_type, device, n_heads=n_heads)
            except Exception as e:
                print(f"  ep={ep} ERROR: {e}")
                continue
            payload = {
                "_meta": {"exp": exp, "agent_type": agent_type, "episode": ep,
                          "eval_seed": int(cfg.get("eval_seed", cfg.get("seed", 0)))},
                "metrics": summary,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, default=float)
            total_done += 1
            dt = time.time() - t_s
            print(f"  ep={ep:5d}  done in {dt:.1f}s")
    print(f"\nAll done in {(time.time()-t0)/60:.1f} min ({total_done} evals)")


if __name__ == "__main__":
    main()
