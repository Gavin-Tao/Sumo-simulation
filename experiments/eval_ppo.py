"""Evaluation script for trained PPO models (stable-baselines3).

Usage:
    python experiments/eval_ppo.py --config experiments/configs/exp49_ppo_1x1_51_WEstr_only.yaml \\
                                    --ckpt models/exp49_ppo_1x1_51_WEstr_only/2026-03-30T12-00-00/ppo_ep00050.zip \\
                                    --episodes 10

Runs N evaluation episodes (deterministic=True), collects EpisodeMetricsCollector each episode,
then prints mean ± std for every metric. Optionally logs to wandb.
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import yaml

# ── Project root ──────────────────────────────────────────────────────────────
os.environ["SUMO_RL_LIBSUMO"] = "0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
probe = PROJECT_ROOT
while not (probe / "sumo_rl").exists() and probe.parent != probe:
    probe = probe.parent
PROJECT_ROOT = probe
sys.path.insert(0, str(PROJECT_ROOT))

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from stable_baselines3 import PPO
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.environment.observations import (
    DefaultObservationFunction,
    PressLightObservationFunction,
    PressLightNormObservationFunction,
    QueuePrioObservationFunction,
    CTBPriorityObservationFunction,
    PriorityObservationFunction,
)

OBS_REGISTRY = {
    "Default":        DefaultObservationFunction,
    "PressLight":     PressLightObservationFunction,
    "PressLightNorm": PressLightNormObservationFunction,
    "QueuePrio":      QueuePrioObservationFunction,
    "CTBPriority":    CTBPriorityObservationFunction,
    "Priority":       PriorityObservationFunction,
}


def run_eval(cfg: dict, ckpt_path: str, n_episodes: int, use_gui: bool,
             wandb_log: bool = False, warmup_steps: int = 0, fixed_seed: bool = False):
    obs_class = OBS_REGISTRY[cfg["observation_class"]]
    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"],
        out_csv_name=None,
        use_gui=use_gui,
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5),
        max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=True,
        yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"],
        observation_class=obs_class,
        sumo_seed=cfg.get("seed", 0),
    )

    model = PPO.load(ckpt_path, env=env)
    print(f"Loaded checkpoint: {ckpt_path}")

    ts_lane_map = {ts: env.traffic_signals[ts].lanes for ts in env.ts_ids}
    base_seed = cfg.get("seed", 0)
    all_episode_metrics: list[dict] = []

    for episode in range(1, n_episodes + 1):
        seed = base_seed if fixed_seed else base_seed + episode - 1
        obs, _ = env.reset(seed)

        mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time)
        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            if steps >= warmup_steps:
                mc.collect_step(env.sumo)

            action, _ = model.predict(obs, deterministic=True)
            print(f"  step={steps:3d}  action={int(action)}")
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            if steps >= warmup_steps:
                episode_reward += float(reward) if reward is not None else 0.0
            steps += 1

        mc.finalize(env.sumo)
        flat = mc.to_flat_dict(prefix="eval")
        flat["eval/episode_reward"] = episode_reward
        flat["eval/steps"] = steps
        all_episode_metrics.append(flat)
        print(f"  episode {episode:3d}/{n_episodes}  reward={episode_reward:.2f}  steps={steps}")

    env.close()

    # ── aggregate mean ± std ──────────────────────────────────────────────────
    keys = all_episode_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([ep[k] for ep in all_episode_metrics], dtype=float)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}

    # ── print ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Evaluation summary  ({n_episodes} episodes)")
    print("=" * 70)
    col_w = max(len(k) for k in summary) + 2
    for k, v in sorted(summary.items()):
        print(f"  {k:<{col_w}}  {v['mean']:>10.4f}  ± {v['std']:.4f}")
    print("=" * 70)

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_dir = Path(ckpt_path).parent
    out_path = out_dir / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "episodes": all_episode_metrics}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── optional wandb ────────────────────────────────────────────────────────
    if wandb_log:
        import wandb
        exp_name = cfg["name"]
        wandb.init(
            project=cfg.get("wandb_project", "sumo-rl"),
            group=exp_name,
            name=f"{exp_name}_eval_{Path(ckpt_path).stem}",
            config=cfg,
            dir="./logs/wandb",
        )
        wandb.log({f"{k}_mean": v["mean"] for k, v in summary.items()})
        wandb.log({f"{k}_std":  v["std"]  for k, v in summary.items()})
        wandb.finish()

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO-RL PPO evaluator (stable-baselines3)")
    parser.add_argument("--config",   required=True,  help="Path to YAML config")
    parser.add_argument("--ckpt",     required=True,  help="Path to .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--gui",      action="store_true", help="Enable SUMO GUI")
    parser.add_argument("--wandb",    action="store_true", help="Log summary to wandb")
    parser.add_argument("--warmup",   type=int, default=20,
                        help="Steps to skip before collecting metrics (default: 20 = 100s)")
    parser.add_argument("--fixed-seed", action="store_true",
                        help="Use the same seed as training (config seed) for all episodes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_eval(cfg, args.ckpt, args.episodes, args.gui,
             wandb_log=args.wandb, warmup_steps=args.warmup, fixed_seed=args.fixed_seed)
