"""PPO training script for SUMO-RL (single-agent / 1x1).

Usage:
    python experiments/train_ppo.py --config experiments/configs/exp49_ppo_1x1_51.yaml

Uses stable-baselines3 PPO with the existing SumoEnvironment (single_agent=True).
Supports the same wandb / EpisodeMetricsCollector logging as train.py.
"""

import argparse
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

os.environ["SUMO_RL_LIBSUMO"] = "1"

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


# ── Callbacks ─────────────────────────────────────────────────────────────────

class SumoMetricsCallback(BaseCallback):
    """Logs per-episode metrics, phase selection, and optionally wandb."""

    def __init__(self, env: SumoEnvironment, cfg: dict,
                 model_dir: str, logging_mode: str,
                 metrics_interval: int, checkpoint_interval: int,
                 verbose: int = 0):
        super().__init__(verbose)
        self.sumo_env      = env
        self.cfg           = cfg
        self.model_dir     = model_dir
        self.logging_mode  = logging_mode
        self.metrics_interval    = metrics_interval
        self.checkpoint_interval = checkpoint_interval

        self.episode       = 0
        self.ts_lane_map   = None
        self.mc            = None
        self.phase_counts  = {}

    def _on_training_start(self) -> None:
        self.ts_lane_map = {
            ts: self.sumo_env.traffic_signals[ts].lanes
            for ts in self.sumo_env.ts_ids
        }
        self._reset_episode()

    def _reset_episode(self):
        self.episode += 1
        do_full = (self.logging_mode == "full") and (self.episode % self.metrics_interval == 0)
        self.mc = EpisodeMetricsCollector(
            self.ts_lane_map, delta_time=self.sumo_env.delta_time
        ) if do_full else None
        self.phase_counts = {ts_id: {} for ts_id in self.sumo_env.ts_ids}

    def _on_step(self) -> bool:
        # Collect metrics before step
        if self.mc is not None:
            try:
                self.mc.collect_step(self.sumo_env.sumo)
            except Exception:
                pass

        # Track phase selection
        for ts_id in self.sumo_env.ts_ids:
            try:
                p = self.sumo_env.traffic_signals[ts_id].green_phase
                self.phase_counts[ts_id][p] = self.phase_counts[ts_id].get(p, 0) + 1
            except Exception:
                pass

        return True

    def _on_rollout_end(self) -> None:
        """Log PPO training stats (entropy, losses) to wandb after each policy update."""
        if self.logging_mode == "none":
            return
        stats = self.model.logger.name_to_value
        log = {}
        for key in ("train/entropy_loss", "train/policy_gradient_loss",
                    "train/value_loss", "train/approx_kl",
                    "train/clip_fraction", "train/explained_variance"):
            if key in stats:
                log[key] = stats[key]
        if log:
            wandb.log(log, step=self.num_timesteps)

    def _on_episode_end(self) -> None:
        """Called at episode boundary."""
        # Full metrics
        if self.mc is not None:
            try:
                self.mc.finalize(self.sumo_env.sumo)
                if self.logging_mode != "none":
                    wandb.log(self.mc.to_flat_dict(prefix="metrics"),
                              step=self.num_timesteps)
            except Exception:
                pass

        # Phase selection log
        phase_log = {}
        for ts_id, counts in self.phase_counts.items():
            total = sum(counts.values()) or 1
            for p, c in counts.items():
                phase_log[f"phase/{ts_id}/phase{p}_ratio"] = c / total

        print(f"[{self.cfg['name']}] ep={self.episode:5d}  "
              f"timesteps={self.num_timesteps}  phases={self.phase_counts}")

        if self.logging_mode != "none":
            wandb.log({"train/episode": self.episode, **phase_log},
                      step=self.num_timesteps)

        # Checkpoint
        if self.episode % self.checkpoint_interval == 0:
            os.makedirs(self.model_dir, exist_ok=True)
            ckpt = os.path.join(self.model_dir, f"ppo_ep{self.episode:05d}")
            self.model.save(ckpt)
            print(f"  → ckpt saved: {ckpt}.zip")

        self._reset_episode()


class EpisodeBoundaryCallback(BaseCallback):
    """Detects episode boundaries for SumoMetricsCallback."""

    def __init__(self, metrics_cb: SumoMetricsCallback, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_cb  = metrics_cb
        self._prev_done  = False

    def _on_step(self) -> bool:
        done = bool(self.locals.get("dones", [False])[0])
        if done:
            self.metrics_cb._on_episode_end()
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: dict, timestamp: str, gpu: int):
    exp_name     = cfg["name"]
    run_id       = f"{exp_name}_{timestamp}"
    model_dir    = os.path.join("./models", exp_name, timestamp)
    logging_mode = cfg.get("logging_mode", "basic")

    if logging_mode != "none":
        wandb.init(
            project=cfg.get("wandb_project", "sumo-rl"),
            group=exp_name,
            name=run_id,
            config=cfg,
            dir="./logs/wandb",
            reinit=True,
        )

    obs_class = OBS_REGISTRY[cfg["observation_class"]]
    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"],
        out_csv_name=None,
        use_gui=cfg.get("use_gui", False),
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5),
        max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=True,          # required for SB3
        yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"],
        observation_class=obs_class,
        sumo_seed=cfg.get("seed", 0),
    )

    total_timesteps = cfg.get("episodes", 5000) * (cfg.get("num_seconds", 1000) // cfg.get("delta_time", 5))

    metrics_cb = SumoMetricsCallback(
        env=env,
        cfg=cfg,
        model_dir=model_dir,
        logging_mode=logging_mode,
        metrics_interval=cfg.get("metrics_interval", 50),
        checkpoint_interval=cfg.get("checkpoint_interval", 5),
    )
    boundary_cb = EpisodeBoundaryCallback(metrics_cb)

    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        device=device,
        # PPO hyperparams from config (with sensible defaults)
        learning_rate=cfg.get("lr", 3e-4),
        n_steps=cfg.get("n_steps", 2048),
        batch_size=cfg.get("batch_size", 64),
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),   # entropy bonus: key for avoiding starvation
        vf_coef=cfg.get("vf_coef", 0.5),
        tensorboard_log=f"./logs/ppo/{exp_name}" if logging_mode != "none" else None,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[metrics_cb, boundary_cb],
        progress_bar=True,
    )

    # Save final model
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "ppo_final"))
    print(f"Final model saved: {model_dir}/ppo_final.zip")

    env.close()
    if logging_mode != "none":
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO-RL PPO trainer (single-agent)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--gpu",    type=int, default=0, help="GPU index; -1 for CPU")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_save_dir = os.path.join("./models", cfg["name"], timestamp)
    os.makedirs(config_save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(config_save_dir, "config.yaml"))

    train(cfg, timestamp, args.gpu)
