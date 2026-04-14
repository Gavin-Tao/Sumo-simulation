"""Unified PPO training script for SUMO-RL (1x1 and 1x3).

Works for any number of intersections via parameter-sharing VecEnv:
  - 1x1: ts_ids has 1 element → n_envs=1, equivalent to single-agent
  - 1x3: ts_ids has 3 elements → n_envs=3, parameter sharing across intersections

One PPO policy is shared across all intersections (same as train.py's DQN design).
Each SUMO step produces n_envs samples in the rollout buffer simultaneously.

Usage:
    python experiments/train_ppo_uni.py --config experiments/configs/exp49_ppo_1x1_51_WEstr_only.yaml
    python experiments/train_ppo_uni.py --config experiments/configs/exp65_ppo_1x3_41_WEstr_only.yaml
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
from stable_baselines3.common.vec_env import VecEnv

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


# ── VecEnv wrapper ─────────────────────────────────────────────────────────────

class SumoVecEnv(VecEnv):
    """Wraps SUMO multi-agent env as SB3 VecEnv.

    Each sub-env corresponds to one intersection. All intersections act
    synchronously (same SUMO simulation): step_async buffers actions,
    step_wait sends them all at once.

    Works for any number of intersections:
      - n_envs=1 (1x1): identical to a standard single-agent VecEnv
      - n_envs=3 (1x3): parameter sharing — all 3 intersections feed
        the same rollout buffer and update the same policy
    """

    def __init__(self, sumo_env: SumoEnvironment):
        self.sumo_env        = sumo_env
        self.ts_ids          = list(sumo_env.ts_ids)
        super().__init__(len(self.ts_ids), sumo_env.observation_space, sumo_env.action_space)
        self._actions        = None
        self._pre_reset_hooks = []   # called before reset while SUMO is still in terminal state

    def reset(self):
        states = self.sumo_env.reset(self.sumo_env.sumo_seed)
        return np.array([states[ts] for ts in self.ts_ids], dtype=np.float32)

    def step_async(self, actions: np.ndarray):
        self._actions = actions

    def step_wait(self):
        action_dict = {ts: int(self._actions[i]) for i, ts in enumerate(self.ts_ids)}
        states, rewards, dones, _ = self.sumo_env.step(action=action_dict)

        obs  = np.array([states[ts]  for ts in self.ts_ids], dtype=np.float32)
        rews = np.array([rewards[ts] for ts in self.ts_ids], dtype=np.float32)
        # dones[ts] is always False in SUMO multi-agent env; only dones["__all__"] is reliable
        all_done = bool(dones["__all__"])
        done_arr = np.full(self.num_envs, all_done, dtype=bool)

        infos = [{} for _ in self.ts_ids]
        if all_done:
            # Fire hooks BEFORE reset — SUMO is still in terminal state here
            for hook in self._pre_reset_hooks:
                hook()
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i]
            obs = self.reset()

        return obs, rews, done_arr, infos

    def close(self):
        self.sumo_env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.sumo_env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.sumo_env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.sumo_env, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [seed] * self.num_envs


# ── Callback ───────────────────────────────────────────────────────────────────

class SumoMetricsCallback(BaseCallback):
    """Logs per-episode metrics, phase selection, and PPO train stats to wandb."""

    def __init__(self, vec_env: SumoVecEnv, cfg: dict, model_dir: str,
                 logging_mode: str, metrics_interval: int,
                 checkpoint_interval: int, verbose: int = 0):
        super().__init__(verbose)
        self.sumo_env            = vec_env.sumo_env
        self.ts_ids              = vec_env.ts_ids
        self.vec_env             = vec_env
        self.cfg                 = cfg
        self.model_dir           = model_dir
        self.logging_mode        = logging_mode
        self.metrics_interval    = metrics_interval
        self.checkpoint_interval = checkpoint_interval
        self.episode             = 0
        self.ts_lane_map         = None
        self.mc                  = None
        self.phase_counts        = {}
        self._terminal_metrics   = None   # metrics finalized before reset, used in _on_episode_end

    def _on_training_start(self) -> None:
        self.ts_lane_map = {ts: self.sumo_env.traffic_signals[ts].lanes for ts in self.ts_ids}
        # Register pre-reset hook so finalize() runs while SUMO is still in terminal state
        self.vec_env._pre_reset_hooks.append(self._finalize_terminal_metrics)
        self._reset_episode()

    def _reset_episode(self):
        self.episode += 1
        do_full = (self.logging_mode == "full") and (self.episode % self.metrics_interval == 0)
        self.mc = EpisodeMetricsCollector(
            self.ts_lane_map, delta_time=self.sumo_env.delta_time
        ) if do_full else None
        self.phase_counts = {ts_id: {} for ts_id in self.ts_ids}

    def _finalize_terminal_metrics(self):
        """Called by SumoVecEnv BEFORE reset — SUMO is still in terminal state."""
        if self.mc is None:
            return
        try:
            self.mc.collect_step(self.sumo_env.sumo)   # collect the final step
            self.mc.finalize(self.sumo_env.sumo)
            self._terminal_metrics = self.mc.to_flat_dict(prefix="metrics")
        except Exception:
            self._terminal_metrics = None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", np.zeros(self.vec_env.num_envs, dtype=bool))

        # Collect intermediate steps only — terminal step is handled by _finalize_terminal_metrics
        if self.mc is not None and not np.any(dones):
            try:
                self.mc.collect_step(self.sumo_env.sumo)
            except Exception:
                pass

        for ts_id in self.ts_ids:
            try:
                p = self.sumo_env.traffic_signals[ts_id].green_phase
                self.phase_counts[ts_id][p] = self.phase_counts[ts_id].get(p, 0) + 1
            except Exception:
                pass

        if np.any(dones):
            self._on_episode_end()

        return True

    def _on_rollout_end(self) -> None:
        if self.logging_mode == "none":
            return
        stats = self.model.logger.name_to_value
        log = {k: stats[k] for k in (
            "train/entropy_loss", "train/policy_gradient_loss",
            "train/value_loss", "train/approx_kl",
            "train/clip_fraction", "train/explained_variance",
        ) if k in stats}
        if log:
            wandb.log(log, step=self.num_timesteps)

    def _on_episode_end(self):
        # Use metrics finalized before reset (by _finalize_terminal_metrics hook)
        if self._terminal_metrics is not None and self.logging_mode != "none":
            wandb.log(self._terminal_metrics, step=self.num_timesteps)
        self._terminal_metrics = None

        phase_log = {}
        for ts_id, counts in self.phase_counts.items():
            total = sum(counts.values()) or 1
            for p, c in counts.items():
                phase_log[f"phase/{ts_id}/phase{p}_ratio"] = c / total

        print(f"[{self.cfg['name']}] ep={self.episode:5d}  "
              f"timesteps={self.num_timesteps}  phases={self.phase_counts}", flush=True)

        if self.logging_mode != "none":
            wandb.log({"train/episode": self.episode, **phase_log}, step=self.num_timesteps)

        if self.episode % self.checkpoint_interval == 0:
            os.makedirs(self.model_dir, exist_ok=True)
            ckpt = os.path.join(self.model_dir, f"ppo_ep{self.episode:05d}")
            self.model.save(ckpt)
            print(f"  → ckpt saved: {ckpt}.zip", flush=True)

        self._reset_episode()


# ── Main ───────────────────────────────────────────────────────────────────────

def train(cfg: dict, timestamp: str, gpu: int):
    exp_name     = cfg["name"]
    model_dir    = os.path.join("./models", exp_name, timestamp)
    logging_mode = cfg.get("logging_mode", "basic")

    if logging_mode != "none":
        wandb.init(
            project=cfg.get("wandb_project", "sumo-rl"),
            group=exp_name,
            name=f"{exp_name}_{timestamp}",
            config=cfg,
            dir="./logs/wandb",
            reinit=True,
        )

    obs_class = OBS_REGISTRY[cfg["observation_class"]]
    sumo_env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"],
        out_csv_name=None,
        use_gui=cfg.get("use_gui", False),
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5),
        max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=False,
        yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"],
        observation_class=obs_class,
        sumo_seed=cfg.get("seed", 0),
    )

    env = SumoVecEnv(sumo_env)

    steps_per_episode = cfg.get("num_seconds", 1000) // cfg.get("delta_time", 5)
    total_timesteps   = cfg.get("episodes", 5000) * steps_per_episode

    metrics_cb = SumoMetricsCallback(
        vec_env=env,
        cfg=cfg,
        model_dir=model_dir,
        logging_mode=logging_mode,
        metrics_interval=cfg.get("metrics_interval", 50),
        checkpoint_interval=cfg.get("checkpoint_interval", 50),
    )

    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        device=device,
        learning_rate=cfg.get("lr", 3e-4),
        n_steps=cfg.get("n_steps", 2000),
        batch_size=cfg.get("batch_size", 200),
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.05),
        vf_coef=cfg.get("vf_coef", 0.5),
        tensorboard_log=f"./logs/ppo/{exp_name}" if logging_mode != "none" else None,
    )

    model.learn(total_timesteps=total_timesteps, callback=[metrics_cb], progress_bar=True)

    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "ppo_final"))
    print(f"Final model saved: {model_dir}/ppo_final.zip", flush=True)

    env.close()
    if logging_mode != "none":
        wandb.finish()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO-RL unified PPO trainer (1x1 and 1x3)")
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
