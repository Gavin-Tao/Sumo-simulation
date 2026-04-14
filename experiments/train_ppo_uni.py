"""Unified PPO training script for SUMO-RL (1x1 and 1x3).

Works for any number of intersections via parameter-sharing VecEnv:
  - 1x1: ts_ids has 1 element → n_envs=1, equivalent to single-agent
  - 1x3: ts_ids has 3 elements → n_envs=3, parameter sharing across intersections

Reward normalisation: VecNormalize(norm_reward=True) stabilises value function
learning — raw priority-pressure rewards can reach magnitude ~1000 per episode,
causing large value_loss and unstable policy updates without normalisation.

Raw episode reward (before normalisation) is tracked separately and logged to
wandb as train/ep_raw_reward_mean, which directly reflects the true objective.

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
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

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
    PriorityNormObservationFunction,
    DiffWaitingObservationFunction,
    PriorityDiffWaitingObservationFunction,
)

OBS_REGISTRY = {
    "Default":              DefaultObservationFunction,
    "PressLight":           PressLightObservationFunction,
    "PressLightNorm":       PressLightNormObservationFunction,
    "QueuePrio":            QueuePrioObservationFunction,
    "CTBPriority":          CTBPriorityObservationFunction,
    "Priority":             PriorityObservationFunction,
    "PriorityNorm":         PriorityNormObservationFunction,
    "DiffWaiting":          DiffWaitingObservationFunction,
    "PriorityDiffWaiting":  PriorityDiffWaitingObservationFunction,
}


# ── VecEnv wrapper ─────────────────────────────────────────────────────────────

class SumoVecEnv(VecEnv):
    """Wraps SUMO multi-agent env as SB3 VecEnv.

    Works for any number of intersections (1x1 or 1x3).

    pre_reset_hooks: list of callables fired BEFORE self.reset() when an
    episode ends — used by the metrics callback to finalise episode stats
    while SUMO is still in its terminal state.

    last_raw_rews: raw (un-normalised) rewards from the last step, stored so
    the callback can accumulate true episode reward independently of VecNormalize.
    """

    def __init__(self, sumo_env: SumoEnvironment):
        self.sumo_env       = sumo_env
        self.ts_ids         = list(sumo_env.ts_ids)
        super().__init__(len(self.ts_ids), sumo_env.observation_space, sumo_env.action_space)
        self._actions            = None
        self.pre_reset_hooks     = []
        self.last_raw_rews       = np.zeros(len(self.ts_ids), dtype=np.float32)
        self.last_actual_actions = np.zeros(len(self.ts_ids), dtype=np.float32)

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
        # Store raw rewards before VecNormalize transforms them
        self.last_raw_rews[:] = rews
        # Store actual executed actions — may differ from chosen actions when
        # min_green is not yet satisfied (SUMO keeps current phase in that case)
        self.last_actual_actions = np.array([
            self.sumo_env.traffic_signals[ts].last_executed_action
            for ts in self.ts_ids
        ], dtype=np.float32)

        # dones[ts] is always False; only dones["__all__"] reliably signals episode end
        all_done = bool(dones["__all__"])
        done_arr = np.full(self.num_envs, all_done, dtype=bool)

        infos = [{} for _ in self.ts_ids]
        if all_done:
            # Fire hooks BEFORE reset — SUMO is still in terminal state here
            for hook in self.pre_reset_hooks:
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
    """Logs per-episode metrics, raw reward, phase selection, and PPO stats.

    Receives both the raw SumoVecEnv (for SUMO/raw-reward access) and the
    VecNormalize wrapper (to save normalisation stats at checkpoints).
    """

    def __init__(self, base_env: SumoVecEnv,
                 norm_env: VecNormalize,
                 cfg: dict, model_dir: str, logging_mode: str,
                 metrics_interval: int, checkpoint_interval: int,
                 verbose: int = 0):
        super().__init__(verbose)
        self.base_env            = base_env
        self.norm_env            = norm_env
        self.sumo_env            = base_env.sumo_env
        self.ts_ids              = base_env.ts_ids
        self.cfg                 = cfg
        self.model_dir           = model_dir
        self.logging_mode        = logging_mode
        self.metrics_interval    = metrics_interval
        self.checkpoint_interval = checkpoint_interval

        self.episode           = 0
        self.ts_lane_map       = None
        self.mc                = None
        self.phase_counts      = {}
        self._terminal_metrics = None   # finalized before reset
        self._ep_raw_reward    = 0.0    # cumulative raw reward for current episode

    def _on_training_start(self) -> None:
        self.ts_lane_map = {ts: self.sumo_env.traffic_signals[ts].lanes for ts in self.ts_ids}
        if self._finalize_terminal_metrics not in self.base_env.pre_reset_hooks:
            self.base_env.pre_reset_hooks.append(self._finalize_terminal_metrics)
        self._reset_episode()
        self._patch_rollout_buffer()

    def _patch_rollout_buffer(self) -> None:
        """Patch rollout_buffer.add to use last_executed_action instead of policy action.

        SUMO ignores the policy's chosen action when min_green is not satisfied,
        executing the previous phase instead. PPO must store the actually executed
        action; otherwise the stored log_prob and advantage are computed for an
        action that never happened, biasing the policy gradient.
        """
        import torch as th
        from stable_baselines3 import PPO as _PPO
        ppo: _PPO = self.model  # type: ignore[assignment]
        # Guard against double-patching if model.learn() is called more than once
        if getattr(ppo.rollout_buffer.add, "_sumo_patched", False):
            return
        original_add = ppo.rollout_buffer.add
        callback = self

        def patched_add(obs, actions, rewards, episode_starts, values, log_probs):
            actual = callback.base_env.last_actual_actions.astype(np.int64)  # (n_envs,)
            if not np.array_equal(actual, actions.flatten().astype(np.int64)):
                _ppo: _PPO = callback.model  # type: ignore[assignment]
                with th.no_grad():
                    obs_t = th.as_tensor(obs,    device=_ppo.device)
                    act_t = th.as_tensor(actual, dtype=th.long, device=_ppo.device)
                    _, new_log_probs, _ = _ppo.policy.evaluate_actions(obs_t, act_t)  # type: ignore[union-attr]
                actions   = actual.reshape(actions.shape).astype(actions.dtype)
                log_probs = new_log_probs.reshape(log_probs.shape)  # keep as tensor; SB3 calls .clone() internally
            return original_add(obs, actions, rewards, episode_starts, values, log_probs)

        patched_add._sumo_patched = True  # type: ignore[attr-defined]
        ppo.rollout_buffer.add = patched_add  # type: ignore[method-assign]

    def _reset_episode(self):
        self.episode += 1
        do_full = (self.logging_mode == "full") and (self.episode % self.metrics_interval == 0)
        self.mc = EpisodeMetricsCollector(
            self.ts_lane_map, delta_time=self.sumo_env.delta_time
        ) if do_full else None
        self.phase_counts        = {ts_id: {} for ts_id in self.ts_ids}
        self._ep_raw_reward      = 0.0
        self._ep_raw_rews_per_ts = np.zeros(len(self.ts_ids), dtype=np.float32)

    # ── Pre-reset hook — SUMO still in terminal state ──────────────────────────

    def _finalize_terminal_metrics(self):
        if self.mc is None:
            return
        try:
            self.mc.collect_step(self.sumo_env.sumo)
            self.mc.finalize(self.sumo_env.sumo)
            self._terminal_metrics = self.mc.to_flat_dict(prefix="metrics")
        except Exception:
            self._terminal_metrics = None

    # ── SB3 hooks ──────────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", np.zeros(self.base_env.num_envs, dtype=bool))

        # Accumulate raw (un-normalised) reward per intersection and overall mean
        self._ep_raw_rews_per_ts += self.base_env.last_raw_rews
        self._ep_raw_reward += float(self.base_env.last_raw_rews.mean())

        # Collect intermediate steps only; terminal step handled by pre-reset hook
        if self.mc is not None and not np.any(dones):
            try:
                self.mc.collect_step(self.sumo_env.sumo)
            except Exception:
                pass

        # Skip phase counting on terminal step: step_wait already called reset(),
        # so traffic_signals now reflect the NEW episode's initial state, not the
        # episode that just ended.
        if not np.any(dones):
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
        # NOTE: _on_rollout_end is called by SB3 INSIDE collect_rollouts(), BEFORE
        # train() is invoked. model.logger.name_to_value therefore holds stats from
        # the *previous* policy update, not the one about to happen.
        # To get the current update's stats we flush the logger after train() via
        # _on_training_end — but SB3 has no per-update hook, so we log here with
        # the understood one-rollout lag (first rollout: stats dict is empty → skipped).
        if self.logging_mode == "none":
            return
        stats = self.model.logger.name_to_value
        log = {k: stats[k] for k in (
            "train/entropy_loss", "train/policy_gradient_loss",
            "train/value_loss", "train/approx_kl",
            "train/clip_fraction", "train/explained_variance",
        ) if k in stats}
        if self.norm_env is not None:
            rms = getattr(self.norm_env, "ret_rms", None) or getattr(self.norm_env, "reward_rms", None)
            if rms is not None:
                log["train/reward_rms_mean"] = float(rms.mean)
                log["train/reward_rms_var"]  = float(rms.var)
        if log:
            wandb.log(log, step=self.num_timesteps)

    def _on_episode_end(self):
        # Log EpisodeMetricsCollector results (finalized before reset)
        if self._terminal_metrics is not None and self.logging_mode != "none":
            wandb.log(self._terminal_metrics, step=self.num_timesteps)
        self._terminal_metrics = None

        # Phase selection ratios
        phase_log = {}
        for ts_id, counts in self.phase_counts.items():
            total = sum(counts.values()) or 1
            for p, c in counts.items():
                phase_log[f"phase/{ts_id}/phase{p}_ratio"] = c / total

        print(f"[{self.cfg['name']}] ep={self.episode:5d}  "
              f"raw_reward={self._ep_raw_reward:.1f}  "
              f"timesteps={self.num_timesteps}  phases={self.phase_counts}", flush=True)

        if self.logging_mode != "none":
            per_ts_reward_log = {
                f"train/ep_raw_reward_{ts_id}": float(self._ep_raw_rews_per_ts[i])
                for i, ts_id in enumerate(self.ts_ids)
            }
            wandb.log({
                "train/episode":          self.episode,
                "train/ep_raw_reward":    self._ep_raw_reward,   # mean across intersections
                **per_ts_reward_log,
                **phase_log,
            }, step=self.num_timesteps)

        if self.episode % self.checkpoint_interval == 0:
            os.makedirs(self.model_dir, exist_ok=True)
            ckpt = os.path.join(self.model_dir, f"ppo_ep{self.episode:05d}")
            self.model.save(ckpt)
            if self.norm_env is not None:
                self.norm_env.save(ckpt + "_vecnorm.pkl")
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

    base_env = SumoVecEnv(sumo_env)

    # Wrap with VecNormalize for reward normalisation.
    # norm_obs=False : PriorityObservationFunction already handles obs scaling.
    # norm_reward=True: running-stats normalise rewards → stable value function.
    # clip_reward    : clip normalised reward to [-clip, +clip].
    norm_env = VecNormalize(
        base_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=cfg.get("clip_reward", 10.0),
        gamma=cfg.get("gamma", 0.99),
    )

    steps_per_episode = cfg.get("num_seconds", 1000) // cfg.get("delta_time", 5)
    # SB3 increments num_timesteps by n_envs per SUMO step, so multiply by n_envs
    # to ensure the correct number of episodes are trained regardless of intersection count.
    # 1x1: n_envs=1 → no change; 1x3: n_envs=3 → 3× to compensate.
    total_timesteps   = cfg.get("episodes", 5000) * steps_per_episode * base_env.num_envs

    metrics_cb = SumoMetricsCallback(
        base_env=base_env,
        norm_env=norm_env,
        cfg=cfg,
        model_dir=model_dir,
        logging_mode=logging_mode,
        metrics_interval=cfg.get("metrics_interval", 50),
        checkpoint_interval=cfg.get("checkpoint_interval", 50),
    )

    device = f"cuda:{gpu}" if gpu >= 0 else "cpu"

    model = PPO(
        policy="MlpPolicy",
        env=norm_env,           # PPO sees normalised rewards
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
    norm_env.save(os.path.join(model_dir, "vecnorm_final.pkl"))
    print(f"Final model saved:        {model_dir}/ppo_final.zip", flush=True)
    print(f"VecNormalize stats saved: {model_dir}/vecnorm_final.pkl", flush=True)

    norm_env.close()
    if logging_mode != "none":
        wandb.finish()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SUMO-RL unified PPO trainer (1x1 and 1x3, reward normalisation)"
    )
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
