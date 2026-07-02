"""MaskablePPO trainer for the Dublin masked-8std setup (exp205-family).

New file — train_ppo_uni.py and train.py stay byte-identical. Reuses the
uni script's SumoVecEnv + SumoMetricsCallback and adds the Dublin machinery
that the old PPO trainer predates:
  * PriorityMovement (B-family) obs + obs_* kwargs plumbing   (as train.py)
  * priority-avg-waiting reward factory + reward_scale/floor  (as train.py)
  * action_meta_file: per-TLS 8-std masks, std<->green maps, movement rebind
    on every reset                                            (as train.py)
  * MaskablePPO (sb3-contrib): the policy samples only valid std actions;
    rollout buffer stores the EXECUTED std action (min_green may override),
    with log-probs re-evaluated under the mask.

Usage:
  cd experiments && python train_ppo_dublin.py --config configs/exp212_dublin11h_531_ppo.yaml [--gpu 0]
"""
import argparse
import functools
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import wandb

os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces  # sb3>=2.x requires gymnasium spaces, not gym

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from train_ppo_uni import SumoVecEnv, SumoMetricsCallback


# ── Dublin plumbing (mirrors train.py) ─────────────────────────────────────────

def build_obs_class(cfg):
    obs_class = {
        "PriorityMovement": obsmod.PriorityMovementObservationFunction,
    }[cfg["observation_class"]]
    kw = {}
    for src, dst in [("obs_fields", "fields"), ("obs_phase_state", "phase_state"),
                     ("priority_source", "priority_source"),
                     ("obs_downstream", "include_downstream"),
                     ("obs_downstream_fields", "downstream_fields"),
                     ("obs_lane_occ", "include_lane_occ"),
                     ("obs_awt_cap", "awt_cap"), ("obs_awt_basis", "awt_basis"),
                     ("obs_slot_stats", "slot_stats")]:
        if src in cfg:
            v = cfg[src]
            kw[dst] = tuple(v) if isinstance(v, list) else v
    return functools.partial(obs_class, **kw) if kw else obs_class


def build_reward_fn(cfg):
    reward_fn = cfg["reward_fn"]
    if reward_fn == "priority-avg-waiting":
        from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
        from sumo_rl.environment.priority_map import load_priority_table
        reward_fn = make_priority_avg_waiting_reward(
            load_priority_table(cfg.get("priority_source")))
    _rs = float(cfg.get("reward_scale", 1.0))
    if _rs != 1.0 and callable(reward_fn):
        _base = reward_fn

        def reward_fn(ts, _f=_base, _s=_rs):  # noqa: F811
            return _f(ts) * _s
    _rf = cfg.get("reward_floor", None)
    if _rf is not None and callable(reward_fn):
        _base2 = reward_fn

        def reward_fn(ts, _f=_base2, _c=float(_rf)):  # noqa: F811
            return max(_f(ts), _c)
    return reward_fn


def load_meta(path):
    meta = json.load(open(path))["tls"]
    ts_mask, std2green, green2std, turnmap = {}, {}, {}, {}
    for tid, t in meta.items():
        ts_mask[tid] = np.array(t["mask"], dtype=bool)
        turnmap[tid] = {int(i): (c[0]["approach"], c[0]["turn"])
                        for i, c in t["links"].items()}
        s2g = np.full(8, -1, dtype=int)
        for a, gi in t["std_to_green_index"].items():
            s2g[int(a)] = gi
        std2green[tid] = s2g
        g2s = np.full(int(s2g.max()) + 1, -1, dtype=int)
        for a in range(8):
            if s2g[a] >= 0:
                g2s[s2g[a]] = a
        green2std[tid] = g2s
    return ts_mask, std2green, green2std, turnmap


# ── VecEnv with std-action masking ─────────────────────────────────────────────

class DublinVecEnv(SumoVecEnv):
    """Actions live in the fixed 8-dim STD space; masks come from the meta.

    step: std -> dense green index before SumoEnvironment.step;
    last_actual_actions: executed green -> canonical STD (the rollout-buffer
    patch stores these, so they must be in the policy's action space);
    reset: rebind obs movements + refresh obs spaces (TrafficSignal objects
    are recreated each episode)."""

    def __init__(self, sumo_env, ts_mask, std2green, green2std, turnmap):
        self.ts_mask, self.std2green = ts_mask, std2green
        self.green2std, self.turnmap = green2std, turnmap
        # rebind BEFORE snapshotting spaces: legacy one-hot must be 8-dim std
        sumo_env.reset(sumo_env.sumo_seed)
        self._rebind(sumo_env)
        super().__init__(sumo_env)
        self.action_space = spaces.Discrete(8)
        self.observation_space = sumo_env.traffic_signals[self.ts_ids[0]].observation_space

    def _rebind(self, env):
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.std_action_map = self.green2std[tid]
            if hasattr(ts.observation_fn, "rebind_movements"):
                ts.observation_fn.rebind_movements(self.turnmap[tid])
            ts.observation_space = ts.observation_fn.observation_space()

    def reset(self):
        self.sumo_env.reset(self.sumo_env.sumo_seed)
        self._rebind(self.sumo_env)
        states = {tid: self.sumo_env.traffic_signals[tid].observation_fn()
                  for tid in self.sumo_env.ts_ids}
        return np.array([states[ts] for ts in self.ts_ids], dtype=np.float32)

    def step_wait(self):
        std = {ts: int(self._actions[i]) for i, ts in enumerate(self.ts_ids)}
        self._actions = np.array(
            [int(self.std2green[ts][std[ts]]) for ts in self.ts_ids])
        obs, rews, done_arr, infos = super().step_wait()
        # executed green -> canonical std (buffer patch consumes this)
        self.last_actual_actions = np.array(
            [float(self.green2std[ts][int(self.last_actual_actions[i])])
             for i, ts in enumerate(self.ts_ids)], dtype=np.float32)
        return obs, rews, done_arr, infos

    def action_masks(self):
        return np.stack([self.ts_mask[ts] for ts in self.ts_ids])

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name == "action_masks":       # get_action_masks() protocol
            return [self.ts_mask[ts] for ts in self.ts_ids]
        return super().env_method(method_name, *args, indices=indices, **kwargs)

    def get_attr(self, attr_name, indices=None):
        if attr_name == "action_masks":         # is_masking_supported() probe
            return [self.ts_mask[ts] for ts in self.ts_ids]
        return super().get_attr(attr_name, indices=indices)


# ── Callback: maskable rollout-buffer patch ────────────────────────────────────

class DublinMetricsCallback(SumoMetricsCallback):
    def _patch_rollout_buffer(self) -> None:
        """Maskable variant of the uni script's executed-action patch: the
        buffer add() carries action_masks, and log-prob re-evaluation must
        pass the masks to the maskable policy."""
        import torch as th
        ppo = self.model
        if getattr(ppo.rollout_buffer.add, "_sumo_patched", False):
            return
        original_add = ppo.rollout_buffer.add
        callback = self

        def patched_add(obs, actions, rewards, episode_starts, values,
                        log_probs, action_masks=None):
            actual = callback.base_env.last_actual_actions.astype(np.int64)
            if not np.array_equal(actual, actions.flatten().astype(np.int64)):
                _ppo = callback.model
                with th.no_grad():
                    obs_t = th.as_tensor(obs, device=_ppo.device)
                    act_t = th.as_tensor(actual, dtype=th.long, device=_ppo.device)
                    masks_t = None
                    if action_masks is not None:
                        masks_t = th.as_tensor(action_masks, device=_ppo.device)
                    _, new_log_probs, _ = _ppo.policy.evaluate_actions(
                        obs_t, act_t, action_masks=masks_t)
                actions = actual.reshape(actions.shape).astype(actions.dtype)
                log_probs = new_log_probs.reshape(log_probs.shape)
            return original_add(obs, actions, rewards, episode_starts, values,
                                log_probs, action_masks=action_masks)

        patched_add._sumo_patched = True  # type: ignore[attr-defined]
        ppo.rollout_buffer.add = patched_add  # type: ignore[method-assign]


# ── Main ───────────────────────────────────────────────────────────────────────

def train(cfg, timestamp, gpu):
    exp_name = cfg["name"]
    model_dir = os.path.join("./models", exp_name, timestamp)
    logging_mode = cfg.get("logging_mode", "basic")
    if logging_mode != "none":
        wandb.init(project=cfg.get("wandb_project", "sumo-rl"), group=exp_name,
                   name=f"{exp_name}_{timestamp}", config=cfg,
                   dir="./logs/wandb", reinit=True)

    sumo_env = SumoEnvironment(
        net_file=cfg["net_file"], route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"], out_csv_name=None,
        use_gui=cfg.get("use_gui", False),
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5), max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False), single_agent=False,
        yellow_time=cfg.get("yellow_time", 2), delta_time=cfg.get("delta_time", 5),
        reward_fn=build_reward_fn(cfg), observation_class=build_obs_class(cfg),
        sumo_seed=cfg.get("seed", 0), sumo_warnings=False)

    ts_mask, std2green, green2std, turnmap = load_meta(cfg["action_meta_file"])
    base_env = DublinVecEnv(sumo_env, ts_mask, std2green, green2std, turnmap)
    norm_env = VecNormalize(base_env, norm_obs=False, norm_reward=True,
                            clip_reward=cfg.get("clip_reward", 10.0),
                            gamma=cfg.get("gamma", 0.99))

    steps_per_episode = cfg.get("num_seconds", 1000) // cfg.get("delta_time", 5)
    total_timesteps = cfg.get("episodes", 800) * steps_per_episode * base_env.num_envs

    cb = DublinMetricsCallback(
        base_env=base_env, norm_env=norm_env, cfg=cfg, model_dir=model_dir,
        logging_mode=logging_mode,
        metrics_interval=cfg.get("metrics_interval", 50),
        checkpoint_interval=cfg.get("checkpoint_interval", 50))

    model = MaskablePPO(
        policy="MlpPolicy", env=norm_env, verbose=0,
        device=(f"cuda:{gpu}" if gpu >= 0 else "cpu"),
        learning_rate=cfg.get("lr", 3e-4), n_steps=cfg.get("n_steps", 720),
        batch_size=cfg.get("batch_size", 720), n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 0.99), gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2), ent_coef=cfg.get("ent_coef", 0.01),
        vf_coef=cfg.get("vf_coef", 0.5),
        tensorboard_log=f"./logs/ppo/{exp_name}" if logging_mode != "none" else None)

    model.learn(total_timesteps=total_timesteps, callback=[cb], progress_bar=True)

    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "ppo_final"))
    norm_env.save(os.path.join(model_dir, "vecnorm_final.pkl"))
    print(f"Final model saved: {model_dir}/ppo_final.zip", flush=True)
    norm_env.close()
    if logging_mode != "none":
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dublin MaskablePPO trainer")
    p.add_argument("--config", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = os.path.join("./models", cfg["name"], timestamp)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(save_dir, "config.yaml"))
    train(cfg, timestamp, args.gpu)
