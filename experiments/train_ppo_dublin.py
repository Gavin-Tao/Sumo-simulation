"""Universal MaskablePPO trainer: 1x1 / 1x3 / Dublin, optional CoLight-orig
GAT coordination (exp209-style). New file — train.py / train_ppo_uni.py /
trainorico.py stay byte-identical; SumoVecEnv/SumoMetricsCallback and the
CoLight GATLayer are imported and reused, not copied.

Modes (all combinations valid):
  * action_meta_file present  -> masked fixed 8-std action space, std<->green
    maps, per-reset movement rebind            (mirrors train.py/trainorico M3)
  * action_meta_file absent   -> plain uniform action space (1x1/1x3 nets),
    masks all-valid (MaskablePPO degenerates to PPO)
  * coordination: colight_orig -> per-junction obs = [own || 4 neighbors]
    (zero rows = missing, neighbor_map order up/down/left/right as trainorico);
    policy features extractor reproduces CoLightOrigQNet exactly up to the
    cat_feat layer (GATLayer imported; Q-head replaced by SB3 pi/vf heads).

DQN-parity guarantees (checked against train.py / trainorico.py):
  * rollout buffer stores the EXECUTED canonical std action (min_green may
    override the sampled one), log-probs re-evaluated UNDER THE MASK;
  * neighbor obs snapshot = same decision step, same as trainorico's nb_obs;
  * reward factory / scale / floor identical to train.py.
Known protocol differences vs the DQN trainers (accepted, PPO-standard):
  * VecNormalize(norm_reward=True) — PPO trains on normalised rewards; raw
    episode reward is logged as train/ep_raw_reward (uni-script convention);
  * no periodic greedy eval episodes — compare via raw-reward curves +
    offline eval, not against DQN's eval_system series.

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
import torch
import torch.nn.functional as F
import yaml
import wandb

os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from gymnasium import spaces  # sb3>=2.x requires gymnasium spaces, not gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.agents.colight_orig.dqn_colight_orig import GATLayer
from train_ppo_uni import SumoVecEnv, SumoMetricsCallback

NB_SLOTS = 4  # neighbor_map order: up, down, left, right (trainorico contract)


# ── Config plumbing (mirrors train.py) ─────────────────────────────────────────

OBS_REGISTRY = {  # superset of the names used by dublin/1x3 configs
    "Priority": "PriorityObservationFunction",
    "PriorityBCA": "PriorityBCAObservationFunction",
    "PriorityMovement": "PriorityMovementObservationFunction",
    "PriorityPhase": "PriorityPhaseObservationFunction",
}


def build_obs_class(cfg):
    obs_class = getattr(obsmod, OBS_REGISTRY[cfg["observation_class"]])
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


def identity_meta(sumo_env):
    """No-meta mode (1x1/1x3): uniform action space, all actions valid."""
    n = sumo_env.action_space.n
    ident = np.arange(n, dtype=int)
    ts_mask = {ts: np.ones(n, dtype=bool) for ts in sumo_env.ts_ids}
    std2green = {ts: ident for ts in sumo_env.ts_ids}
    green2std = {ts: ident for ts in sumo_env.ts_ids}
    return ts_mask, std2green, green2std, None


# ── VecEnv: masked std actions, optional neighbor-concat obs ───────────────────

class MaskedSumoVecEnv(SumoVecEnv):
    """Actions in a fixed std space (8 with meta, env-native without);
    obs optionally concatenated with the 4 neighbors' obs (colight mode).

    DQN-parity: rebind on EVERY reset (TrafficSignal objects are recreated);
    last_actual_actions returned in STD space (rollout patch stores them)."""

    def __init__(self, sumo_env, ts_mask, std2green, green2std, turnmap,
                 neighbor_map=None):
        self.ts_mask, self.std2green = ts_mask, std2green
        self.green2std, self.turnmap = green2std, turnmap
        self.neighbor_map = neighbor_map
        sumo_env.reset(sumo_env.sumo_seed)
        self._rebind(sumo_env)
        super().__init__(sumo_env)
        n_std = len(next(iter(ts_mask.values())))
        self.action_space = spaces.Discrete(n_std)
        own_dim = int(np.prod(
            sumo_env.traffic_signals[self.ts_ids[0]].observation_space.shape))
        self.own_dim = own_dim
        obs_dim = own_dim * (1 + NB_SLOTS) if neighbor_map else own_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,),
                                            dtype=np.float32)

    def _rebind(self, env):
        if self.turnmap is None:
            return
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.std_action_map = self.green2std[tid]
            if hasattr(ts.observation_fn, "rebind_movements"):
                ts.observation_fn.rebind_movements(self.turnmap[tid])
            ts.observation_space = ts.observation_fn.observation_space()

    def _stack(self, states):
        own = {ts: np.asarray(states[ts], dtype=np.float32) for ts in self.ts_ids}
        if not self.neighbor_map:
            return np.stack([own[ts] for ts in self.ts_ids])
        zeros = np.zeros(self.own_dim, dtype=np.float32)
        rows = []
        for ts in self.ts_ids:
            nbs = self.neighbor_map.get(ts) or [None] * NB_SLOTS
            row = [own[ts]] + [own[nb] if nb is not None and nb in own else zeros
                               for nb in nbs[:NB_SLOTS]]
            rows.append(np.concatenate(row))
        return np.stack(rows)

    def reset(self):
        states = self.sumo_env.reset(self.sumo_env.sumo_seed)
        self._rebind(self.sumo_env)
        states = {tid: self.sumo_env.traffic_signals[tid].observation_fn()
                  for tid in self.sumo_env.ts_ids}
        return self._stack(states)

    def step_wait(self):
        action_dict = {ts: int(self.std2green[ts][int(self._actions[i])])
                       for i, ts in enumerate(self.ts_ids)}
        states, rewards, dones, _ = self.sumo_env.step(action=action_dict)
        obs = self._stack(states)
        rews = np.array([rewards[ts] for ts in self.ts_ids], dtype=np.float32)
        self.last_raw_rews[:] = rews
        # executed green -> canonical std (the rollout-buffer patch stores these)
        self.last_actual_actions = np.array(
            [float(self.green2std[ts][
                int(self.sumo_env.traffic_signals[ts].last_executed_action)])
             for ts in self.ts_ids], dtype=np.float32)
        all_done = bool(dones["__all__"])
        done_arr = np.full(self.num_envs, all_done, dtype=bool)
        infos = [{} for _ in self.ts_ids]
        if all_done:
            for hook in self.pre_reset_hooks:
                hook()
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i]
            obs = self.reset()
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


# ── CoLight-orig features extractor (mirror of CoLightOrigQNet ≤ cat_feat) ─────

class CoLightGATExtractor(BaseFeaturesExtractor):
    """Reproduces CoLightOrigQNet.forward exactly up to cat_feat: n_heads
    GATLayers (imported, identical math incl. zero-row missing mask) + own
    encoder, output = concat[own_feat, head aggs] of dim hidden*(n_heads+1).
    The DQN's q_head is replaced by SB3's policy/value heads."""

    def __init__(self, observation_space, own_dim, hidden_dim=128, n_heads=2):
        super().__init__(observation_space, hidden_dim * (n_heads + 1))
        self.own_dim = own_dim
        self.heads = torch.nn.ModuleList(
            [GATLayer(own_dim, hidden_dim) for _ in range(n_heads)])
        self.own_enc = torch.nn.Linear(own_dim, hidden_dim)

    def forward(self, obs):
        own = obs[:, : self.own_dim]
        nbs = obs[:, self.own_dim:].reshape(-1, NB_SLOTS, self.own_dim)
        mask = (nbs.abs().sum(dim=-1, keepdim=True) == 0)   # zero rows = missing
        aggs = [head(own, nbs, mask)[0] for head in self.heads]
        own_feat = F.relu(self.own_enc(own))
        return torch.cat([own_feat] + aggs, dim=-1)


# ── Callback: maskable rollout-buffer patch ────────────────────────────────────

class DublinMetricsCallback(SumoMetricsCallback):
    def _patch_rollout_buffer(self) -> None:
        """Maskable variant of the uni script's executed-action patch: add()
        carries action_masks, and the log-prob re-evaluation must pass the
        masks to the maskable policy (min_green may override the sampled
        action — storing the executed one keeps the gradient unbiased,
        identical rationale to the DQN trainers storing executed actions)."""
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

    if cfg.get("action_meta_file"):
        ts_mask, std2green, green2std, turnmap = load_meta(cfg["action_meta_file"])
    else:
        ts_mask, std2green, green2std, turnmap = identity_meta(sumo_env)

    coordination = cfg.get("coordination")
    assert coordination in (None, "colight_orig"), coordination
    neighbor_map = cfg.get("neighbor_map") if coordination else None
    if coordination:
        assert neighbor_map, "coordination: colight_orig requires neighbor_map"

    base_env = MaskedSumoVecEnv(sumo_env, ts_mask, std2green, green2std,
                                turnmap, neighbor_map=neighbor_map)
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

    policy_kwargs = {}
    if coordination:
        policy_kwargs = dict(
            features_extractor_class=CoLightGATExtractor,
            features_extractor_kwargs=dict(
                own_dim=base_env.own_dim,
                hidden_dim=cfg.get("hidden_dim", 128),
                n_heads=cfg.get("n_heads", 2)))

    model = MaskablePPO(
        policy="MlpPolicy", env=norm_env, verbose=0,
        device=(f"cuda:{gpu}" if gpu >= 0 else "cpu"),
        learning_rate=cfg.get("lr", 3e-4), n_steps=cfg.get("n_steps", 720),
        batch_size=cfg.get("batch_size", 720), n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 0.99), gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2), ent_coef=cfg.get("ent_coef", 0.01),
        vf_coef=cfg.get("vf_coef", 0.5), max_grad_norm=cfg.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
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
    p = argparse.ArgumentParser(description="Universal MaskablePPO trainer (1x1/1x3/Dublin, optional CoLight coordination)")
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
