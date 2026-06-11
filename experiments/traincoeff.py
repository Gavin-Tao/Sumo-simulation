"""Coefficient DQN training script.

Differences from train.py:
  - Uses CoeffDQN: standard MLP Q-network with learnable cooperative weight β.
  - Each agent's obs is augmented: [own_obs | nb_up | nb_down | nb_left | nb_right]
    (zeros for missing neighbors). obs_dim stays fixed regardless of grid size.
  - Effective TD reward: r_eff = r_local + sigmoid(β) * mean(neighbor_rewards)
  - β is an nn.Parameter optimised jointly with Q-network weights.
  - Neighbor map defined in YAML config under `neighbor_map`.
  - Logs β value to wandb each episode.

neighbor_map format in YAML (order: up, down, left, right; null = no neighbor):
    neighbor_map:
      "ts_0": [null, null, null, "ts_1"]
      "ts_1": [null, null, "ts_0", "ts_2"]
      "ts_2": [null, null, "ts_1", null]

Usage:
    python experiments/traincoeff.py --config experiments/configs/expXX_coeff.yaml
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb

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
from sumo_rl.agents.coefficient import CoeffDQN
from sumo_rl.agents.noisy_linear import NoisyLinear
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.environment.observations import (
    DefaultObservationFunction,
    PressLightObservationFunction,
    PressLightNormObservationFunction,
    QueuePrioObservationFunction,
    CTBPriorityObservationFunction,
    PriorityObservationFunction,
    PriorityCtrlObservationFunction,
    PriorityNormObservationFunction,
    DiffWaitingObservationFunction,
    PriorityDiffWaitingObservationFunction,
    PriorityBCAObservationFunction,
    PriorityCtrlBCAObservationFunction,
    PriorityWaitingBCAObservationFunction,
    PriorityPhaseObservationFunction,
    PriorityMovementObservationFunction,
    PriorityLaneTokenObservationFunction,
    PriorityPhaseLegacyCQ,
    PriorityPhaseLegacyCQM,
    PriorityPhaseLegacyCQMM,
    PriorityPhaseElapsedCQ,
    PriorityPhaseElapsedCQM,
    PriorityPhaseElapsedCQMM,
    PriorityMovementLegacyCQ,
    PriorityMovementLegacyCQM,
    PriorityMovementLegacyCQMM,
    PriorityMovementElapsedCQ,
    PriorityMovementElapsedCQM,
    PriorityMovementElapsedCQMM,
    PriorityLaneLegacyCQ,
    PriorityLaneLegacyCQM,
    PriorityLaneLegacyCQMM,
    PriorityLaneElapsedCQ,
    PriorityLaneElapsedCQM,
    PriorityLaneElapsedCQMM,
)

OBS_REGISTRY = {
    "Default":             DefaultObservationFunction,
    "PressLight":          PressLightObservationFunction,
    "PressLightNorm":      PressLightNormObservationFunction,
    "QueuePrio":           QueuePrioObservationFunction,
    "CTBPriority":         CTBPriorityObservationFunction,
    "Priority":            PriorityObservationFunction,
    "PriorityCtrl":        PriorityCtrlObservationFunction,
    "PriorityNorm":        PriorityNormObservationFunction,
    "DiffWaiting":         DiffWaitingObservationFunction,
    "PriorityDiffWaiting": PriorityDiffWaitingObservationFunction,
    "PriorityBCA":            PriorityBCAObservationFunction,
    "PriorityCtrlBCA":        PriorityCtrlBCAObservationFunction,
    "PriorityWaitingBCA":     PriorityWaitingBCAObservationFunction,
    # A/B/T base families (configure via obs_phase_state + obs_fields)
    "PriorityPhase":          PriorityPhaseObservationFunction,
    "PriorityMovement":       PriorityMovementObservationFunction,
    "PriorityLaneToken":      PriorityLaneTokenObservationFunction,
    # Scheme A — phase-level priority buckets (legacy vs perphase-elapsed × field-sets)
    "PriorityPhaseLegacyCQ":    PriorityPhaseLegacyCQ,
    "PriorityPhaseLegacyCQM":   PriorityPhaseLegacyCQM,
    "PriorityPhaseLegacyCQMM":  PriorityPhaseLegacyCQMM,
    "PriorityPhaseElapsedCQ":   PriorityPhaseElapsedCQ,
    "PriorityPhaseElapsedCQM":  PriorityPhaseElapsedCQM,
    "PriorityPhaseElapsedCQMM": PriorityPhaseElapsedCQMM,
    # Scheme B — movement-level priority buckets (12 turning slots)
    "PriorityMovementLegacyCQ":    PriorityMovementLegacyCQ,
    "PriorityMovementLegacyCQM":   PriorityMovementLegacyCQM,
    "PriorityMovementLegacyCQMM":  PriorityMovementLegacyCQMM,
    "PriorityMovementElapsedCQ":   PriorityMovementElapsedCQ,
    "PriorityMovementElapsedCQM":  PriorityMovementElapsedCQM,
    "PriorityMovementElapsedCQMM": PriorityMovementElapsedCQMM,
    # Scheme T (simplified) — lane-level priority buckets (flat, no static geometry yet)
    "PriorityLaneLegacyCQ":    PriorityLaneLegacyCQ,
    "PriorityLaneLegacyCQM":   PriorityLaneLegacyCQM,
    "PriorityLaneLegacyCQMM":  PriorityLaneLegacyCQMM,
    "PriorityLaneElapsedCQ":   PriorityLaneElapsedCQ,
    "PriorityLaneElapsedCQM":  PriorityLaneElapsedCQM,
    "PriorityLaneElapsedCQMM": PriorityLaneElapsedCQMM,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_aug_obs(ts_id: str, states: dict, neighbor_map: dict, obs_dim: int) -> np.ndarray:
    """Return own_obs concatenated with 4 neighbor obs (zeros if absent).
    Shape: [own_dim + 4 * obs_dim].  Neighbor order: [up, down, left, right].
    """
    own = np.array(states[ts_id], dtype=np.float32)
    nb_parts = []
    for nb_id in neighbor_map.get(ts_id, [None, None, None, None]):
        if nb_id is not None and nb_id in states:
            nb_parts.append(np.array(states[nb_id], dtype=np.float32))
        else:
            nb_parts.append(np.zeros(obs_dim, dtype=np.float32))
    return np.concatenate([own] + nb_parts)


def get_nb_rewards(ts_id: str, rewards: dict | None, neighbor_map: dict) -> np.ndarray:
    """Return per-direction neighbor rewards [up, down, left, right].
    Missing or unavailable neighbors are filled with 0.0.
    """
    nb_list = neighbor_map.get(ts_id, [None, None, None, None])
    result = []
    for nb in nb_list:
        if rewards is not None and nb is not None and nb in rewards and rewards[nb] is not None:
            result.append(float(rewards[nb]))
        else:
            result.append(0.0)
    return np.array(result, dtype=np.float32)


def get_noisy_sigma_stats(model: torch.nn.Module) -> dict:
    """Return mean |weight_sigma| for each NoisyLinear layer (for wandb logging)."""
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, NoisyLinear):
            stats[f"noisy/{name}_sigma"] = m.weight_sigma.abs().mean().item()
    return stats


def save_checkpoint(agent: CoeffDQN, episode: int, model_dir: str, filename: str = None) -> str:
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        "policy_state_dict":    agent.q_net.state_dict(),
        "target_state_dict":    agent.target_q_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "beta":                 agent.beta.data,
        "episode":              episode,
    }
    path = os.path.join(model_dir, filename or f"ckpt_ep{episode:05d}.pth")
    torch.save(checkpoint, path)
    return path


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, timestamp: str):
    exp_name = cfg["name"]
    run_id   = f"{exp_name}_{timestamp}"

    model_dir = os.path.join("./models", exp_name, timestamp)
    os.makedirs("./tmux", exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(cfg["cfg_file"]), "output"), exist_ok=True)

    set_seed(cfg.get("seed", 0))
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (no CUDA GPU available)")

    logging_mode = cfg.get("logging_mode", "basic")

    # Neighbor map: ts_id → [up, down, left, right] (None = no neighbor)
    neighbor_map: dict = cfg.get("neighbor_map", {})

    # ── M3 (Dublin): static per-junction action masks (mirrors train.py) ──
    action_meta_file = cfg.get("action_meta_file")
    if action_meta_file and cfg.get("use_per"):
        sys.exit("action_meta_file + use_per: masked PER not implemented yet")
    ts_mask, std2green, green2std, ts_turnmap = {}, {}, {}, {}
    if action_meta_file:
        import json as _json
        with open(action_meta_file) as _f:
            _meta = _json.load(_f)["tls"]
        for _tid, _t in _meta.items():
            ts_mask[_tid] = np.array(_t["mask"], dtype=bool)
            _s2g = np.full(8, -1, dtype=int)
            for _a, _gi in _t["std_to_green_index"].items():
                _s2g[int(_a)] = _gi
            std2green[_tid] = _s2g
            _g2s = np.full(int(_s2g.max()) + 1, -1, dtype=int)
            for _a in range(8):
                if _s2g[_a] >= 0:
                    _g2s[_s2g[_a]] = _a
            green2std[_tid] = _g2s
            ts_turnmap[_tid] = {int(_i): (_c[0]["approach"], _c[0]["turn"])
                                for _i, _c in _t["links"].items()}

    def _masked_reset(_env, _seed):
        _states = _env.reset(int(_seed))
        if action_meta_file:
            for _tid in _env.ts_ids:
                _ts = _env.traffic_signals[_tid]
                _ts.std_action_map = green2std[_tid]
                if hasattr(_ts.observation_fn, "rebind_movements"):
                    _ts.observation_fn.rebind_movements(ts_turnmap[_tid])
                _ts.observation_space = _ts.observation_fn.observation_space()
            _states = {_tid: _env.traffic_signals[_tid].observation_fn()
                       for _tid in _env.ts_ids}
        return _states

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
    # Optional obs params (A/B/T families): bind via partial so env can call obs_class(ts).
    obs_kwargs = {}
    if "obs_fields" in cfg:
        obs_kwargs["fields"] = tuple(cfg["obs_fields"])
    if "obs_phase_state" in cfg:
        obs_kwargs["phase_state"] = cfg["obs_phase_state"]
    if "priority_source" in cfg:
        obs_kwargs["priority_source"] = cfg["priority_source"]
    if "obs_phase_service" in cfg:   # PriorityLaneToken only: lane-identity multi-hot (R1)
        obs_kwargs["include_phase_service"] = bool(cfg["obs_phase_service"])
    if obs_kwargs:
        obs_class = functools.partial(obs_class, **obs_kwargs)
    # Table-driven reward (reward-side dual of φ): resolve to a callable here —
    # it needs this experiment's priority table, so it can't live in the registry.
    reward_fn = cfg["reward_fn"]
    if reward_fn == "priority-avg-waiting":
        from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
        from sumo_rl.environment.priority_map import load_priority_table
        reward_fn = make_priority_avg_waiting_reward(
            load_priority_table(cfg.get("priority_source")))
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
        single_agent=cfg.get("single_agent", False),
        yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5),
        reward_fn=reward_fn,
        observation_class=obs_class,
        sumo_seed=cfg.get("seed", 0),
        sumo_warnings=True,
    )

    episodes            = cfg.get("episodes", 5000)
    checkpoint_interval = cfg.get("checkpoint_interval", 5)
    eval_interval       = cfg.get("eval_interval", 0)      # 0 = disabled
    training_seed       = int(cfg.get("seed", 0))
    eval_seed           = cfg.get("eval_seed", 42)

    for _ in range(1, cfg.get("runs", 1) + 1):
        initial_states = _masked_reset(env, training_seed)
        obs_dim     = env.observation_space.shape[0]
        aug_obs_dim = obs_dim * 5              # own + 4 neighbors

        ts_lane_map  = {ts: env.traffic_signals[ts].signal_controlled_lanes for ts in env.ts_ids}
        always_green = set().union(*(env.traffic_signals[ts].always_green_lanes for ts in env.ts_ids))

        agent = CoeffDQN(
            aug_state_dim=aug_obs_dim,
            hidden_dim=cfg.get("hidden_dim", 64),
            action_dim=(8 if action_meta_file else env.action_space.n),
            learning_rate=cfg.get("lr", 0.01),
            gamma=cfg.get("gamma", 0.99),
            epsilon=cfg.get("epsilon", 0.1),
            target_update=cfg.get("target_update", 10),
            capacity=cfg.get("capacity", 10000),
            mini_size=cfg.get("mini_size", 500),
            batch_size=cfg.get("batch_size", 256),
            eps_start=cfg.get("eps_start", 0.5),
            eps_end=cfg.get("eps_end", 0.01),
            eps_decay=cfg.get("eps_decay", 1000),
            device=device,
            use_noisy=cfg.get("use_noisy", False),
            use_double=cfg.get("use_double", False),
            use_per=cfg.get("use_per", False),
            per_alpha=cfg.get("per_alpha", 0.6),
            per_beta_start=cfg.get("per_beta_start", 0.4),
            per_beta_end=cfg.get("per_beta_end", 1.0),
            per_beta_steps=cfg.get("per_beta_steps", 100_000),
            per_eps=cfg.get("per_eps", 1e-6),
        )

        step_counter = 0
        best_eval_reward = -float("inf")
        best_eval_hist: list = []   # sliding window for the best-ckpt criterion (B2 fix)

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = _masked_reset(env, training_seed)

            done = {"__all__": False}
            phase_counts = {ts_id: {} for ts_id in env.ts_ids}
            ep_losses: list = []

            try:
                while not done["__all__"]:
                    # ── Act (on augmented obs) — cache aug_obs for reuse in store ──
                    actions     = {}
                    aug_obs_now = {}
                    for ts in env.ts_ids:
                        aug_obs_now[ts] = get_aug_obs(ts, initial_states, neighbor_map, obs_dim)
                        if action_meta_file:
                            a_std = agent.take_action(aug_obs_now[ts], mask=ts_mask[ts])
                            actions[ts] = int(std2green[ts][a_std])
                        else:
                            actions[ts] = agent.take_action(aug_obs_now[ts])

                    s, r, done, info = env.step(action=actions)

                    if agent.loss is not None:
                        ep_losses.append(agent.loss)
                    step_counter += 1

                    for ts_id in env.ts_ids:
                        p = env.traffic_signals[ts_id].green_phase
                        phase_counts[ts_id][p] = phase_counts[ts_id].get(p, 0) + 1

                    # ── Store experience ──────────────────────────────────────
                    for ts in env.ts_ids:
                        actual_action  = env.traffic_signals[ts].last_executed_action
                        aug_obs_t      = aug_obs_now[ts]                                    # reuse cached
                        aug_obs_t1     = get_aug_obs(ts, s, neighbor_map, obs_dim)
                        nb_r           = get_nb_rewards(ts, r, neighbor_map)
                        if action_meta_file:
                            agent.replay_buffer.add(
                                aug_obs_t, int(green2std[ts][actual_action]),
                                r[ts], aug_obs_t1, done[ts],
                                nb_r, next_mask=ts_mask[ts],
                            )
                        else:
                            agent.replay_buffer.add(
                                aug_obs_t, actual_action,
                                r[ts], aug_obs_t1, done[ts],
                                nb_r,
                            )

                    initial_states = s

                    # ── Update ────────────────────────────────────────────────
                    if agent.replay_buffer.size() > agent.mini_size:
                        agent.epsilon = (
                            agent.eps_end
                            + (agent.eps_start - agent.eps_end)
                            * math.exp(-1.0 * agent.count / agent.eps_decay)
                        )
                        if agent.use_per:
                            b_s, b_a, b_r, b_ns, b_d, b_nb_r, b_w, b_idx = \
                                agent.replay_buffer.sample(agent.batch_size, beta=agent.current_beta)  # type: ignore[call-arg]
                            agent.update({
                                "states": b_s, "actions": b_a,
                                "next_states": b_ns, "rewards": b_r, "dones": b_d,
                                "nb_rewards": b_nb_r,
                                "weights": b_w, "indices": b_idx,
                            })
                        else:
                            _batch = agent.replay_buffer.sample(agent.batch_size)  # type: ignore[call-arg]
                            if len(_batch) == 7:
                                b_s, b_a, b_r, b_ns, b_d, b_nb_r, b_m = _batch
                            else:
                                b_s, b_a, b_r, b_ns, b_d, b_nb_r = _batch
                                b_m = None
                            _td = {
                                "states": b_s, "actions": b_a,
                                "next_states": b_ns, "rewards": b_r, "dones": b_d,
                                "nb_rewards": b_nb_r,
                            }
                            if b_m is not None:
                                _td["next_masks"] = b_m
                            agent.update(_td)

            except Exception as e:
                import traceback
                print(f"\n[ERROR] SUMO error at episode {episode}, step {step_counter}:")
                traceback.print_exc()
                print(f"[ERROR] Resetting and continuing from episode {episode + 1}...\n")
                try:
                    initial_states = _masked_reset(env, training_seed)
                except Exception as reset_err:
                    print(f"[ERROR] Reset also failed: {reset_err}")
                    raise
                continue

            if agent.start_train and logging_mode != "none":
                bv = agent.beta_value  # [up, down, left, right]
                print(f"[{exp_name}] ep={episode:5d}  epsilon={agent.epsilon:.4f}"
                      f"  β=[{bv[0]:.3f},{bv[1]:.3f},{bv[2]:.3f},{bv[3]:.3f}]")
                if logging_mode == "simple":
                    ep_log: dict = {}
                    if ep_losses:
                        ep_log["train/loss"] = sum(ep_losses) / len(ep_losses)
                else:  # basic / full
                    phase_log = {}
                    for ts_id, counts in phase_counts.items():
                        total = sum(counts.values()) or 1
                        for p, c in counts.items():
                            phase_log[f"phase/{ts_id}/phase{p}_ratio"] = c / total
                    noisy_stats = get_noisy_sigma_stats(agent.q_net) if agent.use_noisy else {}
                    ep_log = {
                        "train/episode":   episode,
                        "train/epsilon":   agent.epsilon,
                        "train/beta_up":   bv[0],
                        "train/beta_down": bv[1],
                        "train/beta_left": bv[2],
                        "train/beta_right":bv[3],
                        **phase_log,
                        **noisy_stats,
                    }
                    if ep_losses:
                        ep_log["train/loss"] = sum(ep_losses) / len(ep_losses)
                # Log PER β annealing (only meaningful when PER on; else it's a constant 1.0)
                if agent.use_per:
                    ep_log["train/per_beta"] = agent.current_beta
                if ep_log:
                    wandb.log(ep_log, step=step_counter)

                if episode % checkpoint_interval == 0:
                    ckpt_path = save_checkpoint(agent, episode, model_dir)
                    print(f"  → ckpt saved: {ckpt_path}")
                    wandb.save(ckpt_path, base_path=".")

            # ── Evaluation episode ────────────────────────────────────────────
            if eval_interval > 0 and episode % eval_interval == 0:
                eps_backup = agent.epsilon
                agent.epsilon = 0.0                          # greedy policy
                agent.q_net.eval()                           # disable noise if use_noisy

                eval_obs = _masked_reset(env, eval_seed)
                eval_done: dict = {"__all__": False}
                eval_mc = EpisodeMetricsCollector(
                    ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green
                )
                eval_ts_reward: dict = {ts: 0.0 for ts in env.ts_ids}
                while not eval_done["__all__"]:
                    eval_mc.collect_step(env.sumo)
                    eval_actions = {}
                    for ts in env.ts_ids:
                        aug = get_aug_obs(ts, eval_obs, neighbor_map, obs_dim)  # type: ignore[arg-type]
                        if action_meta_file:
                            a_std = agent.take_action(aug, mask=ts_mask[ts])
                            eval_actions[ts] = int(std2green[ts][a_std])
                        else:
                            eval_actions[ts] = agent.take_action(aug)
                    eval_obs, eval_rew, eval_done, _ = env.step(action=eval_actions)  # type: ignore[misc]
                    for ts in env.ts_ids:
                        eval_ts_reward[ts] += eval_rew.get(ts, 0.0)

                eval_mc.collect_step(env.sumo)               # capture final step (matches training pattern)
                eval_mc.finalize(env.sumo)
                eval_mean = sum(eval_ts_reward.values()) / len(env.ts_ids)
                # B2 fix (ABT_1X3_RESULTS_2026-06-11 Part 4): the best-ckpt
                # criterion is the SLIDING MEAN of the last K evals, not a
                # single-eval argmax — one lucky eval episode must not freeze
                # the best checkpoint (K = cfg best_eval_window, default 3;
                # comparison starts once the window is full).
                _bw = int(cfg.get("best_eval_window", 3))
                best_eval_hist.append(eval_mean)
                if len(best_eval_hist) > _bw:
                    del best_eval_hist[: len(best_eval_hist) - _bw]
                eval_sliding = sum(best_eval_hist) / len(best_eval_hist)
                if len(best_eval_hist) == _bw and eval_sliding > best_eval_reward:
                    best_eval_reward = eval_sliding
                    save_checkpoint(agent, episode, model_dir, filename="best.pth")
                    # Snapshot the best ckpt's full eval metrics (all scopes x vTypes x all metric fields)
                    # for offline analysis / thesis tables. Overwrites each new-best event.
                    best_metrics = {
                        "_meta": {
                            "episode":          episode,
                            "eval_mean_reward": float(best_eval_reward),
                            "eval_last_reward": float(eval_mean),
                            "best_criterion":   f"sliding{_bw}",
                            "timestamp":        timestamp,
                            "ckpt_filename":    "best.pth",
                        },
                        "metrics": eval_mc.summary(),
                    }
                    with open(os.path.join(model_dir, "best_metrics.json"), "w") as _f:
                        json.dump(best_metrics, _f, indent=2, default=float)
                    print(f"  → best ckpt updated (sliding{_bw}={best_eval_reward:.4f}, last={eval_mean:.4f})")
                if logging_mode != "none":
                    _BASIC = {
                        "avg_stopped_time", "avg_stop_events", "avg_speed",
                        # Metric A: per-vehicle normalized average (each vehicle 1 vote, k_v skip-empty)
                        "avg_stopped_time_per_visit", "avg_stop_events_per_visit",
                        # Metric B: cross-ts mean (each intersection 1 vote, skip ts where n_vehicles==0)
                        "xts_avg_stopped_time", "xts_avg_stop_events", "xts_avg_speed",
                    }
                    _BASIC_SYS = _BASIC | {"completion_rate"}
                    all_mc = {k.replace("eval/", "eval_", 1): v
                              for k, v in eval_mc.to_flat_dict(prefix="eval").items()}
                    if logging_mode == "full":
                        mc_log = all_mc
                    elif logging_mode == "basic":
                        mc_log = {
                            k: v for k, v in all_mc.items()
                            if (k.startswith("eval_system/") and k.split("/")[-1] in _BASIC_SYS)
                            or (not k.startswith("eval_system/") and k.split("/")[-1] in _BASIC)
                        }
                    else:  # simple
                        mc_log = {}
                    eval_reward_log = {f"eval_{ts}/reward": eval_ts_reward[ts] for ts in env.ts_ids}
                    eval_reward_log["eval_system/mean_reward"] = eval_mean
                    eval_reward_log["eval_system/best_reward"] = best_eval_reward
                    wandb.log({**mc_log, **eval_reward_log}, step=step_counter)
                print(f"  → eval ep={episode}")

                agent.epsilon = eps_backup                   # restore training epsilon
                agent.q_net.train()                          # re-enable noise if use_noisy

        env.close()

    if logging_mode != "none":
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoeffDQN cooperative trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_double", action="store_true", default=None,
                        help="Enable Double DQN (overrides YAML if set)")
    parser.add_argument("--use_per", action="store_true", default=None,
                        help="Enable Prioritized Experience Replay (overrides YAML if set)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI > YAML > default(False); write resolved value back into cfg so wandb logs it
    cfg["use_double"] = (
        args.use_double if args.use_double is not None
        else cfg.get("use_double", False)
    )
    cfg["use_per"] = (
        args.use_per if args.use_per is not None
        else cfg.get("use_per", False)
    )

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_save_dir = os.path.join("./models", cfg["name"], timestamp)
    os.makedirs(config_save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(config_save_dir, "config.yaml"))

    train(cfg, timestamp)
