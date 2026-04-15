"""CoLight DQN training script.

Differences from train.py:
  - Uses CoLightDQN (GAT Q-network) instead of DQN.
  - Each agent's input is split into own_obs + neighbor_obs [4, obs_dim].
  - Neighbor map defined in YAML config under `neighbor_map`.
  - Replay buffer stores (own_obs, a, r, next_own_obs, done, nb_obs, next_nb_obs).

neighbor_map format in YAML (order: up, down, left, right; null = no neighbor):
    neighbor_map:
      "ts_0": [null, null, null, "ts_1"]
      "ts_1": [null, null, "ts_0", "ts_2"]
      "ts_2": [null, null, "ts_1", null]

Usage:
    python experiments/traincolight.py --config experiments/configs/expXX_colight.yaml
"""

import argparse
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
from sumo_rl.agents.colight import CoLightDQN
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
    "Default":             DefaultObservationFunction,
    "PressLight":          PressLightObservationFunction,
    "PressLightNorm":      PressLightNormObservationFunction,
    "QueuePrio":           QueuePrioObservationFunction,
    "CTBPriority":         CTBPriorityObservationFunction,
    "Priority":            PriorityObservationFunction,
    "PriorityNorm":        PriorityNormObservationFunction,
    "DiffWaiting":         DiffWaitingObservationFunction,
    "PriorityDiffWaiting": PriorityDiffWaitingObservationFunction,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_nb_obs(ts_id: str, states: dict, neighbor_map: dict, obs_dim: int) -> np.ndarray:
    """Return neighbor obs matrix [4, obs_dim]. Zeros for missing neighbors.
    Neighbor order: [up, down, left, right].
    """
    nb_obs = np.zeros((4, obs_dim), dtype=np.float32)
    for i, nb_id in enumerate(neighbor_map.get(ts_id, [None, None, None, None])):
        if nb_id is not None and nb_id in states:
            nb_obs[i] = np.array(states[nb_id], dtype=np.float32)
    return nb_obs


def save_checkpoint(agent: CoLightDQN, episode: int, model_dir: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        "policy_state_dict":    agent.q_net.state_dict(),
        "target_state_dict":    agent.target_q_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "episode":              episode,
    }
    path = os.path.join(model_dir, f"ckpt_ep{episode:05d}.pth")
    torch.save(checkpoint, path)
    return path


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, timestamp: str):
    exp_name = cfg["name"]
    run_id   = f"{exp_name}_{timestamp}"

    model_dir = os.path.join("./models",  exp_name, timestamp)
    out_csv   = os.path.join("./outputs", exp_name, timestamp, "metrics")
    os.makedirs(os.path.join("./outputs", exp_name, timestamp), exist_ok=True)

    set_seed(cfg.get("seed", 0))
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 and torch.cuda.is_available() else torch.device("cpu")

    logging_mode = cfg.get("logging_mode", "basic")

    # Neighbor map: ts_id → [up, down, left, right] (None = no neighbor)
    neighbor_map: dict = cfg.get("neighbor_map", {})

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
        out_csv_name=out_csv,
        use_gui=cfg.get("use_gui", False),
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5),
        max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=cfg.get("single_agent", False),
        yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"],
        observation_class=obs_class,
        sumo_seed=cfg.get("seed", 0),
        sumo_warnings=True,
    )

    episodes            = cfg.get("episodes", 5000)
    checkpoint_interval = cfg.get("checkpoint_interval", 5)
    metrics_interval    = cfg.get("metrics_interval", 50)

    for run in range(1, cfg.get("runs", 1) + 1):
        initial_states = env.reset(env.sumo_seed)
        obs_dim = env.observation_space.shape[0]   # own obs dimension = nb_dim

        ts_lane_map  = {ts: env.traffic_signals[ts].signal_controlled_lanes for ts in env.ts_ids}
        always_green = set().union(*(env.traffic_signals[ts].always_green_lanes for ts in env.ts_ids))

        agent = CoLightDQN(
            own_dim=obs_dim,
            nb_dim=obs_dim,                         # same obs type for all agents
            hidden_dim=cfg.get("hidden_dim", 64),
            action_dim=env.action_space.n,
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
        )

        step_counter = 0

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset(env.sumo_seed)

            done = {"__all__": False}
            do_full = (logging_mode == "full") and (episode % metrics_interval == 0)
            mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time,
                                         excluded_lanes=always_green) if do_full else None
            phase_counts = {ts_id: {} for ts_id in env.ts_ids}

            try:
                while not done["__all__"]:
                    if mc is not None:
                        mc.collect_step(env.sumo)

                    # ── Act ───────────────────────────────────────────────────
                    actions = {}
                    for ts in env.ts_ids:
                        nb_obs = get_nb_obs(ts, initial_states, neighbor_map, obs_dim)
                        actions[ts] = agent.take_action(initial_states[ts], nb_obs)

                    s, r, done, info = env.step(action=actions)

                    # ── Log ───────────────────────────────────────────────────
                    if logging_mode != "none":
                        log_dict = {k: v for k, v in info.items()}
                        for ts_id in env.ts_ids:
                            if r[ts_id] is not None:
                                log_dict["reward_" + ts_id] = r[ts_id]
                            if agent.loss is not None:
                                log_dict["loss_" + ts_id] = agent.loss
                        wandb.log(log_dict, step=step_counter)
                    step_counter += 1

                    for ts_id in env.ts_ids:
                        p = env.traffic_signals[ts_id].green_phase
                        phase_counts[ts_id][p] = phase_counts[ts_id].get(p, 0) + 1

                    # ── Store experience ──────────────────────────────────────
                    for ts in env.ts_ids:
                        actual_action  = env.traffic_signals[ts].last_executed_action
                        nb_obs_t       = get_nb_obs(ts, initial_states, neighbor_map, obs_dim)
                        nb_obs_t1      = get_nb_obs(ts, s,              neighbor_map, obs_dim)
                        agent.replay_buffer.add(
                            initial_states[ts], actual_action,
                            r[ts], tuple(s[ts]), done[ts],
                            nb_obs_t, nb_obs_t1,
                        )

                    initial_states = s

                    # ── Update ────────────────────────────────────────────────
                    if agent.replay_buffer.size() > agent.mini_size:
                        b_s, b_a, b_r, b_ns, b_d, b_nb, b_next_nb = agent.replay_buffer.sample(agent.batch_size)
                        agent.epsilon = (
                            agent.eps_end
                            + (agent.eps_start - agent.eps_end)
                            * math.exp(-1.0 * agent.count / agent.eps_decay)
                        )
                        agent.update({
                            "states": b_s, "actions": b_a,
                            "next_states": b_ns, "rewards": b_r, "dones": b_d,
                            "nb_obs": b_nb, "next_nb_obs": b_next_nb,
                        })

            except Exception as e:
                import traceback
                print(f"\n[ERROR] SUMO error at episode {episode}, step {step_counter}:")
                traceback.print_exc()
                print(f"[ERROR] Resetting and continuing from episode {episode + 1}...\n")
                try:
                    initial_states = env.reset(env.sumo_seed)
                except Exception as reset_err:
                    print(f"[ERROR] Reset also failed: {reset_err}")
                    raise
                continue

            if env.metrics is not None:
                for metric in env.metrics:
                    env.list_metrics.append(metric)

            if mc is not None:
                mc.collect_step(env.sumo)
                mc.finalize(env.sumo)
                wandb.log(mc.to_flat_dict(prefix="metrics"), step=step_counter)

            if agent.start_train and logging_mode != "none":
                phase_log = {}
                for ts_id, counts in phase_counts.items():
                    total = sum(counts.values()) or 1
                    for p, c in counts.items():
                        phase_log[f"phase/{ts_id}/phase{p}_ratio"] = c / total
                print(f"[{exp_name}] ep={episode:5d}  epsilon={agent.epsilon:.4f}  phases={phase_counts}")
                wandb.log({"train/episode": episode, "train/epsilon": agent.epsilon, **phase_log},
                          step=step_counter)

                if episode % checkpoint_interval == 0:
                    ckpt_path = save_checkpoint(agent, episode, model_dir)
                    print(f"  → ckpt saved: {ckpt_path}")
                    wandb.save(ckpt_path, base_path=".")

        env.txw_save_csv(out_csv, run)
        env.close()

    if logging_mode != "none":
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoLight DQN trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    config_save_dir = os.path.join("./models", cfg["name"], timestamp)
    os.makedirs(config_save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(config_save_dir, "config.yaml"))

    train(cfg, timestamp)
