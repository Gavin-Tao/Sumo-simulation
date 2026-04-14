"""Unified DQN training script for SUMO-RL.

Usage:
    python experiments/train.py --config experiments/configs/exp18_presslight.yaml

Each experiment is fully described by its YAML config file.
Run name in wandb: {exp_name}_{timestamp}  — distinguishes both
  different experiments (by exp_name) and repeated runs (by timestamp).
Checkpoints: ./models/{exp_name}/{timestamp}/ckpt_ep{episode:05d}.pth
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

# ── Project root ──────────────────────────────────────────────────────────────
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
from sumo_rl.agents.dqn_agent_txw import DQN
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.environment.observations import (
    DefaultObservationFunction,
    PressLightObservationFunction,
    PressLightNormObservationFunction,
    QueuePrioObservationFunction,
    CTBPriorityObservationFunction,
    PriorityObservationFunction,
)

# ── Registries ────────────────────────────────────────────────────────────────
OBS_REGISTRY = {
    "Default":         DefaultObservationFunction,
    "PressLight":      PressLightObservationFunction,
    "PressLightNorm":  PressLightNormObservationFunction,
    "QueuePrio":       QueuePrioObservationFunction,
    "CTBPriority":     CTBPriorityObservationFunction,
    "Priority":        PriorityObservationFunction,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(agent: DQN, episode: int, model_dir: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        "policy_state_dict":   agent.q_net.state_dict(),
        "target_state_dict":   agent.target_q_net.state_dict(),
        "optimizer_state_dic": agent.optimizer.state_dict(),
        "episode":             episode,
    }
    path = os.path.join(model_dir, f"ckpt_ep{episode:05d}.pth")
    torch.save(checkpoint, path)
    return path


# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg: dict, timestamp: str):
    exp_name = cfg["name"]
    run_id   = f"{exp_name}_{timestamp}"

    # ── Output paths (all grouped by exp_name / timestamp) ────────────────────
    model_dir = os.path.join("./models",  exp_name, timestamp)
    out_csv   = os.path.join("./outputs", exp_name, timestamp, "metrics")
    os.makedirs(os.path.join("./outputs", exp_name, timestamp), exist_ok=True)

    # ── Seed ──────────────────────────────────────────────────────────────────
    set_seed(cfg.get("seed", 0))
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # logging_mode: "none" | "basic" | "full"
    #   none  – no wandb logging at all
    #   basic – reward / loss / epsilon only  (default)
    #   full  – basic + EpisodeMetricsCollector every metrics_interval episodes
    logging_mode = cfg.get("logging_mode", "basic")

    # ── wandb ─────────────────────────────────────────────────────────────────
    if logging_mode != "none":
        wandb.init(
            project=cfg.get("wandb_project", "sumo-rl"),
            group=exp_name,
            name=run_id,
            config=cfg,
            dir="./logs/wandb",
            reinit=True,
        )

    # ── Environment ───────────────────────────────────────────────────────────
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
    metrics_interval    = cfg.get("metrics_interval", 50)  # collect full metrics every N episodes

    # ── Training ──────────────────────────────────────────────────────────────
    for run in range(1, cfg.get("runs", 1) + 1):
        initial_states = env.reset(env.sumo_seed)
        last_ts_id = list(env.ts_ids)[-1]

        # Build lane map once (lane structure is fixed across episodes)
        ts_lane_map = {ts: env.traffic_signals[ts].lanes for ts in env.ts_ids}

        agent = DQN(
            starting_state=tuple(initial_states[last_ts_id]),
            state_space=env.observation_space.shape[0],
            hidden_dim=cfg.get("hidden_dim", 64),
            action_space=env.action_space.n,
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
            # Only create the collector on episodes where we will log full metrics
            do_full = (logging_mode == "full") and (episode % metrics_interval == 0)
            mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time) if do_full else None
            phase_counts = {ts_id: {} for ts_id in env.ts_ids}

            try:
                while not done["__all__"]:
                    # ── Collect metrics (only on designated episodes) ─────────────
                    if mc is not None:
                        mc.collect_step(env.sumo)

                    # ── Act ───────────────────────────────────────────────────────
                    actions = {ts: agent.take_action(initial_states[ts]) for ts in env.ts_ids}
                    s, r, done, info = env.step(action=actions)

                    # ── Log to wandb per-step (basic & full both log reward/loss) ──
                    if logging_mode != "none":
                        log_dict = {k: v for k, v in info.items()}  # type: ignore[union-attr]
                        for ts_id in env.ts_ids:
                            if r[ts_id] is not None:  # type: ignore[index]
                                log_dict["reward_" + ts_id] = r[ts_id]  # type: ignore[index]
                            if agent.loss is not None:
                                log_dict["loss_" + ts_id] = agent.loss
                        wandb.log(log_dict, step=step_counter)
                    step_counter += 1

                    # ── Track phase selection ─────────────────────────────────────
                    for ts_id in env.ts_ids:
                        p = env.traffic_signals[ts_id].green_phase
                        phase_counts[ts_id][p] = phase_counts[ts_id].get(p, 0) + 1

                    # ── Store experience ──────────────────────────────────────────
                    for ts in env.ts_ids:
                        actual_action = env.traffic_signals[ts].last_executed_action
                        ts_reward = r[ts]  # type: ignore[index]
                        ts_next_state = tuple(s[ts])  # type: ignore[index]
                        ts_done = done[ts]  # type: ignore[index]
                        agent.replay_buffer.add(
                            initial_states[ts], actual_action,
                            ts_reward, ts_next_state, ts_done,
                        )

                    initial_states = s

                    # ── Update ────────────────────────────────────────────────────
                    if agent.replay_buffer.size() > agent.mini_size:
                        b_s, b_a, b_r, b_ns, b_d = agent.replay_buffer.sample(agent.batch_size)
                        agent.epsilon = (
                            agent.eps_end
                            + (agent.eps_start - agent.eps_end)
                            * math.exp(-1.0 * agent.count / agent.eps_decay)
                        )
                        agent.update({
                            "states": b_s, "actions": b_a,
                            "next_states": b_ns, "rewards": b_r, "dones": b_d,
                        })
            except Exception as e:
                import traceback
                print(f"\n[ERROR] SUMO error at episode {episode}, step {step_counter}:")
                traceback.print_exc()
                print(f"[ERROR] Resetting environment and continuing from episode {episode + 1}...\n")
                try:
                    initial_states = env.reset(env.sumo_seed)
                except Exception as reset_err:
                    print(f"[ERROR] Reset also failed: {reset_err}")
                    raise
                continue

            # ── End of episode ────────────────────────────────────────────────
            if env.metrics is not None:
                for metric in env.metrics:
                    env.list_metrics.append(metric)

            # ── Full metrics log (only on designated episodes) ────────────────
            if mc is not None:
                mc.collect_step(env.sumo)   # ← 补采终止步
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
    parser = argparse.ArgumentParser(description="SUMO-RL DQN unified trainer")
    parser.add_argument("--config", default="./configs/exp18_presslight.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0); -1 for CPU")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Timestamp fixed once per launch so all paths are consistent
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Save a copy of the config next to the checkpoints for reproducibility
    config_save_dir = os.path.join("./models", cfg["name"], timestamp)
    os.makedirs(config_save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(config_save_dir, "config.yaml"))

    train(cfg, timestamp)
