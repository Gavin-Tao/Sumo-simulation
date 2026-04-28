"""Unified evaluation script for all DQN variants.

--agent choices:
  dqn      plain DQN (train.py)             obs: [obs_dim]
  coeff    CoeffDQN  (traincoeff.py)         obs: [obs_dim * 5]  (own + 4 neighbours concat)
  colight  CoLightDQN (traincolight.py)      obs: own [obs_dim] + nb [4, obs_dim]  single-head GAT
  orico    CoLightOrigDQN (original CoLight) obs: own [obs_dim] + nb [4, obs_dim]  multi-head GAT

Usage:
    python experiments/eval_unified.py --agent coeff \\
        --config experiments/configs/expXX_coeff.yaml \\
        --ckpt   models/expXX/TIMESTAMP/ckpt_epNNNNN.pth

    python experiments/eval_unified.py --agent orico --n-heads 4 \\
        --config experiments/configs/expXX_orico.yaml \\
        --ckpt   models/expXX/TIMESTAMP/ckpt_epNNNNN.pth
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
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

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.dqn_agent_txw import DQN
from sumo_rl.agents.coefficient import CoeffDQN
from sumo_rl.agents.colight import CoLightDQN
from sumo_rl.agents.colight_orig import CoLightOrigDQN
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
}

AGENT_TYPES = ("dqn", "coeff", "colight", "orico")


# ── Observation helpers ───────────────────────────────────────────────────────

def get_aug_obs(ts_id, states, neighbor_map, obs_dim):
    """CoeffDQN: flat concat [own | nb_up | nb_down | nb_left | nb_right]."""
    own   = np.array(states[ts_id], dtype=np.float32)
    parts = []
    for nb_id in neighbor_map.get(ts_id, [None, None, None, None]):
        if nb_id is not None and nb_id in states:
            parts.append(np.array(states[nb_id], dtype=np.float32))
        else:
            parts.append(np.zeros(obs_dim, dtype=np.float32))
    return np.concatenate([own] + parts)


def get_nb_obs(ts_id, states, neighbor_map, obs_dim):
    """CoLightDQN / CoLightOrigDQN: matrix [4, obs_dim], zeros for missing."""
    nb = np.zeros((4, obs_dim), dtype=np.float32)
    for i, nb_id in enumerate(neighbor_map.get(ts_id, [None, None, None, None])):
        if nb_id is not None and nb_id in states:
            nb[i] = np.array(states[nb_id], dtype=np.float32)
    return nb


# ── Agent factory + checkpoint loader ────────────────────────────────────────

def load_agent(agent_type, ckpt_path, cfg, obs_dim, action_space, device, n_heads):
    common = dict(
        learning_rate=cfg.get("lr", 0.01),
        gamma=cfg.get("gamma", 0.99),
        epsilon=0.0,
        target_update=cfg.get("target_update", 10),
        capacity=1, mini_size=1, batch_size=1,
        eps_start=0.0, eps_end=0.0, eps_decay=1,
        device=device,
    )
    hidden = cfg.get("hidden_dim", 64)
    ckpt   = torch.load(ckpt_path, map_location=device)

    if agent_type == "dqn":
        agent = DQN(
            starting_state=tuple([0.0] * obs_dim),
            state_space=obs_dim,
            hidden_dim=hidden,
            action_space=action_space,
            **common,
        )
        agent.q_net.load_state_dict(ckpt["policy_state_dict"])
        agent.q_net.eval()
        print(f"[dqn] Loaded {ckpt_path}  ep={ckpt.get('episode', '?')}")

    elif agent_type == "coeff":
        agent = CoeffDQN(
            aug_state_dim=obs_dim * 5,
            hidden_dim=hidden,
            action_dim=action_space,
            **common,
        )
        agent.q_net.load_state_dict(ckpt["policy_state_dict"])
        if "beta" in ckpt:
            agent.beta.data = ckpt["beta"]
        agent.q_net.eval()
        bv = torch.sigmoid(agent.beta).tolist()
        print(f"[coeff] Loaded {ckpt_path}  ep={ckpt.get('episode', '?')}"
              f"  β=[{bv[0]:.4f},{bv[1]:.4f},{bv[2]:.4f},{bv[3]:.4f}]")

    elif agent_type == "colight":
        agent = CoLightDQN(
            own_dim=obs_dim, nb_dim=obs_dim,
            hidden_dim=hidden,
            action_dim=action_space,
            **common,
        )
        agent.q_net.load_state_dict(ckpt["policy_state_dict"])
        agent.q_net.eval()
        print(f"[colight] Loaded {ckpt_path}  ep={ckpt.get('episode', '?')}")

    elif agent_type == "orico":
        agent = CoLightOrigDQN(
            obs_dim=obs_dim,
            hidden_dim=hidden,
            action_dim=action_space,
            n_heads=n_heads,
            **common,
        )
        agent.q_net.load_state_dict(ckpt["policy_state_dict"])
        agent.q_net.eval()
        print(f"[orico] Loaded {ckpt_path}  ep={ckpt.get('episode', '?')}"
              f"  n_heads={n_heads}")

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agent


# ── Action helper (hides per-type obs construction) ───────────────────────────

def pick_action(agent_type, agent, ts, states, neighbor_map, obs_dim):
    if agent_type == "dqn":
        return agent.take_action(states[ts])

    elif agent_type == "coeff":
        return agent.take_action(get_aug_obs(ts, states, neighbor_map, obs_dim))

    else:  # colight / orico
        own = np.array(states[ts], dtype=np.float32)
        nb  = get_nb_obs(ts, states, neighbor_map, obs_dim)
        return agent.take_action(own, nb)


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(agent_type, cfg, ckpt_path, n_episodes, use_gui,
             device, n_heads, wandb_log=False, warmup_steps=20,
             fixed_seed=False, debug=False):

    obs_class    = OBS_REGISTRY[cfg["observation_class"]]
    neighbor_map = cfg.get("neighbor_map", {})

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
        single_agent=cfg.get("single_agent", False),
        yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"],
        observation_class=obs_class,
        sumo_seed=cfg.get("seed", 0),
    )

    initial_states = env.reset(env.sumo_seed)
    obs_dim        = env.observation_space.shape[0]
    ts_lane_map    = {ts: env.traffic_signals[ts].signal_controlled_lanes for ts in env.ts_ids}
    always_green   = set().union(*(env.traffic_signals[ts].always_green_lanes for ts in env.ts_ids))

    agent = load_agent(agent_type, ckpt_path, cfg, obs_dim,
                       env.action_space.n, device, n_heads)

    all_episode_metrics: list[dict] = []
    base_seed = cfg.get("seed", 0)

    for episode in range(1, n_episodes + 1):
        seed           = base_seed if fixed_seed else base_seed + episode - 1
        initial_states = env.reset(seed)

        done           = {"__all__": False}
        mc             = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time,
                                                excluded_lanes=always_green)
        episode_reward = 0.0
        steps          = 0

        while not done["__all__"]:
            if steps >= warmup_steps:
                mc.collect_step(env.sumo)

            actions = {
                ts: pick_action(agent_type, agent, ts,
                                initial_states, neighbor_map, obs_dim)
                for ts in env.ts_ids
            }

            if debug:
                for ts in env.ts_ids:
                    ts_obj       = env.traffic_signals[ts]
                    car_q, bus_q = ts_obj.get_lanes_queue_by_type()
                    pre_r        = ts_obj.reward_fn(ts_obj)
                    print(f"  step={steps:3d} | ts={ts} | phase={ts_obj.green_phase} | "
                          f"action={actions[ts]} | pre_r={pre_r:7.1f} | "
                          f"car_q={[f'{v:.2f}' for v in car_q]} "
                          f"bus_q={[f'{v:.2f}' for v in bus_q]}")

            s, r, done, _ = env.step(action=actions)

            if debug:
                for ts in env.ts_ids:
                    print(f"           actual_r={r[ts]:.1f}")

            if steps >= warmup_steps:
                episode_reward += sum(v for v in r.values() if v is not None)
            initial_states = s
            steps         += 1

        mc.finalize(env.sumo)
        flat = mc.to_flat_dict(prefix="eval")
        flat["eval/episode_reward"] = episode_reward
        flat["eval/steps"]          = steps
        all_episode_metrics.append(flat)
        print(f"  episode {episode:3d}/{n_episodes}  reward={episode_reward:.2f}  steps={steps}")

    env.close()

    keys    = all_episode_metrics[0].keys()
    summary = {}
    for k in keys:
        vals       = np.array([ep[k] for ep in all_episode_metrics], dtype=float)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}

    print("\n" + "=" * 70)
    print(f"Evaluation summary  ({n_episodes} episodes)  [{agent_type}]")
    print("=" * 70)
    col_w = max(len(k) for k in summary) + 2
    for k, v in sorted(summary.items()):
        print(f"  {k:<{col_w}}  {v['mean']:>10.4f}  ± {v['std']:.4f}")
    print("=" * 70)

    out_dir  = Path(ckpt_path).parent
    out_path = out_dir / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump({"agent_type": agent_type,
                   "summary":    summary,
                   "episodes":   all_episode_metrics}, f, indent=2)
    print(f"\nSaved: {out_path}")

    if wandb_log:
        import wandb
        exp_name = cfg["name"]
        wandb.init(
            project=cfg.get("wandb_project", "sumo-rl"),
            group=exp_name,
            name=f"{exp_name}_eval_{Path(ckpt_path).stem}",
            config={**cfg, "agent_type": agent_type},
            dir="./logs/wandb",
        )
        wandb.log({f"{k}_mean": v["mean"] for k, v in summary.items()})
        wandb.log({f"{k}_std":  v["std"]  for k, v in summary.items()})
        wandb.finish()

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified DQN evaluator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--agent",      required=True, choices=AGENT_TYPES,
                        help="Agent type: dqn | coeff | colight | orico")
    parser.add_argument("--config",     required=True,          help="Path to YAML config")
    parser.add_argument("--ckpt",       required=True,          help="Path to .pth checkpoint")
    parser.add_argument("--episodes",   type=int,  default=1,   help="Number of eval episodes")
    parser.add_argument("--gui",        action="store_true",    help="Enable SUMO GUI")
    parser.add_argument("--gpu",        type=int,  default=0,   help="GPU index; -1 for CPU")
    parser.add_argument("--n-heads",    type=int,  default=None,
                        help="GAT heads for orico (overrides config n_heads, default 2)")
    parser.add_argument("--wandb",      action="store_true",    help="Log summary to wandb")
    parser.add_argument("--warmup",     type=int,  default=20,  help="Steps before metrics collection")
    parser.add_argument("--fixed-seed", action="store_true",    help="Use config seed for all episodes")
    parser.add_argument("--debug",      action="store_true",    help="Print per-step debug info")
    args = parser.parse_args()

    device = (torch.device(f"cuda:{args.gpu}")
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device("cpu"))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # n_heads: CLI flag > config key > default 2
    n_heads = args.n_heads if args.n_heads is not None else cfg.get("n_heads", 2)

    run_eval(args.agent, cfg, args.ckpt, args.episodes, args.gui, device,
             n_heads=n_heads, wandb_log=args.wandb, warmup_steps=args.warmup,
             fixed_seed=args.fixed_seed, debug=args.debug)
