import argparse
import os
import sys
from datetime import datetime
import torch
import random
import numpy as np
from pathlib import Path
import csv

# 用libsumo
os.environ["SUMO_RL_LIBSUMO"] = "0"
# 或用traci
# os.environ["SUMO_RL_LIBSUMO"] = "0"

# =========================
# SUMO / project path
# =========================
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
probe = PROJECT_ROOT
while not (probe / "sumo_rl").exists() and probe.parent != probe:
    probe = probe.parent
sys.path.insert(0, str(probe))

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.dqn_agent_txw import DQN
from sumo_rl.environment.observations import PressLightObservationFunction
from torch.utils.tensorboard import SummaryWriter


# =========================
# Utils
# =========================
def get_sumo(env):
    if not hasattr(env, "sumo") or env.sumo is None:
        raise RuntimeError("SUMO not started correctly.")
    return env.sumo


# =========================
# Evaluation
# =========================
def run_eval(env, agent, episodes, out_csv, tb_writer):

    headers = [
        "episode",
        "vtype",
        "throughput_completed",   # completed vehicles
        "avg_travel_time",
        "avg_stopped_time",
        "avg_stop_count",
        "avg_speed",
        "departed_total",         # NEW
        "arrived_total",          # NEW
        "running_end",            # NEW
        "teleport_total",         # NEW
    ]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for ep in range(1, episodes + 1):
            print(f"\n========== Eval episode {ep} ==========")

            obs = env.reset(env.sumo_seed)
            sumo = get_sumo(env)

            print(
                f"[CHECK] t={sumo.simulation.getTime()} | "
                f"in-network={len(sumo.vehicle.getIDList())}"
            )

            agent.epsilon = 0.0
            agent.q_net.eval()

            # -------------------------------
            # Episode-level counters (NEW)
            # -------------------------------
            departed_total = 0
            arrived_total = 0
            teleport_total = 0

            # -------------------------------
            # Per-vehicle lifecycle records
            # -------------------------------
            veh_stats = {}
            completed = {}

            done = {"__all__": False}

            while not done["__all__"]:
                actions = {
                    ts: agent.take_action(obs[ts])
                    for ts in env.ts_ids
                }

                obs, reward, done, info = env.step(actions)
                sim_time = sumo.simulation.getTime()

                departed_ids = sumo.simulation.getDepartedIDList()
                arrived_ids = sumo.simulation.getArrivedIDList()

                departed_total += len(departed_ids)
                arrived_total += len(arrived_ids)

                try:
                    teleport_total += sumo.simulation.getStartingTeleportNumber()
                except Exception:
                    pass

                # ---------------------------
                # Departed vehicles
                # ---------------------------
                for vid in departed_ids:
                    vt = sumo.vehicle.getTypeID(vid)
                    veh_stats[vid] = {
                        "vtype": vt,
                        "depart_time": sim_time,
                        "speed_sum": 0.0,
                        "speed_cnt": 0,
                        "stopped_time": 0.0,
                        "stop_count": 0,
                        "prev_speed": None,
                    }

                # ---------------------------
                # In-network update
                # ---------------------------
                for vid in sumo.vehicle.getIDList():
                    if vid not in veh_stats:
                        continue

                    rec = veh_stats[vid]
                    speed = sumo.vehicle.getSpeed(vid)

                    rec["speed_sum"] += speed
                    rec["speed_cnt"] += 1

                    if speed < 0.1:
                        rec["stopped_time"] += env.delta_time

                    if rec["prev_speed"] is not None:
                        if rec["prev_speed"] >= 0.1 and speed < 0.1:
                            rec["stop_count"] += 1

                    rec["prev_speed"] = speed

                # ---------------------------
                # Arrived vehicles
                # ---------------------------
                for vid in arrived_ids:
                    if vid not in veh_stats:
                        continue

                    rec = veh_stats.pop(vid)
                    rec["travel_time"] = sim_time - rec["depart_time"]
                    rec["avg_speed"] = rec["speed_sum"] / max(1, rec["speed_cnt"])
                    completed[vid] = rec

            # episode-end remaining vehicles
            running_end = len(sumo.vehicle.getIDList())

            # ---------------------------------
            # Aggregate by vehicle type
            # ---------------------------------
            type_stats = {}

            for rec in completed.values():
                vt = rec["vtype"]
                r = type_stats.setdefault(
                    vt,
                    {
                        "n": 0,
                        "travel": 0.0,
                        "stop_time": 0.0,
                        "stop_count": 0.0,
                        "speed": 0.0,
                    },
                )

                r["n"] += 1
                r["travel"] += rec["travel_time"]
                r["stop_time"] += rec["stopped_time"]
                r["stop_count"] += rec["stop_count"]
                r["speed"] += rec["avg_speed"]

            for vt, r in type_stats.items():
                r["travel"] /= r["n"]
                r["stop_time"] /= r["n"]
                r["stop_count"] /= r["n"]
                r["speed"] /= r["n"]

                writer.writerow([
                    ep,
                    vt,
                    r["n"],
                    r["travel"],
                    r["stop_time"],
                    r["stop_count"],
                    r["speed"],
                    departed_total,
                    arrived_total,
                    running_end,
                    teleport_total,
                ])

                tb_writer.add_scalar(f"eval/throughput_completed/{vt}", r["n"], ep)

            print(
                f"[EP{ep}] departed={departed_total}, "
                f"arrived={arrived_total}, "
                f"running_end={running_end}, "
                f"teleport={teleport_total}"
            )

    print("\n✅ Evaluation finished")
    print("CSV saved to:", out_csv)


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=1000)
    parser.add_argument("--out", type=str, default="./logs/eval")
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    tb_writer = SummaryWriter(os.path.join(args.out, stamp))

    env = SumoEnvironment(
        net_file="../nets/syc/1x3/Eq_350_BC_Wstr_Estr-Wleft_Eleft/"
                 "Eq_350_100-0-100-0-Bus-Car/1x3_4phases.net.xml",
        route_file="../nets/syc/1x3/Eq_350_BC_Wstr_Estr-Wleft_Eleft/"
                   "Eq_350_100-0-100-0-Bus-Car/"
                   "Eq_350_100-0-100-0-Bus-Car.rou.xml",
        cfg_file="../nets/syc/1x3/Eq_350_BC_Wstr_Estr-Wleft_Eleft/"
                 "Eq_350_100-0-100-0-Bus-Car/"
                 "Eq_350_100-0-100-0-Bus-Car.sumocfg",
        use_gui=True,
        num_seconds=args.seconds,
        min_green=5,
        max_green=50,
        use_max_green=True,
        sumo_seed=seed,
        observation_class=PressLightObservationFunction,
        reward_fn="pressure",
        delta_time=5,
        single_agent=False,
    )

    obs = env.reset(seed)
    ts = env.ts_ids[-1]

    agent = DQN(
        starting_state=tuple(obs[ts]),
        state_space=env.observation_space.shape[0],
        hidden_dim=64,
        action_space=env.action_space.n,
        learning_rate=0.01,
        gamma=0.99,
        epsilon=0.0,
        target_update=10,
        capacity=10000,
        mini_size=500,
        batch_size=256,
        eps_start=0.5,
        eps_end=0.01,
        eps_decay=1000,
        device=torch.device("cpu"),
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    agent.q_net.load_state_dict(ckpt["policy_state_dict"])

    out_csv = os.path.join(args.out, f"eval_results_{stamp}.csv")
    run_eval(env, agent, args.episodes, out_csv, tb_writer)

    env.close()
    tb_writer.close()
