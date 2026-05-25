"""Diagnostic: trace every ambulance individually during a long eval and compare
the raw per-vehicle waiting data against what EpisodeMetricsCollector would record.

If avg_wait reported by the metric ≠ mean of per-vehicle final getAccumulatedWaitingTime,
then there is a bug. If they match, the metric is correct.

Usage:
    python experiments/diagnose_amb_wait.py \
        --config configs/evaluation/eval_exp135_NS20bus_1amb_U.yaml \
        --ckpt   models/exp135_*/*/best.pth \
        --agent  orico
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ["SUMO_RL_LIBSUMO"] = "0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(EXP_DIR)

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("SUMO_HOME not set")

from eval_unified import OBS_REGISTRY, load_agent, pick_action
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.metrics import EpisodeMetricsCollector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--agent", choices=("dqn", "orico", "colight", "coeff"), required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-heads", type=int, default=2)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = glob.glob(args.ckpt)[0] if "*" in args.ckpt else args.ckpt

    device = torch.device(f"cuda:{args.gpu}"
                          if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    obs_class    = OBS_REGISTRY[cfg["observation_class"]]
    neighbor_map = cfg.get("neighbor_map", {})
    eval_seed    = int(cfg.get("seed", 0))

    env = SumoEnvironment(
        net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"],
        out_csv_name=None, use_gui=False,
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5), max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=cfg.get("single_agent", False),
        yellow_time=cfg.get("yellow_time", 2), delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"], observation_class=obs_class,
        sumo_seed=eval_seed,
    )

    initial_states = env.reset(eval_seed)
    obs_dim     = env.observation_space.shape[0]
    ts_lane_map = {ts: env.traffic_signals[ts].signal_controlled_lanes for ts in env.ts_ids}
    always_green = set().union(*(env.traffic_signals[ts].always_green_lanes for ts in env.ts_ids))

    agent = load_agent(args.agent, ckpt_path, cfg, obs_dim,
                       env.action_space.n, device, args.n_heads)

    mc = EpisodeMetricsCollector(
        ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green,
    )

    # Per-ambulance trace state
    amb_traces: dict = {}
    # vid -> { first_seen_t, last_seen_t, max_acc_wait_seen, final_acc_wait,
    #          waiting_time_integral (sum of getWaitingTime > 0 → +delta_time),
    #          completed (bool) }

    sim_time = 0.0
    delta_t  = env.delta_time
    done = {"__all__": False}

    while not done["__all__"]:
        mc.collect_step(env.sumo)

        # Track ambulances directly from SUMO
        for vid in env.sumo.vehicle.getIDList():
            try:
                vtype = env.sumo.vehicle.getTypeID(vid)
            except Exception:
                continue
            if vtype != "ambulance":
                continue
            try:
                acc_wait     = env.sumo.vehicle.getAccumulatedWaitingTime(vid)
                inst_wait    = env.sumo.vehicle.getWaitingTime(vid)
                speed        = env.sumo.vehicle.getSpeed(vid)
            except Exception:
                continue
            if vid not in amb_traces:
                amb_traces[vid] = {
                    "first_seen_t":     sim_time,
                    "last_seen_t":      sim_time,
                    "max_acc_wait":     acc_wait,
                    "final_acc_wait":   acc_wait,
                    "wait_integral":    0.0,    # ∑ delta_t where speed<0.1
                    "n_observations":   0,
                    "exit_t":           None,
                }
            tr = amb_traces[vid]
            tr["last_seen_t"]    = sim_time
            tr["final_acc_wait"] = acc_wait
            tr["max_acc_wait"]   = max(tr["max_acc_wait"], acc_wait)
            tr["n_observations"] += 1
            if speed < 0.1:
                tr["wait_integral"] += delta_t

        actions = {ts: pick_action(args.agent, agent, ts,
                                   initial_states, neighbor_map, obs_dim)
                   for ts in env.ts_ids}
        initial_states, _, done, _ = env.step(action=actions)
        sim_time += delta_t

    mc.collect_step(env.sumo)
    mc.finalize(env.sumo)
    env.close()

    # Find which ambulances completed (exited before sim end)
    for vid in amb_traces:
        # If last_seen_t is well before sim_time and we recorded their exit
        # vehicle was finalized by mc (not active at end)
        if vid in mc._finalized:
            amb_traces[vid]["completed"] = mc._finalized[vid].get("completed", False)
            amb_traces[vid]["mc_last_acc_wait"] = mc._finalized[vid]["last_acc_wait"]
        else:
            amb_traces[vid]["completed"] = False
            amb_traces[vid]["mc_last_acc_wait"] = None

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"DIAGNOSTIC: per-ambulance trace ({len(amb_traces)} ambulances)")
    print("=" * 100)
    print(f"{'vid':>20} {'depart':>7} {'last_t':>7} {'trip_dur':>9} "
          f"{'wait_int':>9} {'final_acc':>10} {'max_acc':>9} {'mc_last_acc':>12} {'completed':>10}")
    for vid, tr in sorted(amb_traces.items(), key=lambda kv: kv[1]["first_seen_t"]):
        trip_dur = tr["last_seen_t"] - tr["first_seen_t"]
        mc_v = tr["mc_last_acc_wait"]
        mc_str = f"{mc_v:>12.2f}" if isinstance(mc_v, (int, float)) else f"{str(mc_v):>12}"
        comp = "Y" if tr["completed"] else "n"
        print(f"{vid:>20} {tr['first_seen_t']:>7.0f} {tr['last_seen_t']:>7.0f} "
              f"{trip_dur:>9.0f} {tr['wait_integral']:>9.0f} "
              f"{tr['final_acc_wait']:>10.2f} {tr['max_acc_wait']:>9.2f} {mc_str} {comp:>10}")

    # Compare metric output with our raw trace
    print("\n" + "=" * 100)
    print("AGGREGATE COMPARISON")
    print("=" * 100)
    n_amb_seen   = len(amb_traces)
    n_completed  = sum(1 for tr in amb_traces.values() if tr["completed"])
    raw_mean_final_acc = np.mean([tr["final_acc_wait"]   for tr in amb_traces.values()])
    raw_mean_max_acc   = np.mean([tr["max_acc_wait"]     for tr in amb_traces.values()])
    raw_mean_wait_int  = np.mean([tr["wait_integral"]    for tr in amb_traces.values()])
    raw_mean_trip      = np.mean([tr["last_seen_t"] - tr["first_seen_t"] for tr in amb_traces.values()])

    # What EpisodeMetricsCollector reports
    summary = mc.summary()
    mc_amb_system = summary["system"]["ambulance"]
    print(f"  ambulances seen via TraCI:                {n_amb_seen}")
    print(f"  completed (exited network):               {n_completed}")
    print(f"  RAW mean(final getAccumulatedWaitingTime): {raw_mean_final_acc:.2f}s  ← capped at 100s by SUMO memory")
    print(f"  RAW mean(max  getAccumulatedWaitingTime):  {raw_mean_max_acc:.2f}s  ← peak in last-100s window")
    print(f"  RAW mean(∑ delta_t where speed<0.1):       {raw_mean_wait_int:.2f}s  ← TRUE total local wait")
    print(f"  RAW mean(trip duration):                   {raw_mean_trip:.2f}s")
    print()
    print(f"  mc.summary system.ambulance:")
    print(f"    n_vehicles      = {mc_amb_system.get('n_vehicles')}")
    print(f"    avg_wait        = {mc_amb_system.get('avg_wait'):.2f}s  ← mean of last_acc_wait stored by mc")
    print(f"    avg_waiting_inc = {mc_amb_system.get('avg_waiting_inc', 'N/A')}")
    print(f"    avg_stopped_time= {mc_amb_system.get('avg_stopped_time'):.2f}s")

    # Show per-ts comparison
    print()
    for ts in env.ts_ids:
        ts_amb = summary[ts]["ambulance"]
        n_ts = ts_amb["n_vehicles"]
        print(f"  mc per-ts [{ts}] ambulance:  n={n_ts}  avg_wait={ts_amb['avg_wait']:.2f}s  "
              f"avg_stopped_time={ts_amb['avg_stopped_time']:.2f}s  "
              f"avg_waiting_inc={ts_amb.get('avg_waiting_inc','N/A')}")


if __name__ == "__main__":
    main()
