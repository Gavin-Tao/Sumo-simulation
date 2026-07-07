"""exp215 (MoE, latest ckpt, greedy, eval_seed): count steps where a TLS has
zero vehicles on ALL its approach lanes yet the executed action switches phase.
Also per-class per-visit metrics via EpisodeMetricsCollector for the same episode."""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
cfg = yaml.safe_load(open("configs/exp215_dublin11h_531_moe_enum.yaml"))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.agents.dqn_agent_txw import DQN
import moe_glue
obs_class = functools.partial(obsmod.PriorityMovementObservationFunction,
    fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
    priority_source=cfg["priority_source"], awt_cap=float(cfg["obs_awt_cap"]),
    awt_basis=cfg["obs_awt_basis"],
    include_downstream=bool(cfg.get("obs_downstream", False)),
    downstream_fields=tuple(cfg.get("obs_downstream_fields", ())),
    include_lane_occ=bool(cfg.get("obs_lane_occ", False)),
    slot_stats=str(cfg.get("obs_slot_stats", "intent")))
env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
    cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
    num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
    max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
    single_agent=False, yellow_time=cfg["yellow_time"], delta_time=cfg["delta_time"],
    reward_fn=make_priority_avg_waiting_reward(load_priority_table(cfg["priority_source"])),
    observation_class=obs_class, sumo_seed=cfg["seed"], sumo_warnings=False)
moe = moe_glue.load_moe_tables(cfg["moe_meta_file"])
env.reset(int(cfg["eval_seed"]))
for tid in env.ts_ids:
    ts = env.traffic_signals[tid]
    ts.observation_fn.rebind_movements(moe["turnmap"][tid])
    ts.observation_space = ts.observation_fn.observation_space()
states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
experts = moe_glue.build_experts(cfg, moe, env)
od = len(next(iter(states.values())))
agent = DQN(starting_state=tuple([0.0]*od), state_space=od, hidden_dim=cfg["hidden_dim"],
    action_space=6, learning_rate=1e-3, gamma=0.95, epsilon=0.0, target_update=10,
    capacity=100, mini_size=10**9, batch_size=1, eps_start=0, eps_end=0, eps_decay=1, device="cpu")
ck = sorted(glob.glob(f"models/{cfg['name']}/2026-07-03T19-29-06/ckpt_ep*.pth"))[-1]
agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
agent.q_net.eval()

ts_lane_map = {t: env.traffic_signals[t].signal_controlled_lanes for t in env.ts_ids}
always_green = set().union(*(env.traffic_signals[t].always_green_lanes for t in env.ts_ids))
mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green)

empty_steps = 0          # (ts, step) with zero vehicles on all approach lanes
empty_switch = 0         # among those, executed action switches the phase
empty_switch_k = []      # gate expert index chosen on those events
total_ts_steps = 0
done = {"__all__": False}
while not done["__all__"]:
    mc.collect_step(env.sumo)
    acts = {}
    for t in env.ts_ids:
        ts = env.traffic_signals[t]
        props, lv = experts.propose(t, ts.sumo, ts.green_phase)
        m = moe_glue.gate_mask(lv, bool(cfg.get("moe_lexicographic", False)),
                               presence=bool(cfg.get("moe_presence_mask", False)))
        k = int(agent.take_action(states[t], mask=m))
        a = int(props[k])
        n = sum(env.sumo.lane.getLastStepVehicleNumber(l) for l in ts_lane_map[t])
        total_ts_steps += 1
        if n == 0:
            empty_steps += 1
            if a != ts.green_phase:
                empty_switch += 1; empty_switch_k.append(k)
        acts[t] = a
    states, r, done, _ = env.step(action=acts)
mc.collect_step(env.sumo); mc.finalize(env.sumo)
s = mc.summary()["system"]
out = {"arm": "215-asis", "ckpt": os.path.basename(ck),
       "total_ts_steps": total_ts_steps, "empty_steps": empty_steps,
       "empty_switch_events": empty_switch,
       "empty_switch_gate_k": np.bincount(empty_switch_k, minlength=6).tolist() if empty_switch_k else [0]*6,
       "teleports": int(env.sumo.simulation.getStartingTeleportNumber())}
for cls in ("car", "bus", "ambulance"):
    out[cls] = {"stops/visit": round(s[cls]["avg_stop_events_per_visit"], 3),
                "stopped_s/visit": round(s[cls]["avg_stopped_time_per_visit"], 3)}
env.close()
print("PROBE", json.dumps(out, ensure_ascii=False))
