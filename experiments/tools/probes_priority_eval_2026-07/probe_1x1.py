"""1x1 batch unified probe: {217|219|220} x {base|blind}, best.pth, greedy,
eval_seed. Prints per-class per-visit metrics + behavior fingerprint + amb
stop classification. blind = obs-only priority table amb 5->1."""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
which, variant = sys.argv[1], sys.argv[2]
CFGS = {"217": "configs/exp217_1x1_531_NS20bus_vanilla_B_movement_legacy_cqm.yaml",
        "219": "configs/exp219_1x1_531_NS20bus_multihead_bnf.yaml",
        "220": "configs/exp220_1x1_531_NS20bus_enumfrap_cqm.yaml"}
cfg = yaml.safe_load(open(CFGS[which]))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.agents.dqn_agent_txw import DQN
obs_kwargs = dict(fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"])
if variant == "blind":
    obs_kwargs["priority_source"] = ({"ambulance": 1, "bus": 3, "car": 1} if "priority_source" in cfg else {"ambulance": 1, "bus": 4, "truck": 2, "car": 1})
elif variant == "w421b":
    # 4-2-1 经由桶: amb→l4 (217/220 的 bus 训练桶), bus→l2 (训练中恒零的"处女桶")
    obs_kwargs["priority_source"] = ({"ambulance": 4, "bus": 2, "car": 1} if "priority_source" in cfg else {"ambulance": 4, "bus": 2, "truck": 2, "car": 1})
elif variant == "swap351":
    # 3-5-1 重排: bus 进 amb 的训练桶, amb 进 bus 的训练桶 (桶待遇交换)
    obs_kwargs["priority_source"] = ({"ambulance": 3, "bus": 5, "car": 1} if "priority_source" in cfg else {"ambulance": 4, "bus": 5, "truck": 2, "car": 1})
elif "priority_source" in cfg:
    obs_kwargs["priority_source"] = cfg["priority_source"]
for src, dst in [("obs_awt_cap","awt_cap"), ("obs_awt_basis","awt_basis")]:
    if src in cfg: obs_kwargs[dst] = cfg[src]
obs_class = functools.partial(obsmod.PriorityMovementObservationFunction, **obs_kwargs)
rf = cfg["reward_fn"]
if rf == "priority-avg-waiting":
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    from sumo_rl.environment.priority_map import load_priority_table
    rf = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))
env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
    cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
    num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
    max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
    single_agent=False, yellow_time=cfg["yellow_time"], delta_time=cfg["delta_time"],
    reward_fn=rf, observation_class=obs_class, sumo_seed=cfg["seed"], sumo_warnings=False)
env.reset(int(cfg["eval_seed"]))
tid = env.ts_ids[0]

if which == "220":
    from frap_glue import load_enum_tables, build_frap_agent
    tables = load_enum_tables(cfg["enum_meta_file"])
    ts = env.traffic_signals[tid]
    ts.observation_fn.rebind_movements(tables["turnmap"][tid])
    ts.observation_space = ts.observation_fn.observation_space()
    states = {tid: ts.observation_fn()}
    agent = build_frap_agent(cfg, tables, env, "cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval(); agent.epsilon = 0.0
    def act(): return int(agent.take_action(states[tid], tid))
else:
    states = {tid: env.traffic_signals[tid].observation_fn()}
    od = len(states[tid])
    A = env.action_space.n if hasattr(env.action_space, "n") else env.traffic_signals[tid].num_green_phases
    if which == "219":
        from multihead_glue import build_multihead_agent
        agent = build_multihead_agent(cfg, starting_state=tuple([0.0]*od),
                                      state_space=od, action_space=A, device="cpu")
    else:
        agent = DQN(starting_state=tuple([0.0]*od), state_space=od, hidden_dim=cfg["hidden_dim"],
            action_space=A, learning_rate=1e-3, gamma=0.99, epsilon=0.0, target_update=10,
            capacity=100, mini_size=10**9, batch_size=1, eps_start=0, eps_end=0, eps_decay=1, device="cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval(); agent.epsilon = 0.0
    def act():
        return int(agent.take_action(states[tid]))

ts_lane_map = {tid: env.traffic_signals[tid].signal_controlled_lanes}
always_green = set(env.traffic_signals[tid].always_green_lanes)
mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green)
lane_tls = {}
for i, links in enumerate(env.sumo.trafficlight.getControlledLinks(tid)):
    for (inlane, _o, _v) in links:
        lane_tls.setdefault(inlane, []).append(i)

from collections import Counter
amb_cls = Counter(); amb_steps = 0
prev = env.traffic_signals[tid].green_phase
hold, holds, switches = 1, [], 0
rew = []; done = {"__all__": False}; step = 0
while not done["__all__"]:
    mc.collect_step(env.sumo)
    sumo = env.sumo
    for vid in sumo.vehicle.getIDList():
        if sumo.vehicle.getTypeID(vid) != "ambulance": continue
        amb_steps += 1
        if sumo.vehicle.getSpeed(vid) < 0.1:
            lane = sumo.vehicle.getLaneID(vid)
            if lane in lane_tls:
                st = sumo.trafficlight.getRedYellowGreenState(tid)
                green = any(st[i] in "Gg" for i in lane_tls[lane])
                pos = sumo.vehicle.getLanePosition(vid)
                ahead = sum(1 for o in sumo.lane.getLastStepVehicleIDs(lane)
                            if o != vid and sumo.vehicle.getLanePosition(o) > pos)
                amb_cls["green_blocked" if (green and ahead>0) else "green_head" if green else "denied_green"] += 1
            else:
                amb_cls["mid_queue"] += 1
    a = act()
    states, r, done, _ = env.step(action={tid: a})
    g = env.traffic_signals[tid].green_phase
    if g != prev: holds.append(hold); switches += 1; hold = 1
    else: hold += 1
    prev = g
    rew.append(float(r[tid])); step += 1
mc.collect_step(env.sumo); mc.finalize(env.sumo)
s = mc.summary()["system"]
env.close()
out = {"arm": f"{which}-{variant}", "ckpt": os.path.basename(os.path.dirname(ck)),
       "ep_return": round(float(np.sum(rew)), 2),
       "switch_rate": round(switches/(step-1), 3),
       "hold_s": {"p50": float(np.percentile([h*5 for h in holds], 50)) if holds else None,
                  "p90": float(np.percentile([h*5 for h in holds], 90)) if holds else None,
                  "max": float(max(holds)*5) if holds else None},
       "amb_stop_cls": dict(amb_cls)}
for cls in ("car", "bus", "ambulance"):
    out[cls] = {"stops/visit": round(s[cls]["avg_stop_events_per_visit"], 3),
                "stopped_s/visit": round(s[cls]["avg_stopped_time_per_visit"], 3)}
print("PROBE", json.dumps(out, ensure_ascii=False))
