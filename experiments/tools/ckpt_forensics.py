"""Checkpoint forensic rollout: greedy policy, deep per-junction/per-vehicle
instrumentation. Usage: python3 ckpt_forensics.py <ckpt_path> <tag> [seed]
Writes /tmp/forensic_<tag>.json. Run from experiments/ cwd.
"""
import sys, os, json, functools
import numpy as np

os.environ.setdefault('SUMO_RL_LIBSUMO', '1')
sys.path.insert(0, '/home/xiaowen/sumo-rl')
sys.path.insert(0, '/home/xiaowen/sumo-rl/experiments')

import torch, yaml
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.agents.dqn_agent_txw import DQN

ckpt_path, tag = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(ckpt_path), "config.yaml")))

# ── masked setup (mirrors eval_unified) ──
meta = json.load(open(cfg["action_meta_file"]))["tls"]
ts_mask, std2green, green2std, ts_turnmap = {}, {}, {}, {}
for tid, t in meta.items():
    ts_mask[tid] = np.array(t["mask"], dtype=bool)
    s2g = np.full(8, -1, dtype=int)
    for a, gi in t["std_to_green_index"].items():
        s2g[int(a)] = gi
    std2green[tid] = s2g
    g2s = np.full(int(s2g.max()) + 1, -1, dtype=int)
    for a in range(8):
        if s2g[a] >= 0:
            g2s[s2g[a]] = a
    green2std[tid] = g2s
    ts_turnmap[tid] = {int(i): (c[0]["approach"], c[0]["turn"])
                       for i, c in t["links"].items()}

OBS = getattr(obsmod, {"PriorityMovement": "PriorityMovementLegacyCQM"}[cfg["observation_class"]])
obs_kwargs = {}
if "priority_source" in cfg: obs_kwargs["priority_source"] = cfg["priority_source"]
if cfg.get("obs_downstream"): obs_kwargs["include_downstream"] = True
if "obs_downstream_fields" in cfg: obs_kwargs["downstream_fields"] = tuple(cfg["obs_downstream_fields"])
reward_fn = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))

env = SumoEnvironment(
    net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"],
    out_csv_name=None, use_gui=False, num_seconds=cfg.get("num_seconds", 3600),
    delta_time=cfg.get("delta_time", 5), yellow_time=cfg.get("yellow_time", 2),
    min_green=cfg.get("min_green", 5), max_green=cfg.get("max_green", 50),
    use_max_green=cfg.get("use_max_green", True),
    single_agent=False, sumo_seed=seed, fixed_ts=False,
    reward_fn=reward_fn,
    observation_class=functools.partial(OBS, **obs_kwargs),
)
states = env.reset(seed)
for tid in env.ts_ids:
    ts = env.traffic_signals[tid]
    ts.std_action_map = green2std[tid]
    ts.observation_fn.rebind_movements(ts_turnmap[tid])
    ts.observation_space = ts.observation_fn.observation_space()
states = {tid: env.traffic_signals[tid].observation_fn() for tid in env.ts_ids}
obs_dim = len(next(iter(states.values())))

agent = DQN(starting_state=tuple([0.0]*obs_dim), state_space=obs_dim,
            hidden_dim=cfg.get("hidden_dim", 128), action_space=8,
            learning_rate=cfg.get("lr", 1e-3), gamma=cfg.get("gamma", 0.99),
            epsilon=0.0, target_update=10, capacity=1, mini_size=1, batch_size=1,
            eps_start=0.0, eps_end=0.0, eps_decay=1, device="cpu")
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
agent.q_net.load_state_dict(ck["policy_state_dict"])
agent.q_net.eval()
agent.epsilon = 0.0
print(f"[{tag}] loaded ep={ck.get('episode','?')} obs_dim={obs_dim}")

sumo = env.traffic_signals[env.ts_ids[0]].sumo
# controlled lane <-> link chars per TS (for green-share tracking)
ts_lanes_order = {tid: sumo.trafficlight.getControlledLanes(tid) for tid in env.ts_ids}

D = {
    "tag": tag, "ckpt": ckpt_path, "episode": int(ck.get("episode", -1)), "seed": seed,
    "ts": {tid: {"act_hist": [0]*8, "switches": 0, "reward": 0.0,
                 "q_chosen_sum": 0.0, "q_absmax": 0.0, "q_top_gap_sum": 0.0}
           for tid in env.ts_ids},
    "timeline": [],   # per 60s: t, running, stopped, teleports, rewsum
}
lane_green = {}   # (tid, lane) -> green steps
lane_qsum, lane_qmax = {}, {}
veh_stop = {}     # vid -> stopped seconds
veh_cls = {}
prev_act = {tid: None for tid in env.ts_ids}
dt = cfg.get("delta_time", 5)
rewsum = 0.0
arrived_total = 0
tele_total = 0
done = {"__all__": False}
step = 0
while not done["__all__"]:
    acts = {}
    for tid in env.ts_ids:
        s = states[tid]; m = ts_mask[tid]
        with torch.no_grad():
            q = agent.q_net(torch.tensor(np.asarray(s, dtype=np.float32)).unsqueeze(0)).squeeze(0).numpy()
        qm = np.where(m, q, -np.inf)
        a = int(qm.argmax())
        acts[tid] = int(std2green[tid][a])
        d = D["ts"][tid]
        d["act_hist"][a] += 1
        if prev_act[tid] is not None and a != prev_act[tid]:
            d["switches"] += 1
        prev_act[tid] = a
        d["q_chosen_sum"] += float(qm[a])
        d["q_absmax"] = max(d["q_absmax"], float(np.abs(q[m]).max()))
        top2 = np.sort(qm[m])[-2:] if m.sum() >= 2 else [0, 0]
        d["q_top_gap_sum"] += float(top2[1] - top2[0])
    states, r, done, _ = env.step(action=acts)
    step += 1
    for tid in env.ts_ids:
        D["ts"][tid]["reward"] += float(r[tid]); rewsum += float(r[tid])
    # lane green + queue + vehicle stop tracking
    for tid in env.ts_ids:
        st = sumo.trafficlight.getRedYellowGreenState(tid)
        for i, ln in enumerate(ts_lanes_order[tid]):
            key = (tid, ln)
            if i < len(st) and st[i] in "Gg":
                lane_green[key] = lane_green.get(key, 0) + 1
            q = sumo.lane.getLastStepHaltingNumber(ln)
            lane_qsum[key] = lane_qsum.get(key, 0) + q
            lane_qmax[key] = max(lane_qmax.get(key, 0), q)
    arrived_total += sumo.simulation.getArrivedNumber()
    tele_total += sumo.simulation.getStartingTeleportNumber()
    vids = sumo.vehicle.getIDList()
    nstop = 0
    for vid in vids:
        if sumo.vehicle.getSpeed(vid) < 0.1:
            nstop += 1
            veh_stop[vid] = veh_stop.get(vid, 0.0) + dt
            if vid not in veh_cls:
                veh_cls[vid] = sumo.vehicle.getTypeID(vid)
    if step % 12 == 0:
        D["timeline"].append(dict(
            t=step*dt, running=len(vids), stopped=nstop,
            teleports=tele_total,
            arrived=arrived_total, rewsum=round(rewsum, 1)))
env.close()

steps = step
for tid in env.ts_ids:
    d = D["ts"][tid]
    d["q_chosen_mean"] = d["q_chosen_sum"] / steps
    d["q_top_gap_mean"] = d["q_top_gap_sum"] / steps
    d["switch_rate"] = d["switches"] / steps
    del d["q_chosen_sum"], d["q_top_gap_sum"], d["switches"]
# starvation: mean queue >=3 AND green share < 3%
starved = []
for (tid, ln), qs in lane_qsum.items():
    mq = qs / steps
    g = lane_green.get((tid, ln), 0) / steps
    if mq >= 3.0 and g < 0.03:
        starved.append(dict(ts=tid, lane=ln, mean_queue=round(mq, 1),
                            max_queue=lane_qmax[(tid, ln)], green_share=round(g, 4)))
D["starved_lanes"] = sorted(starved, key=lambda x: -x["mean_queue"])
D["lane_top_queues"] = sorted(
    [dict(ts=t, lane=l, mean_queue=round(q/steps, 1), max_queue=lane_qmax[(t, l)],
          green_share=round(lane_green.get((t, l), 0)/steps, 3))
     for (t, l), q in lane_qsum.items()], key=lambda x: -x["mean_queue"])[:15]
# vehicle-class aggregates
cls_agg = {}
for vid, sgs in veh_stop.items():
    c = veh_cls.get(vid, "car")
    a = cls_agg.setdefault(c, {"n": 0, "sum": 0.0, "max": 0.0, "trapped600": 0})
    a["n"] += 1; a["sum"] += sgs; a["max"] = max(a["max"], sgs)
    if sgs >= 600: a["trapped600"] += 1
for c, a in cls_agg.items():
    a["mean_stop"] = round(a["sum"]/a["n"], 1); del a["sum"]
D["veh_class"] = cls_agg

json.dump(D, open(f"/tmp/forensic_{tag}.json", "w"), indent=1)
tl = D["timeline"][-1]
print(f"[{tag}] steps={steps} total_reward={rewsum:.0f} arrived={tl['arrived']} "
      f"teleports={tl['teleports']} starved={len(starved)}")
