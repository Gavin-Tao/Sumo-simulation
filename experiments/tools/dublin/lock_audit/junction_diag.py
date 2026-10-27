"""逐路口相位/车道诊断 (只读): 用检查点贪心跑一集, 每决策步对每个路口统计
  blocked_green: 某进口车道有停车, 该车道有运动是绿灯, 但头车要走的运动是红灯 (共享车道头车阻塞) 的步数与受影响停车数;
  empty_green:   某绿灯槽没有车, 同时该路口另有槽排队 (绿灯给了空槽) 的步数;
  starve:        某槽有排队且连续红灯 ≥ 60 s 的事件数与最长秒数;
  phase:         每相位平均持续秒、切换次数。
用法: python junction_diag.py <exp> --ckpt P --seed 123 --out prefix"""
import argparse, collections, json, os, sys
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]; sumo = env.sumo
meta = json.load(open(cfg["action_meta_file"]))["tls"]
name = lambda c: c["approach"] + "-" + c["turn"]
T = {}
for tid in env.ts_ids:
    t = meta[tid]; ts = env.traffic_signals[tid]
    by_edges = {}; lane_slots = collections.defaultdict(set); link_slot = {}
    for i, grp in t["links"].items():
        for c in grp:
            s = name(c); link_slot[int(i)] = s; by_edges[(c["from_edge"], c["to_edge"])] = s; lane_slots[c["from_edge"] + "_" + str(c["from_lane"])].add(s)
    serves = {}
    for g, ph in enumerate(ts.green_phases):
        serves[g] = {link_slot[i] for i in link_slot if i < len(ph.state) and ph.state[i] in "Gg"}
    T[tid] = dict(by_edges=by_edges, lane_slots=lane_slots, serves=serves, lanes=list(ts.lanes), g2s=ts.std_action_map)
def slot_of(tid, vid, lane):
    r = sumo.vehicle.getRoute(vid); k = sumo.vehicle.getRouteIndex(vid); nxt = r[k + 1] if k + 1 < len(r) else None
    return T[tid]["by_edges"].get((lane.rsplit("_", 1)[0], nxt))
st = {tid: dict(blocked_steps=0, blocked_veh=0, empty_green_steps=0, starve_events=0, starve_max=0, red_since={}, phase_dur=collections.defaultdict(list), switches=0, last_phase=None, phase_t0=0.0, steps=0, steps_with_queue=0) for tid in env.ts_ids}
done = {"__all__": False}
while not done["__all__"]:
    now = float(sumo.simulation.getTime())
    acts = {t: act(t, states[t])[0] for t in env.ts_ids}
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]; tt = T[tid]; s_ = st[tid]; served = tt["serves"][ts.green_phase]; s_["steps"] += 1
        g = ts.green_phase
        if s_["last_phase"] is None: s_["last_phase"] = g; s_["phase_t0"] = now
        elif g != s_["last_phase"]:
            s_["phase_dur"][int(tt["g2s"][s_["last_phase"]])].append(now - s_["phase_t0"]); s_["switches"] += 1; s_["last_phase"] = g; s_["phase_t0"] = now
        queued = collections.Counter(); present = collections.Counter(); blocked_here = 0; blocked_veh = 0
        for lane in tt["lanes"]:
            vids = sumo.lane.getLastStepVehicleIDs(lane)
            if not vids: continue
            halted = [v for v in vids if sumo.vehicle.getSpeed(v) < 0.1]
            head = max(vids, key=lambda v: sumo.vehicle.getLanePosition(v))
            for v in vids:
                s = slot_of(tid, v, lane)
                if s is None: continue
                present[s] += 1
                if sumo.vehicle.getSpeed(v) < 0.1: queued[s] += 1
            if halted and len(tt["lane_slots"][lane]) > 1:
                hs = slot_of(tid, head, lane)
                if hs is not None and hs not in served and (tt["lane_slots"][lane] & served):
                    blocked_here = 1; blocked_veh += len(halted)
        s_["blocked_steps"] += blocked_here; s_["blocked_veh"] += blocked_veh
        if queued: s_["steps_with_queue"] += 1
        if queued and any(present[s] == 0 for s in served) and any(queued[s] > 0 for s in queued if s not in served):
            s_["empty_green_steps"] += 1
        for s in list(set(queued) | set(s_["red_since"])):
            if queued.get(s, 0) > 0 and s not in served:
                s_["red_since"].setdefault(s, now)
            else:
                t0 = s_["red_since"].pop(s, None)
                if t0 is not None and now - t0 >= 60: s_["starve_events"] += 1; s_["starve_max"] = max(s_["starve_max"], now - t0)
    states, r, done, _ = env.step(action=acts)
env.close()
out = {}
for tid, s_ in st.items():
    pd = {k: (sum(v) / len(v), len(v)) for k, v in s_["phase_dur"].items()}
    out[tid] = dict(steps=s_["steps"], steps_with_queue=s_["steps_with_queue"], blocked_steps=s_["blocked_steps"], blocked_veh=s_["blocked_veh"], empty_green_steps=s_["empty_green_steps"],
                    starve_events=s_["starve_events"], starve_max=s_["starve_max"], switches=s_["switches"], phase_mean_dur={str(k): round(v[0], 1) for k, v in pd.items()}, phase_n={str(k): v[1] for k, v in pd.items()})
json.dump(dict(exp=a.exp, ckpt_episode=B["ckpt_episode"], seed=a.seed, junctions=out), open(a.out + ".json", "w"), indent=1)
print(f"exp{a.exp} ep{B['ckpt_episode']} seed{a.seed} 完成")
