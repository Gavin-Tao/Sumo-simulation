"""零重训探针: 救护车抢占掩码 —— 决策时若某红灯槽有救护车 (优先级 5 的 count>0), 只允许放行该槽 (救护车等待最长者) 的相位 (最短绿内仍保持)。
与 eval_pervehicle 同样记录每辆车每次进口道访问 (entry/exit/entry_state/first_green/stopped), 只存探针车辆与小汽车汇总。
用法: python amb_preempt_probe.py <exp> --ckpt P --seed s --routes a,b,c [--mask] --out file.json"""
import argparse, os, sys, json, collections, numpy as np, math
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--routes"); ap.add_argument("--mask", action="store_true"); ap.add_argument("--out", required=True); a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed, route_override=a.routes); env, act, states, cfg, agent = B["env"], B["act"], B["states"], B["cfg"], B["agent"]; sumo = env.sumo
meta = json.load(open(cfg["action_meta_file"]))["tls"]; fields = list(cfg["obs_fields"]); ds = list(cfg.get("obs_downstream_fields", [])) if cfg.get("obs_downstream") else []
D = 1 + 5 * len(fields) + 5 * len(ds) + (1 if cfg.get("obs_lane_occ") else 0) + (1 if cfg.get("obs_slot_min_green") else 0); ci, ai = fields.index("count"), fields.index("mean_awt")
prio = dict(cfg.get("obs_priority_source", cfg.get("priority_source")) or {}); amb_lvl = prio.get("ambulance", 5) - 1
APP = ("N", "E", "S", "W"); TURN = ("L", "T", "R"); serves, std_of, mask = {}, {}, {}
for t in env.ts_ids:
    ts = env.traffic_signals[t]; ls = {int(i): APP.index(grp[0]["approach"]) * 3 + TURN.index(grp[0]["turn"]) for i, grp in meta[t]["links"].items()}
    serves[t] = {int(ts.std_action_map[g]): {ls[i] for i in ls if i < len(ph.state) and ph.state[i] in "Gg"} for g, ph in enumerate(ts.green_phases)}
    std_of[t] = {int(ts.std_action_map[g]): g for g in range(len(ts.green_phases))}; mask[t] = np.array(meta[t]["mask"], bool)
n_forced = collections.Counter()
def act2(t, s):
    a_env, k, q = act(t, s)
    if not a.mask: return a_env
    s = np.asarray(s); best, bw = None, -1
    for sl in range(12):
        if s[sl * D] < 0.5 and s[sl * D + 1 + amb_lvl * len(fields) + ci] > 0:
            w = s[sl * D + 1 + amb_lvl * len(fields) + ai]
            if w > bw: best, bw = sl, w
    if best is None: return a_env
    ok = np.array([mask[t][i] and (best in serves[t].get(i, set())) for i in range(8)])
    if not ok.any(): return a_env
    k2 = int(np.nanargmax(np.where(ok, q, np.nan)))
    if k2 != k: n_forced[t] += 1
    return std_of[t][k2]
lanes = {t: [l for l in dict.fromkeys(sumo.trafficlight.getControlledLanes(t))] for t in env.ts_ids}
veh = {}; on = {}
def rec():
    now = sumo.simulation.getTime(); present = set()
    for t, ls in lanes.items():
        for l in ls:
            for vid in sumo.lane.getLastStepVehicleIDs(l):
                key = (vid, t); present.add(key); v = veh.setdefault(vid, {"type": sumo.vehicle.getTypeID(vid), "tls": {}}); sp = sumo.vehicle.getSpeed(vid)
                if key not in on:
                    nt = sumo.vehicle.getNextTLS(vid); st = None; link = None
                    for x in nt:
                        if x[0] == t: link, st = int(x[1]), x[3]; break
                    on[key] = {"entry": now, "entry_state": st, "link": link, "first_green": None, "stopped": 0, "exit": None}
                r = on[key]
                if sp < 0.1: r["stopped"] += 1
                if r["first_green"] is None and r["link"] is not None:
                    for x in sumo.vehicle.getNextTLS(vid):
                        if x[0] == t and x[3] in ("G", "g"): r["first_green"] = now; break
    for key in [k for k in on if k not in present]:
        r = on.pop(key); r["exit"] = now; veh[key[0]]["tls"].setdefault(key[1], []).append(r)
_o = env._sumo_step
def h(): _o(); rec()
env._sumo_step = h
rec(); done = {"__all__": False}
while not done["__all__"]:
    states, _, done, _ = env.step(action={t: act2(t, states[t]) for t in env.ts_ids})
for key, r in on.items(): veh[key[0]]["tls"].setdefault(key[1], []).append(r)
env.close()
S = N = 0
for vid, v in veh.items():
    if vid.startswith("probe_") or v["type"] != "car": continue
    for t, rs in v["tls"].items():
        for r in rs: S += r["stopped"]; N += 1
out = dict(exp=a.exp, seed=a.seed, mask=a.mask, forced=dict(n_forced), others_car_stop=S / max(N, 1), probes={vid: v for vid, v in veh.items() if vid.startswith("probe_")})
json.dump(out, open(a.out, "w")); P = [r for v in out["probes"].values() for rs in v["tls"].values() for r in rs]
print(f"exp{a.exp} seed{a.seed} mask={a.mask}: 探针访问 {len(P)}, 停车/visit {np.mean([r['stopped'] for r in P]):.2f}, 零停 {np.mean([r['stopped']==0 for r in P]):.0%}, 强制改动 {sum(n_forced.values())} 次, 其他小汽车停车/visit {out['others_car_stop']:.3f}")
