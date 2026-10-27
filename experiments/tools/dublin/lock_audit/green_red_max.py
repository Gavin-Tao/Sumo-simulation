"""每路口: 最长连续绿 (同一相位保持), 最长 "有排队的槽连续红灯" (= 最长等待的上界), 以及槽内最大本次访问等待。best 检查点贪心一集。"""
import os, sys, json, collections
os.environ["OMP_NUM_THREADS"]="1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import ckpt_env
exp, ck, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = ckpt_env.build(exp, ckpt=ck, seed=seed); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]; sumo = env.sumo
meta = json.load(open(cfg["action_meta_file"]))["tls"]; APP=("N","E","S","W"); TURN=("L","T","R")
serves, by_edges = {}, {}
for t in env.ts_ids:
    ts = env.traffic_signals[t]; ls = {int(i): APP.index(g[0]["approach"])*3+TURN.index(g[0]["turn"]) for i,g in meta[t]["links"].items()}
    serves[t] = {g: {ls[i] for i in ls if i < len(ph.state) and ph.state[i] in "Gg"} for g, ph in enumerate(ts.green_phases)}
    by_edges[t] = {(c["from_edge"], c["to_edge"]): APP.index(c["approach"])*3+TURN.index(c["turn"]) for grp in meta[t]["links"].values() for c in grp}
tr = env.visit_tracker
green_run = {t: 0 for t in env.ts_ids}; last_g = {t: None for t in env.ts_ids}; max_green = collections.Counter()
red_run = collections.defaultdict(float); max_red = collections.Counter(); max_W = collections.Counter()
done = {"__all__": False}
while not done["__all__"]:
    acts = {t: act(t, states[t])[0] for t in env.ts_ids}
    states, r, done, _ = env.step(action=acts)
    for t in env.ts_ids:
        ts = env.traffic_signals[t]; g = ts.green_phase
        green_run[t] = green_run[t] + 5 if g == last_g[t] else 5; last_g[t] = g; max_green[t] = max(max_green[t], green_run[t])
        # 槽级: 有停车且红灯 → 连续红计时
        queued = set(); 
        for lane in ts.lanes:
            for v in sumo.lane.getLastStepVehicleIDs(lane):
                if sumo.vehicle.getSpeed(v) < 0.1:
                    rt = sumo.vehicle.getRoute(v); k = sumo.vehicle.getRouteIndex(v); nxt = rt[k+1] if k+1 < len(rt) else None
                    s = by_edges[t].get((lane.rsplit("_",1)[0], nxt))
                    if s is not None: queued.add(s)
        for s in range(12):
            if s in queued and s not in serves[t][g]: red_run[(t,s)] += 5; max_red[t] = max(max_red[t], red_run[(t,s)])
            else: red_run[(t,s)] = 0
    for (t, vid, e), w in tr.W.items(): max_W[t] = max(max_W[t], w)
env.close()
short = lambda t: t.split("_")[0]+("_"+t.split("_")[1] if t.startswith("cluster") else "")
json.dump({short(t): dict(max_green=max_green[t], max_red_with_queue=max_red[t], max_visit_wait=max_W[t]) for t in env.ts_ids}, open(sys.argv[4], "w"))
print("done", seed)
