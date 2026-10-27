"""零重训探针: 用 best 检查点贪心评估, 可改最短绿 (--min-green) 或用裁剪过的 meta (--meta) 屏蔽部分相位; 按路口输出 停车秒/visit (5 s 采样) 与访问数。"""
import argparse, os, sys, json, collections
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--min-green", type=int); ap.add_argument("--meta"); ap.add_argument("--out", required=True); a = ap.parse_args()
ov = {}
if a.min_green: ov["min_green"] = a.min_green
if a.meta: ov["action_meta_file"] = a.meta
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed, cfg_override=ov); env, act, states = B["env"], B["act"], B["states"]; sumo = env.sumo
tr = env.visit_tracker; seen = set(); halt = collections.Counter(); sw = collections.Counter(); last = {}
done = {"__all__": False}
while not done["__all__"]:
    acts = {t: act(t, states[t])[0] for t in env.ts_ids}
    states, r, done, _ = env.step(action=acts)
    for k in tr.W: seen.add(k)
    for t in env.ts_ids:
        ts = env.traffic_signals[t]
        for l in ts.lanes: halt[t] += sumo.lane.getLastStepHaltingNumber(l) * 5
        if t in last and last[t] != ts.green_phase: sw[t] += 1
        last[t] = ts.green_phase
remaining = sumo.vehicle.getIDCount(); env.close()
visits = collections.Counter(k[0] for k in seen)
out = {t: dict(visits=visits[t], stop_s=halt[t], stop_per_visit=halt[t] / max(visits[t], 1), switches=sw[t]) for t in env.ts_ids}
tot_v = sum(visits.values()); tot_s = sum(halt.values())
json.dump(dict(exp=a.exp, seed=a.seed, min_green=a.min_green, meta=a.meta, remaining=remaining, network=dict(visits=tot_v, stop_per_visit=tot_s / tot_v), junctions=out), open(a.out, "w"), indent=1)
print(f"seed{a.seed} min_green={a.min_green} meta={'prune' if a.meta else '-'}: 全网 {tot_s/tot_v:.3f} s/visit, 剩余 {remaining}, 1448442 {out['1448442']['stop_per_visit']:.2f} (切 {out['1448442']['switches']}), c21620852 {[v for t,v in out.items() if '21620852' in t][0]['stop_per_visit']:.2f} (切 {[v for t,v in out.items() if '21620852' in t][0]['switches']})")
