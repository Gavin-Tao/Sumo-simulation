"""绿灯时长诊断 (2026-10-11, 只读分析, 不训练): 用检查点在评估种子上跑一集, 记录每路口每段 "两次相位切换之间的时长" (含 2 s 黄灯),
给出逐路口中位/p90/最大绿灯、超过 50/120/300 s 的段数, 以及该路口有需求运动 (本集有车进入的槽) 的最长连续红灯。
用法: python experiments/tools/dublin/green_dur_diag.py <exp> <ckpt.pth> <seed>; 输出 experiments/analysis/data/green_dur_exp<exp>_seed<seed>.json"""
import os, sys, json, numpy as np
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
exp, ckpt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = ckpt_env.build(exp, ckpt, seed); env, act, states = B["env"], B["act"], B["states"]
ts_ids = list(env.ts_ids); t0 = float(env.sumo.simulation.getTime())
prev = {t: None for t in ts_ids}; start = {t: t0 for t in ts_ids}; durs = {t: [] for t in ts_ids}
last_green = {t: {} for t in ts_ids}; red_max = {t: {} for t in ts_ids}; demand = {t: set() for t in ts_ids}
done = {"__all__": False}
while not done["__all__"]:
    a = {t: act(t, states[t])[0] for t in ts_ids}
    states, _, done, _ = env.step(action=a); now = float(env.sumo.simulation.getTime())
    for t in ts_ids:
        ts = env.traffic_signals[t]; g = ts.green_phase; of = ts.observation_fn
        if prev[t] is not None and g != prev[t]: durs[t].append(now - start[t]); start[t] = now
        prev[t] = g
        served = of._phase_serves.get(g, set())
        for s in served: last_green[t][s] = now
        for s, lanes in of._slot_in_lanes.items():
            if any(env.sumo.lane.getLastStepVehicleNumber(l) > 0 for l in lanes): demand[t].add(s)
            red = now - last_green[t].get(s, t0); red_max[t][s] = max(red_max[t].get(s, 0.0), red)
    # 末段
for t in ts_ids: durs[t].append(float(env.sumo.simulation.getTime()) - start[t])
env.close()
out = {}
for t in ts_ids:
    d = np.array(durs[t]); dem = sorted(demand[t]); rm = [red_max[t].get(s, 0.0) for s in dem]
    out[t] = dict(n_greens=int(len(d)), median=float(np.median(d)), p90=float(np.percentile(d, 90)), max=float(d.max()),
                  n_gt50=int((d > 50).sum()), n_gt120=int((d > 120).sum()), n_gt300=int((d > 300).sum()),
                  demand_slots=dem, max_red_demand=float(max(rm)) if rm else 0.0, n_demand_red_gt120=int(sum(1 for x in rm if x > 120)))
OUT = f"/home/xiaowen/sumo-rl/experiments/analysis/data/green_dur_exp{exp}_seed{seed}.json"; json.dump(out, open(OUT, "w"), indent=1)
allg = np.concatenate([np.array(durs[t]) for t in ts_ids])
print(f"[{exp} seed{seed}] 全网 {len(allg)} 段绿灯: 中位 {np.median(allg):.0f} s, p90 {np.percentile(allg,90):.0f} s, 最大 {allg.max():.0f} s; >50 s {int((allg>50).sum())} 段, >120 s {int((allg>120).sum())} 段, >300 s {int((allg>300).sum())} 段; 有需求运动最长连续红灯 {max(v['max_red_demand'] for v in out.values()):.0f} s, 红灯>120 s 的有需求运动 {sum(v['n_demand_red_gt120'] for v in out.values())} 个")
