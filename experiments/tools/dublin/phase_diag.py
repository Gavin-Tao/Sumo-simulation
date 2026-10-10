"""相位行为诊断 (2026-10-10, 只读分析, 不训练): 用检查点在评估种子上跑一集, 记录每路口的切换次数、平均绿灯时长、用到的相位数、各运动槽的最长饿死 (距上次放行的最大秒数)、相位时间占比。
用法: python experiments/tools/dublin/phase_diag.py <exp> <ckpt.pth> <seed>; 输出 json 到脚本旁 (或改 OUT)。"""
import os, sys, json, numpy as np
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
exp, ckpt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = ckpt_env.build(exp, ckpt, seed); env, act, states = B["env"], B["act"], B["states"]
ts_ids = list(env.ts_ids); prev = {t: None for t in ts_ids}; sw = {t: 0 for t in ts_ids}; used = {t: set() for t in ts_ids}
last_green = {t: {} for t in ts_ids}; max_sg = {t: {} for t in ts_ids}; t0 = float(env.sumo.simulation.getTime()); dwell = {t: {} for t in ts_ids}
done = {"__all__": False}; steps = 0
while not done["__all__"]:
    a = {t: act(t, states[t])[0] for t in ts_ids}
    states, _, done, _ = env.step(action=a); steps += 1
    now = float(env.sumo.simulation.getTime())
    for t in ts_ids:
        ts = env.traffic_signals[t]; g = ts.green_phase; used[t].add(g); dwell[t][g] = dwell[t].get(g, 0) + 1
        if prev[t] is not None and g != prev[t]: sw[t] += 1
        prev[t] = g
        served = ts.observation_fn._phase_serves.get(g, set())
        for s in served: last_green[t][s] = now
        for s in range(12):
            if s in ts.observation_fn._slot_cap and ts.observation_fn._slot_cap[s] > 0:
                sg = now - last_green[t].get(s, t0); max_sg[t][s] = max(max_sg[t].get(s, 0), sg)
env.close()
out = {}
for t in ts_ids:
    K = env.traffic_signals[t].num_green_phases
    # 有需求的槽 = 有相位放行过的槽 (slot_cap>0 且在某相位 served 里)
    demand_slots = set().union(*[set(v) for v in env.traffic_signals[t].observation_fn._phase_serves.values()])
    msg = [max_sg[t].get(s, 0) for s in demand_slots]
    out[t] = dict(K=K, used=len(used[t]), switches=sw[t], mean_green=3600 / max(sw[t], 1), max_starv=max(msg) if msg else 0, n_starv_300=sum(1 for x in msg if x > 300), dwell=dwell[t])
OUT = os.environ.get("PHASE_DIAG_OUT", "/home/xiaowen/sumo-rl/experiments/analysis/data"); json.dump(out, open(f"{OUT}/phase_diag_exp{exp}_seed{seed}.json", "w"))
print(f"[{exp} seed{seed}] steps={steps} | 全网: 切换/路口/小时 均 {np.mean([v['switches'] for v in out.values()]):.1f}, 平均绿 {np.mean([v['mean_green'] for v in out.values()]):.1f} s, 用到相位/菜单 {np.mean([v['used']/v['K'] for v in out.values()]):.2f}, 最长饿死均 {np.mean([v['max_starv'] for v in out.values()]):.0f} s, 饿死>300 s 的槽数合计 {sum(v['n_starv_300'] for v in out.values())}")
