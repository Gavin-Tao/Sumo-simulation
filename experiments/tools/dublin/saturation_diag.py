"""饱和度诊断 (2026-10-11, 只读分析, 不训练): 用检查点在评估种子上跑一集, 按路口/进口车道统计
  到达流量 (veh/h, 按车道上新出现的车辆 ID 计), 绿灯份额 (该车道任一 link 为 G/g 的时间占比, 5 s 采样),
  v/c = 到达流量 / (1800 veh/h/车道 × 绿灯份额), 最大排队 (getLastStepHaltingNumber 的最大值),
  周期失效率 = 绿灯段结束时 (最后一个绿灯采样点) 仍有停驶车辆的绿灯段占比。
用法: python experiments/tools/dublin/saturation_diag.py <exp> <ckpt.pth> <seed>; 输出 experiments/analysis/data/saturation_exp<exp>_seed<seed>.json"""
import os, sys, json, numpy as np
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
exp, ckpt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = ckpt_env.build(exp, ckpt, seed); env, act, states = B["env"], B["act"], B["states"]; sumo = env.sumo
ts_ids = list(env.ts_ids); SAT = 1800.0
lane_links = {}; lanes_of = {}
for t in ts_ids:
    cl = sumo.trafficlight.getControlledLinks(t); m = {}
    for i, links in enumerate(cl):
        for (inl, outl, via) in links: m.setdefault(inl, []).append(i)
    lane_links[t] = m; lanes_of[t] = sorted(m)
seen = {t: {l: set() for l in lanes_of[t]} for t in ts_ids}
arr = {t: {l: 0 for l in lanes_of[t]} for t in ts_ids}; green_s = {t: {l: 0.0 for l in lanes_of[t]} for t in ts_ids}
maxq = {t: {l: 0 for l in lanes_of[t]} for t in ts_ids}; prev_green = {t: {l: False for l in lanes_of[t]} for t in ts_ids}
prev_halt = {t: {l: 0 for l in lanes_of[t]} for t in ts_ids}; n_green_end = {t: {l: 0 for l in lanes_of[t]} for t in ts_ids}
n_fail = {t: {l: 0 for l in lanes_of[t]} for t in ts_ids}; resid = {t: {l: [] for l in lanes_of[t]} for t in ts_ids}
done = {"__all__": False}; t_prev = float(sumo.simulation.getTime()); T0 = t_prev
while not done["__all__"]:
    a = {t: act(t, states[t])[0] for t in ts_ids}
    states, _, done, _ = env.step(action=a); now = float(sumo.simulation.getTime()); dt = now - t_prev; t_prev = now
    for t in ts_ids:
        st = sumo.trafficlight.getRedYellowGreenState(t)
        for l in lanes_of[t]:
            ids = set(sumo.lane.getLastStepVehicleIDs(l)); arr[t][l] += len(ids - seen[t][l]); seen[t][l] |= ids
            h = sumo.lane.getLastStepHaltingNumber(l); maxq[t][l] = max(maxq[t][l], h)
            g = any(st[i] in "Gg" for i in lane_links[t][l])
            if g: green_s[t][l] += dt
            if prev_green[t][l] and not g:   # 绿灯段刚结束: 用上一个 (最后一个绿灯) 采样点的停驶数作残余排队
                n_green_end[t][l] += 1; resid[t][l].append(prev_halt[t][l])
                if prev_halt[t][l] > 0: n_fail[t][l] += 1
            prev_green[t][l] = g; prev_halt[t][l] = h
env.close(); H = (t_prev - T0) / 3600.0
out = {}
for t in ts_ids:
    rows = {}
    for l in lanes_of[t]:
        q = arr[t][l] / H; gs = green_s[t][l] / (t_prev - T0)
        vc = (q / (SAT * gs)) if gs > 0 else (float("inf") if q > 0 else 0.0)
        rows[l] = dict(flow_vph=q, green_share=gs, vc=vc, max_queue=maxq[t][l], n_green=n_green_end[t][l], n_fail=n_fail[t][l],
                       fail_rate=(n_fail[t][l] / n_green_end[t][l]) if n_green_end[t][l] else None, mean_resid=float(np.mean(resid[t][l])) if resid[t][l] else None)
    tot = sum(r["flow_vph"] for r in rows.values()); busy = [r for r in rows.values() if r["flow_vph"] >= 30]
    out[t] = dict(entering_vph=tot, n_lanes=len(rows), max_lane_flow=max((r["flow_vph"] for r in rows.values()), default=0),
                  max_vc=max((r["vc"] for r in busy), default=0), max_queue=max((r["max_queue"] for r in rows.values()), default=0),
                  fail_rate_busy=(sum(r["n_fail"] for r in busy) / max(sum(r["n_green"] for r in busy), 1)), lanes=rows)
OUT = f"/home/xiaowen/sumo-rl/experiments/analysis/data/saturation_exp{exp}_seed{seed}.json"; json.dump(out, open(OUT, "w"), indent=1)
print(f"[{exp} seed{seed}] 路口 {len(out)}, 仿真 {H*3600:.0f} s; 全网最大 v/c {max(v['max_vc'] for v in out.values()):.2f}; 失效率≥30% 的路口 {sum(1 for v in out.values() if v['fail_rate_busy']>=0.3)} 个")
