"""1x1 相位使用与放行效率探针 (只读, 2026-10-14)。一集贪心, 逐仿真秒记录:
  每个相位 (按放行集合分为 配对标准 / 单进口标准 / 组合) 的绿灯秒数 (不含黄灯) 与放行车辆数 (非右转运动车道上离开停车线的车);
  饱和放行率 = 放行运动车道上 "上一秒有停车车辆" 的秒里, 每秒离开的车数 (按运动计);
  能切换的决策点: 所选相位类别, Q 第一与第二之差 / 可选 Q 标准差 (决策余量), 余量 < 0.1 的近似平局比例, 前两名里是否有组合相位。
用法: python phase_probe_1x1.py <exp> <ckpt> <seed>   输出一行 JSON (PROBE ...)"""
import sys, json, numpy as np
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
exp, ckpt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = ckpt_env.build(exp, ckpt, seed); env, act, states = B["env"], B["act"], B["states"]; t = list(env.ts_ids)[0]; ts = env.traffic_signals[t]; of = ts.observation_fn
RIGHT = {2, 5, 8, 11}
PAIRED = [frozenset(s) for s in ({1, 7}, {4, 10}, {0, 6}, {3, 9})]; SPLIT = [frozenset(s) for s in ({0, 1}, {6, 7}, {3, 4}, {9, 10})]
serves = {k: frozenset(set(v) - RIGHT) for k, v in of._phase_serves.items()}
g2s = getattr(ts, "std_action_map", None) if B["kind"] == "dqn_masked" else None
def cat_green(g):
    s = serves[g]; return "配对" if s in PAIRED else ("单进口" if s in SPLIT else "组合")
mov_lane = {m: [l for l, ss in of._lane_slots.items() if m in ss][0] for m in range(12) if m not in RIGHT}
stat = {c: {"green_s": 0, "dep": 0, "sat_s": 0, "sat_dep": 0, "n_phase": 0} for c in ("配对", "单进口", "组合")}
prev = {m: set() for m in mov_lane}; prev_halt = {m: 0 for m in mov_lane}
cur_phase = [None]
def hook():
    _orig()
    g = int(ts.green_phase)
    now_ids = {m: set(env.sumo.lane.getLastStepVehicleIDs(l)) for m, l in mov_lane.items()}
    if not ts.is_yellow:
        c = cat_green(g); stat[c]["green_s"] += 1
        for m in serves[g]:
            left = len(prev[m] - now_ids[m]); stat[c]["dep"] += left
            if prev_halt[m] > 0: stat[c]["sat_s"] += 1; stat[c]["sat_dep"] += left
    for m, l in mov_lane.items(): prev[m] = now_ids[m]; prev_halt[m] = env.sumo.lane.getLastStepHaltingNumber(l)
_orig = env._sumo_step; env._sumo_step = hook
dec = {"n": 0, "cat": {"配对": 0, "单进口": 0, "组合": 0}, "marg": [], "tie": 0, "combo_top2": 0}
time_share = {"配对": 0, "单进口": 0, "组合": 0}; seq = []
done = {"__all__": False}
while not done["__all__"]:
    g, k, q = act(t, states[t]); q = np.array(q, float)
    free = ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time
    if free:
        valid = np.flatnonzero(~np.isnan(q)); qv = q[valid]; srt = np.sort(qv)[::-1]; sd = qv.std() + 1e-9
        m = (srt[0] - srt[1]) / sd; dec["n"] += 1; dec["marg"].append(m); dec["tie"] += int(m < 0.1)
        top2 = valid[np.argsort(qv)[::-1][:2]]
        to_g = (lambda kk: int(np.flatnonzero(g2s == kk)[0])) if g2s is not None else (lambda kk: int(kk))
        dec["combo_top2"] += int(any(cat_green(to_g(kk)) == "组合" for kk in top2))
        dec["cat"][cat_green(g)] += 1
    states, _, done, _ = env.step(action={t: g}); time_share[cat_green(int(ts.green_phase))] += 1; seq.append(int(ts.green_phase))
env.close()
runs = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]) + 1
for i in range(len(seq)):
    if i == 0 or seq[i] != seq[i - 1]: stat[cat_green(seq[i])]["n_phase"] += 1
n = sum(time_share.values())
out = dict(exp=exp, ckpt=ckpt, seed=seed, time_share={c: v / n for c, v in time_share.items()},
           rate={c: (s["dep"] / s["green_s"] if s["green_s"] else None) for c, s in stat.items()},
           sat_rate={c: (s["sat_dep"] / s["sat_s"] if s["sat_s"] else None) for c, s in stat.items()},
           green_s={c: s["green_s"] for c, s in stat.items()}, n_phase={c: s["n_phase"] for c, s in stat.items()},
           dec_n=dec["n"], dec_cat=dec["cat"], marg_med=float(np.median(dec["marg"])) if dec["marg"] else None,
           tie_share=dec["tie"] / max(dec["n"], 1), combo_in_top2=dec["combo_top2"] / max(dec["n"], 1))
print("PROBE " + json.dumps(out, ensure_ascii=False))
