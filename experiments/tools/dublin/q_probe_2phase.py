"""Q 边际探针 (2026-10-10, 只读): 在一个 2 相位路口 (默认 389332) 逐决策步记录两个相位的 Q、选择、切换、已持续时间, 以及两相位各自放行运动的排队和 (状态向量的 queue 特征), 用来看决策跟的是排队平衡还是保持。用法: python experiments/tools/dublin/q_probe_2phase.py <exp> <ckpt.pth> <seed> [tls_id]。"""
import os, sys, json, numpy as np
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
from sumo_rl.environment.observations import PRIORITY_LEVELS
exp, ckpt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); T = sys.argv[4] if len(sys.argv) > 4 else "389332"
B = ckpt_env.build(exp, ckpt, seed); env, act, states = B["env"], B["act"], B["states"]
ts = env.traffic_signals[T]; of = ts.observation_fn; serves = of._phase_serves   # phase -> served slots
fields = of.fields; nl = len(PRIORITY_LEVELS); legacy = of._is_legacy
head = (8 + 1) if legacy else 2
phi = len(fields) * nl; psi = nl * len(of.downstream_fields) if of.include_downstream else 0
slot_dim = phi + psi + (1 if of.include_lane_occ else 0) + (1 if of.include_since_green else 0); per = slot_dim + (0 if legacy else 1)
qi = fields.index("queue")
def served_queue(vec, slots):   # Σ_{slot∈served} Σ_level queue/cap
    tot = 0.0
    for s in slots:
        o = head + s * per + (0 if legacy else 1)
        tot += sum(float(vec[o + p * len(fields) + qi]) for p in range(nl))
    return tot
rows = []; prev = None; done = {"__all__": False}
while not done["__all__"]:
    a = {t: act(t, states[t]) for t in env.ts_ids}
    k, _, q = a[T]; qv = {int(i): float(x) for i, x in enumerate(q) if not np.isnan(x)}
    # 8STD: q 是 8 个标准槽, k 是 green index; 把 q 映射到 green index
    if B["kind"] == "dqn_masked":
        g2s = ts.std_action_map; qv = {int(g): float(q[int(g2s[g])]) for g in range(ts.num_green_phases) if g2s[g] >= 0 and not np.isnan(q[int(g2s[g])])}
    ph = sorted(serves.keys()); qA, qB = qv.get(ph[0], np.nan), qv.get(ph[1], np.nan)
    vec = states[T]; QA, QB = served_queue(vec, serves[ph[0]]), served_queue(vec, serves[ph[1]])
    rows.append(dict(t=float(env.sumo.simulation.getTime()), chosen=int(k), cur=int(ts.green_phase), elapsed=float(ts.time_since_last_phase_change), qA=qA, qB=qB, queueA=QA, queueB=QB, switch=int(prev is not None and k != prev)))
    prev = k
    states, _, done, _ = env.step(action={t: a[t][0] for t in env.ts_ids})
env.close()
r = rows; m = np.array([x["qA"] - x["qB"] for x in r]); dq = np.array([x["queueA"] - x["queueB"] for x in r]); sw = np.array([x["switch"] for x in r]); ch = np.array([x["chosen"] for x in r])
scale = np.nanmean([abs(x["qA"]) + abs(x["qB"]) for x in r]) / 2
print(f"[{exp} seed{seed} 389332] 决策 {len(r)} 步, 切换 {sw.sum()} 次, 选相位{sorted(set(ch.tolist()))} 占比 {np.mean(ch==ch.min()):.2f}/{np.mean(ch==ch.max()):.2f}")
print(f"  Q 量级 |Q| 均 {scale:.2f}; Q(A)-Q(B) 的均值 {np.nanmean(m):.3f}, 绝对值中位 {np.nanmedian(np.abs(m)):.3f}, 相对量级 {np.nanmedian(np.abs(m))/scale:.3f}")
print(f"  corr(Q(A)-Q(B), 排队A-排队B) = {np.corrcoef(m, dq)[0,1]:.2f};  切换时刻 |Q差| 中位 {np.nanmedian(np.abs(m[sw==1])) if sw.sum() else float('nan'):.3f}; 切换时 '切向排队更多的一侧' 的比例 {np.mean([(x['chosen']==ph[0]) == (x['queueA']>x['queueB']) for x in r if x['switch']]) if sw.sum() else float('nan'):.2f}")
# 排队更多的一侧是否就是被选的一侧 (逐步)
agree = np.mean([((x["queueA"] > x["queueB"]) == (x["chosen"] == ph[0])) for x in r if abs(x["queueA"] - x["queueB"]) > 0.05])
print(f"  逐步: 选了排队更多一侧的比例 {agree:.2f}; 已持续时间<10 s 时切换的次数 {int(sum(1 for x in r if x['switch'] and x['elapsed'] < 10))}")
OUT = os.environ.get("QPROBE_OUT", "/home/xiaowen/sumo-rl/experiments/analysis/data"); json.dump(rows, open(f"{OUT}/qprobe_{exp}_{T}_seed{seed}.json", "w"))
