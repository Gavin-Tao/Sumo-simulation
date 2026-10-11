"""相位序列诊断 (2026-10-11, 只读): 单路口场景逐决策步记录绿相位, 算有效相位数 (熵)、枚举多出相位占比、运动服务间隔均值/标准差/最大、局部切换比例、每集完整轮次。用法: python experiments/tools/dublin/cycle_diag.py <exp> <ckpt> <seed>"""
(c) 每个运动的服务间隔 (两次被放行之间的秒数) 均值/最大, (d) '局部切换' 比例 (新相位与上一相位共享运动), (e) 每千秒完整轮次数 (8 个非右转运动都被服务一次)。"""
import os, sys, json, numpy as np
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
exp, ckpt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = ckpt_env.build(exp, ckpt, seed); env, act, states = B["env"], B["act"], B["states"]; t = list(env.ts_ids)[0]; ts = env.traffic_signals[t]
serves = {k: set(v) for k, v in ts.observation_fn._phase_serves.items()}
STD = {frozenset({1,2,7,8}), frozenset({4,5,10,11}), frozenset({0,2,6,8}), frozenset({3,5,9,11}), frozenset({0,1,2,5,8,11}) }  # 占位, 下面用放行集合判断是否标准相位
# 标准 8 相位的放行集合 (去掉常绿右转槽 2,5,8,11): 配对 {N T,S T},{E T,W T},{N L,S L},{E L,W L}; 单进口 {N T,N L},{S T,S L},{E T,E L},{W T,W L}
RIGHT = {2, 5, 8, 11}
std_sets = [frozenset(s) for s in ({1,7},{4,10},{0,6},{3,9},{0,1},{6,7},{3,4},{9,10})]
core = {k: frozenset(v - RIGHT) for k, v in serves.items()}
is_std = {k: (core[k] in std_sets) for k in serves}
seq = []; done = {"__all__": False}
while not done["__all__"]:
    a = act(t, states[t])[0]; states, _, done, _ = env.step(action={t: a}); seq.append(int(ts.green_phase))
env.close()
n = len(seq); dt = 5.0
dwell = {k: seq.count(k) for k in set(seq)}; p = np.array([dwell.get(k, 0) / n for k in sorted(serves)]); p = p[p > 0]
eff = float(np.exp(-(p * np.log(p)).sum()))
extra_share = sum(dwell.get(k, 0) for k in serves if not is_std[k]) / n
movs = sorted(set().union(*core.values()))
last = {m: None for m in movs}; gaps = {m: [] for m in movs}
for i, k in enumerate(seq):
    for m in core[k]:
        if last[m] is not None and i - last[m] > 1: gaps[m].append((i - last[m]) * dt)
        last[m] = i
allg=[x for g in gaps.values() for x in g]; gap_mean = float(np.mean(allg)); gap_max = float(max(allg)); gap_std=float(np.std(allg)); gap_cv=gap_std/gap_mean
sw = [(seq[i-1], seq[i]) for i in range(1, n) if seq[i] != seq[i-1]]
partial = np.mean([len(core[a] & core[b]) > 0 for a, b in sw]) if sw else float("nan")
# 完整轮次: 从头扫描, 每当 8 个运动都被服务过一次计一轮
rounds = 0; seen = set()
for k in seq:
    seen |= core[k]
    if len(seen) >= len(movs): rounds += 1; seen = set()
print(f"[{exp} seed{seed}] 步 {n} | 相位数 {len(serves)} 用到 {len(dwell)} 有效相位数(熵) {eff:.1f} | 多出相位占比 {100*extra_share:.0f}% | 切换 {len(sw)} 次, 局部切换(与上一相位共享运动) {100*partial:.0f}% | 运动服务间隔 均 {gap_mean:.0f} s 标准差 {gap_std:.0f} s (CV {gap_cv:.2f}) 最大 {gap_max:.0f} s | 完整轮次 {rounds} (每轮 {n*dt/max(rounds,1):.0f} s)")
print("   相位占比:", sorted(((k, round(100*v/n)) for k, v in dwell.items()), key=lambda x: -x[1])[:8], "| 标准相位 id:", [k for k in sorted(serves) if is_std[k]])
