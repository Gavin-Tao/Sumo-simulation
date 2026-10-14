# 仓库副本 (2026-10-14): 中间文件在 analysis/data/synthesis_1x1_2026-10-14/ (会话中原在临时目录); 逐集行为记录合并为 behav_out.jsonl。
"""1x1 行为与奖励诊断 (只读, 不训练; 2026-10-14)。一集贪心 (或固定周期) 记录:
  训练奖励回报 (未折扣总和、γ=0.99 折扣回报), 评估收集器口径停车时间/visit (与 eval_pervehicle 相同的收集器);
  相位结构: 切换次数/1000 s, 相位时长 (含 2 s 黄灯; 10 s = 最短) 分布, 有效相位数 (熵), 局部切换比例, 多出相位时间占比,
            每个运动的服务间隔 (两次放行之间的秒数) 均值 / 变异系数 / 最大, 完整轮次 (8 个非右转运动各放行一次) 平均时长;
  决策: 能切换 (最短绿已满足) 的决策点上执行切换的比例, 保持余量 (Q 当前 − 最佳其它, 除以可选 Q 的标准差; 只对学习策略);
  浪费: 空放 = 放行运动的进口车道上一辆车都没有、同时有红灯运动的车道上有停车车辆 的决策步比例;
        截断 = 每次切换时被切掉的运动车道上仍在行驶 (速度 > 0.1 m/s) 的车辆数。
用法: python behav_1x1.py <exp> <ckpt|fixed:G[:相位列表]> <seed> [干预]   干预: none | minphase15 (相位不足 15 s 不许切) | paired4 (只许 4 个配对相位)
输出: 一行 JSON (打印到标准输出)。固定周期借 <exp> 的环境 (应为 8STD 路网的配置), 相位列表默认 0,1,2,3 (配对 4 相位, 与 fixed_cycle_eval 相同)。"""
import sys, json, numpy as np
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
from sumo_rl.environment.metrics import EpisodeMetricsCollector

exp, pol, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); interv = sys.argv[4] if len(sys.argv) > 4 else "none"
RIGHT = {2, 5, 8, 11}
fixed = pol.startswith("fixed:"); wq = pol.startswith("wq:"); tsp = pol.startswith("tsp:")
if tsp:  # 固定周期 + 公交/救护车优先 (无学习): tsp:G —— 配对 4 相位、每相位 G 秒; 红灯运动上有停着的优先车 → 提前切给它 (相位满 10 s 后);
         #                                     绿灯运动上有行驶中的优先车 → 延长, 最多再 10 s; 之后从该相位接着轮转
    parts = pol.split(":"); G = float(parts[1]); B = ckpt_env.build(exp, None, seed); ptr = 0
elif wq:   # 加权排队贪心 (无学习): wq:G[:paired|std8] —— 相位至少持续 G 秒 (含黄灯) 才可切换, 选 Σ 停车车辆优先级权重 最大的相位
    parts = pol.split(":"); G = float(parts[1]); WQSET = parts[2] if len(parts) > 2 else "paired"
    B = ckpt_env.build(exp, None, seed)
elif fixed:
    parts = pol.split(":"); G = float(parts[1]); CYC = [int(x) for x in parts[2].split(",")] if len(parts) > 2 else [0, 1, 2, 3]
    B = ckpt_env.build(exp, None, seed)
else:
    B = ckpt_env.build(exp, pol, seed)
env, act, states = B["env"], B["act"], B["states"]; t = list(env.ts_ids)[0]; ts = env.traffic_signals[t]; of = ts.observation_fn
serves = {k: frozenset(set(v) - RIGHT) for k, v in of._phase_serves.items()}
std_sets = [frozenset(s) for s in ({1, 7}, {4, 10}, {0, 6}, {3, 9}, {0, 1}, {6, 7}, {3, 4}, {9, 10})]
paired = [frozenset(s) for s in ({1, 7}, {4, 10}, {0, 6}, {3, 9})]
movs = sorted(set().union(*serves.values()))
slot_lanes = {m: [l for l, ss in of._lane_slots.items() if m in ss] for m in movs}
ts_lane_map = {x: env.traffic_signals[x].signal_controlled_lanes for x in env.ts_ids}
always_green = set().union(*(env.traffic_signals[x].always_green_lanes for x in env.ts_ids))
mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green)
g2s = getattr(ts, "std_action_map", None) if B["kind"] == "dqn_masked" else None
s2g = None
if g2s is not None:
    s2g = {int(k): int(gi) for gi, k in enumerate(g2s) if k >= 0}

seq, rew, free_n, free_sw, hold, trunc, empty = [], [], 0, 0, [], [], 0
comp = {}; TYPE_W = {k: float(v) for k, v in B['cfg']['priority_source'].items()}
done = {"__all__": False}; i = 0; sp = None if not fixed else max(int(round(G / env.delta_time)), 1)
while not done["__all__"]:
    mc.collect_step(env.sumo)
    cur = int(ts.green_phase); el = ts.time_since_last_phase_change; free = el >= ts.min_green + ts.yellow_time
    if fixed:
        a = CYC[(i // sp) % len(CYC)]
    elif tsp:
        PIDS = [k for k in sorted(serves) if serves[k] in paired]
        def pv(k, moving):
            return sum(TYPE_W.get(env.sumo.vehicle.getTypeID(v), 1.0) for m in serves[k] for l in slot_lanes[m] for v in env.sumo.lane.getLastStepVehicleIDs(l)
                       if TYPE_W.get(env.sumo.vehicle.getTypeID(v), 1.0) > 1 and ((env.sumo.vehicle.getSpeed(v) >= 0.1) == moving))
        if cur in PIDS: ptr = PIDS.index(cur)
        if el < 10 - 1e-9:
            a = cur
        else:
            red = {k: pv(k, False) for k in PIDS if k != cur}; kb = max(red, key=red.get)
            if red[kb] > 0 and pv(cur, True) == 0 and pv(cur, False) == 0:
                a = kb
            elif el < G - 1e-9 or (pv(cur, True) > 0 and el < G + 10 - 1e-9):
                a = cur
            else:
                a = PIDS[(ptr + 1) % len(PIDS)]
    elif wq:
        cand = [k for k in serves if (serves[k] in paired if WQSET == "paired" else serves[k] in std_sets)]
        def score(k):
            return sum(TYPE_W.get(env.sumo.vehicle.getTypeID(v), 1.0) for m in serves[k] for l in slot_lanes[m] for v in env.sumo.lane.getLastStepVehicleIDs(l) if env.sumo.vehicle.getSpeed(v) < 0.1)
        if el >= G - 1e-9:
            sc = {k: score(k) for k in cand}; best = max(sc, key=sc.get)
            a = cur if (cur in sc and sc[cur] >= sc[best]) else best
        else:
            a = cur
    else:
        g, k, q = act(t, states[t]); q = np.array(q, float)
        # 干预 (零样本, 只在评估时加): 在可选集合上再加限制后重新取最大
        if interv != "none":
            idx_to_green = (lambda kk: s2g[kk]) if s2g is not None else (lambda kk: kk)
            ok = ~np.isnan(q)
            for kk in np.flatnonzero(ok):
                gg = idx_to_green(kk)
                if interv == "paired4" and serves[gg] not in paired: ok[kk] = False
                if interv == "minphase15" and gg != cur and el < 15 - 1e-9: ok[kk] = False
            if ok.any():
                q2 = np.where(ok, q, np.nan); k = int(np.nanargmax(q2)); g = idx_to_green(k)
            else:
                g = cur
        a = g
        if free:
            valid = np.flatnonzero(~np.isnan(q)); sd = q[valid].std() + 1e-9
            ci = int(g2s[cur]) if g2s is not None else cur
            if ci in valid and len(valid) > 1:
                hold.append(float((q[ci] - np.nanmax(np.delete(q, ci))) / sd))
    if free:
        free_n += 1; free_sw += int(a != cur)
    # 空放: 放行运动车道全空, 且有红灯运动车道上有停车车辆
    served = serves[cur]
    if all(env.sumo.lane.getLastStepVehicleNumber(l) == 0 for m in served for l in slot_lanes[m]) and \
       any(env.sumo.lane.getLastStepHaltingNumber(l) > 0 for m in movs if m not in served for l in slot_lanes[m]):
        empty += 1
    if free and a != cur:
        cut = served - serves[a]
        trunc.append(sum(sum(1 for v in env.sumo.lane.getLastStepVehicleIDs(l) if env.sumo.vehicle.getSpeed(v) > 0.1) for m in cut for l in slot_lanes[m]))
    states, r, done, _ = env.step(action={t: a}); rew.append(float(r[t])); seq.append(int(ts.green_phase)); i += 1
    # 奖励按车类拆分: 与 make_priority_avg_waiting_reward 同一口径 (env.vehicles 已被本步奖励函数更新, 这里只读)
    tot, cntc = {}, {}
    for lane in ts.lanes:
        for vid in env.sumo.lane.getLastStepVehicleIDs(lane):
            lw = env.vehicles.get(vid, {}).get(env.sumo.vehicle.getLaneID(vid), 0.0); w = TYPE_W.get(env.sumo.vehicle.getTypeID(vid), 1.0)
            tot[w] = tot.get(w, 0.0) + lw; cntc[w] = cntc.get(w, 0) + 1
    for w in tot: comp[w] = comp.get(w, 0.0) - w * tot[w] / cntc[w] / 100.0
mc.collect_step(env.sumo); mc.finalize(env.sumo); d = mc.to_flat_dict(prefix="eval"); env.close()
gm = lambda c, m: float(d.get(f"eval/system/{c}/{m}", float("nan")))

n = len(seq); dt = float(env.delta_time)
runs, cnt = [], 1
for j in range(1, n):
    if seq[j] == seq[j - 1]: cnt += 1
    else: runs.append(cnt * dt); cnt = 1
runs.append(cnt * dt); runs_c = runs[1:-1] if len(runs) > 2 else runs   # 去掉首尾不完整的相位
sw = [(seq[j - 1], seq[j]) for j in range(1, n) if seq[j] != seq[j - 1]]
partial = float(np.mean([len(serves[x] & serves[y]) > 0 for x, y in sw])) if sw else float("nan")
dwell = {k: seq.count(k) / n for k in set(seq)}; p = np.array(list(dwell.values())); eff = float(np.exp(-(p * np.log(p)).sum()))
extra = sum(v for k, v in dwell.items() if serves[k] not in std_sets)
last = {m: None for m in movs}; gaps = []
for j, k in enumerate(seq):
    for m in serves[k]:
        if last[m] is not None and j - last[m] > 1: gaps.append((j - last[m]) * dt)
        last[m] = j
seen, rounds, t0 = set(), [], 0
for j, k in enumerate(seq):
    seen |= serves[k]
    if seen >= set(movs): rounds.append((j - t0 + 1) * dt); seen, t0 = set(), j + 1
disc = float(sum(r * 0.99 ** j for j, r in enumerate(rew)))
out = dict(exp=exp, pol=("fixed" if fixed else ("wq-" + WQSET if wq else ("tsp" if tsp else "rl"))), G=(G if (fixed or wq or tsp) else None), ckpt=(None if (fixed or wq or tsp) else pol), ckpt_ep=(0 if (fixed or wq or tsp) else B["ckpt_episode"]), seed=seed, interv=interv,
           ret=float(sum(rew)), ret_disc=disc, ret_car=comp.get(1.0, 0.0), ret_bus=comp.get(3.0, 0.0), ret_amb=comp.get(5.0, 0.0), stop_all=gm("all", "avg_stopped_time_per_visit"), stop_car=gm("car", "avg_stopped_time_per_visit"),
           stop_bus=gm("bus", "avg_stopped_time_per_visit"), stop_amb=gm("ambulance", "avg_stopped_time_per_visit"), ev_all=gm("all", "avg_stop_events_per_visit"),
           switches_per_1000s=len(sw) / (n * dt) * 1000, phase_mean=float(np.mean(runs_c)), phase_med=float(np.median(runs_c)),
           phase_share_10=float(np.mean(np.array(runs_c) <= 10 + 1e-9)), phase_share_ge20=float(np.mean(np.array(runs_c) >= 20 - 1e-9)),
           eff_phases=eff, partial=partial, extra_share=extra, gap_mean=float(np.mean(gaps)), gap_cv=float(np.std(gaps) / np.mean(gaps)), gap_max=float(np.max(gaps)),
           round_mean=float(np.mean(rounds)) if rounds else float("nan"), n_rounds=len(rounds),
           switch_when_free=(free_sw / free_n if free_n else float("nan")), free_share=free_n / n,
           hold_pos=(float(np.mean(np.array(hold) > 0)) if hold else None), hold_med=(float(np.median(hold)) if hold else None),
           empty_share=empty / n, trunc_mean=float(np.mean(trunc)) if trunc else 0.0)
print("BEHAV " + json.dumps(out, ensure_ascii=False))
