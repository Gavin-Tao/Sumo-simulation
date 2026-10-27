"""强制锁死/自然锁死逐步核对 (只读, 不训练): 贪心策略跑一集, 可在指定路口 [force_from, force_until) 强制保持某 std 相位;
每步用 state_reward_debug.Ref 独立重算全网状态与奖励并比对 (统计器 strict); 对指定路口逐步导出: 相位、策略选择与 Q、各槽特征 (还原成秒/辆),
奖励 (环境/重算), SUMO 真值 (各进口车道车数/停车数/最大 getWaitingTime/最大本次访问等待/该车累计等待), 全网每路口最大本次访问等待 (看溢流), 传送计数。
用法: python force_lock_debug.py <exp> --ckpt P --seed 123 --junction cluster_21620852 --force-std 0 --force-from 600 --force-until 2400 --out prefix"""
import argparse, json, math, os, sys, collections
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import ckpt_env, state_reward_debug as srd

ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--junction", required=True); ap.add_argument("--force-std", type=int); ap.add_argument("--force-from", type=float, default=600)
ap.add_argument("--force-until", type=float, default=1e9); ap.add_argument("--out", required=True); ap.add_argument("--dump", help="导出细节的路口 (默认 = --junction)"); ap.add_argument("--routes", help="评估专用路由文件 (逗号分隔, 相对 experiments/)")
a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed, route_override=a.routes); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]; sumo = env.sumo
if env.visit_tracker is None:      # 旧口径配置没有统计器: 挂一个只做测量的 (不影响环境的状态/奖励)
    from sumo_rl.environment.visit_wait import VisitWaitTracker
    _mtr = VisitWaitTracker(sumo, {t: env.traffic_signals[t].lanes for t in env.ts_ids}); _o0 = env._sumo_step
    def _h0(_o=_o0, _m=_mtr): _o(); _m.step()
    env._sumo_step = _h0
else:
    env.visit_tracker.strict = True
meta = json.load(open(cfg["action_meta_file"]))["tls"]; ref = srd.Ref(env, cfg, meta); groups = srd.group_layout(ref)
JF = [t for t in env.ts_ids if t.startswith(a.junction)]; assert len(JF) == 1, JF; JF = JF[0]      # 强制相位的路口
_g2sF = env.traffic_signals[JF].std_action_map; s2gF = {int(_g2sF[g]): g for g in range(len(_g2sF)) if _g2sF[g] >= 0}
J = [t for t in env.ts_ids if t.startswith(a.dump or a.junction)]; assert len(J) == 1, J; J = J[0]   # 导出细节的路口
tsJ = env.traffic_signals[J]; g2s = tsJ.std_action_map
T = ref.T[J]; tr = env.visit_tracker if env.visit_tracker is not None else _mtr
D = len(groups) // 12; assert not ref.header
def slot_index(s):
    base = s * D; idx = base; m = {"is_green": idx}; idx += 1
    for p in srd.LEVELS:
        for f in ref.fields: m[(f, p)] = idx; idx += 1
    for f in ref.ds_fields:
        for p in srd.LEVELS: m[("ds_" + f, p)] = idx; idx += 1
    if ref.lane_occ: m["lane_occ"] = idx; idx += 1
    if ref.since_green: m["since_green"] = idx; idx += 1
    if ref.slot_mg: m["min_green"] = idx; idx += 1
    assert idx == base + D, (idx, base, D); return m
SI = [slot_index(s) for s in range(12)]
delog = (lambda v: math.expm1(v * srd.LOG)) if ref.awt_log else (lambda v: v * 100.0)
name = lambda s: srd.APP[s // 3] + "-" + srd.TURN[s % 3]
lane_slot = {l: sorted(ss) for l, ss in T.lane_slots.items()}
rows = []; maxd_all = collections.defaultdict(float); rdiff_all = 0.0; first_bad = []; last_r = None; last_r_ref = None
done = {"__all__": False}
while not done["__all__"]:
    now = float(sumo.simulation.getTime())
    acts, info = {}, {}
    for t in env.ts_ids:
        a_env, k, q = act(t, states[t]); acts[t] = a_env; info[t] = (int(k), q)
    forced = a.force_std is not None and a.force_from <= now < a.force_until
    if forced: acts[JF] = s2gF[a.force_std]
    sJ = np.asarray(states[J], dtype=np.float64)
    # ---- 该路口当前快照 (决策时刻 s_t) ----
    cur_std = int(g2s[tsJ.green_phase]); k, q = info[J]
    slots = {}
    for s in range(12):
        m = SI[s]; cnt = {p: sJ[m[("count", p)]] * T.slot_cap[s] for p in srd.LEVELS} if ("count", 1) in m else {}
        if not any(v > 0 for v in cnt.values()) and sJ[m["is_green"]] == 0 and (not ref.lane_occ or sJ[m["lane_occ"]] == 0): continue
        d = {"g": int(sJ[m["is_green"]])}
        for p, lab in ((1, "car"), (4, "bus"), (5, "amb")):
            c = cnt.get(p, 0.0)
            if c > 0: d[lab] = (round(c, 1), round(sJ[m[("queue", p)]] * T.slot_cap[s], 1) if ("queue", p) in m else None, round(delog(sJ[m[("mean_awt", p)]]), 1))
        if ref.lane_occ: d["occ"] = round(sJ[m["lane_occ"]], 2)
        if ("ds_count", 1) in m: d["ds_car"] = (round(sJ[m[("ds_count", 1)]] * T.out_cap.get(s, 1.0), 1), round(sJ[m[("ds_queue", 1)]] * T.out_cap.get(s, 1.0), 1))
        if ref.slot_mg: d["mg"] = int(sJ[m["min_green"]])
        slots[name(s)] = d
    lanes = {}
    for lane in T.lanes:
        vids = sumo.lane.getLastStepVehicleIDs(lane)
        if not vids and sumo.lane.getLastStepHaltingNumber(lane) == 0: continue
        ws = [(tr.W.get((J, v, lane.rsplit("_", 1)[0]), -1.0), v) for v in vids]
        wmax, vmax = max(ws) if ws else (0.0, None)
        lanes[lane] = dict(slots=[name(s) for s in lane_slot.get(lane, [])], n=len(vids), halt=int(sumo.lane.getLastStepHaltingNumber(lane)),
                           max_wt=round(max([sumo.vehicle.getWaitingTime(v) for v in vids] or [0.0]), 1), max_W=round(wmax, 1), vmax=vmax,
                           vmax_acc=(round(sumo.vehicle.getAccumulatedWaitingTime(vmax), 1) if vmax else None), vmax_wt=(round(sumo.vehicle.getWaitingTime(vmax), 1) if vmax else None),
                           w_missing=sum(1 for w, _ in ws if w < 0))
    netmax = collections.defaultdict(float); n300 = 0
    for (t, vid, edge), w in tr.W.items():
        if w > netmax[t]: netmax[t] = w
        if w >= 300: n300 += 1
    rows.append(dict(t=now, std=cur_std, policy_k=k, q={str(int(g2s[g])): round(float(q[int(g2s[g])]), 3) for g in range(len(g2s)) if g2s[g] >= 0} if q is not None else None,
                     forced=forced, applied_std=int(g2s[acts[J]]), r_prev=last_r, r_prev_ref=last_r_ref, slots=slots, lanes=lanes,
                     net_maxW={t[:20]: round(w, 1) for t, w in netmax.items() if w >= 60}, n_W300=n300, teleports=int(sumo.simulation.getStartingTeleportNumber())))
    states, r, done, _ = env.step(action=acts)
    now2 = float(sumo.simulation.getTime())
    for t in env.ts_ids:
        got = np.asarray(states[t], dtype=np.float64); exp = ref.expected_state(t); d = np.abs(got - exp)
        for g, x in zip(groups, d): maxd_all[g] = max(maxd_all[g], float(x))
        if d.max() > srd.TOL and len(first_bad) < 20:
            i = int(d.argmax()); first_bad.append((now2, t[:24], i, groups[i], float(got[i]), float(exp[i])))
        rr = ref.expected_reward(t); rd = abs(float(r[t]) - rr); rdiff_all = max(rdiff_all, rd)
        if t == J: last_r, last_r_ref = round(float(r[t]), 5), round(rr, 5)
od, ok_ = ref.own.compare_env_tracker() if ref.own is not None else (None, None)
remaining = sumo.vehicle.getIDCount(); env.close()
summary = dict(exp=a.exp, ckpt=B["ckpt"], ckpt_episode=B["ckpt_episode"], seed=a.seed, junction=J, force_junction=JF, force_std=a.force_std, force_from=a.force_from, force_until=a.force_until,
               state_max_diff_by_group={g: maxd_all[g] for g in sorted(set(groups))}, reward_max_diff=rdiff_all, first_mismatches=first_bad,
               tracker_missing=getattr(env.visit_tracker, 'missing', None), own_vs_env=dict(max_diff=od, key_mismatch=ok_), acc_identity=(dict(checked=ref.own.acc_checked, max_diff=ref.own.acc_max_diff) if ref.own is not None else None),
               stats=dict(ref.stats), remaining_vehicles=remaining, teleports=rows[-1]["teleports"], max_W_J=max(max([l["max_W"] for l in rw["lanes"].values()] or [0]) for rw in rows),
               reward_sum_J=sum(rw["r_prev"] or 0 for rw in rows))
json.dump(dict(summary=summary, rows=rows), open(a.out + ".json", "w"))
print(json.dumps({k: v for k, v in summary.items() if k not in ("first_mismatches",)}, ensure_ascii=False, default=str))
for fb in first_bad: print("不一致:", fb)
