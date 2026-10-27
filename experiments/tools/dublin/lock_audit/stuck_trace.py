"""卡死车辆追踪 (只读): 与 force_lock_debug 同样的强制方案, 每步记录所有本次访问等待 >= 60 s 的车辆的 SUMO 细节
(车道/位置/速度/等待/下一信号灯色/前车/换道状态/当前车道能否继续/路线下一边), 以及该车所在路口的相位。
用法: python stuck_trace.py <exp> --ckpt P --seed 123 --junction J --force-std A --force-from 600 --force-until 2400 --out prefix"""
import argparse, json, os, sys, collections
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--junction"); ap.add_argument("--force-std", type=int); ap.add_argument("--force-from", type=float, default=600)
ap.add_argument("--force-until", type=float, default=1e9); ap.add_argument("--out", required=True); ap.add_argument("--wmin", type=float, default=60)
a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed); env, act, states = B["env"], B["act"], B["states"]; sumo = env.sumo
tr = env.visit_tracker
if tr is None:
    from sumo_rl.environment.visit_wait import VisitWaitTracker
    tr = VisitWaitTracker(sumo, {t: env.traffic_signals[t].lanes for t in env.ts_ids}); _o0 = env._sumo_step
    def _h0(_o=_o0, _m=tr): _o(); _m.step()
    env._sumo_step = _h0
J = None
if a.junction:
    J = [t for t in env.ts_ids if t.startswith(a.junction)]; assert len(J) == 1, J; J = J[0]
    tsJ = env.traffic_signals[J]; g2s = tsJ.std_action_map; s2g = {int(g2s[g]): g for g in range(len(g2s)) if g2s[g] >= 0}
lane_links = {}
def links_of(lane):
    if lane not in lane_links:
        lane_links[lane] = [x[0] for x in sumo.lane.getLinks(lane)]
    return lane_links[lane]
recs = []; done = {"__all__": False}
while not done["__all__"]:
    now = float(sumo.simulation.getTime())
    acts = {t: act(t, states[t])[0] for t in env.ts_ids}
    if J is not None and a.force_std is not None and a.force_from <= now < a.force_until: acts[J] = s2g[a.force_std]
    phases = {t: int(env.traffic_signals[t].std_action_map[env.traffic_signals[t].green_phase]) for t in env.ts_ids}
    for (t, vid, edge), w in list(tr.W.items()):
        if w < a.wmin: continue
        try:
            lane = sumo.vehicle.getLaneID(vid); route = sumo.vehicle.getRoute(vid); ri = sumo.vehicle.getRouteIndex(vid)
            nxt = route[ri + 1] if ri + 1 < len(route) else None
            ntls = sumo.vehicle.getNextTLS(vid); ntls = list(ntls[0]) if ntls else None
            ld = sumo.vehicle.getLeader(vid, 200.0); bl = sumo.vehicle.getBestLanes(vid)
            cur_bl = [b for b in bl if b[0] == lane]; allows = (bool(cur_bl[0][4]) if cur_bl else None)
            lcs = (int(sumo.vehicle.getLaneChangeState(vid, 1)[0]), int(sumo.vehicle.getLaneChangeState(vid, -1)[0]))
            to_lanes = links_of(lane); can_reach = (nxt is None) or any(l.rsplit("_", 1)[0] == nxt for l in to_lanes)
            recs.append(dict(t=now, tls=t[:24], vid=vid, type=sumo.vehicle.getTypeID(vid), W=w, lane=lane, pos=round(sumo.vehicle.getLanePosition(vid), 1), lane_len=round(sumo.lane.getLength(lane), 1),
                             speed=round(sumo.vehicle.getSpeed(vid), 2), wt=sumo.vehicle.getWaitingTime(vid), acc=sumo.vehicle.getAccumulatedWaitingTime(vid),
                             next_tls=ntls, leader=(ld[0], round(ld[1], 1)) if ld else None, allows_cont=allows, lc=lcs, nxt_edge=nxt, can_reach=can_reach,
                             phase_std=phases.get(t), stopped=int(sumo.vehicle.isStopped(vid)), signals=int(sumo.vehicle.getSignals(vid))))
        except Exception as e:
            recs.append(dict(t=now, tls=t[:24], vid=vid, W=w, err=str(e)[:100]))
    states, r, done, _ = env.step(action=acts)
remaining = sumo.vehicle.getIDCount(); env.close()
json.dump(dict(exp=a.exp, ckpt_episode=B["ckpt_episode"], seed=a.seed, junction=J, force_std=a.force_std, remaining=remaining, recs=recs), open(a.out + ".json", "w"))
# 摘要: 每 (路口, 车辆) 的最大 W, 最后一条记录
last = {}
for r in recs: last[(r["tls"], r["vid"])] = r
top = sorted(last.values(), key=lambda r: -r["W"])[:25]
print(f"exp{a.exp} ep{B['ckpt_episode']} remaining={remaining} 记录 {len(recs)} 条; W 最大的 25 (路口, 车辆, 最后一次):")
for r in top:
    print("  ", {k: r.get(k) for k in ("tls", "vid", "type", "W", "t", "lane", "pos", "lane_len", "speed", "wt", "acc", "next_tls", "leader", "allows_cont", "lc", "nxt_edge", "can_reach", "phase_std", "stopped")})
