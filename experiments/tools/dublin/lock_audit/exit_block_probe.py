"""出口阻塞探针 (只读): 用检查点贪心跑一集, 每决策步对每个路口按槽统计 "绿灯给了出口有排队的槽" 的频率, 用来解释下游 ψ 的作用。
对每个路口每个决策步:
  exit_q[slot]  = 该槽出口车道 (observations._slot_out, 与 ψ 同一张表) 上停住 (speed<0.1) 的车数;
  app_q[slot]   = 该槽进口车道上停住的车数 (按车辆路线判槽, 与 ψ 的 φ 同表 _movement_by_edges);
  served        = 当前绿灯相位放行的槽 (_phase_serves)。
统计 (每路口):
  psi_cnt_nz / psi_q_nz: 任一槽出口 count>0 / queue>0 的步数占比 (ψ 的非零率, 464 虽不观测仍按同表计算);
  green_into_blocked: 有绿灯槽 app_q>0 且 exit_q>0 的步数 (绿灯给了出口堵住的槽);
  blocked_alt: 其中另有红灯槽 app_q>0 且 exit_q==0 的步数 (本可以换去出口空的槽);
  veh_green_blocked: 绿灯槽 app_q>0 且 exit_q>0 时停在绿灯槽进口的车数累计 (车·步);
  exit_q_max: 全集最大出口排队; exit_q_mean_when_nz: 出口排队非零时的平均值。
用法: python exit_block_probe.py <exp> --ckpt P --seed 123 --out prefix"""
import argparse, collections, json, os, sys
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]; sumo = env.sumo
OBS = {tid: env.traffic_signals[tid].observation_fn for tid in env.ts_ids}
for tid, o in OBS.items():
    assert hasattr(o, "_slot_out") and hasattr(o, "_phase_serves") and hasattr(o, "_movement_by_edges"), tid
st = {tid: dict(steps=0, psi_cnt_nz=0, psi_q_nz=0, green_into_blocked=0, blocked_alt=0, veh_green_blocked=0, exit_q_max=0, exit_q_nz_sum=0, exit_q_nz_steps=0,
                steps_with_queue=0, per_slot_exit_q_nz=collections.Counter(), per_slot_green_blocked=collections.Counter()) for tid in env.ts_ids}
def slot_of(o, vid, lane):
    r = sumo.vehicle.getRoute(vid); k = sumo.vehicle.getRouteIndex(vid); nxt = r[k + 1] if k + 1 < len(r) else None
    return o._movement_by_edges.get((lane.rsplit("_", 1)[0], nxt))
done = {"__all__": False}
while not done["__all__"]:
    acts = {t: act(t, states[t])[0] for t in env.ts_ids}
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]; o = OBS[tid]; s_ = st[tid]; s_["steps"] += 1
        served = o._phase_serves[ts.green_phase]
        exit_cnt = collections.Counter(); exit_q = collections.Counter()
        for slot, tls_ in o._slot_out.items():
            for tl in tls_:
                vids = sumo.lane.getLastStepVehicleIDs(tl)
                exit_cnt[slot] += len(vids); exit_q[slot] += sum(1 for v in vids if sumo.vehicle.getSpeed(v) < 0.1)
        app_q = collections.Counter()
        for lane in ts.lanes:
            for v in sumo.lane.getLastStepVehicleIDs(lane):
                if sumo.vehicle.getSpeed(v) < 0.1:
                    s = slot_of(o, v, lane)
                    if s is not None: app_q[s] += 1
        if any(exit_cnt.values()): s_["psi_cnt_nz"] += 1
        eq_max = max(exit_q.values()) if exit_q else 0
        if eq_max > 0:
            s_["psi_q_nz"] += 1; s_["exit_q_nz_sum"] += eq_max; s_["exit_q_nz_steps"] += 1; s_["exit_q_max"] = max(s_["exit_q_max"], eq_max)
            for slot, q in exit_q.items():
                if q > 0: s_["per_slot_exit_q_nz"][slot] += 1
        if app_q: s_["steps_with_queue"] += 1
        gb = [s for s in served if app_q.get(s, 0) > 0 and exit_q.get(s, 0) > 0]
        if gb:
            s_["green_into_blocked"] += 1; s_["veh_green_blocked"] += sum(app_q[s] for s in gb)
            for s in gb: s_["per_slot_green_blocked"][s] += 1
            if any(app_q.get(s, 0) > 0 and exit_q.get(s, 0) == 0 for s in app_q if s not in served): s_["blocked_alt"] += 1
    states, r, done, _ = env.step(action=acts)
env.close()
out = {}
for tid, s_ in st.items():
    out[tid] = dict(steps=s_["steps"], steps_with_queue=s_["steps_with_queue"], psi_cnt_nz=s_["psi_cnt_nz"], psi_q_nz=s_["psi_q_nz"], green_into_blocked=s_["green_into_blocked"], blocked_alt=s_["blocked_alt"],
                    veh_green_blocked=s_["veh_green_blocked"], exit_q_max=s_["exit_q_max"], exit_q_mean_when_nz=(s_["exit_q_nz_sum"] / s_["exit_q_nz_steps"] if s_["exit_q_nz_steps"] else 0.0),
                    per_slot_exit_q_nz={str(k): v for k, v in s_["per_slot_exit_q_nz"].items()}, per_slot_green_blocked={str(k): v for k, v in s_["per_slot_green_blocked"].items()})
json.dump(dict(exp=a.exp, ckpt_episode=B["ckpt_episode"], seed=a.seed, junctions=out), open(a.out + ".json", "w"), indent=1)
print(f"exp{a.exp} ep{B['ckpt_episode']} seed{a.seed} 完成")
