"""场景标定检查 (设计文档 SCENARIO_PRIORITY_PAIRS50_DESIGN §2.4 与 §4; 2026-10-28 第二版, 按外部复核修正; 只读, 不训练)。
用模板配置 (默认 exp461) 的环境参数构造环境, 换成指定路网/meta/路由, 用两种与学习无关的控制器跑整集:
  --policy random : 每个决策步在该路口全部有效绿相位里均匀随机;  --policy fixed : 各有效绿相位依次轮转, 每相位绿 G 秒 + 黄 (决策步 5 s, 实际切换在 ≥ G+黄 的第一个决策步)。
第二版修正: (1) 长等待按 **每次访问逐秒记录的最大等待** 统计 (第一版只在集末看统计器, 中途离开的全漏); (2) 车辆→转向一律用 meta 的 (from_edge, to_edge) 表
(第一版用观测类的几何映射, 与 meta 在 cluster_21620852 / cluster_135109528 有 4 处错配); (3) 实验车访问逐秒登记 (分母不漏快速通过), 机会在决策步判断;
(4) 饿死/回溢在集末把仍在持续的事件计入; (5) 回溢改为 "出口车道上停住的车排到车道起点 (队尾距起点 ≤ 10 m) 持续 ≥ 60 s"; (6) 输出换道次数 (lanechange-output) 与实验车到达数; (7) 记每个 RL 路口出口车道上的停车 (车·秒, 按决策步采样 ×5) 与最长出口排队 —— 这些停车不进任何 RL 路口的访问指标 (状态/奖励/逐访问停车/延误都只看进口道), 只进整趟 timeLoss。
用法: python scenario_check.py --net N --meta M --sumocfg C --routes r1,r2 --policy random|fixed --seed 123 --out prefix [--template 461] [--green 20]"""
import argparse, os, sys, json, random, collections, functools, glob, xml.etree.ElementTree as ET
os.environ["OMP_NUM_THREADS"] = "1"; os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"; sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/experiments")
import yaml, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--net", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--sumocfg", required=True); ap.add_argument("--routes", required=True, help="逗号分隔, 绝对或相对仓库根")
ap.add_argument("--template", default="461"); ap.add_argument("--policy", choices=["random", "fixed"], required=True); ap.add_argument("--green", type=float, default=20.0)
ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); ap.add_argument("--lock-s", type=float, default=150.0); ap.add_argument("--exp-types", default="", help="除 pair_ 前缀外, 把这些 vType (逗号分隔, 如 bus,ambulance) 也当作实验车统计机会"); a = ap.parse_args()
EXP_TYPES = {x for x in a.exp_types.split(",") if x}
absr = lambda p: p if os.path.isabs(p) else os.path.join(REPO, p)
a.out = os.path.abspath(a.out); NET = absr(a.net); META = absr(a.meta); CFG = absr(a.sumocfg); ROUTES = ",".join(absr(r) for r in a.routes.split(","))
os.chdir(f"{REPO}/experiments")
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from exp_params import obs_kwargs_from_cfg, env_extra_kwargs, reward_wait_basis, turnmap_from_meta
cfg = yaml.safe_load(open(sorted(glob.glob(f"configs/exp{a.template}_*.yaml"))[0]))
cfg["net_file"] = NET; cfg["action_meta_file"] = META; cfg["cfg_file"] = CFG; cfg["route_file"] = ROUTES
kw = obs_kwargs_from_cfg(cfg); obs_class = functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
reward_fn = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")), wait_basis=reward_wait_basis(cfg))
TRIP = f"{REPO}/experiments/logs/tripinfo_scen_{os.getpid()}.xml"; LC = f"{REPO}/experiments/logs/lanechange_scen_{os.getpid()}.xml"
env = SumoEnvironment(net_file=NET, route_file=ROUTES, cfg_file=CFG, out_csv_name=None, use_gui=False, num_seconds=cfg.get("num_seconds", 3600), min_green=cfg.get("min_green", 5),
                      max_green=cfg.get("max_green", 50), use_max_green=cfg.get("use_max_green", False), single_agent=False, yellow_time=cfg.get("yellow_time", 2), delta_time=cfg.get("delta_time", 5),
                      reward_fn=reward_fn, observation_class=obs_class, sumo_seed=a.seed, sumo_warnings=False,
                      additional_sumo_cmd=f"--tripinfo-output {TRIP} --tripinfo-output.write-unfinished true --lanechange-output {LC}", **env_extra_kwargs(cfg))
states = env.reset(a.seed); sumo = env.sumo; ids = list(env.ts_ids)
meta = json.load(open(META))["tls"]
for tid in ids:   # 与训练一致: 观测槽表按 meta 的 turnmap 重绑 (本脚本的车辆→转向也只用 meta 表)
    env.traffic_signals[tid].observation_fn.rebind_movements(turnmap_from_meta(meta[tid]))
name = lambda c: c["approach"] + "-" + c["turn"]
T = {}
for tid in ids:
    ts = env.traffic_signals[tid]; t = meta[tid]
    link_slot = {int(i): name(grp[0]) for i, grp in t["links"].items()}
    serves = {g: {link_slot[i] for i in link_slot if i < len(ph.state) and ph.state[i] in "Gg"} for g, ph in enumerate(ts.green_phases)}
    by_edges = {}; out_lanes = {}
    for i, grp in t["links"].items():
        for c in grp:
            by_edges[(c["from_edge"], c["to_edge"])] = link_slot[int(i)]
            out_lanes[c["to_edge"] + "_" + str(c["to_lane"])] = sumo.lane.getLength(c["to_edge"] + "_" + str(c["to_lane"]))
    jointly = {(m1, m2) for g in serves for m1 in serves[g] for m2 in serves[g]}
    T[tid] = dict(serves=serves, by_edges=by_edges, lanes=list(ts.lanes), out_lanes=out_lanes, n_green=len(ts.green_phases), jointly=jointly, yellow=ts.yellow_time, min_green=ts.min_green)
def slot_of(tid, vid, lane):
    r = sumo.vehicle.getRoute(vid); k = sumo.vehicle.getRouteIndex(vid); nxt = r[k + 1] if k + 1 < len(r) else None
    return T[tid]["by_edges"].get((lane.rsplit("_", 1)[0], nxt))
rng = random.Random(a.seed); fixed_state = {t: [0, 0.0] for t in ids}; PRIO = {"bus", "ambulance"}
st = {t: dict(starve_events=0, starve_max=0.0, red_since={}, spill_events=0, spill_since={}) for t in ids}
visits = {}; cur_visit = {}; visit_no = collections.Counter(); running = []; pending_3300 = None
tracker = env.visit_tracker
def per_second_hook():
    """每个仿真秒: 登记每次访问 (所有车) 并更新本次访问的最大等待。"""
    now = float(sumo.simulation.getTime()); present = set()
    for tid in ids:
        for lane in T[tid]["lanes"]:
            for v in sumo.lane.getLastStepVehicleIDs(lane):
                key = (v, tid); present.add(key)
                if key not in cur_visit:
                    visit_no[key] += 1; cur_visit[key] = visit_no[key]
                    _ty = sumo.vehicle.getTypeID(v)
                    visits[(v, tid, cur_visit[key])] = dict(type=_ty, t0=now, slot=slot_of(tid, v, lane), max_wait=0.0, opp=False, kinds=set(), states=set(), prio_comp=False, exp=(v.startswith("pair_") or _ty in EXP_TYPES))
                rec = visits[(v, tid, cur_visit[key])]
                w = tracker.get(tid, v, lane) if tracker is not None else sumo.vehicle.getAccumulatedWaitingTime(v)
                if w > rec["max_wait"]: rec["max_wait"] = w
    for key in list(cur_visit):
        if key not in present: del cur_visit[key]
_orig_step = env._sumo_step
def _step_hook(*args, **kwargs):
    r = _orig_step(*args, **kwargs); per_second_hook(); return r
env._sumo_step = _step_hook
done = {"__all__": False}
while not done["__all__"]:
    now = float(sumo.simulation.getTime()); acts = {}
    for tid in ids:
        n = T[tid]["n_green"]
        if a.policy == "random": acts[tid] = rng.randrange(n)
        else:
            ph, t0 = fixed_state[tid]
            if now - t0 >= a.green + T[tid]["yellow"]: ph = (ph + 1) % n; fixed_state[tid] = [ph, now]
            acts[tid] = ph
    for tid in ids:
        ts = env.traffic_signals[tid]; tt = T[tid]; cur = ts.green_phase; served = tt["serves"][cur]; s_ = st[tid]
        min_ok = ts.time_since_last_phase_change >= tt["min_green"] + tt["yellow"]
        present = collections.Counter(); queued = collections.Counter(); prio_on = collections.defaultdict(bool); exp_vehs = []
        for lane in tt["lanes"]:
            for v in sumo.lane.getLastStepVehicleIDs(lane):
                s = slot_of(tid, v, lane)
                if s is None: continue
                present[s] += 1; spd = sumo.vehicle.getSpeed(v)
                if spd < 0.1: queued[s] += 1
                if sumo.vehicle.getTypeID(v) in PRIO: prio_on[s] = True
                if v.startswith("pair_") or sumo.vehicle.getTypeID(v) in EXP_TYPES: exp_vehs.append((v, s, spd))
        for s in list(set(queued) | set(s_["red_since"])):
            if queued.get(s, 0) > 0 and s not in served: s_["red_since"].setdefault(s, now)
            else:
                t0 = s_["red_since"].pop(s, None)
                if t0 is not None and now - t0 >= 60: s_["starve_events"] += 1; s_["starve_max"] = max(s_["starve_max"], now - t0)
        for l in tt["out_lanes"]:      # 回溢: 停住的车的队尾 (最靠近车道起点的停车) 距起点 ≤ 10 m; 另记出口车道上的停车 (车·步, 不进 RL 路口的访问指标)
            halted_pos = [sumo.vehicle.getLanePosition(v) for v in sumo.lane.getLastStepVehicleIDs(l) if sumo.vehicle.getSpeed(v) < 0.1]
            if halted_pos: s_["exit_halt_steps"] = s_.get("exit_halt_steps", 0) + len(halted_pos); s_["exit_queue_max"] = max(s_.get("exit_queue_max", 0.0), tt["out_lanes"][l] - min(halted_pos))
            if halted_pos and min(halted_pos) <= 10.0: s_["spill_since"].setdefault(l, now)
            else:
                t0 = s_["spill_since"].pop(l, None)
                if t0 is not None and now - t0 >= 60: s_["spill_events"] += 1
        if exp_vehs:
            legal = list(range(tt["n_green"])) if min_ok else [cur]
            for v, s, spd in exp_vehs:
                rec = visits.get((v, tid, cur_visit.get((v, tid), 0)))
                if rec is None: continue
                serve_me = [g for g in legal if s in tt["serves"][g]]; not_me = [g for g in legal if s not in tt["serves"][g]]
                if not serve_me or not not_me: continue
                conflict = False; prio_c = False
                for g in not_me:
                    for m2 in tt["serves"][g]:
                        if present.get(m2, 0) > 0 and (s, m2) not in tt["jointly"]: conflict = True; prio_c = prio_c or prio_on[m2]
                if not conflict: continue
                rec["opp"] = True; rec["kinds"].add("green_hold" if s in served else "red_release"); rec["states"].add("queued" if spd < 0.1 else "approach"); rec["prio_comp"] = rec["prio_comp"] or prio_c
    running.append((now, sumo.vehicle.getIDCount()))
    if abs(now - 3300.0) < 2.5: pending_3300 = len(sumo.simulation.getPendingVehicles())
    states, r, done, _ = env.step(action=acts)
now = float(sumo.simulation.getTime())
for tid in ids:    # 集末仍在持续的事件
    s_ = st[tid]
    for s, t0 in s_["red_since"].items():
        if now - t0 >= 60: s_["starve_events"] += 1; s_["starve_max"] = max(s_["starve_max"], now - t0)
    for l, t0 in s_["spill_since"].items():
        if now - t0 >= 60: s_["spill_events"] += 1
env.close()
tis = list(ET.parse(TRIP).getroot().iter("tripinfo")); os.remove(TRIP)
n_lc = sum(1 for _ in ET.parse(LC).getroot().iter("change")) if os.path.exists(LC) else None
if os.path.exists(LC): os.remove(LC)
dep = [float(t.get("departDelay")) for t in tis if float(t.get("depart", -1)) >= 0]
arrived = sum(1 for t in tis if float(t.get("arrival", -1)) >= 0)
pairs_total = sum(1 for t in tis if t.get("id", "").startswith("pair_")); pairs_arrived = sum(1 for t in tis if t.get("id", "").startswith("pair_") and float(t.get("arrival", -1)) >= 0)
planned = 0; in_trip = {t.get("id") for t in tis}; not_ins = 0
for rf in ROUTES.split(","):
    for v in ET.parse(rf).getroot().iter("vehicle"):
        if float(v.get("depart")) <= 3300: planned += 1; not_ins += (v.get("id") not in in_trip)
win = collections.defaultdict(list)
for t, n in running: win[int(t // 600)].append(n)
rw = [float(np.mean(win[k])) for k in sorted(win)]
lock_visits = 0; lock_by = collections.Counter(); long_by_type = collections.Counter(); max_wait_all = 0.0; max_wait_rec = None
for (v, tid, k), rec in visits.items():
    if rec["max_wait"] >= a.lock_s: lock_visits += 1; lock_by[tid] += 1; long_by_type[rec["type"]] += 1
    if rec["max_wait"] > max_wait_all: max_wait_all = rec["max_wait"]; max_wait_rec = (v, tid, rec["max_wait"])
opp = collections.defaultdict(lambda: dict(visits=0, with_opp=0, red_release=0, green_hold=0, approach=0, queued=0, prio_comp=0))
for (v, tid, k), rec in visits.items():
    if not rec["exp"] or rec["slot"] is None: continue
    o = opp[(tid, rec["slot"], rec["type"])]; o["visits"] += 1
    if rec["opp"]:
        o["with_opp"] += 1; o["red_release"] += "red_release" in rec["kinds"]; o["green_hold"] += "green_hold" in rec["kinds"]; o["approach"] += "approach" in rec["states"]; o["queued"] += "queued" in rec["states"]; o["prio_comp"] += rec["prio_comp"]
by_type = collections.defaultdict(lambda: [0, 0])
for (tid, s, typ), o in opp.items(): by_type[typ][0] += o["visits"]; by_type[typ][1] += o["with_opp"]
out = dict(version=2, policy=a.policy, green=a.green, seed=a.seed, net=NET, routes=ROUTES, running_10min=rw, running_growth_last2_vs_34=(np.mean(rw[4:6]) / max(np.mean(rw[2:4]), 1) - 1) if len(rw) >= 6 else None,
           pending_3300=pending_3300, planned_le3300=planned, not_inserted_le3300=not_ins, tripinfo=len(tis), arrived=arrived, pairs_in_tripinfo=pairs_total, pairs_arrived=pairs_arrived,
           depart_delay_mean=float(np.mean(dep)) if dep else None, depart_delay_max=float(np.max(dep)) if dep else None, lane_changes=n_lc,
           visits_total=len(visits), lock_visits=lock_visits, lock_s=a.lock_s, lock_by_junction=dict(lock_by), lock_by_type=dict(long_by_type), max_visit_wait=max_wait_all, max_visit_wait_rec=max_wait_rec,
           starve={t: dict(events=st[t]["starve_events"], max=st[t]["starve_max"]) for t in ids}, spill={t: st[t]["spill_events"] for t in ids},
           exit_halt_veh_s={t: st[t].get("exit_halt_steps", 0) * 5 for t in ids}, exit_queue_max_m={t: round(st[t].get("exit_queue_max", 0.0), 1) for t in ids},
           opportunities={f"{tid}|{s}|{typ}": o for (tid, s, typ), o in opp.items()}, opp_by_type={k: dict(visits=v[0], with_opp=v[1]) for k, v in by_type.items()})
json.dump(out, open(a.out + ".json", "w"), indent=1, ensure_ascii=False)
print(f"{a.policy} seed{a.seed}: 在网 10min 窗均值 {[round(x) for x in rw]} 末两窗/三四窗 {out['running_growth_last2_vs_34']:+.2f}; pending@3300 {pending_3300}; 未插入(≤3300) {not_ins}/{planned}; 出发延迟 {out['depart_delay_mean']:.1f}/{out['depart_delay_max']:.0f}; 到达 {arrived}/{len(tis)} 实验车 {pairs_arrived}/{pairs_total}; 访问 {len(visits)} 其中等待≥{a.lock_s:.0f}s 的 {lock_visits} (最长 {max_wait_all:.0f} s {max_wait_rec}); 饿死 {sum(v['events'] for v in out['starve'].values())} 次 (最长 {max(v['max'] for v in out['starve'].values()):.0f}); 回溢 {sum(out['spill'].values())}; 换道 {n_lc}; 机会 {out['opp_by_type']}")
