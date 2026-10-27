"""场景标定检查 (设计文档 SCENARIO_PRIORITY_PAIRS50_DESIGN §2.4 与 §4, 只读, 不训练)。
用 exp461 的环境参数 (精确版口径, 8STD 分槽 meta) 构造环境, 换成本场景的 sumocfg/路由, 用两种与学习无关的控制器跑整集:
  --policy random : 每个决策步在该路口全部有效绿相位里均匀随机 (环境自己保证最短绿);
  --policy fixed  : 各有效绿相位依次轮转, 每相位 G 秒 (--green, 默认 20);
每集输出 (JSON):
  running: 6 个 10 分钟窗的在网车辆数均值; pending_3300: 3300 s 时已到出发时刻未插入的车数; depart_delay: 平均/最大; not_inserted: 出发时刻 ≤ 3300 s 到 3600 s 仍未插入的车数;
  lock: 任一车辆本次访问等待 ≥ 150 s 的车辆数 (scan_locks 口径, 读 env.visit_tracker); starve: 每路口 "有排队且连续红 ≥ 60 s" 事件数与最长 (junction_diag 口径);
  spill: 每路口出口车道占用 ≥ 80% 持续 ≥ 60 s 的事件数 (exit_block_probe 口径);
  opportunities: 实验车 (id 以 pair_ 开头) 每次路口访问的决策机会 (§4 定义: 合法动作集里既有放行它的也有放行有需求冲突运动的; 红灯提前放行 / 绿灯保持; 接近/排队; 对手含优先级车), 按 (路口, 转向, 车类) 汇总独立访问数与有机会的访问数。
用法: python scenario_check.py --scenario nets/dublin/weekday_pairs50 --policy random --seed 123 --out prefix [--template 461] [--green 20]"""
import argparse, os, sys, json, random, collections, functools, glob, xml.etree.ElementTree as ET
os.environ["OMP_NUM_THREADS"] = "1"; os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"; sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/experiments")
import yaml, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="nets/dublin/weekday_pairs50"); ap.add_argument("--sumocfg", default="dublin_weekday_pairs50.sumocfg")
ap.add_argument("--routes", default="background_cars.rou.xml,pairs50_routes.rou.xml,pairs50_tt1.rou.xml"); ap.add_argument("--template", default="461")
ap.add_argument("--policy", choices=["random", "fixed"], required=True); ap.add_argument("--green", type=float, default=20.0)
ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); a = ap.parse_args()
a.out = os.path.abspath(a.out)
os.chdir(f"{REPO}/experiments")
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from exp_params import obs_kwargs_from_cfg, env_extra_kwargs, reward_wait_basis
cfg = yaml.safe_load(open(sorted(glob.glob(f"configs/exp{a.template}_*.yaml"))[0]))
scen = os.path.join(REPO, a.scenario); routes = ",".join(os.path.join(scen, r) for r in a.routes.split(","))
cfg["cfg_file"] = os.path.join(scen, a.sumocfg); cfg["route_file"] = routes
kw = obs_kwargs_from_cfg(cfg); obs_class = functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
reward_fn = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")), wait_basis=reward_wait_basis(cfg))
TRIP = f"{REPO}/experiments/logs/tripinfo_scen_{os.getpid()}.xml"
env = SumoEnvironment(net_file=os.path.join(REPO, cfg["net_file"].replace("../", "")), route_file=routes, cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                      num_seconds=cfg.get("num_seconds", 3600), min_green=cfg.get("min_green", 5), max_green=cfg.get("max_green", 50), use_max_green=cfg.get("use_max_green", False),
                      single_agent=False, yellow_time=cfg.get("yellow_time", 2), delta_time=cfg.get("delta_time", 5), reward_fn=reward_fn, observation_class=obs_class,
                      sumo_seed=a.seed, sumo_warnings=False, additional_sumo_cmd=f"--tripinfo-output {TRIP} --tripinfo-output.write-unfinished true", **env_extra_kwargs(cfg))
states = env.reset(a.seed); sumo = env.sumo; ids = list(env.ts_ids)
meta = json.load(open(os.path.join(REPO, cfg["action_meta_file"].replace("../", ""))))["tls"]
OBS = {t: env.traffic_signals[t].observation_fn for t in ids}
name = lambda c: c["approach"] + "-" + c["turn"]
# 静态表
T = {}
for tid in ids:
    ts = env.traffic_signals[tid]; o = OBS[tid]; t = meta[tid]
    link_slot = {int(i): name(grp[0]) for i, grp in t["links"].items()}
    serves = {g: {link_slot[i] for i in link_slot if i < len(ph.state) and ph.state[i] in "Gg"} for g, ph in enumerate(ts.green_phases)}
    slot_lanes = collections.defaultdict(set)
    for i, grp in t["links"].items():
        for c in grp: slot_lanes[name(c)].add(c["from_edge"] + "_" + str(c["from_lane"]))
    T[tid] = dict(serves=serves, slot_lanes=slot_lanes, lanes=list(ts.lanes), out=o._slot_out, by_edges=o._movement_by_edges, n_green=len(ts.green_phases),
                  out_cap={l: max(sumo.lane.getLength(l) / 7.5, 1.0) for ss in o._slot_out.values() for l in ss})
rng = random.Random(a.seed); fixed_state = {t: [0, 0.0] for t in ids}   # [相位序号, 该相位开始时间]
PRIO = {"bus", "ambulance"}
SLOT_NAME = {i: f"{ap_}-{tn}" for i, (ap_, tn) in enumerate((ap_, tn) for ap_ in ("N", "E", "S", "W") for tn in ("L", "T", "R"))}   # 槽序号 → "进口-转向" (与 serves 的键一致)
def slot_of(tid, vid, lane):
    r = sumo.vehicle.getRoute(vid); k = sumo.vehicle.getRouteIndex(vid); nxt = r[k + 1] if k + 1 < len(r) else None
    i = T[tid]["by_edges"].get((lane.rsplit("_", 1)[0], nxt))
    return None if i is None or i < 0 else SLOT_NAME[i]
st = {t: dict(starve_events=0, starve_max=0.0, red_since={}, spill_events=0, spill_since={}) for t in ids}
running = []; visits = {}   # (vid, tid) -> dict
done = {"__all__": False}
while not done["__all__"]:
    now = float(sumo.simulation.getTime())
    acts = {}
    for tid in ids:
        ts = env.traffic_signals[tid]; n = T[tid]["n_green"]
        if a.policy == "random": acts[tid] = rng.randrange(n)
        else:
            ph, t0 = fixed_state[tid]
            if now - t0 >= a.green + ts.yellow_time: ph = (ph + 1) % n; fixed_state[tid] = [ph, now]
            acts[tid] = ph
    # ---- 诊断 (在动作施加前, 按当前信号状态) ----
    for tid in ids:
        ts = env.traffic_signals[tid]; tt = T[tid]; cur = ts.green_phase; served = tt["serves"][cur]; s_ = st[tid]
        min_ok = ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time
        present = collections.Counter(); queued = collections.Counter(); prio_on = collections.defaultdict(bool); vehs = []
        for lane in tt["lanes"]:
            for v in sumo.lane.getLastStepVehicleIDs(lane):
                s = slot_of(tid, v, lane)
                if s is None: continue
                present[s] += 1; spd = sumo.vehicle.getSpeed(v)
                if spd < 0.1: queued[s] += 1
                typ = sumo.vehicle.getTypeID(v)
                if typ in PRIO: prio_on[s] = True
                if v.startswith("pair_"): vehs.append((v, s, spd, typ))
        # 饿死
        for s in list(set(queued) | set(s_["red_since"])):
            if queued.get(s, 0) > 0 and s not in served: s_["red_since"].setdefault(s, now)
            else:
                t0 = s_["red_since"].pop(s, None)
                if t0 is not None and now - t0 >= 60: s_["starve_events"] += 1; s_["starve_max"] = max(s_["starve_max"], now - t0)
        # 出口回溢
        for l, cap in tt["out_cap"].items():
            occ = sumo.lane.getLastStepVehicleNumber(l) / cap
            if occ >= 0.8: s_["spill_since"].setdefault(l, now)
            else:
                t0 = s_["spill_since"].pop(l, None)
                if t0 is not None and now - t0 >= 60: s_["spill_events"] += 1
        # 决策机会 (实验车)
        if vehs:
            legal = list(range(tt["n_green"])) if min_ok else [cur]
            for v, s, spd, typ in vehs:
                key = (v, tid); rec = visits.setdefault(key, dict(type=typ, slot=s, steps=0, opp=False, kinds=set(), states=set(), prio_comp=False))
                rec["steps"] += 1
                serve_me = [g for g in legal if s in tt["serves"][g]]; not_me = [g for g in legal if s not in tt["serves"][g]]
                if not serve_me or not_me == []: continue
                # 冲突且有需求的运动 m': 某不放行我的合法相位放行它, 且没有任何有效相位同时放行 s 与 m'
                conflict = False; prio_c = False
                for g in not_me:
                    for m2 in tt["serves"][g]:
                        if present.get(m2, 0) > 0 and not any((s in tt["serves"][h] and m2 in tt["serves"][h]) for h in range(tt["n_green"])):
                            conflict = True; prio_c = prio_c or prio_on[m2]
                if not conflict: continue
                rec["opp"] = True; rec["kinds"].add("green_hold" if s in served else "red_release"); rec["states"].add("queued" if spd < 0.1 else "approach"); rec["prio_comp"] = rec["prio_comp"] or prio_c
    running.append((now, sumo.vehicle.getIDCount()))
    if abs(now - 3300.0) < 2.5: pending_3300 = len(sumo.simulation.getPendingVehicles())
    states, r, done, _ = env.step(action=acts)
# 锁死 (本次访问等待 ≥ 150 s): 读统计器历史最大值需逐步记录; 这里用结束前累计的 W 最大 + 过程中记录
lock_n = 0; lock_by = collections.Counter()
if env.visit_tracker is not None:
    for (ts_id, vid, edge), w in env.visit_tracker.W.items():
        if w >= 150: lock_n += 1; lock_by[ts_id] += 1
env.close()
# tripinfo
tis = list(ET.parse(TRIP).getroot().iter("tripinfo")); os.remove(TRIP)
dep = [float(t.get("departDelay")) for t in tis if float(t.get("depart", -1)) >= 0]
arrived = sum(1 for t in tis if float(t.get("arrival", -1)) >= 0)
wait_max = [float(t.get("waitingTime")) for t in tis]
# 未插入: 路由文件里 depart ≤ 3300 的车辆 − tripinfo 里出现的
planned = 0; in_trip = {t.get("id") for t in tis}; not_ins = 0
for rf in a.routes.split(","):
    for v in ET.parse(os.path.join(scen, rf)).getroot().iter("vehicle"):
        if float(v.get("depart")) <= 3300: planned += 1; not_ins += (v.get("id") not in in_trip)
win = collections.defaultdict(list)
for t, n in running: win[int(t // 600)].append(n)
rw = [float(np.mean(win[k])) for k in sorted(win)]
opp = collections.defaultdict(lambda: dict(visits=0, with_opp=0, red_release=0, green_hold=0, approach=0, queued=0, prio_comp=0))
for (v, tid), rec in visits.items():
    o = opp[(tid, rec["slot"], rec["type"])]; o["visits"] += 1
    if rec["opp"]:
        o["with_opp"] += 1; o["red_release"] += "red_release" in rec["kinds"]; o["green_hold"] += "green_hold" in rec["kinds"]
        o["approach"] += "approach" in rec["states"]; o["queued"] += "queued" in rec["states"]; o["prio_comp"] += rec["prio_comp"]
by_type = collections.defaultdict(lambda: [0, 0])
for (tid, s, typ), o in opp.items(): by_type[typ][0] += o["visits"]; by_type[typ][1] += o["with_opp"]
out = dict(policy=a.policy, green=a.green, seed=a.seed, routes=a.routes, running_10min=rw, running_growth_last2_vs_34=(np.mean(rw[4:6]) / max(np.mean(rw[2:4]), 1) - 1) if len(rw) >= 6 else None,
           pending_3300=locals().get("pending_3300"), planned_le3300=planned, not_inserted_le3300=not_ins, tripinfo=len(tis), arrived=arrived,
           depart_delay_mean=float(np.mean(dep)) if dep else None, depart_delay_max=float(np.max(dep)) if dep else None, trip_wait_gt150=int(sum(1 for w in wait_max if w > 150)),
           lock_visits_ge150=lock_n, lock_by_junction=dict(lock_by), starve={t: dict(events=st[t]["starve_events"], max=st[t]["starve_max"]) for t in ids}, spill={t: st[t]["spill_events"] for t in ids},
           opportunities={f"{tid}|{s}|{typ}": o for (tid, s, typ), o in opp.items()}, opp_by_type={k: dict(visits=v[0], with_opp=v[1]) for k, v in by_type.items()})
json.dump(out, open(a.out + ".json", "w"), indent=1, ensure_ascii=False)
print(f"{a.policy} seed{a.seed}: 在网 10min 窗均值 {[round(x) for x in rw]} 末两窗/三四窗 {out['running_growth_last2_vs_34']:+.2f}; pending@3300 {out['pending_3300']}; 未插入(≤3300) {not_ins}/{planned}; 出发延迟 {out['depart_delay_mean']:.1f}/{out['depart_delay_max']:.0f}; 到达 {arrived}/{len(tis)}; 锁死访问 {lock_n}; 饿死事件 {sum(v['events'] for v in out['starve'].values())} (最长 {max(v['max'] for v in out['starve'].values()):.0f}); 回溢 {sum(out['spill'].values())}; 机会 {out['opp_by_type']}")
