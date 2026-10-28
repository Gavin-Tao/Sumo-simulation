"""任务 8: 标定工具 (scenario_check.py 第二版的扩展副本, 不改共享脚本)。三种与学习无关的控制器:
  fixed  : 每个相位保持 5 个决策步 (25 s: 黄 2 + 绿 23), 依次轮转;
  served : 排队驱动轮转 — 固定顺序, 跳过没有可放行车 (车道头车去向不属于该相位) 的相位; 绿灯最短 10 s, 有可放行车时以 5 s 为步延长, 最长 40 s;
  random : 每个决策步在有效相位里均匀随机 (环境保证最短绿)。
逐秒测量计划 §5 的 C1–C14 (硬判据与诊断指标)。环境参数取模板 exp461 (黄 2 s、决策 5 s、最短绿 5 s), sumocfg 用场景自己的 (不忽略路口阻塞, 开碰撞检查)。
用法: python calib18.py --policy fixed|served|random --seed 123 --routes a,b,c --out prefix [--sumocfg ...]"""
import argparse, os, sys, json, random, collections, functools, glob, math, xml.etree.ElementTree as ET
os.environ["OMP_NUM_THREADS"] = "1"; os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"; sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/experiments")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml, numpy as np
import common18 as C
ap = argparse.ArgumentParser()
ap.add_argument("--policy", choices=["fixed", "served", "random", "prio"], required=True); ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--routes", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sumocfg", default=str(C.OUT / "scenario.sumocfg"))
ap.add_argument("--net", default=str(C.NET)); ap.add_argument("--meta", default=str(C.META))   # 比较其他版本场景时用它自己的路网
# pairs50lf 任务 5 (2026-10-28): 以下参数都不给时, 行为与改动前逐位相同
ap.add_argument("--priority-table", default=None, help="531 / 351 / 111 或 JSON; 同时覆盖控制器、观测 (obs_priority_source) 与奖励 (priority_source)")
ap.add_argument("--manifest", default=None, help="实验车辆清单 JSON {id: {...}}: 按清单统计每类计划/插入/到达/截尾")
ap.add_argument("--end", type=int, default=3600, help="仿真结束时刻; 配对场景标定用 4500 排空")
ap.add_argument("--out-pairs", default=None, help="清单车辆逐车记录 (逐次访问、到达时冲突排队、优先车相遇、车道序列)")
ap.add_argument("--check-priority", action="store_true", help="逐步核对观测分桶与奖励权重是否与优先级表一致")
ap.add_argument("--template", default="461"); ap.add_argument("--lock-s", type=float, default=150.0)
a = ap.parse_args()
absr = lambda p: p if os.path.isabs(p) else str(C.OUT / p)
ROUTES = ",".join(absr(r) for r in a.routes.split(",")); a.out = os.path.abspath(a.out)
os.chdir(f"{REPO}/experiments")
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from exp_params import obs_kwargs_from_cfg, env_extra_kwargs, reward_wait_basis, turnmap_from_meta
cfg = yaml.safe_load(open(sorted(glob.glob(f"configs/exp{a.template}_*.yaml"))[0]))
TABLES = {"531": {"ambulance": 5, "bus": 3, "car": 1}, "351": {"ambulance": 3, "bus": 5, "car": 1}, "111": {"ambulance": 1, "bus": 1, "car": 1}}
if a.priority_table:
    _pt = json.loads(a.priority_table) if a.priority_table.strip().startswith("{") else TABLES[a.priority_table]
    cfg["priority_source"] = dict(_pt)
    if "obs_priority_source" in cfg: cfg["obs_priority_source"] = dict(_pt)
cfg["net_file"] = a.net; cfg["action_meta_file"] = a.meta; cfg["cfg_file"] = a.sumocfg; cfg["route_file"] = ROUTES
kw = obs_kwargs_from_cfg(cfg); obs_class = functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
PRIO_TABLE = load_priority_table(cfg.get("priority_source"))      # 控制器与奖励共用; 观测由 obs_kwargs_from_cfg 读同一配置
reward_fn = make_priority_avg_waiting_reward(PRIO_TABLE, wait_basis=reward_wait_basis(cfg))
tag = f"{os.getpid()}"; LOG = f"{REPO}/experiments/logs"
TRIP, LC, COL = f"{LOG}/c18_trip_{tag}.xml", f"{LOG}/c18_lc_{tag}.xml", f"{LOG}/c18_col_{tag}.xml"
env = SumoEnvironment(net_file=a.net, route_file=ROUTES, cfg_file=a.sumocfg, out_csv_name=None, use_gui=False, num_seconds=a.end, min_green=cfg.get("min_green", 5),
                      max_green=cfg.get("max_green", 50), use_max_green=cfg.get("use_max_green", False), single_agent=False, yellow_time=cfg.get("yellow_time", 2), delta_time=cfg.get("delta_time", 5),
                      reward_fn=reward_fn, observation_class=obs_class, sumo_seed=a.seed, sumo_warnings=False,
                      additional_sumo_cmd=f"--tripinfo-output {TRIP} --tripinfo-output.write-unfinished true --lanechange-output {LC} --collision-output {COL}" + (f" --end {a.end}" if a.end != 3600 else ""), **env_extra_kwargs(cfg))
env.reset(a.seed); sumo = env.sumo; ids = list(env.ts_ids); meta = json.load(open(a.meta))["tls"]
env_ag_lanes = {tid: list(env.traffic_signals[tid].always_green_lanes) for tid in ids}
for tid in ids: env.traffic_signals[tid].observation_fn.rebind_movements(turnmap_from_meta(meta[tid]))
name = lambda c: c["approach"] + "-" + c["turn"]
AG = C.always_green(meta)
T = {}; APPROACH = {}; EXITS = {}; RL_LANES = set(); UPSTREAM = {}
for tid in ids:
    ts = env.traffic_signals[tid]; t = meta[tid]
    link_slot = {int(i): name(g[0]) for i, g in t["links"].items()}
    slot_idx = collections.defaultdict(list)
    for i, s in link_slot.items(): slot_idx[s].append(i)
    serves = {g: {link_slot[i] for i in link_slot if ph.state[i] in "Gg"} for g, ph in enumerate(ts.green_phases)}
    by_edges = {}; out_lanes = {}; slot_lanes = collections.defaultdict(set)
    for i, g in t["links"].items():
        for c in g:
            by_edges[(c["from_edge"], c["to_edge"])] = link_slot[int(i)]
            out_lanes[c["to_edge"] + "_" + str(c["to_lane"])] = sumo.lane.getLength(c["to_edge"] + "_" + str(c["to_lane"]))
            slot_lanes[link_slot[int(i)]].add(c["from_edge"] + "_" + str(c["from_lane"]))
    jointly = {(m1, m2) for g in serves for m1 in serves[g] for m2 in serves[g]}
    lanes = list(dict.fromkeys(ts.lanes))
    T[tid] = dict(serves=serves, by_edges=by_edges, lanes=lanes, out_lanes=out_lanes, n=len(ts.green_phases), jointly=jointly, slot_idx=dict(slot_idx),
                  slot_cap={s: sum(sumo.lane.getLength(l) / 7.5 for l in ls) for s, ls in slot_lanes.items()})
    RL_LANES |= set(lanes); EXITS[tid] = out_lanes
    for l in lanes:
        L = sumo.lane.getLength(l); APPROACH[l] = dict(tid=tid, len=L, edge=l.rsplit("_", 1)[0])
        ups = set()
        for el in sumo.lane.getIDList() if False else []: pass
        UPSTREAM[l] = []
# 上游车道 (连接进入该进口车道的车道), 用于队尾越界判定
for lid in sumo.lane.getIDList():
    if lid.startswith(":"): continue
    for link in sumo.lane.getLinks(lid):
        to = link[0]
        if to in UPSTREAM: UPSTREAM[to].append(lid)
EXIT_LANES = {l for tid in ids for l in EXITS[tid]}
def slot_of(tid, vid, lane):
    r = sumo.vehicle.getRoute(vid); k = sumo.vehicle.getRouteIndex(vid); nxt = r[k + 1] if k + 1 < len(r) else None
    return T[tid]["by_edges"].get((lane.rsplit("_", 1)[0], nxt))
EXP = {"bus", "ambulance"}
MANIFEST = json.load(open(a.manifest)) if a.manifest else {}
MSET = set(MANIFEST)
m_lanes = collections.defaultdict(list)   # 清单车辆的车道序列 (不含路口内部车道, 相邻去重)
m_nonrl = collections.Counter()            # 清单车辆在非 RL 进口车道 (不含路口内部) 上的停车秒数
rng = random.Random(a.seed); step_no = 0
served_state = {t: dict(t0=0.0) for t in ids}
# ---- 状态
visits = {}; cur_visit = {}; visit_no = collections.Counter(); running = []
tracker = env.visit_tracker; tracker_missing = 0; teleports = 0
appr_st = {}
slot_st = {(tid, s): dict(green=False, q=0, snap=set(), green_since_dep=0.0, last_dep=0.0, blocked=False, prev_present=set(), qwas=0, clear_t=None, red_since=None, phase_changes_since_green=0)
           for tid in ids for s in T[tid]["slot_idx"]}
seen_services = collections.Counter()  # vehicle -> green ends experienced while in snapshot
M = collections.defaultdict(float); M_list = collections.defaultdict(list); C4_events = collections.Counter(); green_ends = collections.Counter(); missed = collections.Counter()
residual = collections.Counter(); clears = collections.Counter(); refill = collections.defaultdict(list); blocked_ev = collections.Counter()
lane_q = collections.Counter(); lane_hi = collections.Counter(); plateau_secs = 0; overflow = collections.Counter(); spill_since = {}; spill_ev = collections.Counter(); spill_long = collections.Counter()
nonrl_stop = collections.Counter(); rl_stop = 0.0; exit_stop = 0.0; appr_q_win = collections.defaultdict(lambda: collections.defaultdict(list)); qstep = collections.Counter(); dsteps = collections.Counter()
phase_count = collections.Counter(); last_phase = {t: None for t in ids}
LANE_TO_NODE = {}
def node_of(lane):
    if lane not in LANE_TO_NODE:
        e = lane.rsplit("_", 1)[0]
        try: LANE_TO_NODE[lane] = sumo.lane.getEdgeID(lane) and env.sumo.edge.getToJunction(e) if hasattr(env.sumo.edge, "getToJunction") else e
        except Exception: LANE_TO_NODE[lane] = e
    return LANE_TO_NODE[lane]
# C17: 中段插车审查 — 每秒记录新出发车辆 (车道、时刻); 优先车辆 (公交/救护车) 进入每条车道的时刻; 优先车辆在非 RL 车道上的停车
c17_dep = collections.defaultdict(list)          # lane -> [(t, vid, type)]
c17_prio_enter = []                               # (vid, type, lane, t)
c17_prio_lane = {}
c17_prio_stop = collections.Counter()             # (type, lane) -> 停车秒 (非 RL 进口车道)
def per_second_c17(now):
    for v in sumo.simulation.getDepartedIDList():
        l = sumo.vehicle.getLaneID(v)
        if l: c17_dep[l].append((now, v, sumo.vehicle.getTypeID(v)))
    for v in sumo.vehicle.getIDList():
        ty = sumo.vehicle.getTypeID(v)
        if ty not in EXP: continue
        l = sumo.vehicle.getLaneID(v)
        if not l or l.startswith(":"): continue
        if c17_prio_lane.get(v) != l: c17_prio_enter.append((v, ty, l, now, v not in c17_prio_lane)); c17_prio_lane[v] = l
        if sumo.vehicle.getSpeed(v) < 0.1 and l not in RL_LANES: c17_prio_stop[(ty, l)] += 1

def per_second():
    global tracker_missing, teleports, rl_stop, exit_stop, plateau_secs
    now = float(sumo.simulation.getTime()); teleports += sumo.simulation.getStartingTeleportNumber(); plateau = C.PLATEAU[0] <= now < C.PLATEAU[1]
    per_second_c17(now)
    if plateau: plateau_secs += 1
    present_keys = set()
    for tid in ids:
        st = sumo.trafficlight.getRedYellowGreenState(tid); tt = T[tid]
        slot_vehs = collections.defaultdict(set); slot_halt = collections.defaultdict(set); new_mv = []
        for lane in tt["lanes"]:
            vs = sumo.lane.getLastStepVehicleIDs(lane); halted = []
            for v in vs:
                key = (v, tid); present_keys.add(key); s = slot_of(tid, v, lane)
                if key not in cur_visit:
                    visit_no[key] += 1; cur_visit[key] = visit_no[key]
                    visits[(v, tid, cur_visit[key])] = dict(type=sumo.vehicle.getTypeID(v), t0=now, slot=s, max_wait=0.0, opp=False, kinds=set(), exp=sumo.vehicle.getTypeID(v) in EXP, lane=lane)
                    if v in MSET: new_mv.append((v, s, visits[(v, tid, cur_visit[key])]))
                rec = visits[(v, tid, cur_visit[key])]; rec["t_last"] = now
                w = tracker.get(tid, v, lane) if tracker is not None else 0.0
                if w > rec["max_wait"]: rec["max_wait"] = w
                if s is not None:
                    slot_vehs[s].add(v)
                    if sumo.vehicle.getSpeed(v) < 0.1: slot_halt[s].add(v); halted.append(sumo.vehicle.getLanePosition(v))
            L = APPROACH[lane]["len"]
            if plateau:
                if halted: lane_q[lane] += 1
                if len(halted) * 7.5 > 0.6 * L: lane_hi[lane] += 1
            if halted and min(halted) <= 7.5:     # 队尾到进口起点: 看上游车道末端是否也有停车
                for up in UPSTREAM.get(lane, []):
                    ul = sumo.lane.getLength(up)
                    if any(sumo.vehicle.getSpeed(u) < 0.1 and sumo.vehicle.getLanePosition(u) >= ul - 15 for u in sumo.lane.getLastStepVehicleIDs(up)):
                        overflow[lane] += 1; break
            if plateau: appr_q_win[lane][int((now - C.PLATEAU[0]) // 600)].append(len(halted))
        for v_, s_, rec_ in new_mv:     # 清单车辆到达本路口时: 冲突方向 (不能与它同一相位放行) 的停车数与在场的其他清单车辆
            if s_ is None: rec_["confq"] = None; rec_["enc"] = []; continue
            conf = [s2 for s2 in slot_vehs if (s_, s2) not in tt["jointly"]]
            rec_["confq"] = sum(len(slot_halt[s2]) for s2 in conf)
            rec_["enc"] = sorted(sumo.vehicle.getTypeID(u) for s2 in conf for u in slot_vehs[s2] if u in MSET and u != v_)
        # 真正堵死 (C3): 车道头车自己的转向是绿灯却停着, 累计 ≥30 s, 且该车道 120 s 无车离开
        for lane in tt["lanes"]:
            vs = list(sumo.lane.getLastStepVehicleIDs(lane)); a_ = appr_st.setdefault((tid, lane), dict(prev=set(), last_dep=0.0, gsd=0.0, blocked=False))
            vset = set(vs)
            if a_["prev"] - vset: a_["last_dep"] = now; a_["gsd"] = 0.0; a_["blocked"] = False
            if vs:
                hv = max(vs, key=lambda v: sumo.vehicle.getLanePosition(v)); hs = slot_of(tid, hv, lane)
                if hs is not None and sumo.vehicle.getSpeed(hv) < 0.1 and any(st[i] in "Gg" for i in tt["slot_idx"].get(hs, [])):
                    a_["gsd"] += 1.0
                    if a_["gsd"] >= 30 and now - a_["last_dep"] >= 120 and not a_["blocked"]: a_["blocked"] = True; blocked_ev[(tid, lane)] += 1
            a_["prev"] = vset
        for s in tt["slot_idx"]:
            ss = slot_st[(tid, s)]; g = any(st[i] in "Gg" for i in tt["slot_idx"][s]); q = len(slot_halt[s]); pres = slot_vehs[s]
            if g and not ss["green"]:          # 绿灯开始: 快照排队车
                ss["snap"] = set(slot_halt[s]); ss["phase_changes_since_green"] = 0
            if ss["green"] and not g:          # 绿灯结束
                green_ends[(tid, s)] += 1
                rem = ss["snap"] & pres
                if rem: residual[(tid, s)] += 1
                hit = False
                for v in rem:
                    seen_services[(v, tid)] += 1
                    if seen_services[(v, tid)] == 3: hit = True
                if hit: C4_events[(tid, s)] += 1
            if g and ss["qwas"] > 0 and q == 0: clears[(tid, s)] += 1; ss["clear_t"] = now
            if ss["clear_t"] is not None and q >= max(4.0, 0.6 * tt["slot_cap"].get(s, 1.0)):
                refill[(tid, s)].append(now - ss["clear_t"]); ss["clear_t"] = None
            ss["green"] = g; ss["qwas"] = q; ss["prev_present"] = pres
        for l, Lout in tt["out_lanes"].items():
            hp = [sumo.vehicle.getLanePosition(v) for v in sumo.lane.getLastStepVehicleIDs(l) if sumo.vehicle.getSpeed(v) < 0.1]
            if hp and min(hp) <= 10.0: spill_since.setdefault(l, now)
            else:
                t0 = spill_since.pop(l, None)
                if t0 is not None and now - t0 >= 60: spill_ev[l] += 1
                if t0 is not None and now - t0 >= 300: spill_long[l] += 1
    for key in list(cur_visit):
        if key not in present_keys: del cur_visit[key]
    for v in sumo.vehicle.getIDList():
        if sumo.vehicle.getSpeed(v) >= 0.1: continue
        l = sumo.vehicle.getLaneID(v)
        if not l or l.startswith(":"): continue
        if l in RL_LANES: rl_stop += 1
        else:
            e = l.rsplit("_", 1)[0]; nonrl_stop[e] += 1
            if l in EXIT_LANES: exit_stop += 1
    running.append((now, sumo.vehicle.getIDCount()))
    if MSET:
        for v in sumo.vehicle.getIDList():
            if v not in MSET: continue
            l = sumo.vehicle.getLaneID(v)
            if l and not l.startswith(":") and (not m_lanes[v] or m_lanes[v][-1] != l): m_lanes[v].append(l)
            if l and not l.startswith(":") and l not in RL_LANES and sumo.vehicle.getSpeed(v) < 0.1: m_nonrl[v] += 1
_orig = env._sumo_step
def _hook(*x, **k):
    r = _orig(*x, **k); per_second(); return r
env._sumo_step = _hook
PRIO_W = {t: float(p) for t, p in PRIO_TABLE.items()}       # 车型 → 档位 (与奖励同一张表); 531 下即 5/3/1
_lv = [PRIO_W.get(t, 1.0) for t in ("car", "bus", "ambulance")]
PRIO_TOP = max(_lv); PRIO_PREEMPT = max(_lv) > min(_lv)      # 抢先切灯只给表里最高档位的车; 各档相同 (111) 时不抢先
def prio_action(tid, cur, now):
    """知道车型的启发式 (只用于检验场景是否允许 救护车 < 公交 < 小汽车, 不是 RL): 相位得分 = 该相位能放行的车按 5/3/1 加权
    (车道上的车要它和头车的转向都为绿才算); 有救护车时满最短绿 (7 s) 即切到能放行它的相位并保持; 否则满 10 s 后切到得分更高的相位, 40 s 强制换到其他有车的相位。"""
    tt = T[tid]; n = tt["n"]; el = now - served_state[tid]["t0"]; score = [0.0] * n; amb_need = []
    for lane in tt["lanes"]:
        vs_ = sorted(sumo.lane.getLastStepVehicleIDs(lane), key=lambda v: -sumo.vehicle.getLanePosition(v))
        if not vs_: continue
        hs = slot_of(tid, vs_[0], lane)
        for v in vs_:
            s = slot_of(tid, v, lane); ty = sumo.vehicle.getTypeID(v)
            if s is None or hs is None: continue
            for g in range(n):
                if hs in tt["serves"][g] and s in tt["serves"][g]: score[g] += PRIO_W.get(ty, 1.0)
            if PRIO_PREEMPT and PRIO_W.get(ty, 1.0) == PRIO_TOP: amb_need.append((sumo.vehicle.getLanePosition(v) - sumo.lane.getLength(lane), {hs, s}))
    act = cur
    if amb_need:
        need = max(amb_need, key=lambda x: x[0])[1]
        ok = [g for g in range(n) if need <= tt["serves"][g]] or [g for g in range(n) if (need & tt["serves"][g])]
        if cur not in ok and el >= 7 and ok: act = max(ok, key=lambda g: score[g])
    elif el >= 10:
        best = max(range(n), key=lambda g: score[g])
        if score[best] > score[cur]: act = best
        elif el >= 40:
            others = [g for g in range(n) if g != cur and score[g] > 0]
            if others: act = max(others, key=lambda g: score[g])
    if act != cur: served_state[tid]["t0"] = now
    return act

def phase_demand(tid, g, heads):
    """该相位能放行的车: 有车道的头车去向属于该相位服务的转向。"""
    return any(h in T[tid]["serves"][g] for h in heads)
from sumo_rl.environment.priority_map import DEFAULT_PRIORITY
pchk = dict(obs_checked=0, obs_mismatch=0, reward_checked=0, reward_max_abs_diff=0.0, examples=[])
def check_priority_step(rew):
    """观测: 每辆进口车道上的车, 观测函数给的档位 == 表里它车型的档位; 奖励: 按 fn_visit 公式与同一张表独立重算, 与环境返回值比较。"""
    for tid in ids:
        ts = env.traffic_signals[tid]; ofn = ts.observation_fn; totals = {}; counts = {}
        for lane in ts.lanes:
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                want = PRIO_TABLE.get(sumo.vehicle.getTypeID(vid), DEFAULT_PRIORITY)
                if hasattr(ofn, "_vehicle_priority"):
                    pchk["obs_checked"] += 1
                    if ofn._vehicle_priority(vid) != want:
                        pchk["obs_mismatch"] += 1
                        if len(pchk["examples"]) < 5: pchk["examples"].append((tid, vid, sumo.vehicle.getTypeID(vid), ofn._vehicle_priority(vid), want))
                w = float(want); lw = tracker.get(tid, vid, lane)
                totals[w] = totals.get(w, 0.0) + lw; counts[w] = counts.get(w, 0) + 1
        if tid in rew:
            exp_r = -sum(w * (t_ / counts[w]) for w, t_ in totals.items()) / 100.0
            pchk["reward_checked"] += 1; pchk["reward_max_abs_diff"] = max(pchk["reward_max_abs_diff"], abs(exp_r - rew[tid]))
done = {"__all__": False}
while not done["__all__"]:
    now = float(sumo.simulation.getTime()); acts = {}
    for tid in ids:
        ts = env.traffic_signals[tid]; tt = T[tid]; n = tt["n"]; cur = ts.green_phase
        pres = collections.Counter(); halt = collections.Counter(); exp_vehs = []; prio_on = collections.defaultdict(bool); heads = set()
        for lane in tt["lanes"]:
            vs_ = sumo.lane.getLastStepVehicleIDs(lane)
            if vs_:
                hv = max(vs_, key=lambda v: sumo.vehicle.getLanePosition(v)); hs = slot_of(tid, hv, lane)
                if hs is not None: heads.add(hs)
            for v in vs_:
                s = slot_of(tid, v, lane)
                if s is None: continue
                pres[s] += 1; spd = sumo.vehicle.getSpeed(v)
                if spd < 0.1: halt[s] += 1
                ty = sumo.vehicle.getTypeID(v)
                if ty in EXP: exp_vehs.append((v, s, spd)); prio_on[s] = True
        dsteps[tid] += 1
        if any(halt.values()): qstep[tid] += 1
        if a.policy == "random": act = rng.randrange(n)
        elif a.policy == "fixed": act = (step_no // 5) % n
        elif a.policy == "prio": act = prio_action(tid, cur, now)
        else:
            el = now - served_state[tid]["t0"]; act = cur
            if el >= 10 and not (el < 40 and phase_demand(tid, cur, heads)):
                for k in range(1, n + 1):
                    q = (cur + k) % n
                    if q != cur and phase_demand(tid, q, heads): act = q; break
            if act != cur: served_state[tid]["t0"] = now
        acts[tid] = act
        if last_phase[tid] is not None and act != last_phase[tid]:
            for s in tt["slot_idx"]:
                ss = slot_st[(tid, s)]
                if not ss["green"] and ss["qwas"] > 0:
                    ss["phase_changes_since_green"] += 1
                    if ss["phase_changes_since_green"] == 2 * n: missed[(tid, s)] += 1
        last_phase[tid] = act
        if exp_vehs:   # 优先车辆的放行取舍机会 (pairs50 设计 §4 定义)
            min_ok = ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time
            legal = list(range(n)) if min_ok else [cur]
            for v, s, spd in exp_vehs:
                rec = visits.get((v, tid, cur_visit.get((v, tid), 0)))
                if rec is None: continue
                me = [g for g in legal if s in tt["serves"][g]]; notme = [g for g in legal if s not in tt["serves"][g]]
                if not me or not notme: continue
                if any(pres.get(m2, 0) > 0 and (s, m2) not in tt["jointly"] for g in notme for m2 in tt["serves"][g]):
                    rec["opp"] = True; rec["kinds"].add("green_hold" if s in tt["serves"][cur] else "red_release")
    step_no += 1
    _, rew_, done, _ = env.step(action=acts)
    if a.check_priority: check_priority_step(rew_)
end = float(sumo.simulation.getTime())
tracker_missing = int(getattr(tracker, "missing", 0)) if tracker is not None else 0   # 统计器缺条目时返回 0 并计数, 读内部计数
for l, t0 in spill_since.items():
    if end - t0 >= 60: spill_ev[l] += 1
    if end - t0 >= 300: spill_long[l] += 1
env.close()
# ---- 汇总
tis = {t.get("id"): t for t in ET.parse(TRIP).getroot().iter("tripinfo")}
n_lc = sum(1 for _ in ET.parse(LC).getroot().iter("change")) if os.path.exists(LC) else None
n_col = sum(1 for _ in ET.parse(COL).getroot().iter("collision")) if os.path.exists(COL) else None
for f in (TRIP, LC, COL):
    if os.path.exists(f): os.remove(f)
req = collections.defaultdict(lambda: collections.Counter()); dd = collections.defaultdict(list); unfinished = collections.Counter()
for rf in ROUTES.split(","):
    for v in ET.parse(rf).getroot().iter("vehicle"):
        entry = v.find("route").get("edges").split()[0]; ty = v.get("type"); k = (entry, ty)
        if float(v.get("depart")) > 3300: continue
        req[k]["requested"] += 1
        t = tis.get(v.get("id"))
        if t is None or float(t.get("depart", -1)) < 0: req[k]["not_inserted"] += 1; continue
        req[k]["inserted"] += 1; dd[k].append(float(t.get("departDelay")))
        if float(t.get("arrival", -1)) < 0: req[k]["unfinished"] += 1
C2 = {f"{e}|{ty}": dict(c, mean_delay=round(float(np.mean(dd[(e, ty)])), 2) if dd[(e, ty)] else None, p99_delay=round(float(np.percentile(dd[(e, ty)], 99)), 1) if dd[(e, ty)] else None) for (e, ty), c in req.items()}
all_dd = [x for v in dd.values() for x in v]; tot_req = sum(c["requested"] for c in req.values()); tot_ni = sum(c["not_inserted"] for c in req.values())
win = collections.defaultdict(list)
for t, n in running:
    if C.PLATEAU[0] <= t < C.PLATEAU[1]: win[int((t - C.PLATEAU[0]) // 600)].append(n)
rw = [float(np.mean(win[k])) for k in sorted(win)]
c14 = {}
for e, w in appr_q_win.items():
    ks = sorted(w)
    if len(ks) >= 3:
        first, last = np.mean(w[ks[0]]), np.mean(w[ks[-2]])  # 按单条车道: 第一个窗 (900–1500) 对最后一个完整窗 (2100–2700)
        c14[e] = dict(first=round(float(first), 2), last=round(float(last), 2), grow=round(float(last - first), 2), tid=APPROACH[e]["tid"], edge=APPROACH[e]["edge"])
long_wait = sum(1 for r in visits.values() if r["max_wait"] >= a.lock_s); early_long = sum(1 for r in visits.values() if r["t0"] < 900 and r["max_wait"] >= 90)
ROT = {t: T[t]["n"] * 42.0 for t in ids}     # 一个服务轮转周期: 有效相位数 × (最长绿 40 + 黄 2)
early_rot = sum(1 for (v, tid, k), r in visits.items() if r["t0"] < 900 and r["max_wait"] >= ROT[tid])
r_early = [n for t, n in running if t < 900]; r_plat = [n for t, n in running if C.PLATEAU[0] <= t < C.PLATEAU[1]]
opp = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in visits.values():
    if r["exp"] and r["slot"] is not None and (r["slot"],) and True:
        o = opp[r["type"]]; o[0] += 1; o[1] += r["opp"]; o[2] += "red_release" in r["kinds"]; o[3] += "green_hold" in r["kinds"]
comp_opp = collections.defaultdict(lambda: [0, 0])
for (v, tid, k), r in visits.items():
    if r["exp"] and r["slot"] is not None and (tid, r["slot"]) not in AG:
        comp_opp[r["type"]][0] += 1; comp_opp[r["type"]][1] += r["opp"]
long_lanes = [l for l in APPROACH if APPROACH[l]["len"] >= 60]
# C17 汇总: 插在优先车辆前面 = 某车在车道 L 出发后 15 s 内, 有公交/救护车从上游驶入 L (不是在 L 上出发的那辆自己)
cutins = []
for vid, ty, l, t, first in c17_prio_enter:
    if first: continue   # 优先车辆自己出发的车道 (边界排队), 不是中段插队
    for (td, vd, tyd) in c17_dep.get(l, []):
        if vd != vid and 0 <= t - td <= 15: cutins.append(dict(prio=vid, type=ty, lane=l, t_enter=t, inserted=vd, inserted_type=tyd, dt=round(t - td, 1)))
stop_by_type = collections.defaultdict(float)
for (ty, l), x in c17_prio_stop.items(): stop_by_type[ty] += x
C17 = dict(cutins=cutins, n_cutins={ty: sum(1 for c in cutins if c["type"] == ty) for ty in EXP},
           prio_nonrl_stop_s={k: v for k, v in stop_by_type.items()}, prio_nonrl_stop_top={f"{ty}|{l}": x for (ty, l), x in c17_prio_stop.most_common(15)})
# C16: 与训练/wandb 完全同口径 — 按车道剔除恒绿车道 (env.traffic_signals[tid].always_green_lanes), 不按转向
AGL = {tid: set(env_ag_lanes[tid]) for tid in ids}
c16_t = collections.defaultdict(lambda: [0, 0.0]); c16_m = collections.defaultdict(lambda: [0, 0.0])
for (v, tid, k), r in visits.items():
    if r.get("lane") in AGL[tid]: continue
    a_ = c16_t[r["type"]]; a_[0] += 1; a_[1] += r["max_wait"]
    if r["slot"] is not None: m_ = c16_m[f"{tid}|{r['slot']}|{r['type']}"]; m_[0] += 1; m_[1] += r["max_wait"]
# C15: 每次访问的等待 (按车型; 恒绿转向单列, 与 wandb per-visit 一致不含恒绿), 逐路口、逐转向; 优先车辆逐路口机会
c15_t = collections.defaultdict(lambda: [0, 0.0, 0]); c15_j = collections.defaultdict(lambda: [0, 0.0]); c15_m = collections.defaultdict(lambda: [0, 0.0]); c15_o = collections.defaultdict(lambda: [0, 0])
for (v, tid, k), r in visits.items():
    if r["slot"] is None: continue
    agv = (tid, r["slot"]) in AG; key = f"{r['type']}|{'ag' if agv else 'ctrl'}"
    a_ = c15_t[key]; a_[0] += 1; a_[1] += r["max_wait"]; a_[2] += r["max_wait"] <= 0.0
    if agv: continue
    b_ = c15_j[f"{tid}|{r['type']}"]; b_[0] += 1; b_[1] += r["max_wait"]
    m_ = c15_m[f"{tid}|{r['slot']}|{r['type']}"]; m_[0] += 1; m_[1] += r["max_wait"]
    if r["exp"]: o_ = c15_o[f"{tid}|{r['type']}"]; o_[0] += 1; o_[1] += r["opp"]
out = dict(policy=a.policy, seed=a.seed, routes=ROUTES,
    C1=dict(plateau_windows=rw, growth=(rw[-1] / rw[0] - 1) if len(rw) >= 2 and rw[0] > 0 else None,
            note="计划原口径: 平台期最后一个窗 (2700–3000 s) 相对第一个窗 (900–1500 s)"),
    C2=dict(requested=tot_req, not_inserted=tot_ni, mean_delay=float(np.mean(all_dd)) if all_dd else None, p99_delay=float(np.percentile(all_dd, 99)) if all_dd else None, by_entry_type=C2),
    C3=dict(blocked_events=sum(blocked_ev.values()), blocked=dict((f"{k[0]}|{k[1]}", v) for k, v in blocked_ev.items()), exit_full_300s=dict(spill_long)),
    C3p=dict(long_wait_visits=long_wait, visits=len(visits), share=long_wait / max(1, len(visits))),
    C4=dict(missed=sum(missed.values()), missed_by={f"{k[0]}|{k[1]}": v for k, v in missed.items()}, old_queue_events=sum(C4_events.values()), green_ends=sum(green_ends.values()),
            old_queue_share=sum(C4_events.values()) / max(1, sum(green_ends.values())),
            by_slot={f"{k[0]}|{k[1]}": dict(events=C4_events[k], green_ends=green_ends[k]) for k in green_ends}, worst={f"{k[0]}|{k[1]}": round(C4_events[k] / green_ends[k], 3) for k in sorted(C4_events, key=lambda k: -C4_events[k] / max(1, green_ends[k]))[:8]}),
    C5=dict(by_slot={f"{k[0]}|{k[1]}": dict(clears=clears[k], refill_n=len(refill[k]), refill_lt60=sum(1 for x in refill[k] if x < 60)) for k in clears},
            residual_share=sum(residual.values()) / max(1, sum(green_ends.values())), clears=sum(clears.values()), refill_lt60_share=(sum(1 for v in refill.values() for x in v if x < 60) / max(1, sum(len(v) for v in refill.values())))),
    C6=dict(spill_events=sum(spill_ev.values()), by_lane=dict(spill_ev)),
    C7=dict(long_lanes={l: round(overflow[l] / 3600, 4) for l in long_lanes if overflow[l]}, short_lanes={l: round(overflow[l] / 3600, 4) for l in APPROACH if APPROACH[l]["len"] < 60 and overflow[l]}),
    C8=dict(nonrl_stop_veh_s=sum(nonrl_stop.values()), rl_stop_veh_s=rl_stop, share=sum(nonrl_stop.values()) / max(1.0, rl_stop), top_edges=dict(nonrl_stop.most_common(12))),
    C9=dict(lane_changes=n_lc, collisions=n_col, teleports=teleports, tracker_missing=tracker_missing,
            amb_arrived=sum(1 for i, t in tis.items() if i.startswith("amb") and float(t.get("arrival", -1)) >= 0), amb_total=sum(1 for i in tis if i.startswith("amb"))),
    C10={t: round(qstep[t] / max(1, dsteps[t]), 3) for t in ids},
    C11={ty: dict(visits=o[0], with_opp=o[1], red_release=o[2], green_hold=o[3]) for ty, o in opp.items()} | {f"{ty}_competing": dict(visits=v[0], with_opp=v[1]) for ty, v in comp_opp.items()},
    C12={l: dict(q_share=round(lane_q[l] / max(1, plateau_secs), 3), hi_share=round(lane_hi[l] / max(1, plateau_secs), 3)) for l in APPROACH},
    C13=dict(early_wait_over_rotation=early_rot, early_long_wait_90=early_long, early_list=[(tid, r["slot"], r["type"], round(r["t0"]), round(r["max_wait"])) for (v, tid, k), r in visits.items() if r["t0"] < 900 and r["max_wait"] >= 90], early_vs_plateau=(float(np.mean(r_early)) / max(1.0, float(np.mean(r_plat)))) if r_plat else None),
    C14=c14, C15=dict(by_type_ctrl={k: v for k, v in c15_t.items()}, by_tid_type={k: v for k, v in c15_j.items()}, by_mv_type={k: v for k, v in c15_m.items()}, opp_by_tid_type={k: v for k, v in c15_o.items()}), C17=C17, C16=dict(by_type={k: v for k, v in c16_t.items()}, by_mv_type={k: v for k, v in c16_m.items()}, always_green_lanes=env_ag_lanes))
if a.check_priority:
    out["PRIO_CHECK"] = dict(pchk, expected_table={t: PRIO_TABLE.get(t) for t in ("car", "bus", "ambulance")}, ctrl_table={t: PRIO_W.get(t) for t in ("car", "bus", "ambulance")},
                             obs_class=type(env.traffic_signals[ids[0]].observation_fn).__name__)
if MSET:      # 清单车辆: 每类 计划/插入/到达/截尾 (截尾 = 已插入但到 --end 仍未到达); 逐车记录
    rf_type = {}
    for rf in ROUTES.split(","):
        for v in ET.parse(rf).getroot().iter("vehicle"):
            if v.get("id") in MSET: rf_type[v.get("id")] = v.get("type")
    by_v = collections.defaultdict(list)
    for (v, tid, k), r in visits.items():
        if v in MSET: by_v[v].append(dict(tid=tid, k=k, slot=r["slot"], t0=r["t0"], t_last=r.get("t_last", r["t0"]), wait=r["max_wait"], lane=r.get("lane"), ag_lane=r.get("lane") in AGL[tid],
                                          opp=r["opp"], kinds=sorted(r["kinds"]), confq=r.get("confq"), enc=r.get("enc", [])))
    cnt = collections.defaultdict(collections.Counter); recs = {}
    for vid, ty in rf_type.items():
        t = tis.get(vid); ins = t is not None and float(t.get("depart", -1)) >= 0; arr = ins and float(t.get("arrival", -1)) >= 0
        c = cnt[ty]; c["planned"] += 1; c["inserted"] += int(ins); c["arrived"] += int(arr); c["censored"] += int(ins and not arr); c["not_inserted"] += int(not ins)
        recs[vid] = dict(type=ty, depart=float(t.get("depart")) if ins else None, arrival=float(t.get("arrival")) if arr else None,
                         depart_delay=float(t.get("departDelay")) if ins else None, nonrl_stop_s=m_nonrl.get(vid, 0),
                         visits=sorted(by_v.get(vid, []), key=lambda x: x["t0"]), lanes=m_lanes.get(vid, []))
    out["PAIRS"] = dict(counts={ty: dict(c) for ty, c in cnt.items()}, end=a.end)
    if a.out_pairs: json.dump(recs, open(a.out_pairs, "w"), ensure_ascii=False, default=float)
json.dump(out, open(a.out + ".json", "w"), indent=1, ensure_ascii=False, default=float)
g = lambda x: "—" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
print(f"{a.policy} s{a.seed}: C1 {g(out['C1']['growth'])} | C2 未插入 {tot_ni}/{tot_req} 延迟 {g(out['C2']['mean_delay'])}/{g(out['C2']['p99_delay'])} | C3 堵死 {out['C3']['blocked_events']} 长满 {sum(spill_long.values())} | C3' {long_wait} | "
      f"C4 漏放 {out['C4']['missed']} 旧队列 {out['C4']['old_queue_share']:.3f} | C6 回溢 {out['C6']['spill_events']} | C7 长进口越界>5% {sum(1 for v in out['C7']['long_lanes'].values() if v > 0.05)} | C8 {out['C8']['share']:.2f} | "
      f"C9 换道 {n_lc} 碰撞 {n_col} 瞬移 {teleports} 缺 {tracker_missing} 救护车 {out['C9']['amb_arrived']}/{out['C9']['amb_total']} | C13 超一轮 {early_rot} (≥90 s {early_long}) {g(out['C13']['early_vs_plateau'])}")
