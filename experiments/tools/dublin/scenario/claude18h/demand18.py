"""任务 4 + 7: 逐转向需求目标 (平台期流率, 辆/h) 与小汽车背景车流 (线性规划拟合路线流率 → 整集车数 → 按入口统一排时刻 → 三档嵌套)。
用法: python demand18.py [--scale-file scales.json]   (scales: {tls: s_j}, 缺省 1.0)"""
import argparse, collections, itertools, json, math, random
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, hstack, eye, vstack, csr_matrix
import common18 as C

def conflict_pairs(meta):
    """同一路口里不能在任何有效相位同时放行的转向对 (不同进口)。"""
    out = {}
    for tid, t in meta.items():
        links = {int(i): g for i, g in t["links"].items()}
        mv = collections.defaultdict(set)
        for i, g in links.items(): mv[g[0]["approach"] + "-" + g[0]["turn"]].add(i)
        greens = [x["state"] for x in t["actions"].values() if x.get("valid")]
        served = {m: [any(s[i] == "G" for i in idx) for s in greens] for m, idx in mv.items()}
        pairs = []
        for a, b in itertools.combinations(sorted(mv), 2):
            if a.split("-")[0] == b.split("-")[0]: continue
            if not any(x and y for x, y in zip(served[a], served[b])): pairs.append((a, b))
        out[tid] = pairs
    return out

DESIGN = [(("9101555", "N-T"), 200.0, "用户要求: 9101555 北进口加车"),
          (("9101555", "N-R"), 60.0, "用户要求: 9101555 北进口加车"),
          (("9101555", "W-R"), 80.0, "Digges 支路与北进口形成主支竞争"),
          (("cluster_5166592137_5166592140_557198554_57744042", "N-R"), 40.0, "用户要求: 不留零需求; 例外支路入口 7849022#0")]
DESIGN_KEYS = {k for k, _, _ in DESIGN}

def targets(meta, anchors, buses, amb, zero, scales):
    ag = C.always_green(meta); prior = anchors["prior_old18h"]; A = anchors["anchors"]
    bus_mv = collections.Counter(); amb_mv = collections.Counter()
    for b in buses["buses"]:
        for t, m in b["traversals"]: bus_mv[(t, m)] += 1
    for t, m in amb["route"]["traversals"]: amb_mv[(t, m)] += C.AMB_PER_EP
    movs = collections.defaultdict(list)
    for tid, t in meta.items():
        for g in t["links"].values():
            m = g[0]["approach"] + "-" + g[0]["turn"]
            if m not in movs[tid]: movs[tid].append(m)
    T = {}; ref = {}; notes = {}; topup = []
    for tid in meta:
        p = {m: prior.get(f"{tid}|{m}", {"car": 0, "bus": 0}) for m in movs[tid]}
        pc = {m: v["car"] + v["bus"] for m, v in p.items()}
        trusted = A[tid]["trusted"]
        R = A[tid]["total_18h"] if trusted else sum(pc.values())
        ref[tid] = dict(R=R, source="SCATS 18 点" if trusted else "旧 18h 路由过境量 (无可信站点)", trusted=trusted, scale=scales.get(tid, 1.0))
        if tid == "659799":     # 新补的东、北臂没有先验: SCATS 余量按车道数平分到这两臂的转向
            lanes = collections.Counter()
            for g in meta[tid]["links"].values():
                for c in g:
                    if c["approach"] in ("E", "N"): lanes[(c["approach"], c["from_lane"])] += 0
            per_lane = {}
            for g in meta[tid]["links"].values():
                for c in g:
                    if c["approach"] in ("E", "N"): per_lane.setdefault((c["approach"], c["from_lane"]), set()).add(c["approach"] + "-" + c["turn"])
            rest = max(0.0, R - sum(pc.values())); share = rest / max(1, len(per_lane))
            for k, ms in per_lane.items():
                for m in ms: pc[m] += share / len(ms)
            notes[tid] = f"东、北臂为新补道路, 无旧路由先验; SCATS 余量 {rest:.0f} 辆/h 按 {len(per_lane)} 条车道平分"
        tot = sum(pc.values())
        for m in movs[tid]:
            share = pc[m] / tot if tot > 0 else 0.0
            T[(tid, m)] = 0.0 if (tid, m) in zero else ref[tid]["scale"] * R * share
    # 竞争补量: 路口内没有一对互斥的非恒绿转向都 ≥30 辆/h 时, 给较小一侧补到 min(80, 10% 总量) (总量 <300 的无可信站点路口按设计值 80)
    CP = conflict_pairs(meta)
    for tid in meta:
        pairs = [(a, b) for a, b in CP[tid] if (tid, a) not in ag and (tid, b) not in ag and (tid, a) not in zero and (tid, b) not in zero]
        if not pairs: continue
        tot = sum(T[(tid, m)] for m in movs[tid])
        if any(T[(tid, a)] >= 30 and T[(tid, b)] >= 30 for a, b in pairs): continue
        a, b = max(pairs, key=lambda ab: max(T[(tid, ab[0])], T[(tid, ab[1])]))
        lo = a if T[(tid, a)] < T[(tid, b)] else b
        cap = 80.0 if (not ref[tid]["trusted"] and tot < 300) else min(80.0, 0.1 * tot)
        for m in (a, b) if (not ref[tid]["trusted"] and tot < 300) else (lo,):
            if T[(tid, m)] < cap:
                topup.append(dict(tls=tid, movement=m, before=round(T[(tid, m)], 1), after=round(cap, 1), reason="路口无可学取舍"))
                T[(tid, m)] = cap
    # 设计流量 (用户 2026-10-28 要求), 作为偏离真实逐条记录
    for (tid, m), v, why in DESIGN:
        if (tid, m) in zero: continue
        if T.get((tid, m), 0.0) < v:
            topup.append(dict(tls=tid, movement=m, before=round(T.get((tid, m), 0.0), 1), after=v, reason=why)); T[(tid, m)] = v
    car_T = {k: max(0.0, v - bus_mv[k] - amb_mv[k]) for k, v in T.items()}
    return T, car_T, ref, notes, topup, bus_mv, amb_mv, ag

def all_lanes(r, edges, lane):
    """所有可能的车道路径上的车道 (车辆不换道, 但在路口可能被分到不同目标车道) — 保守口径。"""
    return {x for p in r.lane_paths(edges, lane) for x in p}

def fixed_side_flows(buses, amb_routes, side_lanes, r=None):
    """公交 (每小时车次) 与救护车 (每集辆数) 在支路入口车道上的本地发车与上游穿行 (所有可能车道路径)。"""
    st = collections.Counter(); th = collections.Counter()
    items = [(b["edges"], b["lane"], b["lanes"][0], 1.0) for b in buses["buses"]] + [(e, ln, l0, float(C.AMB_PER_EP)) for e, ln, l0 in amb_routes]
    for edges, ln, l0, w in items:
        lanes = all_lanes(r, edges, ln) if r is not None else set()
        for l in side_lanes:
            if l0 == l: st[l] += w
            elif l in lanes: th[l] += w
    return st, th

def fit(meta, r, routes, car_T, ref, bus_mv, buses, side_lanes, zero, ag, budget_w=1.0, mv_weight=0.15, junction_weight=8.0, side_w=5000.0, prev_major=None, caps=None, lane_factor=None, side_w_=None, amb_routes=()):
    mv_list = sorted(car_T); mi = {m: i for i, m in enumerate(mv_list)}
    side_set = set(side_lanes)
    # 小汽车路线不得从上游穿过支路入口车道 (中段插车车道上只留固定的公交/救护车穿行), 所有可能车道路径保守判断
    use = [d for d in routes if not any(tuple(t) in zero for t in d["traversals"]) and d["entry_type"] in ("boundary", "side")
           and not ((all_lanes(r, d["edges"], d["lane"]) - {d["lanes"][0]}) & side_set)]
    n = len(use); nm = len(mv_list)
    A = lil_matrix((nm, n))
    for j, d in enumerate(use):
        for t in d["traversals"]: A[mi[tuple(t)], j] += 1
    A = A.tocsr(); goal = np.array([car_T[m] for m in mv_list])
    # 转向先验是弱目标 (旧 18h 路由有 37% 的车在 RL 进口道上出发, 先验在相邻路口之间不守恒); 可信路口的总量是强目标
    w = mv_weight * np.array([(0.3 if m in ag else (8.0 if m in DESIGN_KEYS else (0.5 if not ref[m[0]]["trusted"] else 1.0))) / max(40.0, goal[i]) for i, m in enumerate(mv_list)])
    # 路口总量项 (可信路口): |Σ 小汽车 − (s R − 公交)| 权重 2/R
    tls = sorted(meta); ti = {t: k for k, t in enumerate(tls)}; nt = len(tls)
    J = lil_matrix((nt, n)); jgoal = np.zeros(nt); jw = np.zeros(nt)
    bus_j = collections.Counter()
    for (t, m), v in bus_mv.items(): bus_j[t] += v
    for j, d in enumerate(use):
        for t, m in d["traversals"]: J[ti[t], j] += 1
    for t in tls:
        jgoal[ti[t]] = max(0.0, ref[t]["scale"] * ref[t]["R"] - bus_j[t]) if ref[t]["trusted"] else 0.0
        jw[ti[t]] = junction_weight / max(100.0, ref[t]["R"]) if ref[t]["trusted"] else 0.0
    J = J.tocsr()
    # 让行预算: 每个被路线用到的让行连接 → 冲突主路的 (from_lane, to_lane) 连接; 主路流量 (含公交) − 预算 ≤ 超额 (软)
    net = r.net; minor_links = set()
    for d in use:
        for x, y in d["yield_links"]: minor_links.add((x, y))
    def prohibitors(x, y):
        out = set(); L = net.getLane(x)
        for c in L.getOutgoing():
            if c.getToLane().getID() != y: continue
            node = L.getEdge().getToNode()
            for e in node.getIncoming():
                for l2 in e.getLanes():
                    for c2 in l2.getOutgoing():
                        if c2.getState() in ("M", "O", "o") and node.forbids(c2, c) if hasattr(node, "forbids") else False:
                            out.add((l2.getID(), c2.getToLane().getID()))
                        elif c2.getState() == "M" and c2.getFromLane().getEdge() != L.getEdge() and node.areFoes(c2.getJunctionIndex(), c.getJunctionIndex()):
                            out.add((l2.getID(), c2.getToLane().getID()))
        return out
    budgets = {}
    for x, y in sorted(minor_links):
        pro = prohibitors(x, y)
        if not pro: continue
        major_edges = sorted({p[0].rsplit("_", 1)[0] for p in pro})
        B = min(C.YIELD_BUDGET_SPECIAL.get(e, C.YIELD_BUDGET) for e in major_edges)
        budgets[f"{x}->{y}"] = dict(major=sorted(pro), budget=B, node=net.getLane(x).getEdge().getToNode().getID())
    bkeys = sorted(budgets); nb = len(bkeys)
    Bm = lil_matrix((nb, n)); bconst = np.zeros(nb)
    for k, key in enumerate(bkeys):
        pro = set(map(tuple, budgets[key]["major"]))
        for j, d in enumerate(use):
            if any((a, b) in pro for a, b in zip(d["lanes"], d["lanes"][1:])): Bm[k, j] = 1
        for b in buses["buses"]:
            if any((a, c) in pro for a, c in zip(b["lanes"], b["lanes"][1:])): bconst[k] += 1
    Bm = Bm.tocsr()
    # 变量: [r (n), e+ (nm), e- (nm), j+ (nt), j- (nt), b_ex (nb), s_ex (ns)]
    ns = len(side_lanes)
    # 路线代价: 很小的长度与让行项; 支路入口每辆本地发车加小代价 (不鼓励 "多插车稀释穿行比例")
    cost = np.r_[np.array([1e-5 * d["length"] / 1000 + 1e-4 * len(d["yield_links"]) + (2e-3 if d["entry"] in side_lanes else 0.0) for d in use]), w, w, jw, jw, budget_w * np.ones(nb) / 150.0, side_w * np.ones(ns) / 100.0]
    Z = lambda a, b: csr_matrix((a, b))
    Aeq = vstack([hstack([A, -eye(nm), eye(nm), Z(nm, nt), Z(nm, nt), Z(nm, nb), Z(nm, ns)]),
                  hstack([J, Z(nt, nm), Z(nt, nm), -eye(nt), eye(nt), Z(nt, nb), Z(nt, ns)])]).tocsr()
    beq = np.r_[goal, jgoal]
    # 不等式: 入口车道上限; 支路入口上游来车 ≤ 10% 起点量; 让行预算 (软)
    ub_rows = []; ub = []
    entry_use = collections.defaultdict(list)
    for j, d in enumerate(use): entry_use[d["entry"]].append(j)
    bus_entry = collections.Counter(b["entry"] for b in buses["buses"])
    nvar = n + 2 * nm + 2 * nt + nb + ns
    for lane, js in entry_use.items():
        row = lil_matrix((1, nvar))
        for j in js: row[0, j] = 1
        cap = C.LANE_CAP_LONG_VPH if (lane in r.sources and net.getLane(lane).getLength() >= 100) else C.LANE_CAP_VPH
        ub_rows.append(row); ub.append(cap - bus_entry[lane])
    fixed_start, fixed_thr = fixed_side_flows(buses, amb_routes, side_lanes, r)
    route_lanes = [all_lanes(r, d["edges"], d["lane"]) for d in use]
    for k, lane in enumerate(side_lanes):     # 软 (强): 上游穿行 (全部车型, 所有可能车道路径) − 0.1×本地发车 ≤ 超额
        row = lil_matrix((1, nvar))
        for j, d in enumerate(use):
            if d["entry"] == lane: row[0, j] -= 0.1
            elif lane in route_lanes[j]: row[0, j] += 1
        row[0, n + 2 * nm + 2 * nt + nb + k] = -1
        ub_rows.append(row); ub.append(0.1 * fixed_start[lane] - fixed_thr[lane])
    # 让行支路: 用到该让行连接的路线流量 ≤ MINOR_CAP(上一轮的冲突主路流量 + 公交); 特例 2294382#1 (9101554 主路) ≤ 100 (软)
    for k in range(nb):
        key = bkeys[k]; x, y = key.split("->")
        row = lil_matrix((1, nvar))
        if any(e == "2294382#1" for e in {p[0].rsplit("_", 1)[0] for p in budgets[key]["major"]}):
            row[0, :n] = Bm[k].toarray(); row[0, n + 2 * nm + 2 * nt + k] = -1
            ub_rows.append(row); ub.append(100.0 - bconst[k]); continue
        if prev_major is None: continue
        for j, d in enumerate(use):
            if (x, y) in {tuple(z) for z in d["yield_links"]}: row[0, j] = 1
        minor_bus = sum(1 for b in buses["buses"] if (x, y) in set(zip(b["lanes"], b["lanes"][1:])))
        ub_rows.append(row); ub.append(max(0.0, C.MINOR_CAP(prev_major.get(key, 0.0)) - minor_bus))
    # RL 进口车道容量 (硬约束): Σ 经过该车道的流量 (含公交) ≤ 饱和流 × 绿灯比例 × 该车道获绿相位占比 × 利用率
    lane_caps = {}
    for tid, t in meta.items():
        valid = [x["state"] for x in t["actions"].values() if x.get("valid")]
        lane_idx = collections.defaultdict(set); lane_mv = collections.defaultdict(set)
        for i, g in t["links"].items():
            for c in g:
                l = c["from_edge"] + "_" + str(c["from_lane"]); lane_idx[l].add(int(i)); lane_mv[l].add(c["approach"] + "-" + c["turn"])
        mv_idx = collections.defaultdict(set)
        for i, g in t["links"].items():
            for c in g: mv_idx[c["approach"] + "-" + c["turn"]].add(int(i))
        mv_share = {m: sum(1 for st in valid if any(st[i] == "G" for i in idx)) / len(valid) for m, idx in mv_idx.items()}
        for l, idx in lane_idx.items():
            # 共享车道只有头车的转向为绿时才能走: 取车道上各转向中最小的获绿占比 (最不利)
            share = min(mv_share[m] for m in lane_mv[l]) if len(lane_mv[l]) > 1 else sum(1 for st in valid if any(st[i] == "G" for i in idx)) / len(valid)
            util = C.LANE_UTIL["shared" if len(lane_mv[l]) > 1 else "single"]
            lane_caps[l] = C.SAT_FLOW * C.GREEN_FRAC * share * util * (lane_factor or {}).get(l, (lane_factor or {}).get(l.rsplit("_", 1)[0], 1.0))
    bus_lane = collections.Counter(l for b in buses["buses"] for l in b["lanes"])
    for l, capv in lane_caps.items():
        js = [j for j, d in enumerate(use) if l in d["lanes"]]
        if not js: continue
        row = lil_matrix((1, nvar))
        for j in js: row[0, j] = 1
        ub_rows.append(row); ub.append(max(0.0, capv - bus_lane[l]))
    for tid, capv in (caps or {}).items():
        row = lil_matrix((1, nvar)); row[0, :n] = J[ti[tid]].toarray(); ub_rows.append(row); ub.append(capv - bus_j[tid])
    # 最低流量: −(Σ 车道/进口流量) − 松弛 ≤ −(下限 − 公交)  (松弛变量在 nvar 之后追加)
    floors = []
    zero_mv = set(zero)
    for tid, t in meta.items():
        lanes_e = collections.defaultdict(set); lane_ok = {}
        for g in t["links"].values():
            for c in g:
                l = c["from_edge"] + "_" + str(c["from_lane"]); lanes_e[c["from_edge"]].add(l)
                lane_ok[l] = lane_ok.get(l, False) or ((tid, c["approach"] + "-" + c["turn"]) not in zero_mv)
        for e, ls in lanes_e.items():
            ls_ok = [l for l in ls if lane_ok[l]]
            if not ls_ok: continue
            floors.append((ls_ok, min(C.APPROACH_MIN_VPH, sum(lane_caps.get(l, 1e9) for l in ls_ok))))
            for l in ls_ok: floors.append(([l], min(C.LANE_MIN_VPH, lane_caps.get(l, 1e9))))
    nf = len(floors)
    ub_rows = [hstack([rw if rw.shape[1] == nvar else rw, csr_matrix((1, nf))]).tocsr() for rw in ub_rows]
    for k, (ls, fl) in enumerate(floors):
        row = lil_matrix((1, nvar + nf)); sset = set(ls)
        for j, d in enumerate(use):
            if sset & set(d["lanes"]): row[0, j] = -1
        row[0, nvar + k] = -1
        ub_rows.append(row.tocsr()); ub.append(-(fl - sum(bus_lane[l] for l in ls)))
    cost = np.r_[cost, np.ones(nf) * 1.0]      # 最低流量 (用户要求不留空车道) 优先于贴合 SCATS
    Aeq = hstack([Aeq, csr_matrix((Aeq.shape[0], nf))]).tocsr()
    fit.floors = floors
    sol = linprog(cost, A_ub=vstack(ub_rows).tocsr(), b_ub=np.array(ub), A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs")
    assert sol.success, sol.message
    rates = sol.x[:n]; excess = sol.x[n + 2 * nm + 2 * nt: n + 2 * nm + 2 * nt + nb]
    fit.floor_short = [dict(lanes=fl[0], floor=fl[1], short=float(sol.x[nvar + k])) for k, fl in enumerate(floors) if sol.x[nvar + k] > 1]
    realized = A @ rates
    major_now = {bkeys[k]: float((Bm[k] @ rates)[0] + bconst[k]) for k in range(nb)}
    minor_now = {}
    for k in range(nb):
        x, y = bkeys[k].split("->")
        minor_now[bkeys[k]] = float(sum(rates[j] for j, d in enumerate(use) if (x, y) in {tuple(z) for z in d["yield_links"]}))
    for k in range(nb): budgets[bkeys[k]]["minor_flow"] = minor_now[bkeys[k]]; budgets[bkeys[k]]["minor_cap"] = C.MINOR_CAP(major_now[bkeys[k]])
    fit.lane_caps = lane_caps
    return use, rates, dict(zip(map(lambda m: f"{m[0]}|{m[1]}", mv_list), zip(goal.tolist(), realized.tolist()))), \
        {bkeys[k]: dict(budgets[bkeys[k]], used=float((Bm[k] @ rates)[0] + bconst[k]), excess=float(excess[k])) for k in range(nb)}

def schedule(use, counts, buses, amb, tt=1, seed=1028):
    """按入口车道统一排时刻: 公交/救护车 (时刻表 tt) 固定; 小汽车按曲线分层抽样后, 在每条入口车道上顺次保证相邻间隔 ≥ 3 s。"""
    rng = random.Random(seed); cars = []
    for j, (d, n) in enumerate(zip(use, counts)):
        for k in range(n):
            u = (k + rng.random()) / n
            cars.append(dict(id=f"car_{j:04d}_{k:03d}", route=j, entry=d["entry"], want=C.profile_inverse(u), key=rng.random()))
    # 嵌套抽稀用的 key: 同一路线按时间顺序取黄金比例低差异序列 (起点 = 该路线第一辆车原来的随机 key, 不多耗随机数, 1.0 档逐位不变),
    # 使任意时间段、任意路线上 key < L 的比例都贴近 L; 独立均匀 key 在 0.6 档会让平台期后段多留 ~7% 的车, 抬高 C1。
    by_route = collections.defaultdict(list)
    for c in cars: by_route[c["route"]].append(c)
    for cs in by_route.values():
        cs.sort(key=lambda c: c["want"]); off = cs[0]["key"]
        for i, c in enumerate(cs): c["key"] = (off + i * 0.6180339887498949) % 1.0
    fixed = collections.defaultdict(list)
    for b in buses["buses"]: fixed[b["entry"]].append(buses["timetables"][str(tt)][b["id"]] if str(tt) in buses["timetables"] else buses["timetables"][tt][b["id"]])
    amb_entry = amb["route"]["lanes"][0]
    for t in (amb["timetables"][str(tt)] if str(tt) in amb["timetables"] else amb["timetables"][tt]): fixed[amb_entry].append(t)
    by_entry = collections.defaultdict(list)
    for c in cars: by_entry[c["entry"]].append(c)
    shifted = 0
    for lane, cs in by_entry.items():
        cs.sort(key=lambda c: c["want"]); busy = sorted(fixed[lane]); last = -1e9
        for c in cs:
            t = max(c["want"], last + C.MIN_HEADWAY)
            while any(abs(t - b) < C.MIN_HEADWAY for b in busy): t += 0.5
            if t - c["want"] > 0.05: shifted += 1
            c["depart"] = round(t, 1); last = t
    return cars, shifted

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scale-file", default=None); a = ap.parse_args()
    C.check_hashes(); meta = C.load_meta(); r = C.load_router()
    anchors = C.load("anchors_18h.json"); buses = C.load("buses.json"); amb = C.load("ambulances.json"); E = C.load("entries.json")
    routes = C.load("route_pool.json")["routes"]
    zero = {tuple(m) for m in E["zero"]} | {tuple(m) for m in E["zero_lane_loop"]}
    sc = json.load(open(a.scale_file)) if a.scale_file else {}
    caps = sc.pop("caps", {}) if isinstance(sc, dict) else {}
    lane_factor = sc.pop("lane_factor", {}) if isinstance(sc, dict) else {}
    scales = sc
    T, car_T, ref, notes, topup, bus_mv, amb_mv, ag = targets(meta, anchors, buses, amb, zero, scales)
    prev = None
    for it in range(3):    # 让行支路容量依赖主路流量: 迭代 3 轮
        use, rates, mvfit, budgets = fit(meta, r, routes, car_T, ref, bus_mv, buses, E["side_lanes"], zero, ag, prev_major=prev, caps=caps, lane_factor=lane_factor, amb_routes=[(amb["route"]["edges"], amb["route"]["lane"], amb["route"]["lanes"][0])])
        prev = {k: v["used"] for k, v in budgets.items()}
    frac = C.profile_integral() / 3600.0
    counts = [int(math.floor(x * frac + 0.5)) for x in rates]
    cars, shifted = schedule(use, counts, buses, amb, tt=1)
    # 整数化后按平台期流率复核: 入口上限 / 支路入口 10% / 让行预算 (报告)
    plat = collections.Counter(); plat_lane_through = collections.Counter(); plat_side_start = collections.Counter()
    pl0, pl1 = C.PLATEAU; scale_h = 3600.0 / (pl1 - pl0)
    for c in cars:
        if pl0 <= c["depart"] < pl1:
            d = use[c["route"]]; plat[d["entry"]] += 1
            for l in all_lanes(r, d["edges"], d["lane"]) - {d["lanes"][0]}: plat_lane_through[l] += 1
            if d["entry"] in E["side_lanes"]: plat_side_start[d["entry"]] += 1
    bus_entry = collections.Counter(b["entry"] for b in buses["buses"])
    entry_check = {l: round(v * scale_h + bus_entry[l], 1) for l, v in plat.items()}
    fs, ft = fixed_side_flows(buses, [(amb["route"]["edges"], amb["route"]["lane"], amb["route"]["lanes"][0])], E["side_lanes"], r)
    side_check = {l: dict(start=round(plat_side_start[l] * scale_h + fs[l], 1), through=round(plat_lane_through[l] * scale_h + ft[l], 1),
                          car_start=round(plat_side_start[l] * scale_h, 1), car_through=round(plat_lane_through[l] * scale_h, 1), fixed_start=fs[l], fixed_through=ft[l]) for l in E["side_lanes"]}
    # 各路口实现量 (平台期流率, 含公交与救护车) 与参考值
    real = collections.Counter()
    for j, x in enumerate(rates):
        for t, m in use[j]["traversals"]: real[(t, m)] += x
    for k, v in bus_mv.items(): real[k] += v
    for k, v in amb_mv.items(): real[k] += v
    jun = {}
    for tid in meta:
        tot = sum(v for (t, m), v in real.items() if t == tid)
        jun[tid] = dict(ref=ref[tid]["R"], source=ref[tid]["source"], scale=ref[tid]["scale"], realized=round(tot, 1), ratio=round(tot / ref[tid]["R"], 3) if ref[tid]["R"] else None)
    lane_load = collections.Counter()
    for j, x in enumerate(rates):
        for l in use[j]["lanes"]: lane_load[l] += x
    for b in buses["buses"]:
        for l in b["lanes"]: lane_load[l] += 1
    lane_util = {l: dict(load=round(lane_load[l], 1), cap=round(cv, 1)) for l, cv in fit.lane_caps.items()}
    report = dict(scales=scales, caps=caps, lane_factor=lane_factor, floor_short=fit.floor_short, lane_capacity=lane_util, n_routes_used=int(sum(1 for x in counts if x > 0)), cars_episode=len(cars), cars_plateau_rate_total=round(sum(rates), 1), shifted_for_headway=shifted,
                  entry_plateau_vph=entry_check, side_entry=side_check, junctions=jun, topup=topup, notes=notes,
                  movements={f"{t}|{m}": dict(target_total=round(T[(t, m)], 1), car_target=round(car_T[(t, m)], 1), realized_total=round(real[(t, m)], 1)) for (t, m) in sorted(T)},
                  yield_budgets=budgets, zero=[list(m) for m in sorted(zero)])
    C.dump(report, "demand_report.json")
    C.dump(dict(routes=[dict(edges=d["edges"], lane=d["lane"], lanes=d["lanes"], entry=d["entry"], traversals=d["traversals"]) for d in use],
                rates=[float(x) for x in rates], counts=counts, cars=cars), "background.json")
    print(f"路线 {report['n_routes_used']} 条; 平台期小汽车流率合计 {report['cars_plateau_rate_total']:.0f} 辆/h; 整集小汽车 {len(cars)} 辆; 为保证间隔挪动 {shifted} 辆")
    print("各路口 实现/参考 (含公交):")
    for t, v in jun.items(): print(f"  {t.replace('cluster_','c')[:16]:16s} 参考 {v['ref']:5.0f} ({'SCATS' if ref[t]['trusted'] else '旧路由'}) 实现 {v['realized']:6.0f} 比 {v['ratio']}")
    print("入口平台期流率 (含公交) >600:", {l: v for l, v in entry_check.items() if v > 600})
    print("支路入口:", side_check); print("补量:", topup)
    over = {k: dict(major=round(v['used']), minor=round(v['minor_flow']), cap=round(v['minor_cap'])) for k, v in budgets.items() if v['minor_flow'] > v['minor_cap'] + 1}
    print(f"让行连接 {len(budgets)} 处; 支路流量超过估计通过量的 {len(over)} 处:", over)

if __name__ == "__main__":
    main()
