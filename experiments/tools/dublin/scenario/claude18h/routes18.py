"""任务 2 + 3: 入口/出口白名单、穷举零需求判定、候选路线池。
入口: 15 条边界入口 + 按规则选出的支路入口 (只为只从边界入口无法覆盖的转向选); 出口: 15 条无去向的边。
路线: 车道连续 (不换道)、不重访 RL 路口、终点为边界出口。"""
import collections, itertools, json
import networkx as nx
import common18 as C

def build():
    C.check_hashes(); meta = C.load_meta(); r = C.load_router(zonal=False); G = r.g; net = r.net; mv = r.movements
    rl_nodes = set(meta)
    rl_lanes = set()
    for t in meta:
        for e in net.getNode(t).getIncoming() + net.getNode(t).getOutgoing():
            rl_lanes |= {l.getID() for l in e.getLanes()}
    boundary = sorted(l for l in r.sources)
    sinks = sorted({l for l in r.sinks})
    assert not set(boundary) & rl_lanes
    # ---- 穷举 (状态 = 车道 + 已经过的 RL 路口集合)
    def reach_sets(start, tid, forward, sources):
        out = set(); seen = set(); st = [(start, frozenset())]
        while st:
            lane, S = st.pop()
            if (lane, S) in seen: continue
            seen.add((lane, S))
            nbrs = G.successors(lane) if forward else G.predecessors(lane)
            for q in nbrs:
                if forward and q == "__SINK__": out.add(S); continue
                if not forward and q == "__SOURCE__":
                    if lane in sources: out.add(S)
                    continue
                if q.startswith("__"): continue
                key = (lane, q) if forward else (q, lane); j = mv.get(key); S2 = S
                if j:
                    if j[0] == tid or j[0] in S: continue
                    S2 = S | {j[0]}
                st.append((q, S2))
            if not forward and lane in sources: out.add(S)
        return out, len(seen)
    def feasible(sources):
        res = {}; states = 0
        for (a, b), m in mv.items():
            P, n1 = reach_sets(a, m[0], False, sources); F, n2 = reach_sets(b, m[0], True, sources); states += n1 + n2
            res[m] = res.get(m, False) or any(not (p & f) for p in P for f in F)
        return res, states
    feas_b, st_b = feasible(set(boundary))
    need_side = sorted(m for m, ok in feas_b.items() if not ok)
    # ---- 支路入口候选 (静态规则): 非 RL 两端、≥40 m、到第一个 RL 进口不经让行、离第一个 RL 停车线 ≥60 m
    def first_rl(lane):
        """沿车道图向下游走到第一个 RL 进口车道; 返回 [(进口车道, 距离, 经过让行?)]"""
        out = []; st = [(lane, net.getLane(lane).getLength(), False, 0)]; seen = set()
        while st:
            l, d, minor, k = st.pop()
            if (l, minor) in seen or k > 8: continue
            seen.add((l, minor)); e = net.getLane(l).getEdge()
            if e.getToNode().getID() in rl_nodes: out.append((l, d, minor)); continue
            for q in G.successors(l):
                if q.startswith("__"): continue
                st.append((q, d + net.getLane(q).getLength(), minor or G[l][q].get("minor", False), k + 1))
        return out
    cands = []
    for e in net.getEdges(withInternal=False):
        if e.getFromNode().getID() in rl_nodes or e.getToNode().getID() in rl_nodes or not e.getIncoming(): continue
        for l in e.getLanes():
            if not l.allows("passenger") or l.getLength() < 40: continue
            fr = first_rl(l.getID())
            if not fr or any(m for _, _, m in fr) or min(d for _, d, _ in fr) < 60 + 0: continue
            cands.append(l.getID())
    # 为每个只靠边界入口覆盖不了的转向挑一条支路入口: 能覆盖它; 优先选公交/救护车不会穿行的车道 (按全部可能车道路径保守计数), 再看上游车道数
    fixed_thr = collections.Counter()
    try:
        for b in C.load("buses.json")["buses"]:
            for l in {x for p in r.lane_paths(b["edges"], b["lane"]) for x in p[1:]}: fixed_thr[l] += 1
    except FileNotFoundError: pass
    try:   # 救护车路线经过的车道也要避开 (插入的车可能插到救护车前面)
        ch = json.load(open(C.WORK / "amb_route_choice.json"))["edges"]
        for p in r.lane_paths(ch, 0) + r.lane_paths(ch, 1):
            for x in p[1:]: fixed_thr[x] += C.AMB_PER_EP     # 按实际通过量 (每集 3 辆), 与公交同等计入
    except (FileNotFoundError, IndexError): pass
    side = {}
    for m in need_side:
        best = None
        for lane in cands:
            feas, _ = feasible_one(m, set(boundary) | {lane}, G, mv, reach_sets)
            if not feas: continue
            ups = sum(1 for p in G.predecessors(lane) if not p.startswith("__"))
            key = (fixed_thr[lane], ups, -net.getLane(lane).getLength())
            if best is None or key < best[0]: best = (key, lane)
        side[f"{m[0]}|{m[1]}"] = best[1] if best else None
    side_lanes = sorted({v for v in side.values() if v})
    sources = set(boundary) | set(side_lanes)
    feas_all, st_all = feasible(sources)
    zero = sorted(m for m, ok in feas_all.items() if not ok)
    return dict(meta=meta, r=r, boundary=boundary, sinks=sinks, side=side, side_lanes=side_lanes, sources=sources,
                zero=zero, need_side=need_side, states=dict(boundary_only=st_b, with_side=st_all), cands=cands)

def side_rule_ok(r, meta, lane, min_len=40.0, min_dist=60.0):
    """支路入口静态规则: 两端非 RL、车道 ≥40 m、到第一个 RL 进口不经让行连接、离第一个 RL 停车线 ≥60 m (含进口边)。"""
    net = r.net; G = r.g; rl_nodes = set(meta); L = net.getLane(lane); e = L.getEdge()
    if e.getFromNode().getID() in rl_nodes or e.getToNode().getID() in rl_nodes or L.getLength() < min_len: return False
    out = []; st = [(lane, L.getLength(), False, 0)]; seen = set()
    while st:
        l, d, minor, k = st.pop()
        if (l, minor) in seen or k > 8: continue
        seen.add((l, minor)); ee = net.getLane(l).getEdge()
        if ee.getToNode().getID() in rl_nodes: out.append((d, minor)); continue
        for q in G.successors(l):
            if q.startswith("__"): continue
            st.append((q, d + net.getLane(q).getLength(), minor or G[l][q].get("minor", False), k + 1))
    return bool(out) and not any(m for _, m in out) and min(d for d, _ in out) >= min_dist

def feasible_one(m, sources, G, mv, reach_sets):
    ok = False
    for (a, b), mm in mv.items():
        if mm != m: continue
        P, _ = reach_sets(a, m[0], False, sources); F, _ = reach_sets(b, m[0], True, sources)
        ok = ok or any(not (p & f) for p in P for f in F)
    return ok, None

def route_pool(B, k=14, per_mv=40, only=None, only_links=None):
    """每个转向: 删去本路口其他连接后, 前段 (入口→进口车道) 与后段 (出口车道→出口) 各取 k 条最短路, 组合为不重访、车道不重复的路线。"""
    r = B["r"]; G0 = r.g; net = r.net; mv = r.movements
    G = G0.copy()
    for l in list(G.successors("__SOURCE__")):
        if l not in B["sources"]: G.remove_edge("__SOURCE__", l)
    for l in B["side_lanes"]:
        G.add_edge("__SOURCE__", l, weight=net.getLane(l).getLength())
    pool = {}; per = collections.Counter()
    def rl_seq(p): return [mv[(x, y)][0] for x, y in zip(p, p[1:]) if (x, y) in mv]
    by_m = collections.defaultdict(list)
    for (a, b), m in mv.items(): by_m[m].append((a, b))
    for m, links in by_m.items():
        if m in set(B["zero"]): continue
        if only is not None and m not in only: continue
        cand = []
        for a, b in links:
            if only_links is not None and (a, b) not in only_links: continue
            H = G.copy()
            for (x, y), m2 in mv.items():
                if m2[0] == m[0] and (x, y) != (a, b) and H.has_edge(x, y): H.remove_edge(x, y)
            try: pres = list(itertools.islice(nx.shortest_simple_paths(H, "__SOURCE__", a, weight="weight"), k))
            except nx.NetworkXNoPath: pres = []
            try: sufs = list(itertools.islice(nx.shortest_simple_paths(H, b, "__SINK__", weight="weight"), k))
            except nx.NetworkXNoPath: sufs = []
            for p in pres:
                sp = rl_seq(p)
                for s in sufs:
                    ss = rl_seq(s)
                    if set(sp) & set(ss) or len(set(sp)) != len(sp) or len(set(ss)) != len(ss) or set(p) & set(s): continue
                    path = [x for x in p + s if not x.startswith("__")]
                    d = r.describe(path); d["entry"] = path[0]; d["entry_type"] = "side" if path[0] in B["side_lanes"] else "boundary"
                    d["yield_links"] = [(x, y) for x, y in zip(path, path[1:]) if G[x][y].get("minor")]
                    cand.append(d)
        cand.sort(key=lambda d: (len(d["yield_links"]), len(d["traversals"]), d["length"]))
        for d in cand[:per_mv]:
            key = (tuple(d["lanes"]),)
            if key not in pool: pool[key] = d
            per[m] += 1
    routes = list(pool.values())
    for d in routes: d["traversals"] = [list(x) for x in d["traversals"]]
    return routes, per

def construct_link(B, a, b, banned):
    """为指定的转向车道连接 (a→b) 构造一条合规路线: 前段从合法入口到 a, 后段从 b 到出口; 广度优先 (先最短), 不重访 RL 路口,
    不经过 banned 车道 (支路入口车道不可作中途经过), 不经过零需求转向。"""
    r = B["r"]; G = r.g; mv = r.movements; tid = mv[(a, b)][0]; zset = set(B["zero"]); srcs = B["sources"]
    def bfs(start, forward, forbid_tids):
        par = {(start, frozenset()): None}; q = [(start, frozenset())]
        while q:
            lane, S = q.pop(0)
            if not forward and lane in srcs and (lane == start or lane not in banned):
                out = []; st = (lane, S)
                while st is not None: out.append(st[0]); st = par[st]
                return out, S
            for nx_ in (G.successors(lane) if forward else G.predecessors(lane)):
                if forward and nx_ == "__SINK__":
                    out = []; st = (lane, S)
                    while st is not None: out.append(st[0]); st = par[st]
                    return list(reversed(out)), S
                if nx_.startswith("__") or (nx_ in banned and not (not forward and nx_ in srcs)): continue
                key = (lane, nx_) if forward else (nx_, lane); j = mv.get(key); S2 = S
                if j:
                    if j[0] == tid or j[0] in S or j[0] in forbid_tids or j in zset: continue
                    S2 = S | {j[0]}
                if (nx_, S2) in par: continue
                par[(nx_, S2)] = (lane, S); q.append((nx_, S2))
        return None, None
    pre, Sp = bfs(a, False, frozenset())
    if pre is None: return None
    suf, Sf = bfs(b, True, Sp)
    if suf is None: return None
    path = pre + suf
    if len(set(path)) != len(path): return None
    d = r.describe(path); d["entry"] = path[0]; d["entry_type"] = "side" if path[0] in B["side_lanes"] else "boundary"
    d["yield_links"] = [(x, y) for x, y in zip(path, path[1:]) if G[x][y].get("minor")]; d["traversals"] = [list(x) for x in d["traversals"]]
    return d

def construct(B, m):
    """穷举构造: 记录父状态, 返回一条不重访、车道连续的完整路线 (前 k 条最短路组合不出来时用)。"""
    r = B["r"]; G = r.g; mv = r.movements; net = r.net; srcs = B["sources"]; tid = m[0]
    def search(start, forward):
        par = {(start, frozenset()): None}; st = [(start, frozenset())]; ends = {}
        while st:
            lane, S = st.pop(0)          # 广度优先: 每个 RL 集合下先记到最短的前段
            if not forward and lane in srcs: ends.setdefault(S, (lane, S))
            nbrs = G.successors(lane) if forward else G.predecessors(lane)
            for q in nbrs:
                if q == "__SINK__" and forward: ends.setdefault(S, (lane, S)); continue
                if q.startswith("__"): continue
                key = (lane, q) if forward else (q, lane); j = mv.get(key); S2 = S
                if j:
                    if j[0] == tid or j[0] in S: continue
                    S2 = S | {j[0]}
                if (q, S2) in par: continue
                par[(q, S2)] = (lane, S); st.append((q, S2))
        def path(state):
            out = []
            while state is not None: out.append(state[0]); state = par[state]
            return out
        return ends, path
    def fwd_avoid(start, banned, avoid_tids):
        """前段确定后, 后段在不碰前段车道、不经过前段 RL 路口的条件下重新搜 (广度优先, 记父节点)。"""
        par = {(start, frozenset()): None}; q = [(start, frozenset())]
        while q:
            lane, S = q.pop(0)
            for nx_ in G.successors(lane):
                if nx_ == "__SINK__":
                    out = []; st = (lane, S)
                    while st is not None: out.append(st[0]); st = par[st]
                    return list(reversed(out))
                if nx_.startswith("__") or nx_ in banned: continue
                j = mv.get((lane, nx_)); S2 = S
                if j:
                    if j[0] == tid or j[0] in S or j[0] in avoid_tids: continue
                    S2 = S | {j[0]}
                if (nx_, S2) in par: continue
                par[(nx_, S2)] = (lane, S); q.append((nx_, S2))
        return None
    for (a, b), mm in mv.items():
        if mm != m: continue
        P, ppath = search(a, False)
        for Sp, ep in P.items():
            pre = ppath(ep)
            suf = fwd_avoid(b, set(pre), set(Sp))
            if suf is None: continue
            path = pre + suf
            if len(set(path)) != len(path): continue
            d = r.describe(path); d["entry"] = path[0]; d["entry_type"] = "side" if path[0] in B["side_lanes"] else "boundary"
            d["yield_links"] = [(x, y) for x, y in zip(path, path[1:]) if G[x][y].get("minor")]
            d["traversals"] = [list(x) for x in d["traversals"]]
            return d
    return None

def main():
    B = build()
    print("只靠边界入口无路线:", B["need_side"], "状态数", B["states"]["boundary_only"])
    print("支路入口:", B["side"])
    print("零需求转向 (边界 + 支路入口, 允许让行):", B["zero"], "状态数", B["states"]["with_side"])
    routes, per = route_pool(B)
    allm = set(B["r"].movements.values())
    uncovered = sorted(m for m in allm if per[m] == 0 and m not in set(B["zero"]))
    if uncovered:   # 穷举判定可行但前 k 条最短路没组合出不重访路线的转向: 加大搜索
        print("加大搜索:", uncovered)
        more, per2 = route_pool(B, k=150, per_mv=10, only=set(uncovered))
        seen = {tuple(d["lanes"]) for d in routes}
        routes += [d for d in more if tuple(d["lanes"]) not in seen]
        for m in uncovered: per[m] += per2[m]
        uncovered = sorted(m for m in allm if per[m] == 0 and m not in set(B["zero"]))
    for m in list(uncovered):
        d = construct(B, m)
        if d is not None:
            routes.append(d); per[m] += 1; print("穷举构造补路线:", m, "经过 RL", [t[0][:12] for t in d["traversals"]], "让行", len(d["yield_links"]))
    uncovered = sorted(m for m in allm if per[m] == 0 and m not in set(B["zero"]))
    # 按车道覆盖: 每条 RL 进口车道至少要有一条路线的车道路径真正走它 (按转向取前 k 条最短路时, 绕一点的车道会被挤掉)
    # 只认合规路线 (不经过零需求转向、小汽车不从上游穿过支路入口车道), 与 demand18 的路线筛选一致
    zset = set(B["zero"]); side_set = set(B["side_lanes"])
    def usable(d):
        if any(tuple(t) in zset for t in d["traversals"]): return False
        al = {x for p in B["r"].lane_paths(d["edges"], d["lane"]) for x in p}
        return not ((al - {d["lanes"][0]}) & side_set)
    used = {l for d in routes if usable(d) for l in d["lanes"]}
    miss_links = {(a, b) for (a, b), m in B["r"].movements.items() if a not in used and m not in set(B["zero"])}
    if miss_links:
        more, _ = route_pool(B, k=60, per_mv=10, only={B["r"].movements[l] for l in miss_links}, only_links=miss_links)
        seen = {tuple(d["lanes"]) for d in routes}
        add = [d for d in more if tuple(d["lanes"]) not in seen and any(a in d["lanes"] for a, b in miss_links) and usable(d)]
        routes += add
        still = {(a, b) for a, b in miss_links if a not in {l for d in routes if usable(d) for l in d["lanes"]}}
        for a, b in sorted(still):
            d = construct_link(B, a, b, side_set)
            if d is not None and usable(d): routes.append(d); add.append(d); print("  构造:", a, "→", [t[0][:12] + " " + t[1] for t in d["traversals"]], "让行", len(d["yield_links"]))
        print("按车道补路线:", sorted({a for a, b in miss_links}), "补了", len(add), "条; 仍未覆盖:", sorted({a for a, b in miss_links} - {l for d in routes if usable(d) for l in d["lanes"]}))
    # 穷举可行、但从现有入口出发的唯一路线要绕街区重复经过同一车道的转向: 放宽支路入口规则 (允许入口到 RL 进口之间有一次让行连接),
    # 为它单独选一条例外支路入口并构造路线 (用户 2026-10-28: "可以自己构建车辆")
    r = B["r"]; relaxed = {}
    for m in list(uncovered):
        best = None
        for (a, b), mm in r.movements.items():
            if mm != m: continue
            for lp in r.g.predecessors(a):
                if lp.startswith("__"): continue
                L = r.net.getLane(lp); e = L.getEdge()
                if e.getFromNode().getID() in B["meta"] or e.getToNode().getID() in B["meta"] or L.getLength() < 40: continue
                if L.getLength() + r.net.getLane(a).getLength() < 60: continue
                if best is None or L.getLength() > best[1]: best = (lp, L.getLength())
        if best:
            B["sources"] = set(B["sources"]) | {best[0]}; B["side_lanes"] = sorted(set(B["side_lanes"]) | {best[0]})
            d = construct(B, m)
            if d is not None:
                routes.append(d); per[m] += 1; relaxed[f"{m[0]}|{m[1]}"] = best[0]
                print("例外支路入口:", best[0], "→", m, "经过 RL", [t[0][:12] for t in d["traversals"]], "让行", len(d["yield_links"]))
    uncovered = sorted(m for m in allm if per[m] == 0 and m not in set(B["zero"]))
    zero_loop = uncovered
    print("候选路线", len(routes), "; 未覆盖 (非零需求) 转向:", uncovered)
    C.dump(dict(boundary=B["boundary"], sinks=B["sinks"], side=B["side"], side_lanes=B["side_lanes"], side_relaxed=relaxed, side_candidates=B["cands"],
                need_side=[list(m) for m in B["need_side"]], zero=[list(m) for m in B["zero"]], zero_lane_loop=[list(m) for m in zero_loop], search_states=B["states"],
                lane_cap_vph=C.LANE_CAP_VPH, min_headway_s=C.MIN_HEADWAY), "entries.json")
    C.dump(dict(routes=routes), "route_pool.json")

if __name__ == "__main__":
    main()
