"""任务 5: GTFS 18 点公交投影到 NET18B。
每辆车: 取原路线在新网上可不换道走通的最长连续段 → 用车道图路由器延伸到边界入口与边界出口 → 若重访 RL 路口, 截断到第一次经过之后再接出口。
班次浮动起始 1.0 (全部保留); 7 份时刻表各做 ±60 s 抖动 (固定种子)。"""
import collections, json, random, xml.etree.ElementTree as ET
import common18 as C

SRC = C.NETS / "weekday_18h_15_southclosed_corridor/bus_weekday_18h.rou.xml"

def lane_ok_paths(r, edges):
    """返回每个出发车道能否不换道走完 edges。"""
    out = {}
    e0 = r.net.getEdge(edges[0])
    for l in e0.getLanes():
        if r.lane_paths(edges, l.getIndex()):
            out[l.getIndex()] = True
    return out

def project(r, edges):
    have = [e for e in edges if r.net.hasEdge(e)]
    best = None
    # 最长的连续可通行子段
    n = len(edges)
    for i in range(n):
        for j in range(n, i, -1):
            seg = edges[i:j]
            if not all(r.net.hasEdge(e) for e in seg): continue
            if any(r.net.getEdge(a) not in r.net.getEdge(b).getIncoming() for a, b in zip(seg, seg[1:])): continue
            lanes = lane_ok_paths(r, seg)
            if lanes:
                if best is None or len(seg) > len(best[0]): best = (seg, min(lanes))
                break
    return best

def rl_revisit_cut(d):
    seen = set(); keep = None
    for k, (tid, mv) in enumerate(d["traversals"]):
        if tid in seen: return k
        seen.add(tid)
    return None

def main():
    from routes18 import side_rule_ok
    C.check_hashes(); r = C.load_router(zonal=False); meta = C.load_meta(); ag = C.always_green(meta)
    out = []; report = collections.Counter(); changes = []
    vehs = []
    for v in ET.parse(SRC).getroot().iter("vehicle"):
        edges = v.find("route").get("edges").split(); b = project(r, edges); vehs.append((v, edges, b))
    # 公交原出发边满足支路入口静态规则的, 作为公交入口保留 (真实公交从模型外街道进入, 不往上游补经过 RL 路口的前段)
    bus_entries = set(); rl_nodes = set(meta)
    def own_path_ok(seg, lane):
        """沿公交自己的车道路径: 出发边两端非 RL、车道 ≥40 m、到第一个 RL 进口之间无让行连接、距第一个 RL 停车线 ≥60 m。"""
        e0 = r.net.getEdge(seg[0]); L0 = e0.getLanes()[lane]
        if e0.getFromNode().getID() in rl_nodes or e0.getToNode().getID() in rl_nodes or L0.getLength() < 40: return False
        paths = r.lane_paths(seg, lane)
        if not paths: return False
        p = paths[0]; d = 0.0
        for x, y in zip(p, p[1:] + [None]):
            d += r.net.getLane(x).getLength()
            if r.net.getLane(x).getEdge().getToNode().getID() in rl_nodes: return d >= 60
            if y is not None and r.g.has_edge(x, y) and r.g[x][y].get("minor"): return False
        return False
    try: excluded = set(json.load(open(C.WORK / "bus_entry_excluded.json")))   # 按全部车型核对 "上游穿行 ≤ 本地发车 10%" 不合规的原出发边
    except FileNotFoundError: excluded = set()
    for v, edges, b in vehs:
        if b is None: continue
        seg, lane = b; l0 = f"{seg[0]}_{lane}"
        if l0 not in r.sources and l0 not in excluded and own_path_ok(seg, lane): bus_entries.add(l0)
    for l0 in bus_entries:
        r.sources.add(l0); r.g.add_edge("__SOURCE__", l0, weight=r.net.getLane(l0).getLength())
    r.prefixes.cache_clear()
    for v, edges, b in vehs:
        if b is None: report["dropped_no_segment"] += 1; changes.append(dict(id=v.get("id"), change="dropped")); continue
        seg, lane = b
        try: d = r.extend(seg, lane)
        except ValueError: report["dropped_extend"] += 1; changes.append(dict(id=v.get("id"), change="dropped_extend")); continue
        cut = rl_revisit_cut(d)
        if cut is not None:
            # 截断: 保留到第 cut 个 RL 经过之前那段的出口边, 再接出口
            tr_edges = []; k = 0
            for x, y in zip(d["lanes"], d["lanes"][1:]):
                tr_edges.append(x)
                if (x, y) in r.movements:
                    k += 1
                    if k == cut: break
            core = [l.rsplit("_", 1)[0] for l in tr_edges]
            core = [e for i, e in enumerate(core) if i == 0 or core[i - 1] != e]
            d2 = None
            for cand in [core[:-1], core[:-2]]:
                if len(cand) < 2: continue
                try:
                    dd = r.extend(cand, d["lane"])
                    if rl_revisit_cut(dd) is None: d2 = dd; break
                except ValueError: pass
            if d2 is None: report["dropped_revisit"] += 1; changes.append(dict(id=v.get("id"), change="dropped_revisit")); continue
            d = d2; report["truncated_revisit"] += 1; changes.append(dict(id=v.get("id"), change="truncated_revisit"))
        if d["edges"] != edges: report["route_changed"] += 1
        assert not any(l in d["lanes"][1:] for l in [d["lanes"][0]])
        out.append(dict(id=v.get("id"), line=v.get("id").split("_")[1], depart=float(v.get("depart")), edges=d["edges"], lane=d["lane"],
                        lanes=d["lanes"], traversals=[list(t) for t in d["traversals"]], minor=d["minor"], entry=d["lanes"][0]))
    # 竞争运动接触: 每个路口是否有公交经过非恒绿转向
    comp = collections.defaultdict(int)
    for b in out:
        for tid, mv in b["traversals"]:
            if (tid, mv) not in ag: comp[tid] += 1
    no_comp = sorted(t for t in meta if comp[t] == 0)
    # 7 份时刻表: ±60 s 抖动
    tts = {}
    for k in range(1, C.N_TT + 1):
        rng = random.Random(1000 + k)
        tts[k] = {b["id"]: round(min(3500.0, max(0.0, b["depart"] + rng.uniform(-60, 60))), 1) for b in out}
    C.dump(dict(buses=out, timetables=tts, bus_side_entries=sorted(bus_entries), bus_entries_excluded=sorted(excluded), report=dict(report), changes=changes, competing_visits_by_junction=dict(comp), junctions_without_bus_on_competing=no_comp,
                lines=dict(collections.Counter(b["line"] for b in out)), float_factor=1.0), "buses.json")
    print("公交", len(out), "/ 262;", dict(report)); print("公交支路入口 (保留原出发边):", sorted(bus_entries)); print("各路口公交经过竞争转向次数:", {t.replace('cluster_','c')[:12]: comp[t] for t in meta}); print("没有公交经过竞争转向的路口:", no_comp)

if __name__ == "__main__":
    main()
