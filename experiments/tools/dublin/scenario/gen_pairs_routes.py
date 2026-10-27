"""实验车辆 (50 对公交/救护车) 路线与时刻表生成器 (设计文档 SCENARIO_PRIORITY_PAIRS50_DESIGN §3, 2026-10-27)。
路线: 从 18h 真实小汽车路由里抽完整路线 (每条经过 ≥ 3 个 RL 路口), 贪心集合覆盖 "必覆盖转向" 清单 (--must), 优先不经过非 RL 让行口 (非 RL 节点上 state 'm' 的连接),
每个路口 ≥ 2 条, 两类车分配逐条相同。车道一致: 路线从 18h 车辆自带的 departLane 起不换道可走完 (生成时核对)。
时刻表: 第一对 t0 出发, 之后每 gap 秒一对; 对内同路线错开 offset 秒, 奇数对公交先、偶数对救护车先; id 固定 pair_kk_bus / pair_kk_amb; departSpeed=max。
输出: pairs50_routes.rou.xml (路线定义), pairs50_tt<k>.rou.xml (时刻表, k = --tt-seed), pairs50_unseen_routes.rou.xml (未见路线 + 一份时刻表), pairs_report.json。
用法: python gen_pairs_routes.py --scenario nets/dublin/weekday_pairs50 [--n-routes 10] [--pairs 50] [--tt-seed 1]"""
import argparse, json, os, random, collections, xml.etree.ElementTree as ET
import sumolib

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="nets/dublin/weekday_pairs50"); ap.add_argument("--net", default="nets/dublin/dublin_8std_splitslot.net.xml")
ap.add_argument("--meta", default="nets/dublin/dublin_8std_splitslot_meta.json"); ap.add_argument("--car-routes", default="nets/dublin/weekday_18h_reroute_v2/car_weekday_18h.rou.xml")
ap.add_argument("--n-routes", type=int, default=10); ap.add_argument("--max-routes", type=int, default=12); ap.add_argument("--pairs", type=int, default=50)
ap.add_argument("--t0", type=float, default=60.0); ap.add_argument("--gap", type=float, default=50.0); ap.add_argument("--offset", type=float, default=90.0)
ap.add_argument("--route-seed", type=int, default=1); ap.add_argument("--tt-seed", type=int, default=1); ap.add_argument("--min-rl", type=int, default=3)
a = ap.parse_args()
net = sumolib.net.readNet(a.net); meta = json.load(open(a.meta))["tls"]; rl = set(meta)
MUST = [("1430847303", "W-T"), ("659802", "S-R|S-T"), ("1448436", "S-R"), ("26165126", "W-R"), ("133307283", "N-R"), ("389281", "W-R"), ("389332", "W-R"),
        ("cluster_1448439_2464758361", "S-R"), ("cluster_1998326407_2876012598_4781977262_658429", "N-T"), ("cluster_2498303858_3624079904_6141082777_6141082778_9936781", "S-R"),
        ("cluster_5166592137_5166592140_557198554_57744042", "W-R"), ("5166592132", "S-R"), ("32236377", "S-L"),
        ("1448442", "S-L|N-R|E-T|N-T"), ("cluster_21620852_2383668110_2383668118_4009885977", "S-L|S-R|W-T|S-T"), ("cluster_135109528_9101656", "E-T|S-T|N-R|S-L"),
        ("135109534", "E-T|S-R"), ("26165090", "E-R|N-T")]
conn_mv = {}; minor_nodes = set()
for tid, t in meta.items():
    link_slot = {int(i): grp[0]["approach"] + "-" + grp[0]["turn"] for i, grp in t["links"].items()}
    for fl, tl, idx in net.getTLS(tid).getConnections():
        if idx in link_slot: conn_mv[(fl.getID(), tl.getID())] = (tid, link_slot[idx])

def analyse(edges, depart_lane):
    """沿路线不换道追踪车道; 返回 (车道一致?, [(tid, mv)], 经过的非 RL 让行连接数, 经过的非 RL 节点列表)"""
    lanes = {depart_lane}; trav = []; yields = 0; nodes = []
    for e1, e2 in zip(edges, edges[1:]):
        ea, eb = net.getEdge(e1), net.getEdge(e2); nxt = set(); node = ea.getToNode(); got = False
        for c in ea.getOutgoing().get(eb, []):
            if c.getFromLane().getIndex() in lanes:
                nxt.add(c.getToLane().getIndex()); got = True
                if node.getType() == "traffic_light":
                    k = (c.getFromLane().getID(), c.getToLane().getID())
                    if k in conn_mv and conn_mv[k] not in trav: trav.append(conn_mv[k])
                else:
                    if c.getState() == "m": yields += 1
        if not got: return False, trav, yields, nodes
        if node.getType() != "traffic_light": nodes.append(node.getID())
        lanes = nxt
    return True, trav, yields, nodes

root = ET.parse(a.car_routes).getroot(); pool = []
for v in root.iter("vehicle"):
    edges = v.find("route").get("edges").split(); dl = int(v.get("departLane", 0))
    ok, trav, yields, nodes = analyse(edges, dl)
    if ok and len({t for t, _ in trav}) >= a.min_rl:
        pool.append(dict(edges=edges, depart_lane=dl, trav=trav, yields=yields, nodes=nodes, src=v.get("id")))
# 去重 (同路线同 departLane)
uniq = {};
for r in pool: uniq.setdefault((tuple(r["edges"]), r["depart_lane"]), r)
pool = list(uniq.values()); print(f"18h 小汽车路由里 ≥{a.min_rl} 个 RL 路口且车道一致的不同路线 {len(pool)} 条 (不经让行口的 {sum(1 for r in pool if r['yields']==0)})")
def satisfies(r, must):
    tid, alts = must; return any((tid, m) in r["trav"] for m in alts.split("|"))
# 贪心覆盖: 每轮选覆盖未满足 must 最多、让行口最少、路口最多的路线
rng = random.Random(a.route_seed); chosen = []; unmet = list(MUST); junc_count = collections.Counter()
while len(chosen) < a.max_routes:
    def score(r):
        return (sum(1 for m in unmet if satisfies(r, m)), -r["yields"], len({t for t, _ in r["trav"]}), rng.random())
    best = max(pool, key=score)
    if score(best)[0] == 0 and len(chosen) >= a.n_routes: break
    if best in chosen: break
    chosen.append(best); pool.remove(best); unmet = [m for m in unmet if not satisfies(best, m)]
    for t, _ in best["trav"]: junc_count[t] += 1
    if len(chosen) >= a.n_routes and not unmet: break
# 路口 ≥ 2 条的补充
for tid in meta:
    if junc_count[tid] < 2 and len(chosen) < a.max_routes:
        c = [r for r in pool if any(t == tid for t, _ in r["trav"])]
        if c:
            best = max(c, key=lambda r: (-r["yields"], len({t for t, _ in r["trav"]}))); chosen.append(best); pool.remove(best)
            for t, _ in best["trav"]: junc_count[t] += 1
def synth_routes(must, max_depth=6, max_yields=0):
    """为 18h 路由里没有的转向合成路线: 从该运动的连接出发, 向上游/下游沿连接扩展 (非 RL 节点只走 'M', RL 节点任意), 直到 ≥ min_rl 个 RL 路口且起点边 ≥ 40 m。"""
    tid, alts = must; outs = []
    for (fl_id, tl_id), key in conn_mv.items():
        if key[0] != tid or key[1] not in alts.split("|"): continue
        fl, tl = net.getLane(fl_id), net.getLane(tl_id)
        def up(edge, lane_idx, depth):
            res = [([edge.getID()], lane_idx)] if edge.getLength() >= 40 else []
            if depth == 0: return res
            for ue, cs in edge.getIncoming().items():
                for c in cs:
                    if c.getToLane().getIndex() != lane_idx: continue
                    if edge.getFromNode().getType() != "traffic_light" and c.getState() != "M": continue
                    for pre, dl in up(ue, c.getFromLane().getIndex(), depth - 1): res.append((pre + [edge.getID()], dl))
            return res
        def down(edge, lane_idx, depth):
            res = [[edge.getID()]]
            if depth == 0: return res
            for de, cs in edge.getOutgoing().items():
                for c in cs:
                    if c.getFromLane().getIndex() != lane_idx: continue
                    if edge.getToNode().getType() != "traffic_light" and c.getState() != "M": continue
                    for suf in down(de, c.getToLane().getIndex(), depth - 1): res.append([edge.getID()] + suf)
            return res
        for pre, dl in up(fl.getEdge(), fl.getIndex(), max_depth):
            for suf in down(tl.getEdge(), tl.getIndex(), max_depth):
                edges = pre + suf; ok, trav, yields, nodes = analyse(edges, dl)
                if ok and yields <= max_yields and len({t for t, _ in trav}) >= a.min_rl:
                    outs.append(dict(edges=edges, depart_lane=dl, trav=trav, yields=yields, nodes=nodes, src=f"synth:{tid[:12]}:{key[1]}"))
    # 路口数多、边数少的优先
    outs.sort(key=lambda r: (r["yields"], -len({t for t, _ in r["trav"]}), len(r["edges"])))
    return outs[:1]
for m in list(unmet):
    syn = synth_routes(m) or synth_routes(m, max_depth=8, max_yields=1)     # 先不经让行口; 不行再允许 1 处
    if syn and len(chosen) < a.max_routes + 2:
        chosen.append(syn[0]); unmet.remove(m)
        for t, _ in syn[0]["trav"]: junc_count[t] += 1
        print(f"  合成路线覆盖 {m}: RL 路口 {len({t for t,_ in syn[0]['trav']})}, 边数 {len(syn[0]['edges'])}")
unseen = []
for r in sorted(pool, key=lambda r: (r["yields"], -len({t for t, _ in r["trav"]}))):
    if len(unseen) >= a.n_routes: break
    trav_r = set(r["trav"])
    if all(len(trav_r & set(c["trav"])) / max(len(trav_r | set(c["trav"])), 1) < 0.5 for c in chosen) and not any(tuple(r["edges"]) == tuple(c["edges"]) for c in chosen): unseen.append(r)
os.makedirs(a.scenario, exist_ok=True)
def write_routes(path, routes, prefix):
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<!-- 实验车共享路线 (gen_pairs_routes.py, route_seed=%d): 来自 18h 小汽车路由, 车道一致 -->\n<routes>\n' % a.route_seed)
        for i, r in enumerate(routes, 1):
            fh.write('    <route id="%s%02d" edges="%s"/>  <!-- 来自 %s departLane %d, RL 路口 %d, 让行连接 %d -->\n' % (prefix, i, " ".join(r["edges"]), r["src"], r["depart_lane"], len({t for t, _ in r["trav"]}), r["yields"]))
        fh.write("</routes>\n")
def write_tt(path, routes, prefix, seed, extra_comment=""):
    rng = random.Random(seed); order = []
    for k in range(a.pairs):
        order.append((k, routes[k % len(routes)], k % len(routes)))
    rng.shuffle(order)   # 哪一对走哪条路线的次序按种子打乱 (每条路线仍恰好 pairs/len(routes) 对)
    vehicles = []
    for slot, (k, r, ri) in enumerate(order):
        t_first = a.t0 + slot * a.gap; bus_first = (slot % 2 == 0)
        tb = t_first if bus_first else t_first + a.offset; ta = t_first + a.offset if bus_first else t_first
        vehicles.append(dict(id=f"pair_{slot+1:02d}_bus", type="bus", depart=tb, route=f"{prefix}{ri+1:02d}", lane=r["depart_lane"]))
        vehicles.append(dict(id=f"pair_{slot+1:02d}_amb", type="ambulance", depart=ta, route=f"{prefix}{ri+1:02d}", lane=r["depart_lane"]))
    vehicles.sort(key=lambda v: v["depart"])
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<!-- 实验车时刻表 (gen_pairs_routes.py tt_seed=%d): %d 对, t0=%g gap=%g offset=%g, 奇偶对交替先后; 路线见 %s%s -->\n<routes>\n' % (seed, a.pairs, a.t0, a.gap, a.offset, prefix, extra_comment))
        for v in vehicles:
            fh.write('    <vehicle id="%s" type="%s" depart="%.1f" departLane="%d" departSpeed="max" route="%s"/>\n' % (v["id"], v["type"], v["depart"], v["lane"], v["route"]))
        fh.write("</routes>\n")
    return vehicles
write_routes(os.path.join(a.scenario, "pairs50_routes.rou.xml"), chosen, "r")
tt = write_tt(os.path.join(a.scenario, f"pairs50_tt{a.tt_seed}.rou.xml"), chosen, "r", a.tt_seed)
if unseen:
    write_routes(os.path.join(a.scenario, "pairs50_unseen_routes.rou.xml"), unseen, "u")
    write_tt(os.path.join(a.scenario, "pairs50_unseen_tt1.rou.xml"), unseen, "u", 1001, " (未见路线)")
cov = collections.Counter();
for r in chosen:
    for k in r["trav"]: cov[k] += 1
rep = dict(route_seed=a.route_seed, tt_seed=a.tt_seed, n_routes=len(chosen), unmet_must=[list(m) for m in unmet], last_depart=max(v["depart"] for v in tt),
           routes=[dict(id=f"r{i:02d}", src=r["src"], depart_lane=r["depart_lane"], n_rl=len({t for t, _ in r["trav"]}), yields=r["yields"], non_rl_nodes=r["nodes"], traversals=[list(k) for k in r["trav"]]) for i, r in enumerate(chosen, 1)],
           junction_routes={t: junc_count[t] for t in meta}, movement_coverage={f"{k[0]}|{k[1]}": v for k, v in cov.items()}, unseen=[dict(src=r["src"], n_rl=len({t for t, _ in r["trav"]}), yields=r["yields"]) for r in unseen])
json.dump(rep, open(os.path.join(a.scenario, "pairs_report.json"), "w"), indent=1, ensure_ascii=False)
print(f"选 {len(chosen)} 条路线; 未满足的必覆盖: {unmet}; 路口覆盖条数: " + ", ".join(f"{t[:14]}:{junc_count[t]}" for t in meta))
for i, r in enumerate(chosen, 1): print(f"  r{i:02d} 来自 {r['src']}: RL 路口 {len({t for t,_ in r['trav']})}, 让行连接 {r['yields']}, 非 RL 节点 {len(r['nodes'])}, 转向 {[f'{t[:10]}:{m}' for t,m in r['trav']]}")
print(f"时刻表: {len(tt)} 辆, 最后出发 {max(v['depart'] for v in tt):.0f} s; 未见路线 {len(unseen)} 条")
