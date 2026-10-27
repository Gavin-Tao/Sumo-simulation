"""背景车流生成器 (设计文档 SCENARIO_PRIORITY_PAIRS50_DESIGN §2.2–2.3; 2026-10-28 第二版, 按外部复核修正)。
读逐路口逐转向需求表 (demand_targets.json), 按路网连接生成合法、车道一致 (不换道也能走完) 的路线, 每辆车显式 departLane, departSpeed=max,
出发时刻 0–3600 s 均匀随机 (固定种子), vType=car; 一条路线经过几个 RL 路口就贡献几个 (路口, 转向); 用非负最小二乘 (NNLS) 在候选路线上分配车数
(**只用 NNLS, 不再贪心补流** —— 第一版 NNLS 之后误跑了贪心, 多加 70 辆); 输出实际逐转向需求与偏差清单 (统一阈值 ±15%)。
路线规则: 起点边 ≥ MINLEN (否则向上游回溯, 非 RL 节点只走有路权连接 'M', 向上游最多再经过 max_cross 个 RL 路口); **路线结尾必须穿过下游 RL 路口直到
边的下游节点不是 RL 路口** —— 到深度上限或接不下去仍通往 RL 路口的候选一律丢弃 (第一版有 48 辆车结束在 RL 进口边上)。
用法: python gen_background_demand.py --scenario nets/dublin/weekday_pairs50_14 --net ... --meta ... [--seed 1] [--minlen 40]"""
import argparse, json, os, random, collections
import numpy as np
import sumolib
from scipy.optimize import nnls

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", required=True); ap.add_argument("--net", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--targets", default=None)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--minlen", type=float, default=40.0); ap.add_argument("--max-ext", type=int, default=4)
ap.add_argument("--max-down", type=int, default=4, help="路线结尾向下游最多再穿过几个 RL 路口"); ap.add_argument("--max-cross", type=int, default=1, help="向上游最多再经过几个别的 RL 路口 (无候选的转向自动放宽到 +1)")
ap.add_argument("--extra-weight", type=float, default=0.3, help="需求表外转向在 NNLS 里的权重 (目标 0)"); ap.add_argument("--max-cands", type=int, default=60)
ap.add_argument("--out", default="background_cars.rou.xml"); ap.add_argument("--end", type=float, default=3600.0); ap.add_argument("--tol", type=float, default=0.15)
a = ap.parse_args()
net = sumolib.net.readNet(a.net); meta = json.load(open(a.meta))["tls"]
targets = json.load(open(a.targets or os.path.join(a.scenario, "demand_targets.json")))["targets"]
for tid in targets: assert tid in meta, f"需求表里的路口 {tid} 不在 meta 里"

conn_mv = {}                                     # (from_lane_id, to_lane_id) -> (tid, movement)
for tid, t in meta.items():
    link_slot = {int(i): grp[0]["approach"] + "-" + grp[0]["turn"] for i, grp in t["links"].items()}
    for fl, tl, idx in net.getTLS(tid).getConnections():
        if idx in link_slot: conn_mv[(fl.getID(), tl.getID())] = (tid, link_slot[idx])
mv_conns = collections.defaultdict(list)
for (f, t), key in conn_mv.items(): mv_conns[key].append((net.getLane(f), net.getLane(t)))

def extend_upstream(edge, lane_idx, ext_left, crossed, max_cross):
    out = []
    if edge.getLength() >= a.minlen or ext_left == 0:
        out.append(([edge.getID()], lane_idx, []))
        if edge.getLength() >= a.minlen: return out
    node = edge.getFromNode()
    for up_edge, conns in edge.getIncoming().items():
        for c in conns:
            if c.getToLane().getIndex() != lane_idx: continue
            if node.getType() == "traffic_light":
                key = (c.getFromLane().getID(), c.getToLane().getID())
                if key not in conn_mv or len(crossed) >= max_cross: continue
                trav = [conn_mv[key]]
            else:
                if c.getState() != "M": continue
                trav = []
            for pre, dl, tr in extend_upstream(up_edge, c.getFromLane().getIndex(), ext_left - 1, crossed + trav, max_cross):
                out.append((pre + [edge.getID()], dl, tr + trav))
    return out

def extend_downstream(edge, lane_idx, depth):
    """返回 [(edges_suffix, traversals)]; 只返回结尾边通往非 RL 节点的后缀, 否则空列表 (该候选作废)。"""
    node = edge.getToNode()
    if node.getType() != "traffic_light": return [([edge.getID()], [])]
    if depth == 0: return []
    out = []
    for de, conns in edge.getOutgoing().items():
        for c in conns:
            if c.getFromLane().getIndex() != lane_idx: continue
            key = conn_mv.get((c.getFromLane().getID(), c.getToLane().getID()))
            if key is None: continue
            for suf, tr in extend_downstream(de, c.getToLane().getIndex(), depth - 1):
                out.append(([edge.getID()] + suf, [key] + tr))
    return out

cands = {}
for key, conns in mv_conns.items():
    lst = []
    for mc in (a.max_cross, a.max_cross + 1):
        for fl, tl in conns:
            f = fl.getEdge(); t = tl.getEdge()
            for pre, dl, tr in extend_upstream(f, fl.getIndex(), a.max_ext, [], mc):
                if net.getEdge(pre[0]).getLength() < a.minlen: continue
                for suf, trd in extend_downstream(t, tl.getIndex(), a.max_down):
                    lst.append(dict(edges=pre + suf, depart_lane=dl, traversals=sorted(set(tr + trd + [key])), start_len=net.getEdge(pre[0]).getLength(), lane=fl.getIndex()))
        if lst: break
    seen = set(); uniq = []
    for c in sorted(lst, key=lambda c: (len(c["traversals"]), -c["start_len"], len(c["edges"])))[:a.max_cands]:
        k = (tuple(c["edges"]), c["depart_lane"])
        if k in seen: continue
        seen.add(k); uniq.append(c)
    cands[key] = uniq
for key, lst in cands.items():
    for c in lst: assert net.getEdge(c["edges"][-1]).getToNode().getType() != "traffic_light", c

jtag = {tid: tid.replace("cluster_", "c")[:11] for tid in meta}; assert len(set(jtag.values())) == len(jtag)
target_of = {(tid, mv): float(n) for tid, d in targets.items() for mv, n in d.items()}
missing = [k for k in target_of if not cands.get(k)]
routes = []; rkeys = {}
for key, lst in cands.items():
    for c in lst:
        rk = (tuple(c["edges"]), c["depart_lane"])
        if rk in rkeys: continue
        rkeys[rk] = len(routes); routes.append(dict(edges=list(c["edges"]), lane=c["depart_lane"], trav=c["traversals"], key=key))
mv_all = sorted({k for r in routes for k in r["trav"]} | set(target_of))
idx = {k: i for i, k in enumerate(mv_all)}; A = np.zeros((len(mv_all), len(routes))); b = np.zeros(len(mv_all)); w = np.ones(len(mv_all))
for j, r in enumerate(routes):
    for k in r["trav"]: A[idx[k], j] = 1.0
for k, v in target_of.items(): b[idx[k]] = v
for k in mv_all:
    if k not in target_of: w[idx[k]] = a.extra_weight
x, res = nnls(A * w[:, None], b * w, maxiter=50000)
counts = np.floor(x + 0.5).astype(int)
rng = random.Random(a.seed); vehicles = []; realized = collections.Counter(); route_use = collections.Counter()
for j, r in enumerate(routes):
    n = int(counts[j])
    if n <= 0: continue
    key = r["key"]
    for i in range(n):
        vehicles.append(dict(id=f"bg_{jtag[key[0]]}_{key[1]}_{route_use[key]}_{i}", depart=rng.uniform(0, a.end), edges=r["edges"], lane=r["lane"]))
    route_use[key] += 1
    for k in r["trav"]: realized[k] += n
vehicles.sort(key=lambda v: v["depart"])
os.makedirs(a.scenario, exist_ok=True)
with open(os.path.join(a.scenario, a.out), "w") as fh:
    fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<!-- 人造背景车流: gen_background_demand.py (第二版, 只用 NNLS) seed=%d minlen=%g net=%s; 需求表 demand_targets.json; vType 见 vtypes_same.add.xml -->\n<routes>\n' % (a.seed, a.minlen, os.path.basename(a.net)))
    for v in vehicles:
        fh.write('    <vehicle id="%s" type="car" depart="%.1f" departLane="%d" departSpeed="max">\n        <route edges="%s"/>\n    </vehicle>\n' % (v["id"], v["depart"], v["lane"], " ".join(v["edges"])))
    fh.write("</routes>\n")
rows = [(tid, mv, n, realized[(tid, mv)]) for tid, d in targets.items() for mv, n in d.items()]
dev = [x for x in rows if x[2] and abs(x[3] - x[2]) / x[2] > a.tol]
end_rl = sum(1 for v in vehicles if net.getEdge(v["edges"][-1]).getToNode().getType() == "traffic_light")
rep = {"version": 2, "seed": a.seed, "minlen": a.minlen, "net": a.net, "n_vehicles": len(vehicles), "nnls_routes": len(routes), "nnls_used": int((counts > 0).sum()), "nnls_residual": float(res),
       "target_total": int(sum(x[2] for x in rows)), "realized_total": int(sum(x[3] for x in rows)), "missing_candidates": [list(k) for k in missing],
       "deviation_gt_tol": [dict(tid=x[0], mv=x[1], target=x[2], realized=x[3]) for x in dev], "tol": a.tol,
       "realized": {f"{x[0]}|{x[1]}": dict(target=x[2], realized=x[3]) for x in rows}, "extra_traversals": {f"{k[0]}|{k[1]}": v for k, v in realized.items() if k not in target_of},
       "vehicles_ending_on_rl_approach": end_rl, "routes_by_n_rl": dict(collections.Counter(len(r["trav"]) for r in routes))}
json.dump(rep, open(os.path.join(a.scenario, "background_report.json"), "w"), indent=1, ensure_ascii=False)
print(f"车辆 {len(vehicles)} (NNLS 候选 {len(routes)} 用到 {rep['nnls_used']} 残差 {res:.1f}); 目标合计 {rep['target_total']} 实现 {rep['realized_total']}; 无候选 {len(missing)}; 偏差>{a.tol:.0%} 的转向 {len(dev)}/{len(rows)}; 结束在 RL 进口边的车 {end_rl}")
for k in missing: print("  无候选:", k)
for g in dev: print("  偏差:", g)
print("候选路线按经过 RL 路口数:", rep["routes_by_n_rl"], " 表外过境:", rep["extra_traversals"])
