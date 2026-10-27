"""背景车流生成器 (设计文档 SCENARIO_PRIORITY_PAIRS50_DESIGN §2.2–2.3, 2026-10-27)。
读逐路口逐转向需求表 (demand_targets.json), 按路网连接生成合法、车道一致 (不换道也能走完) 的路线, 每辆车显式 departLane, departSpeed=max,
出发时刻 0–3600 s 均匀随机 (固定种子), vType=car; 做需求记账 (一条路线经过几个 RL 路口就贡献几个 (路口, 转向)), 输出实际逐转向需求与缺口清单。
路线规则: 目标运动的进口边 f 长 ≥ MINLEN 则从 f 起步; 否则向上游延伸 (车道回溯: 只用 "上游车道 → 所需进口车道" 的连接), 经过的非 RL 节点必须是本路线有路权
(连接 state 'M'), 经过 RL 路口则记为该路口的一个转向 (最多再经过 1 个 RL 路口); 路线在目标运动的出口边结束 (车辆在边末离网)。
用法: python gen_background_demand.py --scenario nets/dublin/weekday_pairs50 [--seed 1] [--minlen 60] [--out background_cars.rou.xml]"""
import argparse, json, os, random, collections, sys
import sumolib

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="nets/dublin/weekday_pairs50"); ap.add_argument("--net", default="nets/dublin/dublin_8std_splitslot.net.xml")
ap.add_argument("--meta", default="nets/dublin/dublin_8std_splitslot_meta.json"); ap.add_argument("--targets", default=None)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--minlen", type=float, default=40.0); ap.add_argument("--max-ext", type=int, default=4)
ap.add_argument("--max-down", type=int, default=3, help="路线结尾向下游最多再穿过几个 RL 路口 (直到边的下游节点不是 RL 路口)"); ap.add_argument("--max-cross", type=int, default=1, help="路线最多再经过几个别的 RL 路口 (无候选的转向自动放宽到 +1)"); ap.add_argument("--tol", type=float, default=0.15, help="别的路口转向允许的超额比例"); ap.add_argument("--tol2", type=float, default=0.4, help="第二轮补缺口时的超额容忍")
ap.add_argument("--out", default="background_cars.rou.xml"); ap.add_argument("--end", type=float, default=3600.0)
a = ap.parse_args()
net = sumolib.net.readNet(a.net); meta = json.load(open(a.meta))["tls"]
targets = json.load(open(a.targets or os.path.join(a.scenario, "demand_targets.json")))["targets"]
rl = set(meta)

# (from_lane_id, to_lane_id) -> (tid, movement)
conn_mv = {}
for tid, t in meta.items():
    link_slot = {int(i): grp[0]["approach"] + "-" + grp[0]["turn"] for i, grp in t["links"].items()}
    for fl, tl, idx in net.getTLS(tid).getConnections():
        if idx in link_slot: conn_mv[(fl.getID(), tl.getID())] = (tid, link_slot[idx])
mv_conns = collections.defaultdict(list)          # (tid, mv) -> [(from_lane, to_lane)]
for (f, t), key in conn_mv.items(): mv_conns[key].append((net.getLane(f), net.getLane(t)))

def extend_upstream(edge, lane_idx, ext_left, crossed, max_cross):
    """从 edge 的 lane_idx 车道向上游找起点: 返回 [(edges_prefix, depart_lane, extra_traversals)] 候选 (可能多条)。"""
    out = []
    if edge.getLength() >= a.minlen or ext_left == 0:
        out.append(([edge.getID()], lane_idx, []));
        if edge.getLength() >= a.minlen: return out
    node = edge.getFromNode()
    for up_edge, conns in edge.getIncoming().items():
        for c in conns:
            if c.getToLane().getIndex() != lane_idx: continue
            if node.getType() == "traffic_light":
                key = (c.getFromLane().getID(), c.getToLane().getID())
                if key not in conn_mv or len(crossed) >= max_cross: continue        # 最多再经过 max_cross 个 RL 路口
                trav = [conn_mv[key]]
            else:
                if c.getState() != "M": continue                               # 非 RL 节点只走有路权的连接
                trav = []
            for pre, dl, tr in extend_upstream(up_edge, c.getFromLane().getIndex(), ext_left - 1, crossed + trav, max_cross):
                out.append((pre + [edge.getID()], dl, tr + trav))
    return out

def extend_downstream(edge, lane_idx, depth):
    """路线结尾: 若 edge 通往 RL 路口, 继续穿过它 (记一次转向), 直到边的下游节点不是 RL 路口或达到深度; 返回 [(edges_suffix, traversals)]。"""
    node = edge.getToNode()
    if node.getType() != "traffic_light" or depth == 0:
        return [([edge.getID()], [])]
    out = []
    for de, conns in edge.getOutgoing().items():
        for c in conns:
            if c.getFromLane().getIndex() != lane_idx: continue
            key = conn_mv.get((c.getFromLane().getID(), c.getToLane().getID()))
            if key is None: continue
            for suf, tr in extend_downstream(de, c.getToLane().getIndex(), depth - 1):
                out.append(([edge.getID()] + suf, [key] + tr))
    return out or [([edge.getID()], [])]
cands = {}   # (tid, mv) -> [dict(edges, depart_lane, traversals)]
for key, conns in mv_conns.items():
    lst = []
    for mc in (a.max_cross, a.max_cross + 1):          # 无候选时放宽一级
        for fl, tl in conns:
            f = fl.getEdge(); t = tl.getEdge()
            for pre, dl, tr in extend_upstream(f, fl.getIndex(), a.max_ext, [], mc):
                if net.getEdge(pre[0]).getLength() < a.minlen: continue
                for suf, trd in extend_downstream(t, tl.getIndex(), a.max_down):
                    lst.append(dict(edges=pre + suf, depart_lane=dl, traversals=sorted(set(tr + trd + [key])), start_len=net.getEdge(pre[0]).getLength(), lane=fl.getIndex()))
        if lst: break
    # 去重, 少经过别的路口的优先, 起点长的优先
    seen = set(); uniq = []
    for c in sorted(lst, key=lambda c: (len(c["traversals"]), -c["start_len"], len(c["edges"])))[:60]:
        k = (tuple(c["edges"]), c["depart_lane"])
        if k in seen: continue
        seen.add(k); uniq.append(c)
    cands[key] = uniq

jtag = {tid: (tid.replace('cluster_', 'c')[:11]) for tid in meta}      # 车辆 id 里的路口短标签 (唯一)
ap_nnls = True
assert len(set(jtag.values())) == len(jtag), jtag
remaining = {(tid, mv): float(n) for tid, d in targets.items() for mv, n in d.items()}
missing = [k for k in remaining if k not in cands or not cands[k]]
target_of = dict(remaining)
rng = random.Random(a.seed); vehicles = []; realized = collections.Counter(); route_use = collections.Counter()
if ap_nnls:
    # 非负最小二乘: A[运动, 路线] = 该路线经过该运动 (0/1), b = 目标; min ||A x - b||, x ≥ 0; 目标表外的转向也进 A (目标 0, 权重小) 以抑制额外负荷
    import numpy as np
    from scipy.optimize import nnls
    routes = []; rkeys = {}
    for key, lst in cands.items():
        for c in lst:
            rk = (tuple(c["edges"]), c["depart_lane"])
            if rk in rkeys: continue
            rkeys[rk] = len(routes); routes.append(dict(edges=list(c["edges"]), lane=c["depart_lane"], trav=c["traversals"], key=key))
    mv_all = sorted({k for r in routes for k in r["trav"]} | set(remaining))
    idx = {k: i for i, k in enumerate(mv_all)}; A = np.zeros((len(mv_all), len(routes))); b = np.zeros(len(mv_all)); w = np.ones(len(mv_all))
    for j, r in enumerate(routes):
        for k in r["trav"]: A[idx[k], j] = 1.0
    for k, v in remaining.items(): b[idx[k]] = v
    for k in mv_all:
        if k not in remaining: w[idx[k]] = 0.3                       # 表外转向: 目标 0, 弱权重 (允许少量过境)
    x, res = nnls(A * w[:, None], b * w, maxiter=20000)
    counts = np.floor(x + 0.5).astype(int)
    for j, r in enumerate(routes):
        n = int(counts[j]); 
        if n <= 0: continue
        key = r["key"]
        for i in range(n):
            vehicles.append(dict(id=f"bg_{jtag[key[0]]}_{key[1]}_{route_use[key]}_{i}", depart=rng.uniform(0, a.end), edges=r["edges"], lane=r["lane"]))
        route_use[key] += 1
        for k in r["trav"]: remaining[k] -= n; realized[k] += n
    print(f"NNLS: 候选路线 {len(routes)}, 用到 {int((counts > 0).sum())} 条, 残差 {res:.1f}")
    order = []                                                           # 跳过贪心
forced = {k: all(len(c["traversals"]) > 1 for c in cands[k]) for k in cands}      # 候选全都要穿过别的路口的转向先分配
order = sorted(remaining, key=lambda k: (0 if forced.get(k) else 1, len(cands.get(k, [])), -remaining[k]))

for key in order:
    if key not in cands: continue
    lanes = sorted({c["lane"] for c in cands[key]})
    # 按车道平分: 逐车道分配, 每车道用 "不超额" 的候选
    per_lane = {ln: remaining[key] / len(lanes) for ln in lanes}
    for ln in lanes:
        need = per_lane[ln]
        for c in [c for c in cands[key] if c["lane"] == ln]:
            if need <= 0.5: break
            extra = [k2 for k2 in c["traversals"] if k2 != key]
            room = [remaining.get(k2, 0) + a.tol * target_of.get(k2, 0) for k2 in extra]   # 别的路口允许超额 tol
            if any(r <= 0 for r in room): continue
            n = int(round(min([need] + room)))
            if n <= 0: continue
            for i in range(n):
                vehicles.append(dict(id=f"bg_{jtag[key[0]]}_{key[1]}_{route_use[key]}_{i}", depart=rng.uniform(0, a.end), edges=c["edges"], lane=c["depart_lane"]))
            route_use[key] += 1; need -= n
            for k2 in c["traversals"]: remaining[k2] -= n; realized[k2] += n
# 第二轮: 对实现 < 90% 的转向放宽别路口超额容忍 (tol2) 再补
for key in order:
    if key not in cands or remaining[key] <= 0.1 * target_of[key]: continue
    need = remaining[key]
    for c in cands[key]:
        if need <= 0.5: break
        extra = [k2 for k2 in c["traversals"] if k2 != key]
        room = [remaining.get(k2, 0) + a.tol2 * target_of.get(k2, 0) for k2 in extra]
        if any(r <= 0 for r in room): continue
        n = int(round(min([need] + room)))
        if n <= 0: continue
        for i in range(n):
            vehicles.append(dict(id=f"bg_{jtag[key[0]]}_{key[1]}_{route_use[key]}_{i}", depart=rng.uniform(0, a.end), edges=c["edges"], lane=c["depart_lane"]))
        route_use[key] += 1; need -= n
        for k2 in c["traversals"]: remaining[k2] -= n; realized[k2] += n
vehicles.sort(key=lambda v: v["depart"])
os.makedirs(a.scenario, exist_ok=True)
with open(os.path.join(a.scenario, a.out), "w") as fh:
    fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<!-- 人造背景车流: gen_background_demand.py seed=%d minlen=%g; 需求表 demand_targets.json; vType 见 vtypes_same.add.xml -->\n<routes>\n' % (a.seed, a.minlen))
    for v in vehicles:
        fh.write('    <vehicle id="%s" type="car" depart="%.1f" departLane="%d" departSpeed="max">\n        <route edges="%s"/>\n    </vehicle>\n' % (v["id"], v["depart"], v["lane"], " ".join(v["edges"])))
    fh.write("</routes>\n")
# 报告
rows = []
for tid, d in targets.items():
    for mv, n in d.items():
        r = realized[(tid, mv)]; rows.append((tid, mv, n, r, r / n if n else 0))
gaps = [x for x in rows if x[4] < 0.9 or x[4] > 1.15]
rep = {"seed": a.seed, "minlen": a.minlen, "n_vehicles": len(vehicles), "target_total": int(sum(x[2] for x in rows)), "realized_total": int(sum(x[3] for x in rows)),
       "missing_candidates": [list(k) for k in missing], "gaps": [dict(tid=x[0], mv=x[1], target=x[2], realized=x[3]) for x in gaps],
       "realized": {f"{x[0]}|{x[1]}": dict(target=x[2], realized=x[3]) for x in rows},
       "extra_traversals": {f"{k[0]}|{k[1]}": v for k, v in realized.items() if k not in target_of}}
json.dump(rep, open(os.path.join(a.scenario, "background_report.json"), "w"), indent=1, ensure_ascii=False)
print(f"车辆 {len(vehicles)}  目标合计 {rep['target_total']}  实现合计 {rep['realized_total']}  无候选转向 {len(missing)}  缺口/超额转向 {len(gaps)}")
for k in missing: print("  无候选:", k)
for g in gaps: print("  缺口/超额:", g)
cover = collections.Counter(len(c["traversals"]) for key in cands for c in cands[key]); print("候选路线按经过 RL 路口数:", dict(cover))
