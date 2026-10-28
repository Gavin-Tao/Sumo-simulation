"""任务 1: 配对车辆的候选路线池。规则同 18h 救护车选线 (ambulances18.score_route), 放宽 RL 数量门槛:
边界入口、全程无非 RL 让行连接、≥3 个 RL 路口 (不重访)、≥2 个竞争转向 (非恒绿转向、非零需求、进口车道不是车道级恒绿)、
入口到第一个 RL ≥ 100 m; 不穿过小汽车支路入口车道。另记: 入口车道平台期负荷 (18h 定稿, 背景不变所以照用)、分流节点 (本车道到下一条边有多条可选车道, SUMO 自选)。
候选 = 18h 路线池 + 按车道图重新组合的更长路线 (routes18.route_pool, 每转向前后段各取 k 条最短路, 不截断)。"""
import collections, time
import common50 as P
from common50 import C18
from routes18 import build, route_pool
from ambulances18 import first_rl_dist


def ag_lanes(net, meta):
    """车道级恒绿: 该车道所有受控连接在每个有效相位都为绿 (训练环境 always_green_lanes 同义), 不进 per-visit 指标。"""
    out = set()
    for tid, t in meta.items():
        greens = [x["state"] for x in t["actions"].values() if x.get("valid")]
        idx = collections.defaultdict(set)
        for c in net.getTLS(tid).getConnections(): idx[c[0].getID()].add(c[2])
        out |= {l for l, ii in idx.items() if all(all(s[i] in "Gg" for i in ii) for s in greens)}
    return out


def fanout_nodes(net, lanes):
    """路线上 "本车道到下一条边有不止一条可选车道" 的节点 (路线文件管不了 SUMO 在这里选哪条)。"""
    fan = []
    for x, y in zip(lanes, lanes[1:]):
        to = {c.getToLane().getID() for c in net.getLane(x).getOutgoing()}
        if len({l for l in to if l.rsplit("_", 1)[0] == y.rsplit("_", 1)[0]}) > 1:
            fan.append(net.getLane(x).getEdge().getToNode().getID())
    return fan


def main(k=20):
    P.check_inputs(); t0 = time.time()
    B = build(); r = B["r"]; net = r.net; meta = B["meta"]
    zero = {tuple(m) for m in B["zero"]}; boundary = set(B["boundary"])
    pool, _ = route_pool(B, k=k, per_mv=10 ** 9)
    print(f"route_pool k={k}: {len(pool)} 条, {time.time() - t0:.0f} s")
    seen = {}
    for d in pool + C18.load("route_pool.json")["routes"]: seen.setdefault(tuple(d["lanes"]), d)
    load = C18.load("demand_report.json")["entry_plateau_vph"]
    side = set(C18.load("entries.json")["side_lanes"])    # 小汽车支路入口车道: 配对路线不得穿过 (与 18h 小汽车路线同一规则, 避免支路插车插到优先车前面)
    agl = ag_lanes(net, meta); ag_mv = C18.always_green(meta)
    out = []
    for d in seen.values():
        lanes = d["lanes"]
        if lanes[0] not in boundary or d["yield_links"] or set(lanes) & side: continue
        tids = [t for t, _ in d["traversals"]]
        if len(tids) < 3 or len(set(tids)) != len(tids): continue
        comp = [[t, m, x] for x, y in zip(lanes, lanes[1:]) if (x, y) in r.movements for t, m in [tuple(r.movements[(x, y)])]
                if (t, m) not in ag_mv and (t, m) not in zero and x not in agl]
        if len(comp) < 2: continue
        f = first_rl_dist(r, meta, lanes)
        if f < P.FIRST_RL_MIN_M: continue
        out.append(dict(edges=d["edges"], lanes=lanes, traversals=[list(x) for x in d["traversals"]], length=d["length"], entry=lanes[0], entry_type="boundary",
                        yield_links=[], competing=comp, first_rl_m=round(f, 1), entry_load=round(load.get(lanes[0], 0.0), 1), fanout_nodes=fanout_nodes(net, lanes)))
    # 未见路线 (只用于评估) 的候选: 规则放宽到 ≥2 个 RL 路口、≥1 个竞争转向, 其余同上
    unseen_pool = []
    for d in seen.values():
        lanes = d["lanes"]
        if lanes[0] not in boundary or d["yield_links"] or set(lanes) & side: continue
        tids = [t for t, _ in d["traversals"]]
        if len(tids) < 2 or len(set(tids)) != len(tids): continue
        comp = [[t, m, x] for x, y in zip(lanes, lanes[1:]) if (x, y) in r.movements for t, m in [tuple(r.movements[(x, y)])]
                if (t, m) not in ag_mv and (t, m) not in zero and x not in agl]
        if not comp or first_rl_dist(r, meta, lanes) < P.FIRST_RL_MIN_M: continue
        unseen_pool.append(dict(edges=d["edges"], lanes=lanes, traversals=[list(x) for x in d["traversals"]], length=d["length"], entry=lanes[0],
                                competing=comp, entry_load=round(load.get(lanes[0], 0.0), 1), fanout_nodes=fanout_nodes(net, lanes)))
    cov = collections.Counter(t for d in out for t, _, _ in d["competing"])
    P.dump(dict(candidates=out, unseen_pool=unseen_pool, boundary_entries=sorted(boundary), k=k, n_pool=len(seen), coverage=dict(cov),
                uncovered=sorted(set(meta) - set(cov))), "pair_candidates.json")
    print(f"候选 {len(out)} 条 (池 {len(seen)}); 入口负荷 ≤ {P.ENTRY_LOAD_MAX:.0f}: {sum(1 for d in out if d['entry_load'] <= P.ENTRY_LOAD_MAX)}; "
          f"无分流: {sum(1 for d in out if not d['fanout_nodes'])}; 竞争转向覆盖 {len(cov)}/{len(meta)} 路口, 未覆盖 {sorted(set(meta) - set(cov))}")


if __name__ == "__main__":
    main()
