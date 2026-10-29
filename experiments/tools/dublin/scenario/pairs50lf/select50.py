"""任务 2: 实测选线。每条候选放探针车 (每集 10 辆, 900–2400 s 均匀出发, 与背景及其他探针车联合找空档),
在冻结的背景上用固定轮转与排队驱动各跑 3 个种子 (标定用, 与第 7–13 份无关), 记录: 机会占比 (固定轮转)、
实际车道轨迹、第 99 分位行程时间、非 RL 停车、出发延迟; 中性条件下与小汽车的等待差只报告。
选线函数 choose() 只读 SELECT_FIELDS (不读任何等待结果): 覆盖优先的贪心, K = 10 (不足时如实报告)。"""
import collections, json, os, random, subprocess, sys
import numpy as np
import common50 as P

PY = sys.executable; CAL = str(P.TOOLS18 / "calib18.py"); D = P.WORK / "select"
MAX_PER_ENTRY = 3   # 同一入口最多 3 条路线: 分散优先车负荷, 避免一条走廊上优先车扎堆
MAX_PAIRS_PER_LANE = 14   # 每条竞争进口车道上的配对数上限 (第二版 18 对挤过 1448442 的 58 m 短进口 24349650, 固定轮转下回溢到上游非 RL 车道)
MAX_PER_APPROACH = 3   # 同一条竞争进口车道最多 3 条路线经过 (第一版 6 条路线挤过 1448442 的 24349650_0, 排队驱动下积队增长超 C14)
N_PROBE = 10; GROUP = 7; SEEDS = (123, 124, 125); POLICIES = ("fixed", "served")
SELECT_FIELDS = ("id", "entry", "lanes", "competing", "fanout_nodes", "entry_load",
                 "opp_share", "lane_paths_identical", "nonrl_stop_mean_s", "depart_delay_mean_s")


def probe_files(cands, unseen):
    """候选按入口轮流分组 (同一入口的候选分散到各组), 每组一个探针路由文件 + 清单。"""
    D.mkdir(parents=True, exist_ok=True)
    allc = [("c", i, d) for i, d in enumerate(cands)] + [("u", i, d) for i, d in enumerate(unseen)]
    by_entry = collections.defaultdict(list)
    for x in allc: by_entry[x[2]["entry"]].append(x)
    order = []
    queues = [list(v) for v in by_entry.values()]
    while any(queues):
        for q in queues:
            if q: order.append(q.pop(0))
    groups = [order[i::int(np.ceil(len(order) / GROUP))] for i in range(int(np.ceil(len(order) / GROUP)))]
    bg = P.bg_entry_times(); rng = random.Random(2810); files = []
    for gi, grp in enumerate(groups):
        vehs = []; man = {}; taken = collections.defaultdict(list)
        plan = []
        for kind, i, d in grp:
            for j in range(N_PROBE):
                plan.append((900.0 + (j + rng.random()) / N_PROBE * 1500.0, kind, i, j, d))
        for t, kind, i, j, d in sorted(plan, key=lambda x: x[0]):
            lane = d["lanes"][0]; t2 = P.snap(t, bg.get(lane, []) + taken[lane]); taken[lane].append(t2)
            vid = f"probe_{kind}{i:02d}_{j}"; man[vid] = dict(kind=kind, cand=i, type="bus")
            vehs.append(dict(id=vid, type="bus", depart=t2, lane=int(lane.rsplit("_", 1)[1]), edges=d["edges"]))
        f = D / f"probes_g{gi}.rou.xml"; P.write_vehicles(f, vehs, "pairs50lf 探针车 (只用于选线)")
        json.dump(man, open(D / f"probes_g{gi}_manifest.json", "w")); files.append(gi)
    return files


def run_all(groups):
    procs = []
    for gi in groups:
        for pol in POLICIES:
            for s in SEEDS:
                nm = f"g{gi}_{pol}_{s}"
                if (D / f"{nm}_pairs.json").exists(): continue
                routes = ",".join([str(P.OUT / f) for f in P.BG_DEFAULT] + [str(D / f"probes_g{gi}.rou.xml")])
                procs.append(subprocess.Popen([PY, CAL, "--policy", pol, "--seed", str(s), "--routes", routes, "--manifest", str(D / f"probes_g{gi}_manifest.json"), "--sumocfg", str(P.OUT / "scenario.sumocfg"),
                                               "--out", str(D / nm), "--out-pairs", str(D / f"{nm}_pairs.json")], cwd=str(P.TOOLS18),
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS="1")))
                while sum(p.poll() is None for p in procs) >= 8:
                    procs[[p.poll() is None for p in procs].index(True)].wait()
    for p in procs:
        if p.wait() != 0: raise RuntimeError("探针运行失败")


def lanes_follow_plan(r, plan):
    """车道轨迹符合计划: (1) 逐秒观测到的车道按顺序都在计划车道里 (不到 1 s 就开过的短车道可能没被观测到, 但不能出现计划外的车道);
    (2) 每次 RL 路口访问的进口车道都在计划车道里。分流处若 SUMO 选了别的车道, 两条都会失败。"""
    it = iter(plan); plan_set = set(plan)
    return all(l in it for l in r["lanes"]) and all(x["lane"] in plan_set for x in r["visits"])


def measure(cands, unseen, groups):
    """逐候选汇总探针车记录。"""
    recs = collections.defaultdict(list)       # (kind, i) → [(policy, seed, rec)]
    car_wait = collections.defaultdict(list)   # policy → 小汽车每次访问等待 (车道口径) 的集均值
    for gi in groups:
        man = json.load(open(D / f"probes_g{gi}_manifest.json"))
        for pol in POLICIES:
            for s in SEEDS:
                nm = f"g{gi}_{pol}_{s}"; pr = json.load(open(D / f"{nm}_pairs.json")); run = json.load(open(D / f"{nm}.json"))
                n, tot = run["C16"]["by_type"]["car"]; car_wait[pol].append(tot / max(1, n))
                for vid, r in pr.items(): recs[(man[vid]["kind"], man[vid]["cand"])].append((pol, s, r))
    out = {}
    for key, lst in recs.items():
        d = cands[key[1]] if key[0] == "c" else unseen[key[1]]
        comp = {(t, m, l) for t, m, l in d["competing"]}
        cv = [x for pol, s, r in lst if pol == "fixed" for x in r["visits"] if (x["tid"], x["slot"], x["lane"]) in comp]
        tt = [r["arrival"] - r["depart"] for pol, s, r in lst if r["arrival"] is not None]
        wv = [x["wait"] for pol, s, r in lst if pol == "served" for x in r["visits"] if not x["ag_lane"]]
        out[key] = dict(opp_share=sum(x["opp"] for x in cv) / max(1, len(cv)), n_comp_visits=len(cv),
                        lane_paths_identical=all(lanes_follow_plan(r, d["lanes"]) for _, _, r in lst), n_probes=len(lst),
                        n_arrived=len(tt), tt99_s=float(np.percentile(tt, 99)) if tt else None,
                        nonrl_stop_mean_s=float(np.mean([r["nonrl_stop_s"] for _, _, r in lst])),
                        depart_delay_mean_s=float(np.mean([r["depart_delay"] for _, _, r in lst if r["depart_delay"] is not None])),
                        report_only=dict(wait_served_mean_s=float(np.mean(wv)) if wv else None, car_wait_served_mean_s=float(np.mean(car_wait["served"]))))
    return out


def choose(stats, K=10):
    """覆盖优先的贪心。只读 SELECT_FIELDS。合格: 机会 ≥ 0.7、车道轨迹确定、非 RL 停车 ≤ 2 s、入口负荷 ≤ 450 或实测出发延迟 ≤ 2 s; 每个入口最多 MAX_PER_ENTRY 条, 每条竞争进口车道最多 MAX_PER_APPROACH 条。"""
    ok = [c for c in stats if c["opp_share"] >= P.OPP_MIN and c["lane_paths_identical"] and c["nonrl_stop_mean_s"] <= 2.0
          and (c["entry_load"] <= P.ENTRY_LOAD_MAX or c["depart_delay_mean_s"] <= 2.0)]
    chosen = []; covered = set()
    def gain(c): return len({t for t, _, _ in c["competing"]} - covered)
    while len(chosen) < K:
        per_entry = collections.Counter(x["entry"] for x in chosen)
        per_lane = collections.Counter(l for x in chosen for _, _, l in x["competing"])
        rest = [c for c in ok if c not in chosen and per_entry[c["entry"]] < MAX_PER_ENTRY and all(per_lane[l] < MAX_PER_APPROACH for _, _, l in c["competing"])]
        if not rest: break
        best = max(rest, key=lambda c: (gain(c), len(c["competing"]), -len(c["fanout_nodes"]), c["id"]))
        if gain(best) == 0:     # 覆盖已尽: 补足 K, 偏好与已选路线终点路口不同的
            ends = {c["competing"][-1][0] for c in chosen}
            best = max(rest, key=lambda c: (c["competing"][-1][0] not in ends, c["entry"] not in {x["entry"] for x in chosen}, len(c["competing"]), -len(c["fanout_nodes"]), c["id"]))
        chosen.append(best); covered |= {t for t, _, _ in best["competing"]}
    return chosen


def allocate(chosen_all):
    """50 对分到 K 条路线, 每条 4 或 6 对 (偶数), 且每条竞争进口车道上的配对数 ≤ MAX_PAIRS_PER_LANE。
    依次试 K = 10, 11, 12 (取覆盖顺序的前 K 条), 在满足上限的分配里优先给竞争转向多的路线 6 对; 返回 (K, {序号: 对数})。"""
    import itertools
    for K in range(P.K_ROUTES[0], P.K_ROUTES[1] + 1):
        sub = chosen_all[:K]; n6 = (P.N_PAIRS - 4 * K) // 2
        if len(sub) < K or n6 < 0 or 4 * K + 2 * n6 != P.N_PAIRS: continue
        order = sorted(range(K), key=lambda i: (-len(sub[i]["competing"]), i))
        for six in itertools.combinations(order, n6):
            alloc = {i: (6 if i in six else 4) for i in range(K)}
            load = collections.Counter()
            for i, c in enumerate(sub):
                for _, _, l in c["competing"]: load[l] += alloc[i]
            if max(load.values()) <= MAX_PAIRS_PER_LANE: return K, alloc
    raise RuntimeError("K = 10–12 都无法在每条进口车道 ≤ %d 对的上限下分出 50 对" % MAX_PAIRS_PER_LANE)


def rl_in_edges(lanes, net, meta):
    """路线经过的所有 RL 进口边 (车道所在边的终点是 RL 路口), 不只竞争转向的进口。"""
    return {l.rsplit("_", 1)[0] for l in lanes if not l.startswith(":") and net.getLane(l).getEdge().getToNode().getID() in meta}


def pick_unseen(unseen, stats_u, chosen, n=3):
    import sumolib
    net = sumolib.net.readNet(str(P.NET)); meta = P.load_meta()
    used = set().union(*[rl_in_edges(c["lanes"], net, meta) for c in chosen]); out = []
    for i in sorted(range(len(unseen)), key=lambda i: (not stats_u.get(i, {}).get("lane_paths_identical", False), -len(unseen[i]["competing"]), i)):
        ins = rl_in_edges(unseen[i]["lanes"], net, meta)
        if ins & used or any(ins & rl_in_edges(u["lanes"], net, meta) for u in out): continue
        out.append(dict(unseen[i], measured=stats_u.get(i)))
        if len(out) == n: break
    return out


def main():
    P.check_inputs()
    C = P.load("pair_candidates.json"); cands, unseen = C["candidates"], C["unseen_pool"]
    groups = probe_files(cands, unseen); run_all(groups); m = measure(cands, unseen, groups)
    stats = [dict(id=i, entry=d["entry"], lanes=d["lanes"], competing=d["competing"], fanout_nodes=d["fanout_nodes"], entry_load=d["entry_load"], **{k: v for k, v in m[("c", i)].items()})
             for i, d in enumerate(cands)]
    chosen_all = choose(stats, K=P.K_ROUTES[1])
    K, alloc = allocate(chosen_all); chosen = chosen_all[:K]
    routes = [dict(id=f"r{j + 1:02d}", cand=c["id"], edges=cands[c["id"]]["edges"], lanes=c["lanes"], entry=c["entry"], entry_load=c["entry_load"],
                   competing=c["competing"], fanout_nodes=c["fanout_nodes"], traversals=cands[c["id"]]["traversals"], n_pairs=alloc.get(j),
                   measured={k: m[("c", c["id"])][k] for k in ("opp_share", "n_comp_visits", "lane_paths_identical", "tt99_s", "nonrl_stop_mean_s", "depart_delay_mean_s", "n_probes", "n_arrived")},
                   report_only=m[("c", c["id"])]["report_only"]) for j, c in enumerate(chosen)]
    unseen_sel = pick_unseen(unseen, {i: m[("u", i)] for i in range(len(unseen)) if ("u", i) in m}, chosen)
    meta = P.load_meta()
    cov = collections.Counter(t for r in routes for t, _, _ in r["competing"])
    rej = {i: [k for k, bad in (("opp", s["opp_share"] < P.OPP_MIN), ("lanes", not s["lane_paths_identical"]), ("nonrl", s["nonrl_stop_mean_s"] > 2.0),
                                  ("entry", s["entry_load"] > P.ENTRY_LOAD_MAX and s["depart_delay_mean_s"] > 2.0)) if bad] for i, s in enumerate(stats)}
    P.dump(dict(routes=routes, unseen=unseen_sel, coverage=dict(cov), uncovered=sorted(set(meta) - set(cov)),
                candidates_measured=[{k: v for k, v in s.items() if k not in ("lanes",)} for s in stats], rejected={i: r for i, r in rej.items() if r}), "pair_routes.json")
    print(f"合格 {sum(1 for r in rej.values() if not r)}/{len(stats)}; 选 {len(routes)} 条; 覆盖 {len(cov)}/{len(meta)}; 未覆盖 {sorted(set(meta) - set(cov))}")
    for r in routes: print(r["id"], r["entry"][:22], "对数", r["n_pairs"], "竞争", len(r["competing"]), {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r["measured"].items()})
    print("淘汰原因:", collections.Counter(x for r in rej.values() for x in r))


if __name__ == "__main__":
    main()
