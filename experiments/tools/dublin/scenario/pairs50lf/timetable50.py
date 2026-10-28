"""任务 3: 时刻表、镜像、换车型孪生、未见路线文件与 id 清单。
- 出发窗: 路线 r = [900, 3000 − 第 99 分位行程时间 − 对内间隔 − 120 s] (§5.2)。
- 对内间隔: 在标定时刻表上试 60/90/120 s, 逐路线取 "同对两车同时在同一 RL 进口" ≤ 10% 的最小值 (中性控制器, §5.3)。
- 每条路线 n 对在出发窗内分层抽样; 路线内交替先后 (首对车型随机), 对数为偶数 → 两类先行次数相等 (§5.4)。
- 配对车辆按块联合找空档 (同路线相邻两对为一块, 块内实际对内间隔相同, ∈ [Δ, Δ+40 s]): 与同入口车道背景车及已排配对车辆相距 ≥ 3 s, 首车不出出发窗 (§3; 实施时由逐车改为按块, 见进度记录)。
- 第 1–6 份 (训练/标定): 单份优先车相遇不对称 > 20% 或有效冲突不足时换种子重抽 (§5.5); 第 7–13 份用预定种子一次生成, 不仿真。
- id 不含车型: t{k:02d}{m}_p{NN}_{x|y}, m = '' 原份 / 'm' 镜像 / 'c' 孪生 / 'u' 未见 / 'um' 未见镜像; 清单 pairs_manifest.json。"""
import collections, json, os, random, subprocess, sys
import numpy as np
import common50 as P

PY = sys.executable; CAL = str(P.TOOLS18 / "calib18.py"); D = P.WORK / "tt"
OTHER = {"bus": "ambulance", "ambulance": "bus"}
CHECK_POLICIES = ("fixed", "served"); CHECK_SEEDS = (123, 124)
ASYM_MAX = 0.20; MIN_CONFLICT_PER_CLASS = 20; MAX_REDRAW = 30; COPRESENCE_MAX = 0.10; UNSEEN_DELTA = 180.0; UNSEEN_PAIRS = 4; MAX_PAIR_SHIFT = 150.0; D_SLACK = 40.0


def windows(routes, deltas):
    return {r["id"]: (float(P.PLATEAU[0]), float(P.PLATEAU[1] - r["measured"]["tt99_s"] - deltas[r["id"]] - P.WINDOW_MARGIN)) for r in routes}


def plan_pairs(routes, deltas, win, seed, bg, taken=None):
    """一份时刻表的配对计划 (找空档后)。taken: 已占用的入口车道时刻 (给未见路线用)。"""
    rng = random.Random(seed); raw = []
    for r in routes:
        lo, hi = win[r["id"]]; n = r["n_pairs"]; t0 = rng.choice(["bus", "ambulance"])
        for j in range(n):
            raw.append(dict(route=r["id"], t_plan=round(lo + (j + rng.random()) / n * (hi - lo), 1), x_type=t0 if j % 2 == 0 else OTHER[t0],
                            delta=deltas[r["id"]], lane=r["lanes"][0], edges=r["edges"]))
    raw.sort(key=lambda p: (p["t_plan"], p["route"]))
    for nn, p in enumerate(raw, 1): p["pair"] = nn
    taken = collections.defaultdict(list, {k: list(v) for k, v in (taken or {}).items()})
    # 按块找空档: 同一路线上时间相邻的两对 (一对公交先、一对救护车先) 为一块; 两对各自可挪, 但实际对内间隔取同一个值 d ∈ [Δ, Δ + D_SLACK]。
    # 块内公交出发时刻之和 = tA + (tB + d) = 救护车之和 → 路线内两类平均出发时刻严格相等; 所有车与同车道背景及已排配对车辆相距 ≥ 3 s, 首车不出出发窗。
    blocks = []
    for r in routes:
        ps = [p for p in raw if p["route"] == r["id"]]
        blocks += [(ps[i], ps[i + 1]) for i in range(0, len(ps), 2)]
    blocks.sort(key=lambda b: (b[0]["t_plan"], b[0]["route"]))
    def shifts():
        yield 0.0
        for kk in range(1, int(MAX_PAIR_SHIFT / 0.5) + 1): yield kk * 0.5; yield -kk * 0.5
    for A, B in blocks:
        l = A["lane"]; lo, hi = win[A["route"]]; busy = bg.get(l, []) + taken[l]; dgrid = [round(A["delta"] + 0.5 * i, 1) for i in range(int(D_SLACK / 0.5) + 1)]
        ok = lambda t, extra=(): all(abs(t - x) >= P.MIN_HEADWAY for x in busy) and all(abs(t - x) >= P.MIN_HEADWAY for x in extra)
        found = None
        for sA in shifts():
            ta = round(A["t_plan"] + sA, 1)
            if not (lo <= ta <= hi and ok(ta)): continue
            dA = [d for d in dgrid if ok(round(ta + d, 1))]
            if not dA: continue
            for sB in shifts():
                tb = round(B["t_plan"] + sB, 1)
                if not (lo <= tb <= hi): continue
                for d in dA:
                    ex = (ta, round(ta + d, 1))
                    if ok(tb, ex) and ok(round(tb + d, 1), ex): found = (ta, tb, d, sA, sB); break
                if found: break
            if found: break
        if not found: raise RuntimeError(f"{A['route']} 第 {A['pair']}/{B['pair']} 对在 ±{MAX_PAIR_SHIFT} s 内找不到空档")
        ta, tb, d, sA, sB = found
        for p, t, s_ in ((A, ta, sA), (B, tb, sB)):
            p["t_x"], p["t_y"], p["shift"], p["offset"] = t, round(t + d, 1), s_, d
            taken[l] += [p["t_x"], p["t_y"]]
    return raw, taken


def vehicles(pairs, k, mark, kind):
    """kind: orig / mirror / twin。返回 (车辆列表, 清单条目)。"""
    vs = []; man = {}
    for p in pairs:
        for slot, t in (("x", p["t_x"]), ("y", p["t_y"])):
            orig = p["x_type"] if slot == "x" else OTHER[p["x_type"]]
            ty = {"orig": orig, "mirror": OTHER[orig], "twin": "car"}[kind]
            vid = f"t{k:02d}{mark}_p{p['pair']:02d}_{slot}"
            vs.append(dict(id=vid, type=ty, depart=t, lane=int(p["lane"].rsplit("_", 1)[1]), edges=p["edges"]))
            man[vid] = dict(tt=k, mirror=kind == "mirror", twin=kind == "twin", route=p["route"], pair=p["pair"], slot=slot, type=ty, orig_type=orig)
    return vs, man


def simulate(name, route_file, man_file, policies=CHECK_POLICIES, seeds=CHECK_SEEDS, end=3600):
    """中性控制器下跑一份配对文件, 返回 {(policy, seed): 逐车记录}。"""
    procs = {}
    for pol in policies:
        for s in seeds:
            nm = f"{name}_{pol}_{s}"
            if (D / f"{nm}_pairs.json").exists(): continue
            routes = ",".join([str(P.OUT / f) for f in P.BG_DEFAULT] + [str(route_file)])
            procs[nm] = subprocess.Popen([PY, CAL, "--policy", pol, "--seed", str(s), "--routes", routes, "--manifest", str(man_file), "--end", str(end),
                                          "--out", str(D / nm), "--out-pairs", str(D / f"{nm}_pairs.json")], cwd=str(P.TOOLS18),
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS="1"))
    for nm, pr in procs.items():
        if pr.wait() != 0: raise RuntimeError(f"{nm} 运行失败")
    return {(pol, s): json.load(open(D / f"{name}_{pol}_{s}_pairs.json")) for pol in policies for s in seeds}


def copresence(recs, man):
    """逐路线: 同对两车在同一 RL 路口的进口上停留区间有重叠的对数占比。"""
    hit = collections.Counter(); tot = collections.Counter()
    for r in recs.values():
        pairs = collections.defaultdict(dict)
        for vid, x in r.items(): pairs[(man[vid]["route"], man[vid]["pair"])][man[vid]["slot"]] = x
        for (rid, _), xy in pairs.items():
            if len(xy) < 2: continue
            tot[rid] += 1
            vx = {v["tid"]: (v["t0"], v["t_last"]) for v in xy["x"]["visits"]}
            hit[rid] += any(a0 <= b1 and b0 <= a1 for v in xy["y"]["visits"] if v["tid"] in vx for (a0, a1), (b0, b1) in [(vx[v["tid"]], (v["t0"], v["t_last"]))])
    return {rid: hit[rid] / tot[rid] for rid in tot}


def encounters(recs, man):
    """四种相遇 (到达 RL 路口时冲突方向上有另一辆配对车辆) 与每类每集有效冲突访问数。"""
    enc = collections.Counter(); conf = collections.Counter(); n = len(recs)
    for r in recs.values():
        for vid, x in r.items():
            ty = man[vid]["type"]
            for v in x["visits"]:
                if v["enc"]: conf[ty] += 1
                for e in v["enc"]: enc[f"{ty}|{e}"] += 1
    ab, ba, aa, bb = enc["ambulance|bus"], enc["bus|ambulance"], enc["ambulance|ambulance"], enc["bus|bus"]
    asym = max(abs(ab - ba) / max(1.0, (ab + ba) / 2), abs(aa - bb) / max(1.0, (aa + bb) / 2))
    return dict(counts=dict(enc), conflict_visits_per_ep={t: conf[t] / n for t in ("bus", "ambulance")}, asym=asym)


def write_set(pairs, k, kinds, prefix, man_all):
    files = {}
    for kind, mark, suf in kinds:
        vs, man = vehicles(pairs, k, mark, kind); man_all.update(man)
        f = P.OUT / f"{prefix}{k}{suf}.rou.xml"; P.write_vehicles(f, vs, f"pairs50lf 第 {k} 份 {kind}"); files[kind] = f
    return files


def main():
    P.check_inputs(); D.mkdir(parents=True, exist_ok=True)
    S = P.load("pair_routes.json"); routes = S["routes"]; unseen = S["unseen"]; bg = P.bg_entry_times()
    # ---- 对内间隔标定 (标定时刻表, 种子 TRAIN_SEED_BASE)
    cal = {}
    for dl in P.DELTA_CANDIDATES:
        deltas = {r["id"]: dl for r in routes}; win = windows(routes, deltas)
        pairs, _ = plan_pairs(routes, deltas, win, P.TRAIN_SEED_BASE, bg); man = {}
        vs, man = vehicles(pairs, 0, f"d{int(dl)}", "orig")
        f = D / f"delta{int(dl)}.rou.xml"; P.write_vehicles(f, vs, "对内间隔标定"); json.dump(man, open(D / f"delta{int(dl)}_man.json", "w"))
        recs = simulate(f"delta{int(dl)}", f, D / f"delta{int(dl)}_man.json", seeds=(123, 124, 125))
        cal[dl] = {pol: copresence({k_: v for k_, v in recs.items() if k_[0] == pol}, man) for pol in CHECK_POLICIES}
    deltas = {}
    for r in routes:
        ok = [dl for dl in P.DELTA_CANDIDATES if all(cal[dl][pol].get(r["id"], 1.0) <= COPRESENCE_MAX for pol in CHECK_POLICIES)]
        deltas[r["id"]] = min(ok) if ok else max(P.DELTA_CANDIDATES)
    win = windows(routes, deltas)
    print("对内间隔:", deltas); print("共现 (固定轮转/排队驱动):", {dl: {pol: {k: round(v, 2) for k, v in c.items()} for pol, c in x.items()} for dl, x in cal.items()})
    # ---- 第 1–13 份
    man_all = {}; T = {}
    kinds = (("orig", "", ""), ("mirror", "m", "_mirror"), ("twin", "c", "_as_car"))
    for k in range(1, P.N_TT + 1):
        if k in P.HOLDOUT_SEEDS:
            seed = P.HOLDOUT_SEEDS[k]; pairs, taken = plan_pairs(routes, deltas, win, seed, bg)
            T[k] = dict(seed=seed, rejected=0, simulated_for_selection=False, pairs=pairs)
        else:
            rej = 0; best = None
            for att in range(MAX_REDRAW):
                seed = P.TRAIN_SEED_BASE + 100 * k + att; pairs, taken = plan_pairs(routes, deltas, win, seed, bg)
                vs, man = vehicles(pairs, k, f"s{att}", "orig")
                f = D / f"tt{k}_s{att}.rou.xml"; P.write_vehicles(f, vs, "相遇检查"); json.dump(man, open(D / f"tt{k}_s{att}_man.json", "w"))
                e = encounters(simulate(f"tt{k}_s{att}", f, D / f"tt{k}_s{att}_man.json"), man)
                good = e["asym"] <= ASYM_MAX and min(e["conflict_visits_per_ep"].values()) >= MIN_CONFLICT_PER_CLASS
                if best is None or e["asym"] < best[1]["asym"]: best = (seed, e, pairs, taken)
                if good: break
                rej += 1
            seed, e, pairs, taken = best
            T[k] = dict(seed=seed, rejected=rej, simulated_for_selection=True, encounters=e, pairs=pairs, passed=e["asym"] <= ASYM_MAX)
        write_set(pairs, k, kinds, "pairs_tt", man_all)
        if k >= 8:     # 未见路线 (评估用): 与本份配对车辆一起找空档
            uroutes = [dict(u, id=f"u{i + 1}", n_pairs=UNSEEN_PAIRS) for i, u in enumerate(unseen)]
            ud = {u["id"]: UNSEEN_DELTA for u in uroutes}; uw = windows(uroutes, ud)
            up, _ = plan_pairs(uroutes, ud, uw, 9000 + k, bg, taken)
            for kind, mark, suf in (("orig", "u", ""), ("mirror", "um", "_mirror")):
                vs, man = vehicles(up, k, mark, kind); man_all.update(man)
                P.write_vehicles(P.OUT / f"pairs_unseen_tt{k}{suf}.rou.xml", vs, f"pairs50lf 未见路线 第 {k} 份 {kind}")
            T[k]["unseen_pairs"] = up
    json.dump(man_all, open(P.OUT / "pairs_manifest.json", "w"), ensure_ascii=False, indent=0)
    P.dump(dict(deltas=deltas, windows=win, delta_calibration={str(k): v for k, v in cal.items()}, timetables={str(k): v for k, v in T.items()}), "timetables.json")
    for k in range(1, P.N_TT + 1):
        t = T[k]; print(k, "种子", t["seed"], "重抽", t["rejected"], ("相遇不对称 %.2f 冲突/类/集 %s" % (t["encounters"]["asym"], {a: round(b, 1) for a, b in t["encounters"]["conflict_visits_per_ep"].items()})) if "encounters" in t else "(留出, 未仿真)",
              "整对最大移动", max(abs(p["shift"]) for p in t["pairs"]))


if __name__ == "__main__":
    main()
