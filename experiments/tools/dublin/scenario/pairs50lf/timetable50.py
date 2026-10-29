"""任务 3: 时刻表、镜像、换车型孪生、未见路线文件与 id 清单。
- 出发窗: 路线 r = [900, 3000 − 第 99 分位行程时间 − 对内间隔 − 120 s] (§5.2)。
- 对内间隔: 在标定时刻表上试 60/90/120 s, 逐路线取 "同对两车同时在同一 RL 进口" ≤ 10% 的最小值 (中性控制器, §5.3)。
- 第三版 (2026-10-29, 审阅后重做): 每份时刻表自身镜像对称 —— 前半段每条路线 n/2 对在全局前半窗内分层抽样, 后半段为前半段整体晚 600 s 并对调车型 (见 plan_pairs)。
- 配对车辆只与其他配对车辆保持 ≥ 3 s; 与背景挨得近时由 SUMO 正常排队插入 (出发延迟实测)。
- 第 1–6 份 (训练/标定): 单份优先车相遇不对称 > 20% 或有效冲突不足时换种子重抽 (§5.5); 第 7–13 份用预定种子一次生成, 不仿真。
- id 不含车型: t{k:02d}{m}_p{NN}_{x|y}, m = '' 原份 / 'm' 镜像 / 'c' 孪生 / 'u' 未见 / 'um' 未见镜像; 清单 pairs_manifest.json。"""
import collections, json, math, os, random, subprocess, sys
import numpy as np
import common50 as P

PY = sys.executable; CAL = str(P.TOOLS18 / "calib18.py"); D = P.WORK / "tt"
OTHER = {"bus": "ambulance", "ambulance": "bus"}
CHECK_POLICIES = ("fixed", "served"); CHECK_SEEDS = (123, 124)
ASYM_MAX = 0.20; MIN_CONFLICT_PER_CLASS = 20; MAX_REDRAW = 30; COPRESENCE_MAX = 0.10; UNSEEN_DELTA = 180.0; UNSEEN_PAIRS = 4; MAX_PAIR_SHIFT = 150.0; MIRROR_SHIFT = 600.0; HALF_GAP = 120.0; MAX_DELTA_ITERS = 4


def windows(routes, deltas):
    return {r["id"]: (float(P.PLATEAU[0]), float(P.PLATEAU[1] - r["measured"]["tt99_s"] - deltas[r["id"]] - P.WINDOW_MARGIN)) for r in routes}


def plan_pairs(routes, deltas, win, seed, bg=None, taken=None):
    """一份时刻表的配对计划, 自身镜像对称 (2026-10-29 第三版, 审阅后重做):
    前半段 = 每条路线一半的对, 放在全局前半窗 [900, 900+W]; 后半段 = 前半段每一对整体晚 MIRROR_SHIFT 秒、公交与救护车对调。
    于是单份时刻表里每一种结构 (同路线结伴、跨路线共现、优先车相遇、出发时刻、固定轮转的相位对齐) 都恰好一次由公交、一次由救护车承担;
    每条路线上两类的先行次数与平均出发时刻严格相等。MIRROR_SHIFT = 600 s 是各路口固定轮转周期 (50/75/100/150 s) 最小公倍数 300 s 的倍数。
    W = min_r(出发窗上限) − 900 − MIRROR_SHIFT, 使后半段也落在每条路线自己的出发窗内。
    配对车辆只与其他配对车辆保持 ≥ 3 s (同入口车道); 不再为背景车找空档 (密集入口上两半同时找空档不可行), 与背景挨得近时由 SUMO 正常排队插入, 出发延迟实测报告。
    bg 参数保留只为接口兼容, 不再使用。taken: 已占用的入口车道时刻 (给未见路线用)。"""
    rng = random.Random(seed); raw = []
    lo = float(P.PLATEAU[0]); min_hi = min(win[r["id"]][1] for r in routes)
    # 前半段窗: 起点 900; 终点 = min(各路线出发窗上限的最小值 − 600, 900 + 600 − Δ_r − HALF_GAP) —— 前半段的后车也比后半段首车早 ≥ HALF_GAP 出发, 两半不交错
    hi_r = {r["id"]: min(min_hi - MIRROR_SHIFT, lo + MIRROR_SHIFT - deltas[r["id"]] - HALF_GAP) for r in routes}
    assert min(hi_r.values()) > lo, ("前半段窗为空", hi_r)
    for r in routes:
        h = r["n_pairs"] // 2; t0 = rng.choice(["bus", "ambulance"]); hi = hi_r[r["id"]]
        for j in range(h):
            raw.append(dict(route=r["id"], t_plan=round(lo + (j + rng.random()) / h * (hi - lo), 1), x_type=t0 if j % 2 == 0 else OTHER[t0],
                            delta=deltas[r["id"]], lane=r["lanes"][0], edges=r["edges"], half="base"))
    raw.sort(key=lambda p: (p["t_plan"], p["route"]))
    taken = collections.defaultdict(list, {k: list(v) for k, v in (taken or {}).items()})
    out = []
    for p in raw:     # 前半段一对与它在后半段的镜像对一起挪: 四辆车都与同车道已排配对车辆相距 ≥ 3 s, 前半段首车不出前半窗
        l = p["lane"]; busy = taken[l]; d = p["delta"]
        for kk in range(0, 2 * int(MAX_PAIR_SHIFT / 0.5) + 1):
            s_ = (kk + 1) // 2 * 0.5 * (1 if kk % 2 else -1) if kk else 0.0
            tx = round(p["t_plan"] + s_, 1)
            ts = (tx, round(tx + d, 1), round(tx + MIRROR_SHIFT, 1), round(tx + MIRROR_SHIFT + d, 1))
            if lo <= tx <= hi_r[p["route"]] and all(abs(t - b) >= P.MIN_HEADWAY for t in ts for b in busy): break
        else:
            raise RuntimeError(f"{p['route']} 在 ±{MAX_PAIR_SHIFT} s 内找不到与其他配对车辆相距 ≥ 3 s 的时刻")
        taken[l] += list(ts)
        base = dict(p, t_x=ts[0], t_y=ts[1], shift=s_, offset=d)
        mirr = dict(p, t_plan=round(p["t_plan"] + MIRROR_SHIFT, 1), t_x=ts[2], t_y=ts[3], shift=s_, offset=d, x_type=OTHER[p["x_type"]], half="mirror")
        out += [base, mirr]
    out.sort(key=lambda p: (p["t_x"], p["route"]))
    for nn, p in enumerate(out, 1): p["pair"] = nn
    for p in out:     # 记下每对在另一半段的对应对号
        q = next(x for x in out if x is not p and x["route"] == p["route"] and abs(abs(x["t_x"] - p["t_x"]) - MIRROR_SHIFT) < 1e-6 and x["x_type"] != p["x_type"] and x["half"] != p["half"])
        p["twin_pair"] = q["pair"]
    return out, taken


def adjacency_weights(pairs):
    """每条路线上按时间相邻、同一类的两车: 按间隔加权 exp(−间隔/30 s) 计入该类 (越近越可能结伴、共用绿灯)。"""
    w = {"bus": 0.0, "ambulance": 0.0}; by_r = collections.defaultdict(list)
    for p in pairs:
        by_r[p["route"]] += [(p["t_x"], p["x_type"]), (p["t_y"], OTHER[p["x_type"]])]
    for vs in by_r.values():
        vs.sort()
        for (t1, c1), (t2, c2) in zip(vs, vs[1:]):
            if c1 == c2: w[c1] += math.exp(-(t2 - t1) / 30.0)
    return w


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
            procs[nm] = subprocess.Popen([PY, CAL, "--policy", pol, "--seed", str(s), "--routes", routes, "--manifest", str(man_file), "--end", str(end), "--sumocfg", str(P.OUT / "scenario.sumocfg"),
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
    print("初始对内间隔 (标定时刻表 delta*):", deltas)
    kinds = (("orig", "", ""), ("mirror", "m", "_mirror"), ("twin", "c", "_as_car"))
    # ---- 第 1–6 份 (训练/标定): 相遇不对称时换种子重抽; 用同一批仿真统计同对共现, 某条路线在第 1 份或第 1–6 份合计 > 10% 就把它的对内间隔提一档并全部重来
    delta_hist = []
    for it in range(MAX_DELTA_ITERS):
        win = windows(routes, deltas); T = {}; cp_tot = collections.defaultdict(lambda: [0.0, 0]); cp_tt1 = {}
        for k in range(1, 7):
            rej = 0; best = None
            for att in range(MAX_REDRAW):
                seed = P.TRAIN_SEED_BASE + 100 * k + att; pairs, taken = plan_pairs(routes, deltas, win, seed, bg)
                tag = f"i{it}_tt{k}_s{att}"; vs, man = vehicles(pairs, k, f"x{it}s{att}", "orig")
                f = D / f"{tag}.rou.xml"; P.write_vehicles(f, vs, "相遇与共现检查"); json.dump(man, open(D / f"{tag}_man.json", "w"))
                recs = simulate(tag, f, D / f"{tag}_man.json"); e = encounters(recs, man)
                good = e["asym"] <= ASYM_MAX and min(e["conflict_visits_per_ep"].values()) >= MIN_CONFLICT_PER_CLASS
                if best is None or e["asym"] < best[1]["asym"]: best = (seed, e, pairs, taken, recs, man)
                if good: break
                rej += 1
            seed, e, pairs, taken, recs, man = best
            cps = {pol: copresence({k_: v for k_, v in recs.items() if k_[0] == pol}, man) for pol in CHECK_POLICIES}
            for pol, c in cps.items():
                for rid, v in c.items(): cp_tot[(rid, pol)][0] += v; cp_tot[(rid, pol)][1] += 1
            if k == 1: cp_tt1 = cps
            T[k] = dict(seed=seed, rejected=rej, simulated_for_selection=True, encounters=e, pairs=pairs, passed=e["asym"] <= ASYM_MAX, copresence=cps)
        bad = sorted({rid for (rid, pol), (sm, n) in cp_tot.items() if sm / n > COPRESENCE_MAX} | {rid for pol, c in cp_tt1.items() for rid, v in c.items() if v > COPRESENCE_MAX})
        delta_hist.append(dict(iter=it, deltas=dict(deltas), bad=bad))
        print(f"第 {it} 轮对内间隔 {deltas}; 同对共现超 10% 的路线 {bad}")
        if not bad or it == MAX_DELTA_ITERS - 1: break      # 最后一轮不再提间隔, 保证第 1–6 份与最终间隔一致 (未收敛的路线如实记录)
        for rid in bad:
            nxt = [d for d in P.DELTA_CANDIDATES if d > deltas[rid]]
            if nxt: deltas[rid] = nxt[0]
    man_all = {}
    for k in range(1, 7):
        T[k]["adjacency_w"] = adjacency_weights(T[k]["pairs"]); write_set(T[k]["pairs"], k, kinds, "pairs_tt", man_all)
    # ---- 第 7–13 份: 用最终对内间隔与预定种子一次生成, 不仿真
    for k in range(7, P.N_TT + 1):
        seed = P.HOLDOUT_SEEDS[k]; pairs, taken = plan_pairs(routes, deltas, win, seed, bg)
        T[k] = dict(seed=seed, rejected=0, simulated_for_selection=False, pairs=pairs)
        T[k]["adjacency_w"] = adjacency_weights(pairs)
        write_set(pairs, k, kinds, "pairs_tt", man_all)
        if k >= 8:     # 未见路线 (评估用): 与本份配对车辆一起排同车道间隔
            uroutes = [dict(u, id=f"u{i + 1}", n_pairs=UNSEEN_PAIRS) for i, u in enumerate(unseen)]
            ud = {u["id"]: UNSEEN_DELTA for u in uroutes}; uw = windows(uroutes, ud)
            up, _ = plan_pairs(uroutes, ud, uw, 9000 + k, bg, taken)
            for kind, mark, suf in (("orig", "u", ""), ("mirror", "um", "_mirror")):
                vs, man = vehicles(up, k, mark, kind); man_all.update(man)
                P.write_vehicles(P.OUT / f"pairs_unseen_tt{k}{suf}.rou.xml", vs, f"pairs50lf 未见路线 第 {k} 份 {kind}")
            T[k]["unseen_pairs"] = up
    json.dump(man_all, open(P.OUT / "pairs_manifest.json", "w"), ensure_ascii=False, indent=0)
    P.dump(dict(deltas=deltas, windows=win, delta_calibration={str(k): v for k, v in cal.items()}, delta_iterations=delta_hist, timetables={str(k): v for k, v in T.items()}), "timetables.json")
    for k in range(1, P.N_TT + 1):
        t = T[k]; print(k, "种子", t["seed"], "重抽", t["rejected"], ("相遇不对称 %.2f 冲突/类/集 %s" % (t["encounters"]["asym"], {a: round(b, 1) for a, b in t["encounters"]["conflict_visits_per_ep"].items()})) if "encounters" in t else "(留出, 未仿真)",
              "整对最大移动", max(abs(p["shift"]) for p in t["pairs"]))


if __name__ == "__main__":
    main()
