"""任务 6: 标定与验收 (计划 §7)。只用标定时刻表 (第 1–6 份及第 1 份的镜像); 第 7–13 份不跑。结果写 build/final_calibration.json。
- 硬判据 (继承 18h): 排队驱动 / 固定轮转各 5 种子, 1.0 档 + 第 1 份, 3600 s; 随机策略 5 种子压力测试; 0.6 / 0.8 档各 3 种子。
- P1 可互换性: 排队驱动 / 固定轮转分别, 第 1 份 (单份), 排空到 4500 s; 独立单位 = 整集: 每种子 "公交类均值 − 救护车类均值"
  (类均值 = 该类车辆 "每次访问平均等待" 的车辆平均, 剔除恒绿车道), 对逐种子差值求 90% t 区间; 种子数按预跑方差定 (区间半宽 ≤ 0.3 s)。
- P2 镜像一致性: 同种子 (123): 固定轮转 / 排队驱动 第 1 份对第 1′ 份; 知道车型的控制器 531 + 第 1 份 对 351 + 第 1′ 份; 111 + 第 1 份 对 111 + 第 1′ 份。
- P3 相遇: 第 1 份 + 第 1′ 份, 中性控制器各 5 种子; P4 机会: 固定轮转, 第 1–6 份 × 5 种子; P5 中性差距 (报告); P6 知道车型的控制器 531/351/111 × 5;
  P7 同对共现; P8 非 RL 停车; P9 车道轨迹; P10 超窗与截尾。"""
import collections, json, math, os, subprocess, sys
import numpy as np
from scipy import stats as sst
import common50 as P
from select50 import lanes_follow_plan
from check_calib50 import diff

PY = sys.executable; CAL = str(P.TOOLS18 / "calib18.py"); D = P.WORK / "final"
MAN = P.OUT / "pairs_manifest.json"; SWAP = {"bus": "ambulance", "ambulance": "bus"}
SEEDS5 = range(123, 128); P1_PILOT = 20; P1_MAX = 150; P1_HALF = 0.3


def routes_of(k=1, mirror=False, level="100"):
    bg = [str(P.OUT / f"background_cars_{level}.rou.xml"), str(P.OUT / "background_gtfs_as_car.rou.xml")]
    return ",".join(bg + [str(P.OUT / f"pairs_tt{k}{'_mirror' if mirror else ''}.rou.xml")])


class Runner:
    def __init__(self): self.q = []; D.mkdir(parents=True, exist_ok=True)

    def add(self, name, policy, seed, routes, end=3600, table=None):
        if (D / f"{name}.json").exists(): return
        cmd = [PY, CAL, "--policy", policy, "--seed", str(seed), "--routes", routes, "--manifest", str(MAN), "--end", str(end),
               "--out", str(D / name), "--out-pairs", str(D / f"{name}_pairs.json"), "--sumocfg", str(P.OUT / "scenario.sumocfg")] + (["--priority-table", table] if table else [])
        self.q.append((name, cmd))

    def run(self, par=8):
        procs = []
        for name, cmd in self.q:
            procs.append((name, subprocess.Popen(cmd, cwd=str(P.TOOLS18), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS="1"))))
            while sum(p.poll() is None for _, p in procs) >= par:
                next(p for _, p in procs if p.poll() is None).wait()
        for name, p in procs:
            if p.wait() != 0: raise RuntimeError(f"{name} 运行失败")
        self.q = []


def J(name): return json.load(open(D / f"{name}.json"))
def PR(name): return json.load(open(D / f"{name}_pairs.json"))


def class_means(recs):
    """类均值 = 该类车辆 "每次访问平均等待 (剔除恒绿车道)" 的车辆平均 (与 wandb per-visit 同义)。"""
    per = collections.defaultdict(list)
    for r in recs.values():
        w = [v["wait"] for v in r["visits"] if not v["ag_lane"]]
        if w: per[r["type"]].append(sum(w) / len(w))
    return {t: float(np.mean(v)) for t, v in per.items()}


def keyed(recs, man, mirror):
    """按 (路线, 对号, 先后位置) 重新编键, 并把车型与相遇车型按镜像映射回原份口径, 用于逐车比较。"""
    out = {}
    for vid, r in recs.items():
        m = man[vid]; r = dict(r)
        r["type"] = SWAP.get(r["type"], r["type"]) if mirror else r["type"]
        r["visits"] = [dict(v, enc=sorted(SWAP.get(e, e) if mirror else e for e in v["enc"])) for v in r["visits"]]
        out[f"{m['route']}|{m['pair']}|{m['slot']}"] = r
    return out


def hard_summary(names):
    ds = [J(n) for n in names]
    c14 = [max([v["grow"] - max(0.2 * v["first"], 2.0) for v in d["C14"].values()] or [0]) for d in ds]
    return dict(n=len(ds), C1_growth=[d["C1"]["growth"] for d in ds], C2_mean_delay=[d["C2"]["mean_delay"] for d in ds],
                C3_blocked=[d["C3"]["blocked_events"] for d in ds], C3p_long=[d["C3p"]["long_wait_visits"] for d in ds], C4_missed=[d["C4"]["missed"] for d in ds],
                C6_spill=[d["C6"]["spill_events"] for d in ds], C7_long_over_5pct=[sum(1 for v in d["C7"]["long_lanes"].values() if v > 0.05) for d in ds],
                C8=[round(d["C8"]["share"], 3) for d in ds], C9_lc_col_tp_missing=[[d["C9"]["lane_changes"], d["C9"]["collisions"], d["C9"]["teleports"], d["C9"]["tracker_missing"]] for d in ds],
                C13=[d["C13"]["early_wait_over_rotation"] for d in ds], C14_worst_excess=[round(x, 2) for x in c14],
                C16_by_type={t: round(sum(d["C16"]["by_type"][t][1] for d in ds) / max(1, sum(d["C16"]["by_type"][t][0] for d in ds)), 3) for t in ("car", "bus", "ambulance")},
                C17_cutins=[d["C17"]["n_cutins"] for d in ds], pairs_counts=[d["PAIRS"]["counts"] for d in ds])


def main():
    P.check_inputs(); man = json.load(open(MAN)); routes = {r["id"]: r for r in P.load("pair_routes.json")["routes"]}
    R = Runner()
    # ---- 硬判据 + P4/P5/P7/P8/P9 用的运行
    for pol in ("served", "fixed", "random"):
        for s in SEEDS5: R.add(f"h_{pol}_{s}", pol, s, routes_of(1))
    for lv in ("060", "080"):
        for pol in ("served", "fixed"):
            for s in range(123, 126): R.add(f"h{lv}_{pol}_{s}", pol, s, routes_of(1, level=lv))
    for k in range(2, 7):
        for s in SEEDS5: R.add(f"opp_tt{k}_{s}", "fixed", s, routes_of(k))
    # ---- P1 (排空), P2, P3, P6
    for pol in ("served", "fixed"):
        for s in range(123, 123 + P1_PILOT): R.add(f"p1_{pol}_{s}", pol, s, routes_of(1), end=P.END_DRAIN)
        for s in SEEDS5: R.add(f"p3m_{pol}_{s}", pol, s, routes_of(1, mirror=True), end=P.END_DRAIN)
    R.add("p2_prio531_orig", "prio", 123, routes_of(1), end=P.END_DRAIN, table="531")
    R.add("p2_prio351_mirr", "prio", 123, routes_of(1, mirror=True), end=P.END_DRAIN, table="351")
    R.add("p2_prio111_orig", "prio", 123, routes_of(1), end=P.END_DRAIN, table="111")
    R.add("p2_prio111_mirr", "prio", 123, routes_of(1, mirror=True), end=P.END_DRAIN, table="111")
    for tab in ("531", "351", "111"):
        for s in SEEDS5: R.add(f"p6_{tab}_{s}", "prio", s, routes_of(1), table=tab)
    for pol in ("served", "fixed"): R.add(f"p1_{pol}_0", pol, 0, routes_of(1), end=P.END_DRAIN)     # 训练用的 SUMO 种子 (exp470–472 seed: 0), 只报告
    R.run()
    # P1: 按预跑方差补种子
    def p1_diffs(pol, n): return [(lambda cm: cm["bus"] - cm["ambulance"])(class_means(PR(f"p1_{pol}_{s}"))) for s in range(123, 123 + n)]
    need = {}
    for pol in ("served", "fixed"):     # 逐轮补种子: 每轮按已有种子重算方差, 直到 90% 区间半宽 ≤ P1_HALF 或到上限
        n = P1_PILOT
        while True:
            d = p1_diffs(pol, n)
            if n >= P1_MAX or sst.t.ppf(0.95, n - 1) * float(np.std(d, ddof=1)) / math.sqrt(n) <= P1_HALF: break
            n2 = min(P1_MAX, n + 10)
            for s in range(123 + n, 123 + n2): R.add(f"p1_{pol}_{s}", pol, s, routes_of(1), end=P.END_DRAIN)
            R.run(); n = n2
        need[pol] = n
    F = {}
    F["hard"] = {pol: hard_summary([f"h_{pol}_{s}" for s in SEEDS5]) for pol in ("served", "fixed", "random")}
    F["levels"] = {f"L{lv}_{pol}": hard_summary([f"h{lv}_{pol}_{s}" for s in range(123, 126)]) for lv in ("060", "080") for pol in ("served", "fixed")}
    F["P1"] = {}
    for pol in ("served", "fixed"):
        d = p1_diffs(pol, need[pol]); n = len(d); m = float(np.mean(d)); se = float(np.std(d, ddof=1) / math.sqrt(n)); h = float(sst.t.ppf(0.95, n - 1) * se)
        F["P1"][pol] = dict(n_seeds=n, mean_diff_bus_minus_amb=m, se=se, ci90=[m - h, m + h], margin=0.5, per_seed=[round(x, 3) for x in d])
    # P2
    F["P2"] = {}
    for a_, b_, key in (("p1_fixed_123", "p3m_fixed_123", "fixed_orig_vs_mirror"), ("p1_served_123", "p3m_served_123", "served_orig_vs_mirror"),
                        ("p2_prio531_orig", "p2_prio351_mirr", "prio531_orig_vs_prio351_mirror"), ("p2_prio111_orig", "p2_prio111_mirr", "prio111_orig_vs_mirror")):
        dd = diff(keyed(PR(a_), man, False), keyed(PR(b_), man, True))
        nd = diff({k: J(a_)[k] for k in ("C1", "C8")}, {k: J(b_)[k] for k in ("C1", "C8")})
        F["P2"][key] = dict(identical=dd is None and nd is None, diff=dd or nd)
    # P3
    enc = collections.Counter(); conf = collections.Counter(); single = []; nep = 0
    for pol in ("served", "fixed"):
        for s in SEEDS5:
            for nm in (f"p1_{pol}_{s}", f"p3m_{pol}_{s}"):
                e1 = collections.Counter(); nep += 1
                for vid, r in PR(nm).items():
                    for v in r["visits"]:
                        if v["enc"]: conf[r["type"]] += 1
                        for e in v["enc"]: enc[f"{r['type']}|{e}"] += 1; e1[f"{r['type']}|{e}"] += 1
                if nm.startswith("p1_"): single.append(abs(e1["ambulance|bus"] - e1["bus|ambulance"]) / max(1, (e1["ambulance|bus"] + e1["bus|ambulance"]) / 2))
    F["P3"] = dict(union_counts=dict(enc), conflict_visits_per_ep={t: conf[t] / nep for t in ("bus", "ambulance")}, single_tt_asym_ab=[round(x, 3) for x in single])
    # P4 (固定轮转, 第 1–6 份 × 5 种子)
    comp = {rid: {(t, m, l) for t, m, l in r["competing"]} for rid, r in routes.items()}
    hit = collections.Counter(); tot = collections.Counter(); wopp = collections.Counter(); neps = 0
    for nm in [f"h_fixed_{s}" for s in SEEDS5] + [f"opp_tt{k}_{s}" for k in range(2, 7) for s in SEEDS5]:
        neps += 1
        for vid, r in PR(nm).items():
            rid = man[vid]["route"]
            for v in r["visits"]:
                if (v["tid"], v["slot"], v["lane"]) in comp[rid]: tot[rid] += 1; hit[rid] += v["opp"]
                if v["opp"]: wopp[r["type"]] += 1
    F["P4"] = dict(by_route={rid: hit[rid] / max(1, tot[rid]) for rid in routes}, with_opp_per_class_per_ep={t: wopp[t] / neps for t in ("bus", "ambulance")})
    # P5 (报告)
    sv = hard_summary([f"h_served_{s}" for s in SEEDS5])["C16_by_type"]
    F["P5"] = dict(reported=True, served_C16_by_type=sv, pair_minus_car_s=round((sv["bus"] + sv["ambulance"]) / 2 - sv["car"], 3),
                   flag_to_user=(sv["bus"] + sv["ambulance"]) / 2 - sv["car"] > 1.0)
    # P6 (能力检查)
    p6 = {}
    for tab in ("531", "351", "111"):
        per = [J(f"p6_{tab}_{s}")["C16"]["by_type"] for s in SEEDS5]
        m = {t: float(np.mean([x[t][1] / max(1, x[t][0]) for x in per])) for t in ("car", "bus", "ambulance")}
        p6[tab] = dict(mean=m, per_seed=[{t: round(x[t][1] / max(1, x[t][0]), 2) for t in ("car", "bus", "ambulance")} for x in per])
    F["P6"] = dict(reported=True, tables=p6,
                   order_531=p6["531"]["mean"]["ambulance"] < p6["531"]["mean"]["bus"] < p6["531"]["mean"]["car"],
                   order_351=p6["351"]["mean"]["bus"] < p6["351"]["mean"]["ambulance"] < p6["351"]["mean"]["car"],
                   mirror_gap_s=abs(p6["531"]["mean"]["ambulance"] - p6["351"]["mean"]["bus"]))
    F["P1"]["training_seed0"] = {pol: (lambda cm: cm["bus"] - cm["ambulance"])(class_means(PR(f"p1_{pol}_0"))) for pol in ("served", "fixed")}
    # P11: 同类同进口共现 (审阅 2026-10-28 新增): 一次访问期间, 同一 RL 进口边上有另一辆同类配对车辆的访问占比, 两类应相近
    share = {}
    for pol in ("served", "fixed"):
        hit = collections.Counter(); tot = collections.Counter()
        for s in SEEDS5:
            by_e = collections.defaultdict(list)
            for vid, r in PR(f"p1_{pol}_{s}").items():
                for v in r["visits"]: by_e[(v["tid"], v["lane"].rsplit("_", 1)[0])].append((v["t0"], v["t_last"], r["type"], vid))
            for lst in by_e.values():
                for a0, a1, ty, vid in lst:
                    tot[ty] += 1; hit[ty] += any(b0 <= a1 and a0 <= b1 for b0, b1, ty2, vid2 in lst if ty2 == ty and vid2 != vid)
        share[pol] = {t: hit[t] / max(1, tot[t]) for t in ("bus", "ambulance")}
    F["P11"] = dict(same_class_copresence=share, max_abs_diff=max(abs(v["bus"] - v["ambulance"]) for v in share.values()))
    # P12: 中间档镜像差 (531 下公交 对 351 下救护车, 知道车型的控制器), 与最高档 (P6 mirror_gap_s) 同一门槛
    F["P12"] = dict(mid_mirror_gap_s=abs(p6["531"]["mean"]["bus"] - p6["351"]["mean"]["ambulance"]))
    # P7 / P8 / P9 / P10
    from timetable50 import copresence
    cp = {pol: copresence({(pol, s): PR(f"p1_{pol}_{s}") for s in SEEDS5}, man) for pol in ("served", "fixed")}
    F["P7"] = dict(by_policy=cp, max_by_route={rid: max(cp[p].get(rid, 0.0) for p in cp) for rid in routes})
    nr = collections.defaultdict(list); viol = 0; cen = 0; late = 0; ddl = collections.defaultdict(list); nrp = collections.defaultdict(list)
    for pol in ("served", "fixed"):
        for s in SEEDS5:
            for vid, r in PR(f"p1_{pol}_{s}").items():
                nr[r["type"]].append(r["nonrl_stop_s"]); nrp[(pol, r["type"])].append(r["nonrl_stop_s"])
                if r["depart_delay"] is not None: ddl[r["type"]].append(r["depart_delay"])
                viol += not lanes_follow_plan(r, routes[man[vid]["route"]]["lanes"])
                cen += r["depart"] is not None and r["arrival"] is None
                late += r["arrival"] is not None and r["arrival"] > P.PLATEAU[1]
    F["P8"] = dict(nonrl_stop_mean_s={t: float(np.mean(v)) for t, v in nr.items()},
                   by_policy={pol: {t: float(np.mean(nrp[(pol, t)])) for t in ("bus", "ambulance")} for pol in ("served", "fixed")},
                   note="固定轮转下多出的停车主要在 4919535#1_0: 1448442 的 58 m 短进口在 6 相位固定周期下排队回溢到上游 (RL 路口延误漏到进口车道之外), 两类相同")
    F["P9"] = dict(violations=viol)
    F["P10"] = dict(reported=True, censored_neutral=cen, arrived_after_3000_neutral=late, runs=2 * len(SEEDS5),
                    depart_delay_mean_s={t: float(np.mean(v)) for t, v in ddl.items()}, depart_delay_max_s={t: float(np.max(v)) for t, v in ddl.items()})
    P.dump(F, "final_calibration.json")
    for pol in ("served", "fixed", "random"):
        h = F["hard"][pol]; print(pol, "C1", [round(x, 3) for x in h["C1_growth"]], "C3", sum(h["C3_blocked"]), "C4", sum(h["C4_missed"]), "C6", sum(h["C6_spill"]), "C7", sum(h["C7_long_over_5pct"]),
                                  "C9", h["C9_lc_col_tp_missing"][0], "C14", max(h["C14_worst_excess"]), "C16", h["C16_by_type"], "C17", h["C17_cutins"])
    print("P1", {p: (v["n_seeds"], round(v["mean_diff_bus_minus_amb"], 3), [round(x, 3) for x in v["ci90"]]) for p, v in F["P1"].items() if p in ("served", "fixed")})
    print("P2", {k: v["identical"] for k, v in F["P2"].items()})
    print("P3", F["P3"]["union_counts"], {k: round(v, 1) for k, v in F["P3"]["conflict_visits_per_ep"].items()})
    print("P4", {k: round(v, 2) for k, v in F["P4"]["by_route"].items()}, {k: round(v, 1) for k, v in F["P4"]["with_opp_per_class_per_ep"].items()})
    print("P5", F["P5"]); print("P6", {t: {k: round(v, 2) for k, v in x["mean"].items()} for t, x in p6.items()}, F["P6"]["order_531"], F["P6"]["order_351"], round(F["P6"]["mirror_gap_s"], 3))
    print("P1 训练种子 0:", {k: round(v, 3) for k, v in F["P1"]["training_seed0"].items()}); print("P11", F["P11"]); print("P12", F["P12"])
    print("P7", {k: round(v, 2) for k, v in F["P7"]["max_by_route"].items()}); print("P8", F["P8"]); print("P9", F["P9"]); print("P10", F["P10"])


if __name__ == "__main__":
    main()
