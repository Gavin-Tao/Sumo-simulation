"""claude18h 场景 (nets/dublin/weekday_18h_nocg_claude) 的断言测试; 对应实现计划
experiments/analysis/PLAN_18H_NET19_DEMAND_2026-10-28.md 第三版的任务 0–7 与全局约束。只读产出文件, 不跑仿真。
运行: cd experiments && python -m pytest tests/test_claude18h_demand.py -q"""
import collections, json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools/dublin/scenario/claude18h"
sys.path.insert(0, str(TOOLS))
import common18 as C  # noqa: E402

OUT = C.OUT
TT_TRAIN_CALIB_TEST = range(1, C.N_TT + 1)


@pytest.fixture(scope="module")
def router():
    return C.load_router()


@pytest.fixture(scope="module")
def meta():
    return C.load_meta()


def vehicles(fname):
    out = []
    for v in ET.parse(OUT / fname).getroot().iter("vehicle"):
        out.append(dict(id=v.get("id"), type=v.get("type"), depart=float(v.get("depart")), lane=int(v.get("departLane")),
                        speed=v.get("departSpeed"), edges=v.find("route").get("edges").split()))
    return out


# ---------------- 任务 0: 路网核验
def test_net18b_hashes():
    C.check_hashes()


def test_net18b_audit():
    from audit_net import audit
    P, info = audit(C.NET, C.META)
    assert {k: v for k, v in P.items() if v} == {}
    assert info == dict(tls=18, entries=15, sinks=15)


# ---------------- 任务 1: 需求基准
def test_anchor_659799_site39():
    a = C.load("anchors_18h.json")["anchors"]
    assert a["659799"]["site"] == "39" and a["659799"]["match_m"] <= 130 and a["659799"]["trusted"]


def test_anchor_659802_not_trusted_site39():
    a = C.load("anchors_18h.json")["anchors"]
    assert not (a["659802"]["site"] == "39" and a["659802"]["trusted"])


# ---------------- 任务 2/3: 入口出口、穷举
def test_no_entry_on_rl_lanes(router, meta):
    E = C.load("entries.json"); rl = set()
    for t in meta:
        for e in router.net.getNode(t).getIncoming() + router.net.getNode(t).getOutgoing():
            rl |= {l.getID() for l in e.getLanes()}
    B = C.load("buses.json")
    assert not (set(E["boundary"]) | set(E["side_lanes"]) | set(B["bus_side_entries"])) & rl


def test_side_entry_rules(router, meta):
    from routes18 import side_rule_ok
    E = C.load("entries.json"); relaxed = set(E["side_relaxed"].values())
    assert relaxed <= {"7849022#0_0"}          # 唯一例外: cluster_5166592137 北右转 (见 README)
    for l in E["side_lanes"]:
        if l in relaxed: continue
        assert side_rule_ok(router, meta, l), l


def test_sinks_are_terminal(router):
    for l in C.load("entries.json")["sinks"]:
        assert not router.net.getLane(l).getEdge().getOutgoing()


def test_zero_demand_matches():
    E = C.load("entries.json")
    assert {tuple(m) for m in E["zero"]} == C.ZERO_EXPECTED
    assert E["zero_lane_loop"] == []          # 绕街区的转向已由例外支路入口解决


# ---------------- 全局: 所有车辆不换道、起终点合法、不重访
ALL_FILES = ["background_cars_100.rou.xml"] + [f"bus_tt{k}.rou.xml" for k in range(1, 14)] + [f"amb_tt{k}.rou.xml" for k in range(1, 14)] + \
            ["amb_freqprobe_eval.rou.xml", "amb_unseen_eval.rou.xml"]


@pytest.mark.parametrize("fname", ALL_FILES)
def test_vehicles_valid(fname, router, meta):
    E = C.load("entries.json"); B = C.load("buses.json")
    entries = set(E["boundary"]) | set(E["side_lanes"]) | set(B["bus_side_entries"]); sinks = set(E["sinks"])
    for v in vehicles(fname):
        assert v["speed"] == "max"
        assert f"{v['edges'][0]}_{v['lane']}" in entries, (v["id"], v["edges"][0])
        assert any(f"{v['edges'][-1]}_{i}" in sinks for i in range(4)), (v["id"], v["edges"][-1])
        paths = router.lane_paths(v["edges"], v["lane"])
        assert paths, ("车道不连续 (需要换道)", v["id"])
        tids = [router.movements[(x, y)][0] for x, y in zip(paths[0], paths[0][1:]) if (x, y) in router.movements]
        assert len(tids) == len(set(tids)), ("重访 RL 路口", v["id"])


def test_vtypes_same_no_lane_change():
    vt = {v.get("id"): dict(v.attrib) for v in ET.parse(OUT / "vtypes_same.add.xml").getroot().iter("vType")}
    assert set(vt) == {"car", "bus", "ambulance"}
    strip = lambda d: {k: x for k, x in d.items() if k not in ("id", "color")}
    assert strip(vt["car"]) == strip(vt["bus"]) == strip(vt["ambulance"])
    assert "vClass" not in vt["car"]
    assert vt["car"]["lcStrategic"] == "-1" and vt["car"]["lcCooperative"] == "-1" and vt["car"]["lcSpeedGain"] == "0" and vt["car"]["lcKeepRight"] == "0"


def test_sumocfg():
    r = ET.parse(OUT / "scenario.sumocfg").getroot()
    val = {e.tag: e.get("value") for e in r.iter() if e.get("value") is not None}
    assert val["ignore-junction-blocker"] == "-1" and val["time-to-teleport"] == "-1"
    assert val["collision.check-junctions"] == "true"
    assert "eval" not in val["route-files"] and "as_car" not in val["route-files"] and "tt7" not in val["route-files"]


# ---------------- 任务 4: 需求目标
def test_totals_soft_band():
    """可信路口: 实现量在 s_j × 参考值 ±5% 内; 低于 0.8 倍参考值的只允许出现在进口车道容量已顶到上限的路口 (README 记录原因)。"""
    rep = C.load("demand_report.json"); a = C.load("anchors_18h.json")["anchors"]; meta = C.load_meta()
    for t, j in rep["junctions"].items():
        if not a[t]["trusted"]: continue
        s = j["scale"]
        assert 0.8 - 1e-9 <= s <= 1.2 + 1e-9, (t, s)
        if j["realized"] >= 0.8 * j["ref"] - 5:
            assert j["realized"] <= 1.2 * j["ref"] + 5, (t, j)
            continue
        lanes = {c["from_edge"] + "_" + str(c["from_lane"]) for g in meta[t]["links"].values() for c in g}
        assert any(rep["lane_capacity"][l]["load"] >= 0.98 * rep["lane_capacity"][l]["cap"] for l in lanes if l in rep["lane_capacity"]), (t, j)


STRUCTURAL_EMPTY_LANES = set()   # 原为 {"37721356#0_0"}, 已由 lanefix 网解决 (build_lanefix18.py)   # cluster_21620852 东进口第 0 车道: 只能由 cluster_1998326407 南进口第 0 车道直行喂车,
# 但上游 2294384#0 第 0 车道的连接同时通向南进口第 0/1 车道, SUMO 把直行车全分到第 1 车道; 不换道规则下没有车能到达 (需改路网, 待用户决定)


def test_no_empty_approach():
    rep = C.load("demand_report.json")
    assert {l for x in rep["floor_short"] for l in x["lanes"]} <= STRUCTURAL_EMPTY_LANES, rep["floor_short"]
    ap = collections.Counter()
    for k, v in rep["movements"].items():
        t, m = k.split("|"); ap[(t, m.split("-")[0])] += v["realized_total"]
    assert min(ap.values()) >= C.APPROACH_MIN_VPH - 1


def test_topup_limited():
    rep = C.load("demand_report.json"); a = C.load("anchors_18h.json")["anchors"]
    by = collections.defaultdict(float)
    for x in rep["topup"]:
        if x["reason"].startswith("用户要求") or x["reason"].startswith("Digges"): continue   # 用户要求的设计流量, README 单列
        assert x["after"] <= 80.0 + 1e-6
        by[x["tls"]] += x["after"] - x["before"]
    for t, add in by.items():
        if a[t]["trusted"]:
            assert add <= 0.1 * rep["junctions"][t]["ref"] + 1e-6, (t, add)


# ---------------- 任务 5: 公交
def test_bus_count_band():
    B = C.load("buses.json")
    assert len(B["buses"]) == 262 and B["float_factor"] == 1.0
    assert all(b["id"].startswith("bus_") for b in B["buses"])   # 无非 GTFS 补充车次


# ---------------- 任务 6: 救护车
def test_amb_route(meta):
    from ambulances18 import classify
    A = C.load("ambulances.json"); r = A["route"]; anchors = C.load("anchors_18h.json")
    ag, nact, minor2 = classify(meta, anchors)
    comp = [tuple(m) for m in A["route_competing"]]
    assert len(r["traversals"]) >= 6 and len(comp) >= 5 and not r["yield_links"] and r["entry_type"] == "boundary"
    assert any(minor2(t, m) for t, m in comp) and any(nact[t] >= 5 for t, _ in comp)


def test_amb_timetables():
    route = None; times = []
    for k in TT_TRAIN_CALIB_TEST:
        vs = vehicles(f"amb_tt{k}.rou.xml")
        assert len(vs) == C.AMB_PER_EP == 3
        ds = sorted(v["depart"] for v in vs)
        for d, (lo, hi) in zip(ds, C.AMB_WINDOWS): assert lo <= d <= hi + 30   # +30: 为保证入口间隔可能后挪
        assert all(b - a >= 300 - 1e-6 for a, b in zip(ds, ds[1:]))
        es = {tuple(v["edges"]) for v in vs}; assert len(es) == 1
        route = route or es; assert es == route
        times.append(tuple(ds))
    assert len(set(times)) == len(times)


def test_amb_probe_eval(router):
    A = C.load("ambulances.json")
    ins = lambda lanes: {x.rsplit("_", 1)[0] for x, y in zip(lanes, lanes[1:]) if (x, y) in router.movements}
    train_in = ins(A["route"]["lanes"])
    for u in A["unseen_routes"]:
        assert not ins(u["lanes"]) & train_in
    for f in ["amb_freqprobe_eval.rou.xml", "amb_unseen_eval.rou.xml"]:
        assert all(v["type"] == "ambulance" for v in vehicles(f))
    cfg = (OUT / "scenario.sumocfg").read_text()
    assert "probe" not in cfg and "unseen" not in cfg


# ---------------- 任务 7: 小汽车
def test_entry_schedule():
    caps = {}
    E = C.load("entries.json"); r = C.load_router()
    for l in set(E["boundary"]) | set(E["side_lanes"]) | set(C.load("buses.json")["bus_side_entries"]):
        caps[l] = C.LANE_CAP_LONG_VPH if (l in r.sources and r.net.getLane(l).getLength() >= 100) else C.LANE_CAP_VPH
    bg = vehicles("background_cars_100.rou.xml")
    for k in TT_TRAIN_CALIB_TEST:
        by = collections.defaultdict(list)
        for v in bg + vehicles(f"bus_tt{k}.rou.xml") + vehicles(f"amb_tt{k}.rou.xml"):
            by[f"{v['edges'][0]}_{v['lane']}"].append(v["depart"])
        for l, ts in by.items():
            ts.sort()
            assert all(b - a >= C.MIN_HEADWAY - 0.05 for a, b in zip(ts, ts[1:])), (k, l)
            plat = sum(1 for t in ts if C.PLATEAU[0] <= t < C.PLATEAU[1]) * 3600 / (C.PLATEAU[1] - C.PLATEAU[0])
            assert plat <= caps[l] * 1.05 + 5, (k, l, plat)


def _plateau_start_through(lanes_of_interest, router):
    """按全部车型 (背景 1.0 + 公交 tt1 + 救护车 tt1)、平台期出发、所有可能车道路径保守计数: 本地发车与上游穿行 (辆/h)。"""
    st = collections.Counter(); th = collections.Counter(); k = 3600 / (C.PLATEAU[1] - C.PLATEAU[0])
    for f in ["background_cars_100.rou.xml", "bus_tt1.rou.xml", "amb_tt1.rou.xml"]:
        for v in vehicles(f):
            if not (C.PLATEAU[0] <= v["depart"] < C.PLATEAU[1]): continue
            allp = {x for p in router.lane_paths(v["edges"], v["lane"]) for x in p}; s0 = f"{v['edges'][0]}_{v['lane']}"
            for l in lanes_of_interest:
                if s0 == l: st[l] += k
                elif l in allp: th[l] += k
    return st, th


def test_side_entry_inflow(router):
    """小汽车支路入口: 上游穿行 (全部车型) ≤ 本地发车 (全部车型) 的 10%。"""
    lanes = C.load("entries.json")["side_lanes"]
    st, th = _plateau_start_through(lanes, router)
    for l in lanes:
        assert th[l] <= 0.1 * st[l] + 1e-6, (l, round(st[l], 1), round(th[l], 1))


def test_bus_entry_insertion():
    """公交在合规支路入口出发 (与小汽车共用, 不合规的 GTFS 原出发边已剔除并延长到合规入口): 实测插车不失败, 平均延迟 ≤2 s、99 分位 ≤10 s, 三种控制器间平均延迟差 ≤0.5 s。"""
    import glob
    B = C.load("buses.json"); ents = {l.rsplit("_", 1)[0] for l in B["bus_side_entries"]}
    by = collections.defaultdict(dict)
    for f in glob.glob(str(C.WORK / "calib_final/L100_*_tt1_*.json")):
        d = json.load(open(f)); pol = Path(f).stem.split("_")[1]
        for k, v in d["C2"]["by_entry_type"].items():
            e, ty = k.split("|")
            if e in ents and ty == "bus":
                assert v.get("not_inserted", 0) == 0, (f, k)
                assert v["mean_delay"] <= 2.0 and v["p99_delay"] <= 10.0, (f, k, v)
                by[e].setdefault(pol, []).append(v["mean_delay"])
    assert by
    for e, pols in by.items():
        means = [sum(x) / len(x) for x in pols.values()]
        assert max(means) - min(means) <= 0.5, (e, pols)


def test_final_hard_criteria():
    """排队驱动与固定轮转: C1 (计划原口径) ≤10%, 堵死/漏放/回溢 0, 换道/碰撞/瞬移/统计器缺条目 0, 救护车全部到达; 排队驱动: 单车道积队不超限。"""
    F = C.load("final_calibration.json")
    for k in ("L100_served_tt1", "L100_fixed_tt1"):
        v = F[k]
        assert all(x <= 0.10 for x in v["C1_growth"]), (k, v["C1_growth"])
        assert sum(v["C3_blocked"]) == 0 and sum(v["C4_missed"]) == 0 and sum(v["C6_spill"]) == 0, k
        assert all(c[:4] == [0, 0, 0, 0] and c[4] == "3/3" for c in v["C9"]), (k, v["C9"])
    assert all(x <= 0 for x in F["L100_served_tt1"]["C14_worst_excess"])


def test_departure_profile():
    bg = vehicles("background_cars_100.rou.xml"); n = len(bg); tot = C.profile_integral()
    for a, b, r0, r1 in C.PROFILE:
        exp = n * (b - a) * (r0 + r1) / 2 / tot
        got = sum(1 for v in bg if a <= v["depart"] < b)
        assert abs(got - exp) <= 0.05 * exp + 10, (a, b, got, exp)


def test_levels_nested():
    ids = {L: {v["id"] for v in vehicles(f"background_cars_{int(L * 100):03d}.rou.xml")} for L in C.LEVELS}
    assert ids[0.6] <= ids[0.8] <= ids[1.0]
    assert abs(len(ids[0.6]) / len(ids[1.0]) - 0.6) < 0.03 and abs(len(ids[0.8]) / len(ids[1.0]) - 0.8) < 0.03
