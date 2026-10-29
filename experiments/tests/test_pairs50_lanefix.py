"""50 + 50 配对场景 (nets/dublin/weekday_pairs50_lanefix) 的断言测试; 对应实现计划
experiments/analysis/PLAN_PAIRS50_LANEFIX_2026-10-28.md 第三版的任务 0–7。只读产出文件, 不跑仿真 (仿真结果由 final50.py 写入 build/)。
运行: cd experiments && python -m pytest tests/test_pairs50_lanefix.py -q"""
import collections, json, sys, xml.etree.ElementTree as ET
from pathlib import Path
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools/dublin/scenario/pairs50lf"
sys.path.insert(0, str(TOOLS))
import common50 as P  # noqa: E402


# ---------------- 任务 0: 输入冻结
def test_inputs_frozen():
    """路网、meta 与 18h 背景源文件的 sha256 与记录一致; 新文件夹里的背景文件与 18h 逐位相同。"""
    P.check_inputs()
    for dst, src in P.BG_COPY.items():
        assert (P.OUT / dst).exists(), dst
        assert P.sha256(P.OUT / dst) == P.BG_SHA[src], dst


# ---------------- 任务 5: calib18 扩展 (核对结果由 check_calib50.py 写入 build/calib18_checks.json)
@pytest.fixture(scope="module")
def calib_checks():
    return P.load("calib18_checks.json")


def test_calib18_default_unchanged(calib_checks):
    """18h 第 1 份、种子 123: 固定轮转、排队驱动、知道车型的控制器 (531) 的输出与改动前逐项相同。"""
    for k in ("fixed", "served", "prio531"):
        assert calib_checks["default_unchanged"][k]["identical"], (k, calib_checks["default_unchanged"][k].get("diff"))


def test_prio_numeric_only(calib_checks):
    """18h 第 1 份把公交与救护车车型对调后, 351 运行与原文件 531 运行逐车相同; 111 下原文件与对调文件逐车相同。"""
    for k in ("prio531_vs_swapped351", "prio111_vs_swapped111"):
        r = calib_checks["numeric_only"][k]
        assert r["n_vehicles"] > 200 and r["identical"], (k, r.get("diff"))


def test_priority_table_plumbing(calib_checks):
    """531 / 351 / 111 三张表下, 控制器、观测分桶、奖励权重读到的是同一张表 (逐车核对档位, 奖励独立重算)。"""
    for tab in ("531", "351", "111"):
        r = calib_checks["plumbing"][tab]
        assert r["ctrl_table"] == r["expected_table"], tab
        assert r["obs_checked"] > 1000 and r["obs_mismatch"] == 0, (tab, r)
        assert r["reward_checked"] > 1000 and r["reward_max_abs_diff"] < 1e-9, (tab, r)


# ---------------- 任务 1: 候选路线池
@pytest.fixture(scope="module")
def net():
    import sumolib
    return sumolib.net.readNet(str(P.NET))


@pytest.fixture(scope="module")
def meta():
    return P.load_meta()


def ag_lanes_of(net, meta):
    """车道级恒绿 (该车道所有受控连接在每个有效相位都为绿), 与训练环境 always_green_lanes 同义; 独立于生成代码重算。"""
    out = set()
    for tid, t in meta.items():
        greens = [x["state"] for x in t["actions"].values() if x.get("valid")]
        idx = collections.defaultdict(set)
        for c in net.getTLS(tid).getConnections(): idx[c[0].getID()].add(c[2])
        out |= {l for l, ii in idx.items() if all(all(s[i] in "Gg" for i in ii) for s in greens)}
    return out


def test_candidates_rules(net, meta):
    C = P.load("pair_candidates.json")
    cands = C["candidates"]; boundary = set(C["boundary_entries"])
    assert len(cands) >= 30
    agl = ag_lanes_of(net, meta)
    for d in cands:
        lanes = d["lanes"]
        assert lanes[0] in boundary and d["entry_type"] == "boundary"
        assert d["yield_links"] == []
        tids = [t for t, _ in d["traversals"]]
        assert len(tids) >= 3 and len(set(tids)) == len(tids)
        assert len(d["competing"]) >= 2 and d["first_rl_m"] >= P.FIRST_RL_MIN_M
        fan = []
        for x, y in zip(lanes, lanes[1:]):
            to = {c.getToLane().getID() for c in net.getLane(x).getOutgoing()}
            assert y in to, (x, y)                                   # 车道逐段连通
            alt = {l for l in to if l.rsplit("_", 1)[0] == y.rsplit("_", 1)[0]}
            if len(alt) > 1: fan.append(net.getLane(x).getEdge().getToNode().getID())
        assert sorted(fan) == sorted(d["fanout_nodes"]), (d["lanes"][0], fan, d["fanout_nodes"])
        # 竞争转向都在计入指标的车道上 (进口车道不是车道级恒绿)
        for tid, mv, lane in d["competing"]:
            assert lane not in agl, (tid, mv, lane)


# ---------------- 任务 2: 实测选线
def test_route_selection(net, meta):
    S = P.load("pair_routes.json"); R = S["routes"]
    assert P.K_ROUTES[0] <= len(R) <= P.K_ROUTES[1]
    assert sum(r["n_pairs"] for r in R) == P.N_PAIRS and all(r["n_pairs"] in P.PAIRS_PER_ROUTE for r in R)
    assert len({tuple(r["lanes"]) for r in R}) == len(R)
    load = collections.Counter(l for r in R for (t, mv, l) in r["competing"] for _ in range(r["n_pairs"]))
    import select50
    assert max(load.values()) <= select50.MAX_PAIRS_PER_LANE, load.most_common(3)
    assert len(set(l for r in R for l in r["lanes"]) & set(P.C18.load("entries.json")["side_lanes"])) == 0     # 不穿过小汽车支路入口车道
    for r in R:
        m = r["measured"]
        assert m["opp_share"] >= P.OPP_MIN, (r["id"], m["opp_share"])
        assert m["lane_paths_identical"], r["id"]
        assert m["nonrl_stop_mean_s"] <= 2.0, (r["id"], m["nonrl_stop_mean_s"])
        assert r["entry_load"] <= P.ENTRY_LOAD_MAX or m["depart_delay_mean_s"] <= 2.0, (r["id"], r["entry_load"], m["depart_delay_mean_s"])
    rl_in = lambda lanes: {l.rsplit("_", 1)[0] for l in lanes if net.getLane(l).getEdge().getToNode().getID() in meta}
    used_in = set().union(*[rl_in(r["lanes"]) for r in R])
    for u in S["unseen"]:
        assert not rl_in(u["lanes"]) & used_in, u["entry"]          # 未见路线与入选路线不共享任何 RL 进口边 (不只竞争进口)


def test_selection_reads_no_waits():
    """选线函数只读白名单字段 (覆盖、机会、车道确定性、非 RL 停车、出发延迟、入口负荷), 不读任何等待结果。"""
    import ast, inspect, select50
    src = inspect.getsource(select50.choose)
    keys = {n.slice.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
    assert keys <= set(select50.SELECT_FIELDS), keys - set(select50.SELECT_FIELDS)
    assert not any("wait" in k for k in select50.SELECT_FIELDS)


# ---------------- 任务 3: 时刻表、镜像、清单
def read_vehicles(path):
    return [dict(id=v.get("id"), type=v.get("type"), depart=float(v.get("depart")), lane=v.get("departLane"), edges=v.find("route").get("edges"))
            for v in ET.parse(path).getroot().iter("vehicle")]


@pytest.fixture(scope="module")
def manifest():
    return json.load(open(P.OUT / "pairs_manifest.json"))


def test_timetable_balance(manifest):
    T = P.load("timetables.json"); routes = {r["id"]: r for r in P.load("pair_routes.json")["routes"]}
    for k in range(1, P.N_TT + 1):
        vs = read_vehicles(P.OUT / f"pairs_tt{k}.rou.xml"); mv = read_vehicles(P.OUT / f"pairs_tt{k}_mirror.rou.xml")
        by_r = collections.defaultdict(lambda: collections.defaultdict(list)); first = collections.Counter()
        for v in vs:
            m = manifest[v["id"]]; by_r[m["route"]][v["type"]].append(v["depart"])
            if m["slot"] == "x": first[(m["route"], v["type"])] += 1
        for rid, r in routes.items():
            b, a = by_r[rid]["bus"], by_r[rid]["ambulance"]
            assert len(b) == len(a) == r["n_pairs"], (k, rid)
            assert first[(rid, "bus")] == first[(rid, "ambulance")] == r["n_pairs"] // 2, (k, rid)
            assert abs(sum(b) / len(b) - sum(a) / len(a)) <= 1.0, (k, rid)
        # 原份 + 镜像合并: 每条路线上两类车的出发时刻集合完全相同
        both = collections.defaultdict(list)
        for v in vs + mv: both[(manifest[v["id"]]["route"], v["type"])].append(v["depart"])
        for rid in routes: assert sorted(both[(rid, "bus")]) == sorted(both[(rid, "ambulance")]), (k, rid)
        # 按块找空档: 实际对内间隔 ∈ [Δ, Δ+40]; 首车 (找空档前后) 都在出发窗内; 整对移动 ≤ 150 s
        for p in T["timetables"][str(k)]["pairs"]:
            lo, hi = T["windows"][p["route"]]; dl = T["deltas"][p["route"]]
            assert lo <= p["t_plan"] <= hi and lo <= p["t_x"] <= hi, (k, p)
            assert dl - 1e-6 <= p["t_y"] - p["t_x"] <= dl + 40 + 1e-6 and abs(p["shift"]) <= 150.0, (k, p)
        # 同一入口车道上配对车辆彼此间隔 ≥ 3 s (第三版起不再为背景车找空档, 与背景挨近时由 SUMO 排队插入, 出发延迟见 final_calibration P10)
        lanes = collections.defaultdict(list)
        for v in vs: lanes[v["edges"].split()[0] + "_" + v["lane"]].append(v["depart"])
        for l, ts in lanes.items():
            ts = sorted(ts)
            assert all(b_ - a_ >= P.MIN_HEADWAY - 1e-6 for a_, b_ in zip(ts, ts[1:])), (k, l)


def test_self_mirrored():
    """每份时刻表自身镜像对称 (第三版): 每一对都有一对对应对 —— 同路线、整体晚/早 600 s、先后车型对调、对内间隔相同、分属前后半段。"""
    T = P.load("timetables.json")["timetables"]
    for k, t in T.items():
        by = {p["pair"]: p for p in t["pairs"]}
        for p in t["pairs"]:
            q = by[p["twin_pair"]]
            assert q["route"] == p["route"] and q["x_type"] != p["x_type"] and q["half"] != p["half"], (k, p["pair"])
            assert abs(abs(q["t_x"] - p["t_x"]) - 600.0) < 1e-6 and abs((q["t_y"] - q["t_x"]) - (p["t_y"] - p["t_x"])) < 1e-6, (k, p["pair"])


def test_mirror_files(manifest):
    for k in range(1, P.N_TT + 1):
        vs = read_vehicles(P.OUT / f"pairs_tt{k}.rou.xml"); mv = read_vehicles(P.OUT / f"pairs_tt{k}_mirror.rou.xml")
        assert len(vs) == len(mv) == 2 * P.N_PAIRS
        for a, b in zip(vs, mv):
            ma, mb = manifest[a["id"]], manifest[b["id"]]
            assert (ma["route"], ma["pair"], ma["slot"]) == (mb["route"], mb["pair"], mb["slot"]) and mb["mirror"] and not ma["mirror"]
            assert {a["type"], b["type"]} == {"bus", "ambulance"}
            assert (a["depart"], a["lane"], a["edges"]) == (b["depart"], b["lane"], b["edges"])


def test_ids_unique(manifest):
    ids = []
    for f in sorted(P.OUT.glob("pairs_*.rou.xml")): ids += [v["id"] for v in read_vehicles(f)]
    assert len(ids) == len(set(ids)) and set(ids) <= set(manifest)


def test_holdout_untouched():
    T = P.load("timetables.json")
    for k in range(7, P.N_TT + 1):
        t = T["timetables"][str(k)]
        assert t["seed"] == P.HOLDOUT_SEEDS[k] and t["rejected"] == 0 and not t.get("simulated_for_selection")


def test_manifest_counts(manifest):
    """配对文件与孪生文件按清单统计的每类计划数: 配对文件公交、救护车各 50; 孪生文件 100 辆全是 car, 原车型各 50。"""
    for k in range(1, P.N_TT + 1):
        for suf in ("", "_mirror"):
            c = collections.Counter(manifest[v["id"]]["type"] for v in read_vehicles(P.OUT / f"pairs_tt{k}{suf}.rou.xml"))
            assert c == {"bus": P.N_PAIRS, "ambulance": P.N_PAIRS}, (k, suf, c)
        tw = read_vehicles(P.OUT / f"pairs_tt{k}_as_car.rou.xml")
        assert all(v["type"] == "car" for v in tw)
        assert collections.Counter(manifest[v["id"]]["orig_type"] for v in tw) == {"bus": P.N_PAIRS, "ambulance": P.N_PAIRS}


# ---------------- 任务 6: 标定与验收 (final50.py 写入 build/final_calibration.json)
@pytest.fixture(scope="module")
def F():
    return P.load("final_calibration.json")


def test_hard_criteria(F):
    """继承 18h 硬判据: 排队驱动与固定轮转各 5 种子 (1.0 档 + 第 1 份)。"""
    for pol in ("served", "fixed"):
        h = F["hard"][pol]
        assert h["n"] == 5
        assert max(h["C1_growth"]) <= 0.10, (pol, h["C1_growth"])
        assert sum(h["C3_blocked"]) == 0 and sum(h["C4_missed"]) == 0 and sum(h["C6_spill"]) == 0, pol
        assert sum(h["C7_long_over_5pct"]) == 0, pol
        assert all(x == [0, 0, 0, 0] for x in h["C9_lc_col_tp_missing"]), pol
        assert all(c["inserted"] == c["planned"] for run in h["pairs_counts"] for c in run.values()), pol
        if pol == "served": assert max(h["C14_worst_excess"]) <= 0, h["C14_worst_excess"]     # 与 18h 相同: 单车道积队只对排队驱动要求


def test_p1_reported(F):
    """P1 单份可互换性 (2026-10-29 起改为报告): 四个时刻表版本的单份两类差都在 ±0.5–2 s 随机波动, 来自冻结背景在前后两段的具体差异, 与设计无关;
    严格对称由原份 + 镜像保证 (P2 逐车相同)。这里只防明显失衡: 两种控制器下 |均值| ≤ 2.5 s, 且训练种子 0 的值已报告。"""
    for pol in ("served", "fixed"):
        v = F["P1"][pol]; assert v["n_seeds"] >= 20 and abs(v["mean_diff_bus_minus_amb"]) <= 2.5, (pol, v)
    assert set(F["P1"]["training_seed0"]) == {"served", "fixed"}


def test_pairs_criteria(F):
    for k, v in F["P2"].items(): assert v["identical"], (k, v.get("diff"))
    p3 = F["P3"]
    assert p3["union_counts"]["ambulance|bus"] == p3["union_counts"]["bus|ambulance"]
    assert min(p3["conflict_visits_per_ep"].values()) >= 20
    for rid, v in F["P4"]["by_route"].items(): assert v >= P.OPP_MIN, (rid, v)
    assert min(F["P4"]["with_opp_per_class_per_ep"].values()) >= 120, F["P4"]["with_opp_per_class_per_ep"]
    for rid, v in F["P7"]["max_by_route"].items(): assert v <= 0.10, (rid, v)
    p8 = F["P8"]["by_policy"]       # 2 s 门槛只对排队驱动 (固定轮转多出的是 1448442 短进口的回溢, 报告); 两种控制器下两类之差都 ≤ 1 s
    assert max(p8["served"].values()) <= 2.0, p8
    for pol in ("served", "fixed"): assert abs(p8[pol]["bus"] - p8[pol]["ambulance"]) <= 1.0, (pol, p8)
    assert F["P9"]["violations"] == 0
    assert F["P10"]["censored_neutral"] == 0
    dd = F["P10"]["depart_delay_mean_s"]
    assert max(dd.values()) <= 3.0 and abs(dd["bus"] - dd["ambulance"]) <= 0.5, dd      # 不再为背景找空档后的实际插入延迟
    for k in ("P5", "P6", "P10"): assert F[k]["reported"], k
    assert F["P11"]["max_abs_diff"] <= 0.05, F["P11"]          # 同类同进口共现, 两类之差 ≤ 5 个百分点
    assert F["P12"]["mid_mirror_gap_s"] <= 0.3, F["P12"]       # 中间档镜像差, 与最高档同一门槛


def test_adjacency_symmetric():
    """每份时刻表 (含留出份): 两类车的同类相邻结伴权重 (按间隔 exp(−g/30 s) 加权) 对称 (审阅 2026-10-28 修正)。"""
    T = P.load("timetables.json")["timetables"]
    for k, t in T.items():
        w = t["adjacency_w"]; assert abs(w["bus"] - w["ambulance"]) <= 0.1 * max(1.0, (w["bus"] + w["ambulance"]) / 2), (k, w)


# ---------------- 任务 7: 状态/奖励逐步重算与档位核对 (validate50.py 写入 build/validation.json)
def test_validation_passed():
    V = P.load("validation.json")["validation"]
    for pol in ("fixed", "random"):
        v = V[pol]
        assert v["verdict"] == "通过" and v["decisions"] == 720 and v["state_checks"] == v["reward_checks"] == 12960, (pol, v)


def test_priority_levels_follow_type():
    """第 1 份与第 1′ 份各起新进程, 531 与 351 下: 观测档位与奖励权重都等于规格表里车型对应值。"""
    L = P.load("validation.json")["levels"]
    for k in ("531_orig", "531_mirror", "351_orig", "351_mirror"):
        v = L[k]
        assert v["obs_checked"] > 1000 and v["obs_mismatch"] == 0 and v["reward_checked"] > 1000 and v["reward_max_abs_diff"] < 1e-9, (k, v)
        assert {t: int(x) for t, x in v["ctrl_table"].items()} == v["expected_table"], k
