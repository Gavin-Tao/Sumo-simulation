"""claude18h 公共定义 (实现计划 experiments/analysis/PLAN_18H_NET19_DEMAND_2026-10-28.md 第三版)。
路网 NET18B = dublin_8std_splitslot18_southclosed_corridor_nocg (sha256 锁定); 产出目录 nets/dublin/weekday_18h_nocg_claude。"""
import hashlib, json, os, sys
from pathlib import Path
REPO = Path("/home/xiaowen/sumo-rl")
NETS = REPO / "nets/dublin"
# 2026-10-28 晚: 用户要求解决结构性空车道 → 在 nocg 网上只改非 RL 节点 658464 的一条连接 (build_lanefix18.py); 18 个路口动作表与 nocg 逐项一致
BASE_NET = NETS / "dublin_8std_splitslot18_southclosed_corridor_nocg.net.xml"      # 用户的原 NET18B, 未改动
BASE_NET_SHA = "18497bb0ba1f467400c6eda591c69405e69ceaf2420f653a579d8e9cc3c1c626"
NET = NETS / "dublin_8std_splitslot18_southclosed_corridor_nocg_lanefix.net.xml"
META = NETS / "dublin_8std_splitslot18_southclosed_corridor_nocg_lanefix_meta.json"
NET_SHA = "bc6ed2dbc9dd875f888addeaa006b05f3162eed7d82e1891c2a0e52ea7fabc97"
META_SHA = "71f661803b34fb5494e8862d6b11a4c14555044d7fb1e377e82d809e1d1cf910"
OUT = NETS / "weekday_18h_nocg_claude"
WORK = OUT / "build"
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(NETS / "build_corridor15"))      # routing.LaneRoutes
sys.path.insert(0, str(REPO / "experiments/tools/dublin"))  # common (SCATS / UTM)

CG = "cluster_1426051358_1426100603_3206752725_3783453672_3783453675"
ZERO_EXPECTED = {("389281", "S-R"), ("389332", "E-L")}   # lanefix 网上 659805 N-L 有了合规路线 (nocg 原网上是 3 个)
SIDE_NEEDED = {("cluster_1998326407_2876012598_4781977262_658429", "S-R"),
               ("cluster_2498303858_3624079904_6141082777_6141082778_9936781", "S-R"),
               ("cluster_5166592137_5166592140_557198554_57744042", "W-R")}

# 时间曲线 (计划 §3.4): (起, 止, 起点相对率, 终点相对率); 积分 = 3060 s = 0.85 h
PROFILE = [(0, 300, 0.4, 0.4), (300, 900, 0.4, 1.0), (900, 3000, 1.0, 1.0), (3000, 3600, 0.7, 0.7)]
PLATEAU = (900, 3000)
LANE_CAP_VPH = 600       # 每条入口车道平台期发车上限 (起始值)
LANE_CAP_LONG_VPH = 900  # 长度 ≥100 m 的边界入口放宽 (偏离计划: 600 时 74233405#1 等单车道入口卡住 SCATS 量; 插车影响由 C2 检验)
SAT_FLOW = 1800.0; GREEN_FRAC = 0.92          # 饱和流率 (辆/h/车道); 扣黄灯后的绿灯比例
LANE_UTIL = {"single": 0.5, "shared": 0.45}     # 进口车道利用率上限: 单一转向 / 多转向共用 (偏离计划: 由 "只报告" 改为线性规划硬约束, 标定显示共享单车道进口是插车延迟主因)
APPROACH_MIN_VPH = 100.0   # 每个 RL 进口平台期最低流量 (用户 2026-10-28: 不要让进口一直没车)
LANE_MIN_VPH = 30.0        # 每条进口车道最低流量 (有可行转向的车道)
MINOR_CAP = lambda major: max(0.0, 0.6 * (1100.0 - major))   # 非 RL 让行支路的通过量估计 (按冲突主路流量), 取代 "主路 ≤150" 预算
MIN_HEADWAY = 3.0        # 同一入口车道相邻计划出发间隔
YIELD_BUDGET = 150.0     # 让行口冲突主路平台期流率上限 (起始值)
YIELD_BUDGET_SPECIAL = {"2294382#1": 100.0}
LEVELS = (0.6, 0.8, 1.0)
N_TT = 7                 # 时刻表: 1–5 训练, 6 标定, 7 最终测试
AMB_PER_EP = 3
AMB_WINDOWS = [(900, 1500), (1500, 2100), (2100, 2700)]

def profile_integral():
    return sum((b - a) * (r0 + r1) / 2 for a, b, r0, r1 in PROFILE)

def profile_rate(t):
    for a, b, r0, r1 in PROFILE:
        if a <= t < b:
            return r0 + (r1 - r0) * (t - a) / (b - a)
    return 0.0

def profile_inverse(u):
    """u ∈ [0,1) → 时刻 t, 使累计份额 = u (按曲线的分段线性密度精确求逆)。"""
    total = profile_integral(); target = u * total; acc = 0.0
    for a, b, r0, r1 in PROFILE:
        seg = (b - a) * (r0 + r1) / 2
        if acc + seg >= target:
            rem = target - acc; L = b - a; k = (r1 - r0) / L
            if abs(k) < 1e-12:
                return a + rem / r0
            # r0*x + k x^2/2 = rem
            import math
            x = (-r0 + math.sqrt(r0 * r0 + 2 * k * rem)) / k
            return a + x
        acc += seg
    return PROFILE[-1][1] - 1e-6

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def check_hashes():
    assert sha256(NET) == NET_SHA, "NET18B 路网哈希不符, 先重跑任务 0"
    assert sha256(META) == META_SHA, "META18B 哈希不符, 先重跑任务 0"

def load_meta():
    return json.load(open(META))["tls"]

def load_router(zonal=False):
    from routing import LaneRoutes
    return LaneRoutes(str(NET), str(META), zonal_sources=zonal)

def always_green(meta):
    out = set()
    for tid, t in meta.items():
        greens = [a["state"] for a in t["actions"].values() if a.get("valid")]
        slots = {}
        for i, g in t["links"].items():
            for c in g:
                slots.setdefault(c["approach"] + "-" + c["turn"], set()).add(int(i))
        for mv, idx in slots.items():
            if all(any(s[i] in "Gg" for i in idx) for s in greens):
                out.add((tid, mv))
    return out

def dump(obj, name):
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / name
    json.dump(obj, open(p, "w"), indent=1, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else str(o))
    return p

def load(name):
    return json.load(open(WORK / name))
