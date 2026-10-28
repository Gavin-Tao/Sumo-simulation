"""pairs50lf 公共定义 (实现计划 experiments/analysis/PLAN_PAIRS50_LANEFIX_2026-10-28.md 第三版)。
路网与 meta 同 18h 定稿 (lanefix); 背景 = 18h 定稿背景逐位复制 (只读); 产出目录 nets/dublin/weekday_pairs50_lanefix。"""
import hashlib, json, sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
TOOLS18 = TOOLS.parent / "claude18h"
sys.path.insert(0, str(TOOLS18))
import common18 as C18  # noqa: E402  (路网路径、sha、车道路由、恒绿判定等全部沿用 18h)

REPO = C18.REPO; NETS = C18.NETS
NET, META, NET_SHA, META_SHA = C18.NET, C18.META, C18.NET_SHA, C18.META_SHA
SRC18 = C18.OUT                                   # 18h 定稿场景 (只读)
OUT = NETS / "weekday_pairs50_lanefix"
WORK = OUT / "build"

# 背景: 新文件夹里的文件名 → 18h 源文件名; 源文件 sha256 冻结 (2026-10-28 18h 定稿)
BG_COPY = {"background_cars_060.rou.xml": "background_cars_060.rou.xml",
           "background_cars_080.rou.xml": "background_cars_080.rou.xml",
           "background_cars_100.rou.xml": "background_cars_100.rou.xml",
           "background_gtfs_as_car.rou.xml": "bus_tt1_as_car.rou.xml",       # 262 辆 GTFS 公交以 car 身份 (D5)
           "vtypes_same.add.xml": "vtypes_same.add.xml"}
BG_SHA = {"background_cars_060.rou.xml": "25fa3f0e6c5938d6df1bb1a12462279715c1fca5266926d7f63ebcfaafad136e",
          "background_cars_080.rou.xml": "99430378aa803cb5c15f896cd9cd7d2eb5214828fae2c823ebadbb9a55521c15",
          "background_cars_100.rou.xml": "fb9ffdfded06a1306a6cc72afe1c06cf1a2e7e8bc45b5f8fb5ad86bff354fb0a",
          "bus_tt1_as_car.rou.xml": "80662abf9339ff5a7faef5f3557ddff21b4e305e6220d4089611190cb0cce4fc",
          "vtypes_same.add.xml": "911b25c75c2bdf278eb476319e844ea094c7346e819e8ffd9cc865993b267cd0"}
BG_DEFAULT = ("background_cars_100.rou.xml", "background_gtfs_as_car.rou.xml")   # D7: 所有时刻表共用

N_PAIRS = 50
K_ROUTES = (10, 12)
PAIRS_PER_ROUTE = (4, 6)          # 每条路线对数为偶数
PLATEAU = C18.PLATEAU             # (900, 3000)
MIN_HEADWAY = C18.MIN_HEADWAY     # 3 s
END_DRAIN = 4500                  # 标定/评估排空上限 (D6)
DELTA_CANDIDATES = (60.0, 90.0, 120.0, 180.0, 240.0, 300.0)   # 120 以上: 长路线在 6 相位路口 (固定轮转一轮约 162 s) 会追上前车
WINDOW_MARGIN = 120.0
ENTRY_LOAD_MAX = 450.0            # 18h 救护车入口规则
FIRST_RL_MIN_M = 100.0
OPP_MIN = 0.70
N_TT = 13                         # 1–5 训练, 6 标定, 7 最终测试, 8–13 评估
CALIB_TT = range(1, 7)            # 调参只用这些
HOLDOUT_SEEDS = {k: 7000 + k for k in range(7, 14)}   # 第 7–13 份: 预先写定, 不筛选、不重抽
TRAIN_SEED_BASE = 5100            # 第 1–6 份的起始种子 (可重抽, 记录被拒数)
PRIORITY_TABLES = {"531": {"ambulance": 5, "bus": 3, "car": 1},
                   "351": {"ambulance": 3, "bus": 5, "car": 1},
                   "111": {"ambulance": 1, "bus": 1, "car": 1}}


def sha256(p):
    return C18.sha256(p)


def check_inputs():
    """路网、meta、18h 背景源文件的 sha256 必须与冻结值一致。"""
    C18.check_hashes()
    for src, h in BG_SHA.items():
        got = sha256(SRC18 / src)
        if got != h:
            raise RuntimeError(f"18h 源文件被改动: {src} {got} != {h}")


def load_meta():
    return C18.load_meta()


def load_router(zonal=False):
    return C18.load_router(zonal)


def dump(obj, name):
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(WORK / name, "w"), indent=1, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, set) else str(o))


def load(name):
    return json.load(open(WORK / name))


def bg_entry_times(files=BG_DEFAULT):
    """背景车辆在各入口车道上的计划出发时刻 (车道 → 升序列表); 背景冻结, 配对车辆按它找空档。"""
    import collections, xml.etree.ElementTree as ET
    out = collections.defaultdict(list)
    for f in files:
        for v in ET.parse(OUT / f).getroot().iter("vehicle"):
            out[v.find("route").get("edges").split()[0] + "_" + v.get("departLane")].append(float(v.get("depart")))
    return {k: sorted(v) for k, v in out.items()}


def snap(t, taken, step=0.5, gap=MIN_HEADWAY):
    """离 t 最近的、与 taken 里所有时刻相距 ≥ gap 的时刻 (先试 t, 再向后、向前交替各走 step)。"""
    k = 0
    while True:
        for c in ((t,) if k == 0 else (t + k * step, t - k * step)):
            if c >= 0 and all(abs(c - x) >= gap for x in taken): return round(c, 1)
        k += 1


def write_vehicles(path, vehicles, comment):
    """vehicles: [{id, type, depart, lane (出发车道序号), edges}]; 按 (出发时刻, 写入顺序) 排序, 镜像文件因此与原份逐行对应。"""
    import xml.etree.ElementTree as ET
    root = ET.Element("routes"); root.append(ET.Comment(comment))
    for i, v in sorted(enumerate(vehicles), key=lambda x: (x[1]["depart"], x[0])):
        e = ET.SubElement(root, "vehicle", id=v["id"], type=v["type"], depart=f"{v['depart']:.1f}", departLane=str(v["lane"]), departSpeed="max")
        ET.SubElement(e, "route", edges=" ".join(v["edges"]))
    ET.indent(root, space="    "); ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
