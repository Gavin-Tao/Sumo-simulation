"""18 个 RL 路口位置图 (2026-10-27): 路网灰线 + 路口按 best×10 停车/visit 着色 (单色序列), 锁死起点加黑圈, 标路口号与街名。
用法: python junction_map.py <exp> --out png"""
import argparse, glob, json, math, re, collections, os, sys, html
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK SC", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
import sumolib
import matplotlib.font_manager as fm
try:
    CJK = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"); CJK.get_name()
except Exception:
    CJK = None
ap = argparse.ArgumentParser(); ap.add_argument("exp", default="459"); ap.add_argument("--out", required=True); a = ap.parse_args()
NET = "../nets/dublin/dublin_8std_splitslot.net.xml"; net = sumolib.net.readNet(NET)
meta = json.load(open("../nets/dublin/dublin_8std_splitslot_meta.json"))["tls"]
short = lambda t: t.split("_")[0] + ("_" + t.split("_")[1] if t.startswith("cluster") else "")
# 停车/visit (best×10 逐秒)
files = sorted(glob.glob(f"analysis/data/pervehicle/exp{a.exp}_ep*_seed*_best_coll1s.json")); V = collections.Counter(); St = collections.Counter()
for f in files:
    d = json.load(open(f))
    for vid, v in d["vehicles"].items():
        for t, visits in v["tls"].items():
            for r in visits: V[t] += 1; St[t] += r["stopped"]
spv = {t: St[t] / V[t] for t in V}
LOCK_NEW = {"cluster_21620852_2383668110_2383668118_4009885977", "659802", "389281"}; LOCK_OLD = {"cluster_1998326407_2876012598_4781977262_658429"}
fig, ax = plt.subplots(figsize=(13, 10), dpi=150); ax.set_facecolor("#ffffff")
xs, ys = [], []
for e in net.getEdges():
    sh = e.getShape(); x = [p[0] for p in sh]; y = [p[1] for p in sh]; xs += x; ys += y
    ax.plot(x, y, color="#d9d9d9", lw=0.8, zorder=1)
pts = {t: net.getNode(t).getCoord() for t in meta}
cmap = plt.get_cmap("Blues"); vmax = max(spv.values()); norm = plt.Normalize(0, vmax)
for t, (x, y) in pts.items():
    c = cmap(0.15 + 0.85 * norm(spv.get(t, 0)))
    ax.scatter([x], [y], s=170, color=c, edgecolor="#ffffff", linewidth=1.5, zorder=3)
    if t in LOCK_NEW: ax.scatter([x], [y], s=420, facecolor="none", edgecolor="#000000", linewidth=1.6, zorder=4)
    if t in LOCK_OLD: ax.scatter([x], [y], s=420, facecolor="none", edgecolor="#000000", linewidth=1.6, linestyle=(0, (2, 2)), zorder=4)
    names = sorted({html.unescape(e.getName()) for e in net.getNode(t).getIncoming() if e.getName()}); nm = " / ".join(n.replace("Street", "St").replace("Square", "Sq") for n in names[:2])
    bold = t in sorted(spv, key=lambda k: -St[k])[:4]
    OFF = {"1448442": (-12, -4, "right"), "389281": (-14, -14, "right"), "32236377": (10, 16, "left"), "135109534": (-12, -24, "right"), "cluster_135109528_9101656": (16, -12, "left"),
           "26165090": (14, -20, "left"), "389332": (8, 14, "left"), "1448436": (-12, 2, "right"), "26165126": (-12, 4, "right"), "cluster_1448439_2464758361": (12, 4, "left")}
    dx, dy, ha = OFF.get(t, (9, 7, "left"))
    ax.annotate(f"{short(t)}\n{nm}\n{spv.get(t, 0):.2f} s", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7.2, color="#1f1f1f", fontweight=("bold" if bold else "normal"), zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="none", alpha=0.8))
pad = 120; ax.set_xlim(min(p[0] for p in pts.values()) - pad, max(p[0] for p in pts.values()) + pad + 150); ax.set_ylim(min(p[1] for p in pts.values()) - pad, max(p[1] for p in pts.values()) + pad)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([]); cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01); cb.set_label("停车时间 / visit (s), 新541 best × 10 种子" if CJK else "Stopped time / visit (s), new-541 best x 10 seeds", fontsize=9, fontproperties=CJK)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", color="#000000", markerfacecolor="none", markersize=11, lw=0, markeredgewidth=1.4, label="实线黑圈: 新口径扫描里的锁死起点 (cluster_21620852, 659802, 389281)"),
                   Line2D([], [], marker="o", color="#888888", markerfacecolor="none", markersize=11, lw=0, markeredgewidth=1.4, label="虚线黑圈: 旧口径扫描里的锁死落点 (cluster_1998326407, gneE62 侧进口)")],
          loc="lower left", fontsize=8, frameon=True, prop=CJK)
ax.set_title("都柏林 18 个 RL 信号路口: 位置、停车时间与锁死落点 (weekday 18h 改道第二版)" if CJK else "Dublin: 18 RL signalised junctions, stopped time and lock locations (weekday 18h reroute v2)", fontsize=11, fontproperties=CJK)
ax.text(0.99, 0.01, "粗体 = 停车秒数最多的 4 个路口; 灰线 = 路网; 北在上" if CJK else "bold = 4 junctions with most stopped seconds; grey = network; north up", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555", fontproperties=CJK)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight"); print("saved", a.out)
