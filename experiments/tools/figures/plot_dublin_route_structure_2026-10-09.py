"""Dublin 三时段 × 三车类 路线结构图 (2026-10-09 用户问 "各自经过哪些路口, 路线上有啥具体区别")。
输入: experiments/analysis/data/dublin_route_structure_2026-10-09.json (route_structure_2026-10-09.py 生成) + nets/dublin/dublin_enum.net.xml (几何)。
输出 (experiments/analysis/figures/dublin_routes/):
  junction_visits_map_2026-10-09_en.png     3×3: 行 = 02h/11h/18h, 列 = car/bus/ambulance; RL 路口圆点面积 ∝ 每小时经过量, 灰底为路网
  junction_visits_heatmap_2026-10-09_en.png 18 个 RL 路口 × 9 列 (车类×时段) 每小时经过量
  ambulance_routes_2026-10-09_en.png        三时段救护车路线逐条画在路网上, 黄圆 = 路线上的 RL 路口 (标 id), 红 × = 路线上的非 RL 路口
  ambulance_routes_11h_each_2026-10-09_en.png  11h 四辆救护车一辆一张 (2×2), 路口按顺序编号, [RL] / (priority) 标注
  bus_routes_2026-10-09_en.png              三时段全部公交路线 (GTFS), 同样标注
  car_routes_top_2026-10-09_en.png          三时段流量最大的 12 条小汽车路线, 同样标注
  car_routes_top25_2026-10-09_en.png        同上, 前 25 条 (2026-10-09 用户问 18h 左上走廊为何不在图里: 18h 该走廊最大一条路线排第 13)
  edge_flows_car_bus_2026-10-09_en.png      2×3: 小汽车 / 公交 的逐边每小时流量 (线宽 ∝ 流量), 三时段并排"""
import os, sys, json, collections, xml.etree.ElementTree as ET
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, "experiments/tools/dublin"); import sumolib
R = "nets/dublin"; OUT = "experiments/analysis/figures/dublin_routes"; os.makedirs(OUT, exist_ok=True)
D = json.load(open("experiments/analysis/data/dublin_route_structure_2026-10-09.json"))
net = sumolib.net.readNet(f"{R}/dublin_enum.net.xml"); RL = D["rl_junctions"]
WIN = {"02h": ("weekday_02h", "amb_weekday_02h_probe11h"), "11h": ("weekday_11h", "amb_weekday_11h"), "18h": ("weekday_18h_reroute_v2", "amb_weekday_18h_probe11h")}
WL = {"02h": "Dublin 02:00", "11h": "Dublin 11:00", "18h": "Dublin 18:00 (v2)"}
CL = {"car": "car", "bus": "bus", "ambulance": "ambulance"}; COL = {"car": "#c0392b", "bus": "#1f5fbf", "ambulance": "#1a9850"}
SHORT = {j: (j if len(j) < 14 else j.split("_")[0] + "_" + j.split("_")[1][:6] + "…") for j in RL}
edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
segs = [np.array(e.getShape()) for e in edges]; xy = {j: net.getNode(j).getCoord() for j in RL}
xs = np.concatenate([s[:, 0] for s in segs]); ys = np.concatenate([s[:, 1] for s in segs]); X0, X1, Y0, Y1 = xs.min() - 30, xs.max() + 30, ys.min() - 30, ys.max() + 30

def base(ax):
    ax.add_collection(LineCollection(segs, colors="#cccccc", linewidths=0.8, zorder=1)); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def fig_visits_map():
    fig, axs = plt.subplots(3, 3, figsize=(16.5, 15))
    for i, w in enumerate(WIN):
        for k, c in enumerate(CL):
            ax = axs[i, k]; base(ax); jv = D["summary"][w][c]["junction_visits"]
            for j in RL:
                v = jv.get(j, 0); x, y = xy[j]
                if v > 0: ax.scatter(x, y, s=18 + 2.2 * v ** 0.85 if c != "ambulance" else 60 + 80 * v, color=COL[c], alpha=0.75, edgecolor="k", linewidth=0.4, zorder=3)
                else: ax.scatter(x, y, s=14, facecolor="white", edgecolor="#888888", linewidth=0.6, zorder=3)
                if c == "ambulance" or v >= (150 if c == "car" else 30): ax.annotate(f"{v}", (x, y), fontsize=7, ha="center", va="bottom", xytext=(0, 5 + 0.08 * v ** 0.85 if c != "ambulance" else 9), textcoords="offset points", zorder=4)
            s = D["summary"][w][c]; ax.set_title(f"{WL[w]} - {c}\n{s['n']} veh/h, {s['distinct_routes']} routes, {s['junctions_touched']}/18 RL junctions", fontsize=9.5)
    fig.suptitle("RL junction visits per hour by window and vehicle class (dot area ∝ visits; hollow = never visited by this class)", fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{OUT}/junction_visits_map_2026-10-09_en.png", dpi=170); plt.close(fig)

def fig_heatmap():
    cols = [(c, w) for c in CL for w in WIN]; M = np.array([[D["summary"][w][c]["junction_visits"].get(j, 0) for c, w in cols] for j in RL], float)
    order = np.argsort(-M[:, :3].sum(1)); M = M[order]; labels = [SHORT[RL[i]] for i in order]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for k, (c, w) in enumerate(cols):
        col = M[:, k]; mx = max(M[:, [kk for kk, (cc, _) in enumerate(cols) if cc == c]].max(), 1)
        for r in range(len(RL)):
            v = col[r]; ax.add_patch(plt.Rectangle((k, r), 1, 1, color=plt.cm.Reds(0.15 + 0.8 * (v / mx) ** 0.6) if v > 0 else "#f4f4f4"))
            ax.text(k + 0.5, r + 0.5, f"{int(v)}", ha="center", va="center", fontsize=8, color="black" if v / mx < 0.6 else "white")
    ax.set_xlim(0, 9); ax.set_ylim(len(RL), 0); ax.set_xticks(np.arange(9) + 0.5); ax.set_xticklabels([f"{c}\n{w}" for c, w in cols], fontsize=9)
    ax.set_yticks(np.arange(len(RL)) + 0.5); ax.set_yticklabels(labels, fontsize=8.5)
    for x in (3, 6): ax.axvline(x, color="k", lw=1.2)
    ax.set_title("Visits per hour at each RL junction (rows sorted by car visits; colour scaled within each vehicle class)", fontsize=11); ax.tick_params(length=0)
    plt.tight_layout(); fig.savefig(f"{OUT}/junction_visits_heatmap_2026-10-09_en.png", dpi=170); plt.close(fig)

def fig_ambulance():
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5)); cmap = ["#1a9850", "#d73027", "#4575b4", "#f46d43", "#7b3294"]
    for i, w in enumerate(WIN):
        ax = axs[i]; base(ax); ambs = D["ambulances"][w]
        for j in RL: ax.scatter(*xy[j], s=16, facecolor="white", edgecolor="#777777", linewidth=0.6, zorder=3)
        seen = set()
        for k, a in enumerate(ambs):
            pts = np.concatenate([np.array(net.getEdge(e).getShape()) for e in a["edges"]])
            ax.plot(pts[:, 0], pts[:, 1], color=cmap[k % len(cmap)], lw=3.2 if k == 0 else 2.2, alpha=0.9, zorder=4, label=f"{a['id']}: depart {a['depart']:.0f} s, {a['length']} m, {len(a['junctions'])} RL junctions")
            ax.scatter(pts[0, 0], pts[0, 1], marker="^", s=70, color=cmap[k % len(cmap)], edgecolor="k", zorder=5); ax.scatter(pts[-1, 0], pts[-1, 1], marker="s", s=60, color=cmap[k % len(cmap)], edgecolor="k", zorder=5)
            for j in a["junctions"]:
                ax.scatter(*xy[j], s=55, color="#ffd92f", edgecolor="k", linewidth=0.8, zorder=6)
                if j not in seen: ax.annotate(SHORT[j], xy[j], fontsize=7.5, xytext=(4, 4), textcoords="offset points", zorder=7); seen.add(j)
        ax.legend(loc="lower left", fontsize=8, frameon=True); ax.set_title(f"{WL[w]}: ambulance routes (▲ entry, ■ exit, yellow = RL junction on route)", fontsize=10)
    fig.suptitle("Ambulance routes per window. 02h and 18h use the 11h amb_1 probe route (George's St corridor); 11h has four different routes", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/ambulance_routes_2026-10-09_en.png", dpi=170); plt.close(fig)

JTYPE = {j.get("id"): j.get("type") for j in ET.parse(f"{R}/dublin_enum.net.xml").getroot().findall("junction")}
def route_nodes(edges):
    """路线经过的内部节点 (每条边的终点, 不含最后一条边): [(node, is_RL, type)]"""
    out = []
    for e in edges[:-1]:
        n = net.getEdge(e).getToNode().getID(); out.append((n, n in RL, JTYPE.get(n, "?")))
    return out

def draw_routes(ax, routes, title, label_rl=True, label_nonrl=False, legend=True, lw=2.4):
    """routes: [(label, edges, color)]. RL 路口 = 黄色实心圆 (标 id); 路线经过的非 RL 路口 (priority 等) = 红色 ×; 其余 RL 路口 = 空心圆。"""
    base(ax)
    for j in RL: ax.scatter(*xy[j], s=22, facecolor="white", edgecolor="#555555", linewidth=0.7, zorder=3)
    seen_rl, seen_non = set(), set()
    for k, (lab, edges, col) in enumerate(routes):
        pts = np.concatenate([np.array(net.getEdge(e).getShape()) for e in edges])
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, alpha=0.85, zorder=4, label=lab)
        ax.scatter(pts[0, 0], pts[0, 1], marker="^", s=55, color=col, edgecolor="k", linewidth=0.6, zorder=5); ax.scatter(pts[-1, 0], pts[-1, 1], marker="s", s=45, color=col, edgecolor="k", linewidth=0.6, zorder=5)
        for n, is_rl, t in route_nodes(edges):
            c = net.getNode(n).getCoord()
            if is_rl:
                ax.scatter(*c, s=70, color="#ffd92f", edgecolor="k", linewidth=0.9, zorder=6)
                if label_rl and n not in seen_rl: ax.annotate(SHORT[n], c, fontsize=7.5, xytext=(4, 4), textcoords="offset points", zorder=7); seen_rl.add(n)
            elif t not in ("dead_end",):
                ax.scatter(*c, s=55, marker="x", color="#d62728", linewidth=1.6, zorder=6)
                if label_nonrl and n not in seen_non: ax.annotate(f"{n[:16]} ({t})", c, fontsize=6.5, color="#d62728", xytext=(4, -9), textcoords="offset points", zorder=7); seen_non.add(n)
    if legend and routes: ax.legend(loc="lower left", fontsize=7.5, frameon=True)
    ax.set_title(title, fontsize=10)

def fig_ambulance_v2():
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.8)); cmap = ["#1a9850", "#d73027", "#4575b4", "#f46d43", "#7b3294"]
    for i, w in enumerate(WIN):
        routes = [(f"{a['id']}: depart {a['depart']:.0f} s, {a['length']} m, {len(a['junctions'])} RL junctions", a["edges"], cmap[k % 5]) for k, a in enumerate(D["ambulances"][w])]
        draw_routes(axs[i], routes, f"{WL[w]}: ambulance routes", label_nonrl=False, lw=3)   # 非 RL 路口只画红 ×, 名单见文档 (标签会重叠)
    fig.suptitle("Ambulance routes. Yellow circle = RL-controlled junction on the route (labelled); red × = junction on the route NOT controlled by RL (priority / yield); hollow circle = other RL junction; ▲ entry, ■ exit", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/ambulance_routes_2026-10-09_en.png", dpi=170); plt.close(fig)

def fig_ambulance_11h_each():
    """11h 四辆救护车一辆一张 (用户令 2026-10-09): 2×2, 每张只画一条轨迹, RL 路口黄圆标 id, 非 RL 路口红 × 标 id (标签交错避免重叠)."""
    ambs = D["ambulances"]["11h"]; cmap = ["#1a9850", "#d73027", "#4575b4", "#f46d43"]
    fig, axs = plt.subplots(2, 2, figsize=(22, 19))
    for k, a in enumerate(ambs):
        ax = axs[k // 2, k % 2]; base(ax)
        for j in RL: ax.scatter(*xy[j], s=24, facecolor="white", edgecolor="#555555", linewidth=0.7, zorder=3)
        pts = np.concatenate([np.array(net.getEdge(e).getShape()) for e in a["edges"]])
        ax.plot(pts[:, 0], pts[:, 1], color=cmap[k], lw=3.4, alpha=0.9, zorder=4)
        ax.scatter(pts[0, 0], pts[0, 1], marker="^", s=90, color=cmap[k], edgecolor="k", zorder=5); ax.scatter(pts[-1, 0], pts[-1, 1], marker="s", s=75, color=cmap[k], edgecolor="k", zorder=5)
        n_rl = n_non = 0
        for i, (n, is_rl, t) in enumerate(route_nodes(a["edges"])):
            c = net.getNode(n).getCoord(); off = [(8, 8), (8, -13), (-70, 16), (-70, -20)][i % 4]
            if is_rl:
                n_rl += 1; ax.scatter(*c, s=90, color="#ffd92f", edgecolor="k", linewidth=0.9, zorder=6)
                ax.annotate(f"{n_rl + n_non}. {SHORT[n]} [RL]", c, fontsize=8, weight="bold", xytext=off, textcoords="offset points", zorder=7)
            elif t != "dead_end":
                n_non += 1; ax.scatter(*c, s=70, marker="x", color="#d62728", linewidth=1.8, zorder=6)
                ax.annotate(f"{n_rl + n_non}. {n[:18]} ({t})", c, fontsize=7, color="#d62728", xytext=off, textcoords="offset points", zorder=7)
        # 用户令 2026-10-09: 画完整路网, 不按路线放大
        ax.set_title(f"Dublin 11:00 - {a['id']}: depart {a['depart']:.0f} s, {a['length']} m, {len(a['junctions'])} RL junctions (yellow) + {n_non} non-RL junctions (red ×)", fontsize=10)
    fig.suptitle("Dublin 11:00 ambulance routes, one per panel. Numbers = order along the route; [RL] = controlled by the RL agent; (priority) = unsignalised yield junction; ▲ entry, ■ exit", fontsize=11.5)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{OUT}/ambulance_routes_11h_each_2026-10-09_en.png", dpi=170); plt.close(fig)

def fig_bus_routes():
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.8)); cm = plt.cm.tab20
    for i, w in enumerate(WIN):
        folder, _ = WIN[w]; cnt = collections.Counter()
        for v in ET.parse(f"{R}/{folder}/bus_weekday_{w}.rou.xml").getroot().iter("vehicle"): cnt[tuple(v.find("route").get("edges").split())] += 1
        routes = [(f"{n}/h", list(e), cm(k % 20)) for k, (e, n) in enumerate(cnt.most_common())]
        draw_routes(axs[i], routes, f"{WL[w]}: {len(cnt)} distinct bus routes, {sum(cnt.values())} buses/h", legend=False, lw=2.0)
    fig.suptitle("Bus routes (GTFS). Yellow circle = RL junction on a bus route; red × = non-RL junction on a bus route; hollow circle = RL junction never used by buses", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/bus_routes_2026-10-09_en.png", dpi=170); plt.close(fig)

def fig_car_routes(top=12):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.8)); cm = plt.cm.tab20
    for i, w in enumerate(WIN):
        folder, _ = WIN[w]; cnt = collections.Counter()
        for v in ET.parse(f"{R}/{folder}/car_weekday_{w}.rou.xml").getroot().iter("vehicle"): cnt[tuple(v.find("route").get("edges").split())] += 1
        tot = sum(cnt.values()); topr = cnt.most_common(top); cov = sum(n for _, n in topr)
        routes = [(f"{n}/h", list(e), cm(k % 20)) for k, (e, n) in enumerate(topr)]
        draw_routes(axs[i], routes, f"{WL[w]}: top {top} of {len(cnt)} car routes = {100*cov/tot:.0f}% of {tot} cars/h", legend=False, lw=1.8)
    fig.suptitle(f"Most used car routes (top {top} per window). Yellow circle = RL junction on these routes; red × = non-RL junction on these routes", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/car_routes_top{'' if top == 12 else top}_2026-10-09_en.png", dpi=170); plt.close(fig)

def fig_edge_flows():
    fig, axs = plt.subplots(2, 3, figsize=(18, 11.5))
    for i, w in enumerate(WIN):
        folder, _ = WIN[w]
        for r, (c, f) in enumerate((("car", f"car_weekday_{w}"), ("bus", f"bus_weekday_{w}"))):
            cnt = collections.Counter()
            for v in ET.parse(f"{R}/{folder}/{f}.rou.xml").getroot().iter("vehicle"):
                for e in v.find("route").get("edges").split(): cnt[e] += 1
            ax = axs[r, i]; base(ax); mx = max(cnt.values())
            ss = [np.array(net.getEdge(e).getShape()) for e in cnt]; lw = [0.4 + 6.0 * (cnt[e] / mx) for e in cnt]
            ax.add_collection(LineCollection(ss, colors=COL[c], linewidths=lw, alpha=0.8, zorder=2))
            for j in RL: ax.scatter(*xy[j], s=12, facecolor="white", edgecolor="#555555", linewidth=0.6, zorder=3)
            ax.set_title(f"{WL[w]} - {c}: {sum(1 for _ in ET.parse(f'{R}/{folder}/{f}.rou.xml').getroot().iter('vehicle'))} vehicles/h on {len(cnt)} edges (max {mx}/h on one edge)", fontsize=9.5)
    fig.suptitle("Edge flows per hour (line width ∝ vehicles/h, scaled per panel). Top: cars. Bottom: buses. White dots = RL junctions", fontsize=12.5)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{OUT}/edge_flows_car_bus_2026-10-09_en.png", dpi=170); plt.close(fig)

if __name__ == "__main__":
    fig_visits_map(); fig_heatmap(); fig_ambulance_v2(); fig_ambulance_11h_each(); fig_bus_routes(); fig_car_routes(); fig_car_routes(top=25); fig_edge_flows(); print("saved 8 figures to", OUT)
