"""第一层: 同一臂内各类车被区别对待的证据 (best × 10 逐秒逐车记录): 零停占比、停车/visit、进入到首绿、遇红到达的条件指标; 逐路口公交对小汽车。
用法 (experiments/ 下): python tools/dublin/priority_evidence/tier1_class_metrics.py"""
import json, glob, collections, numpy as np
NAME = {"458": "旧111", "456": "旧521", "449": "旧531", "457": "旧541", "460": "新111", "462": "新521", "461": "新531", "459": "新541"}
short = lambda t: t.split("_")[0] + ("_" + t.split("_")[1] if t.startswith("cluster") else "")
def load(e):
    V = collections.defaultdict(list)   # (junction, class) -> list of (stopped, red_at_entry, first_green_wait or None)
    files = sorted(glob.glob(f"analysis/data/pervehicle/exp{e}_ep*_seed*_best_coll1s.json"))
    for f in files:
        d = json.load(open(f))
        for vid, v in d["vehicles"].items():
            for t, visits in v["tls"].items():
                for r in visits:
                    V[(t, v["type"])].append((r["stopped"], r["entry_state"] in ("r", "R", "y", "Y"), None if r["first_green"] is None else r["first_green"] - r["entry"]))
    return V, len(files)
def agg(rows):
    n = len(rows); s = np.array([r[0] for r in rows], float); red = np.array([r[1] for r in rows], bool); fg = np.array([r[2] if r[2] is not None else np.nan for r in rows], float)
    return dict(n=n, zero=float((s == 0).mean()), stop=float(s.mean()), fg=float(np.nanmean(fg)) if np.isfinite(fg).any() else float("nan"), red_share=float(red.mean()),
                red_stop=float(s[red].mean()) if red.any() else float("nan"), red_fg=float(np.nanmean(fg[red])) if np.isfinite(fg[red]).any() else float("nan"), red_zero=float((s[red] == 0).mean()) if red.any() else float("nan"))
out = {}; md = ["# 第一层: 同一臂内各类车的信号待遇 (best × 10 种子, 逐秒逐车)\n", "指标: 零停 = 访问时一秒都没停的占比; 停车/visit; 首绿 = 进入进口道到本运动第一次变绿的秒数 (进入时已绿为 0); 遇红 = 进入时本运动是红/黄灯的访问占比; 遇红后… = 只统计遇红到达的访问。\n",
              "| 臂 | 类 | 访问/集 | 零停 | 停车/visit | 首绿 (s) | 遇红占比 | 遇红后零停 | 遇红后停车/visit | 遇红后首绿 (s) |", "|---|---|---|---|---|---|---|---|---|---|"]
for e in ["458", "456", "449", "457", "460", "462", "461", "459"]:
    V, n = load(e); out[NAME[e]] = {}
    for c in ("car", "bus", "ambulance"):
        rows = [r for (t, cc), rs in V.items() if cc == c for r in rs]
        if not rows: continue
        a = agg(rows); out[NAME[e]][c] = a
        md.append(f"| {NAME[e]} | {c} | {a['n']/n:.0f} | {a['zero']:.0%} | {a['stop']:.2f} | {a['fg']:.2f} | {a['red_share']:.0%} | {a['red_zero']:.0%} | {a['red_stop']:.2f} | {a['red_fg']:.2f} |")
    # 逐路口 公交 vs 小汽车 (公交访问最多的 8 个路口)
    bj = sorted({t for (t, c) in V if c == "bus"}, key=lambda t: -len(V[(t, "bus")]))[:8]; out[NAME[e]]["junctions"] = {}
    for t in bj:
        ab, ac = agg(V[(t, "bus")]), agg(V[(t, "car")]); out[NAME[e]]["junctions"][short(t)] = dict(bus=ab, car=ac)
md.append("\n## 逐路口: 公交对小汽车 (公交访问最多的 8 个路口; 括号内为遇红到达后的停车/visit 与首绿)\n")
for name in ["新541", "新531", "新521", "新111", "旧541", "旧111"]:
    md.append(f"\n### {name}\n\n| 路口 | 公交 访问/集 | 公交 零停 | 公交 停车/visit (遇红后) | 公交 首绿 (遇红后) | 小汽车 零停 | 小汽车 停车/visit (遇红后) | 小汽车 首绿 (遇红后) |\n|---|---|---|---|---|---|---|---|")
    for j, d in out[name]["junctions"].items():
        b, c = d["bus"], d["car"]
        md.append(f"| {j} | {b['n']/10:.0f} | {b['zero']:.0%} | {b['stop']:.2f} ({b['red_stop']:.2f}) | {b['fg']:.1f} ({b['red_fg']:.1f}) | {c['zero']:.0%} | {c['stop']:.2f} ({c['red_stop']:.2f}) | {c['fg']:.1f} ({c['red_fg']:.1f}) |")
open("analysis/data/priority_evidence_2026-10-27/tier1_class_metrics.md", "w").write("\n".join(md) + "\n"); json.dump(out, open("analysis/data/priority_evidence_2026-10-27/tier1_class_metrics.json", "w"), indent=1, ensure_ascii=False)
print("\n".join(md[:40]))
