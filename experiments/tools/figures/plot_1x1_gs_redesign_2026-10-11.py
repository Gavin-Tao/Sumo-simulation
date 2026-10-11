"""1x1 GS 重设计总表 (2026-10-11 用户令): 四个面板, 每格 mean ± s.d. (10 评估种子 123-132), 括号 = 相对该面板第一列的变化, 绿 = 行内最小。
口径与 plot_five_scenarios_2026-09-30.py 的 1x1 块逐式相同:
  停车时间/visit、停车次数/visit = 收集器 eval/system/<class>/avg_*_per_visit (5 s 采样);
  延误/visit = 整趟 timeLoss / k_v (该车被收集器看到的路口数, k_v>0), 按车类对车辆取均值再对种子取均值 (网格全是 RL 路口, 与 RL 路口延误等价);
  J(531) = Σ share_1x1 × weight × 指标 (share = {car .965, bus .033, amb .0017}, weight = {1,3,5})。
与主图 1x1 块的差别: 这里用全部 10 个种子 (主图剔了 Tukey 种子 128); JSON 里同时给出剔 128 的敏感性结果。
面板: (1) 8STD 观测消融 274/327/326/316/320; (2) GS FRAP 压力打分 2×2 的一半 316/317/318/319; (3) GS FRAP 学习打分 316/263/323/324/325; (4) 菜单条件化 316/322/321/323。
用法 (仓库根目录): python experiments/tools/figures/plot_1x1_gs_redesign_2026-10-11.py [--out-suffix X]
输出: experiments/analysis/figures/grid1x1_gs_redesign_2026-10-11_en.{png,json}
"""
import glob, json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
OUT_SUFFIX = _arg("--out-suffix", "")
FIG = "experiments/analysis/figures"; PV = "experiments/analysis/data/pervehicle"
SH = {"car": .965, "bus": .033, "ambulance": .0017}; WT = {"car": 1, "bus": 3, "ambulance": 5}
CLASSES = ("car", "bus", "ambulance", "all")
PANELS = [
    ("8STD, observation ablation", [("8STD perphase, no new obs\nexp274 (main-figure arm)", 274, 1325), ("+ log-wait only\nexp327", 327, 1300), ("+ since-green only\nexp326", 326, 1485),
                                    ("+ both\nexp316", 316, 1475), ("+ both + green-elapsed\nexp320", 320, 805)]),
    ("GS-ENUM with FRAP, pressure scoring", [("8STD + both obs\nexp316", 316, 1475), ("FRAP pressure\nexp317", 317, 960), ("+ header into encoder\nexp318", 318, 335),
                                             ("+ green-elapsed\nexp319", 319, 1275)]),
    ("GS-ENUM with FRAP, learned scoring", [("8STD + both obs\nexp316", 316, 1475), ("old GS, no new obs\nexp263 (seed 1)", 263, 1370), ("FRAP learned\nexp323", 323, 1330),
                                            ("+ header into encoder\nexp324", 324, 1365), ("+ green-elapsed\nexp325", 325, 500)]),
    ("Menu-conditioned Q", [("8STD + both obs\nexp316", 316, 1475), ("menu-Q, standard 8 phases\nexp322", 322, 1375), ("menu-Q, enumerated 12\nexp321", 321, 1470),
                            ("FRAP learned, enumerated 12\nexp323", 323, 1330)]),
]
GREEN = "#cfe9cf"; GREY = "#f3f3f3"; HEAD = "#dde3ea"

def load(e, ep):
    return {int(f.split("_seed")[1].split("_")[0]): json.load(open(f)) for f in glob.glob(f"{PV}/exp{e}_ep{ep:05d}_seed*_best_coll.json")}

def delay_per_seed(d):
    cv = d["collector"]["vehicles"]; acc = {c: [] for c in CLASSES}
    for vid, rec in d["vehicles"].items():
        t = rec.get("trip"); k = cv.get(vid, {}).get("k_v", 0)
        if not t or k <= 0 or rec.get("type") not in acc: continue
        x = float(t["timeLoss"]) / k; acc[rec["type"]].append(x); acc["all"].append(x)
    return {c: (float(np.mean(v)) if v else np.nan) for c, v in acc.items()}

def panel_rows(arms, seeds_filter=None):
    data = [load(e, ep) for _, e, ep in arms]
    seeds = sorted(set.intersection(*[set(x) for x in data]))
    if seeds_filter: seeds = [s for s in seeds if s not in seeds_filter]
    rows = []
    for m, label, jlabel, kind in (("avg_stopped_time_per_visit", "Stopped time / visit (s)", "J(531) weighted stopped time", "t"),
                                  ("avg_stop_events_per_visit", "Stop events / visit", "J(531) weighted stop events", "e")):
        vals = {c: [np.array([X[s]["collector"]["summary"].get(f"eval/system/{c}/{m}", np.nan) for s in seeds]) for X in data] for c in CLASSES}
        for c in CLASSES: rows.append([f"{label} - {c}", [(float(np.nanmean(v)), float(np.nanstd(v))) for v in vals[c]], kind])
        J = [sum(SH[c] * WT[c] * np.nan_to_num(vals[c][i]) for c in SH) for i in range(len(arms))]
        rows.append([jlabel, [(float(j.mean()), float(j.std())) for j in J], "J" + kind])
    dl = [[delay_per_seed(X[s]) for s in seeds] for X in data]
    vals = {c: [np.array([r[c] for r in D]) for D in dl] for c in CLASSES}
    for c in CLASSES: rows.append([f"Delay (time loss) / visit (s) - {c}", [(float(np.nanmean(v)), float(np.nanstd(v))) for v in vals[c]], "d"])
    J = [sum(SH[c] * WT[c] * np.nan_to_num(vals[c][i]) for c in SH) for i in range(len(arms))]
    rows.append(["J(531) weighted delay", [(float(j.mean()), float(j.std())) for j in J], "Jd"])
    return rows, seeds

def main():
    dump = {}; tables = []
    for title, arms in PANELS:
        rows, seeds = panel_rows(arms); rows_x, _ = panel_rows(arms, seeds_filter={128})
        dump[title] = {"seeds": seeds, "arms": [(a[0].replace("\n", " "), a[1], a[2]) for a in arms],
                       "rows": [{"label": r[0], "mean_sd": r[1]} for r in rows], "rows_excl_seed128": [{"label": r[0], "mean_sd": r[1]} for r in rows_x]}
        cells, colors = [], []
        for lab, ms, kind in rows:
            d = 3 if kind in ("e", "Je") else 2; means = [m for m, _ in ms]; best = int(np.nanargmin(means)); base = means[0]
            col = [GREY] * (1 + len(arms)); col[1 + best] = GREEN
            txt = [lab] + [f"{m:.{d}f} ± {sd:.{d}f}" + (f" ({100 * (m / base - 1):+.0f}%)" if i > 0 and base > 0 else "") for i, (m, sd) in enumerate(ms)]
            cells.append(txt); colors.append(col)
        tables.append((title, arms, cells, colors))
        print(f"\n== {title} (n={len(seeds)}) ==")
        for row in cells: print("  " + " | ".join(f"{x:26s}" if i else f"{x:40s}" for i, x in enumerate(row)))
    nrows = sum(len(t[2]) + 2 for t in tables)
    fig = plt.figure(figsize=(17, 0.265 * nrows + 1.2))
    gs = fig.add_gridspec(len(tables), 1, height_ratios=[len(t[2]) + 2 for t in tables], hspace=0.10, top=0.965, bottom=0.045, left=0.03, right=0.99)
    for ax_i, (title, arms, cells, colors) in enumerate(tables):
        ax = fig.add_subplot(gs[ax_i]); ax.axis("off")
        ncol = 1 + len(arms); w0 = 0.28; wi = (1 - w0) / len(arms)
        tbl = ax.table(cellText=cells, colLabels=["Metric"] + [a[0] for a in arms], cellColours=colors, colColours=["#c9d3df"] * ncol,
                       loc="upper center", cellLoc="center", colWidths=[w0] + [wi] * len(arms))
        tbl.auto_set_font_size(False); tbl.scale(1, 1.4)
        for (r, c), cell in tbl.get_celld().items():
            t = cell.get_text(); t.set_fontsize(9.5)
            if r == 0: t.set_weight("bold"); t.set_fontsize(9.5); cell.set_height(cell.get_height() * 1.9)
            if c == 0 and r > 0: t.set_ha("left"); cell.PAD = 0.012
        ax.set_title(f"1x1: {title}", fontsize=11.5, loc="left", pad=4, weight="bold")
    fig.suptitle("1x1 grid: GS redesign study, mean ± s.d. per visit over evaluation seeds 123-132 (in brackets: change vs first column; green = lowest in row)", fontsize=12, y=0.985)
    note = ("Best checkpoint of each run (sliding-3 criterion), evaluated with the merged evaluator. 'both obs' = since-green + log-scale wait features. "
            "Panel 1 isolates the two observation features on 8STD; panels 2-3 are the 2x2 of FRAP scoring (pressure / learned) x information fix (header into encoder / per-movement green-elapsed);\n"
            "panel 4 holds the Q-network fixed (menu-conditioned MLP, one hidden layer of 128 like 8STD) and varies only the phase menu. Delay = whole-trip time loss / junction visits (grid: every junction is RL). "
            "The main five-scenario figure's 1x1 block excluded Tukey seed 128; here all 10 seeds are used (JSON carries the excl-128 variant). Ambulance = 1 veh/h flows (underpowered).")
    fig.text(0.03, 0.006, note, fontsize=8.3, color="#333333", va="bottom", ha="left")
    out = f"{FIG}/grid1x1_gs_redesign{OUT_SUFFIX}_2026-10-11_en.png"; plt.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)
    json.dump(dump, open(out.replace(".png", ".json"), "w"), default=float, indent=1)

if __name__ == "__main__": main()
