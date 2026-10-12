"""统一状态逐项消融总表 (2026-10-11 用户令 "验证什么是最好的 state"): 两个面板, 每格 mean ± s.d. (10 评估种子 123-132), 括号 = 相对该面板基线的变化, 绿 = 行内最小。
面板 1 (1x1, 基线 327 = 8STD perphase + 对数等待): 274 (无对数等待) / 327 / 335 (legacy 头部) / 336 (+下游 ψ) / 337 (+lane_occ) / 316 (+since_green) / 320 (+since_green+green_elapsed)
面板 2 (1x3, 基线 338 = 统一状态): 275 (无对数等待) / 338 / 339 (−ψ) / 340 (−lane_occ)
缺数据的臂自动跳过 (列标注 "未完成")。口径同 plot_1x1_gs_redesign: 停车时间/visit, 停车次数/visit (收集器), 整趟 timeLoss/k_v, J(531)。
用法 (仓库根目录): python experiments/tools/figures/plot_state_ablation_2026-10-11.py
输出: experiments/analysis/figures/state_ablation_grid_2026-10-12_en.{png,json} (2026-10-12: 加第三面板 输入对等的两种头; 第三面板基线列 = exp352)"""
import glob, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG = "experiments/analysis/figures"; PV = "experiments/analysis/data/pervehicle"
SH = {"1x1": {"car": .965, "bus": .033, "ambulance": .0017}, "1x3": {"car": .95, "bus": .045, "ambulance": .005}}; WT = {"car": 1, "bus": 3, "ambulance": 5}
CL = ("car", "bus", "ambulance", "all")
PANELS = [("1x1: unified-state component ablation (base = exp327: perphase header + log-wait)", "1x1",
           [("no log-wait\nexp274", 274), ("BASE: perphase + log-wait\nexp327", 327), ("legacy header\nexp335", 335), ("+ downstream ψ\nexp336", 336), ("+ lane_occ\nexp337", 337),
            ("+ since-green\nexp316", 316), ("+ since-green + green-elapsed\nexp320", 320)]),
          ("1x1: input-matched head comparison (no header; time feature only inside the 12 slots)", "1x1",
           [("8STD header+log-wait\n(ref.) exp327", 327), ("8STD no header\n192-d exp352", 352), ("FRAP same 192-d\nexp347", 347),
            ("8STD + green time\ncontinuous exp349", 349), ("FRAP + green time\ncontinuous exp348", 348),
            ("8STD + green time\nphase-reset exp351", 351), ("FRAP + green time\nphase-reset exp350", 350)]),
          ("1x3: unified-state component ablation (base = exp338: unified state)", "1x3",
           [("no log-wait (main-figure arm)\nexp275", 275), ("BASE: unified state\nexp338", 338), ("− downstream ψ\nexp339", 339), ("− lane_occ\nexp340", 340)])]
def best_files(e):
    fs = glob.glob(f"{PV}/exp{e}_ep*_seed*_best_coll.json")
    if not fs: return {}
    ep = max(int(f.split("_ep")[1].split("_")[0]) for f in fs)     # 同一臂可能有多个 best 评估, 取最新 ep
    return {int(f.split("_seed")[1].split("_")[0]): json.load(open(f)) for f in fs if f"_ep{ep:05d}_" in f}
def delay(d):
    cv = d["collector"]["vehicles"]; acc = {c: [] for c in CL}
    for vid, rec in d["vehicles"].items():
        t = rec.get("trip"); k = cv.get(vid, {}).get("k_v", 0)
        if not t or k <= 0 or rec.get("type") not in acc: continue
        x = float(t["timeLoss"]) / k; acc[rec["type"]].append(x); acc["all"].append(x)
    return {c: (float(np.mean(v)) if v else np.nan) for c, v in acc.items()}
def rows_for(arms, sc):
    data = {e: best_files(e) for _, e in arms}; present = [(n, e) for n, e in arms if data[e]]
    seeds = sorted(set.intersection(*[set(data[e]) for _, e in present])) if present else []
    rows = []
    for m, lab, kind in (("avg_stopped_time_per_visit", "Stopped time / visit (s)", "t"), ("avg_stop_events_per_visit", "Stop events / visit", "e")):
        vals = {c: {e: np.array([data[e][s]["collector"]["summary"].get(f"eval/system/{c}/{m}", np.nan) for s in seeds]) for _, e in present} for c in CL}
        for c in CL: rows.append((f"{lab} - {c}", {e: (float(np.nanmean(v)), float(np.nanstd(v))) for e, v in vals[c].items()}, kind))
        J = {e: sum(SH[sc][c] * WT[c] * np.nan_to_num(vals[c][e]) for c in SH[sc]) for _, e in present}
        rows.append((f"J(531) weighted {lab.split(' /')[0].lower()}", {e: (float(j.mean()), float(j.std())) for e, j in J.items()}, "J" + kind))
    dl = {e: [delay(data[e][s]) for s in seeds] for _, e in present}
    for c in CL: rows.append((f"Delay (time loss) / visit (s) - {c}", {e: (float(np.nanmean([x[c] for x in D])), float(np.nanstd([x[c] for x in D]))) for e, D in dl.items()}, "d"))
    return rows, seeds, present
def main():
    dump = {}; tables = []
    for title, sc, arms in PANELS:
        rows, seeds, present = rows_for(arms, sc); base = (352 if "input-matched" in title else 327) if sc == "1x1" else 338
        dump[title] = {"seeds": seeds, "arms": [(n.replace("\n", " "), e) for n, e in arms], "present": [e for _, e in present],
                       "rows": [{"label": r[0], "mean_sd": {str(k): v for k, v in r[1].items()}} for r in rows]}
        cells, colors = [], []
        for lab, ms, kind in rows:
            d = 3 if kind in ("e", "Je") else 2; means = {e: m for e, (m, _) in ms.items()}; bm = min(means.values()) if means else np.nan; b0 = means.get(base, np.nan)
            txt = [lab]; col = ["#f3f3f3"]
            for n, e in arms:
                if e not in ms: txt.append("not finished"); col.append("#ffffff"); continue
                m, sd = ms[e]; rel = f" ({100 * (m / b0 - 1):+.0f}%)" if (e != base and b0 > 0) else ""
                txt.append(f"{m:.{d}f} ± {sd:.{d}f}{rel}"); col.append("#cfe9cf" if abs(m - bm) < 1e-12 else "#f3f3f3")
            cells.append(txt); colors.append(col)
        tables.append((title, arms, cells, colors, len(seeds)))
        print(f"\n== {title} (n={len(seeds)}; present {[e for _, e in present]}) ==")
        for row in cells: print("  " + " | ".join(f"{x:24s}" if i else f"{x:40s}" for i, x in enumerate(row)))
    nrows = sum(len(t[2]) + 2 for t in tables)
    fig = plt.figure(figsize=(21, 0.27 * nrows + 1.6)); gs = fig.add_gridspec(len(tables), 1, height_ratios=[len(t[2]) + 2 for t in tables], hspace=0.12, top=0.955, bottom=0.05, left=0.03, right=0.99)
    for i, (title, arms, cells, colors, n) in enumerate(tables):
        ax = fig.add_subplot(gs[i]); ax.axis("off"); w0 = 0.24; wi = (1 - w0) / len(arms)
        tbl = ax.table(cellText=cells, colLabels=["Metric"] + [a[0] for a in arms], cellColours=colors, colColours=["#c9d3df"] * (1 + len(arms)), loc="upper center", cellLoc="center", colWidths=[w0] + [wi] * len(arms))
        tbl.auto_set_font_size(False); tbl.scale(1, 1.4)
        for (r, c), cell in tbl.get_celld().items():
            t = cell.get_text(); t.set_fontsize(9)
            if r == 0: t.set_weight("bold"); cell.set_height(cell.get_height() * 1.9)
            if c == 0 and r > 0: t.set_ha("left"); cell.PAD = 0.012
        ax.set_title(f"{title}  [n={n} seeds]", fontsize=11, loc="left", pad=4, weight="bold")
    fig.suptitle("Unified state: per-component ablation on grids, mean ± s.d. per visit over evaluation seeds 123-132 (brackets: change vs BASE column; green = lowest in row)", fontsize=12, y=0.985)
    fig.text(0.03, 0.006, "Best checkpoint (sliding-3) of each run, merged evaluator, 10 seeds. Unified state = perphase header [min_green_ok, elapsed] + per movement [is_green, φ 5 priority buckets × (count, queue, log-scale wait), ψ downstream 5 × (count, queue), lane_occ]. "
             "1x1 has no downstream RL junction, so ψ there reads exit edges. Ambulance = 1 veh/h (underpowered). Fixed 15 s cycle on 1x1 = 18.32 (see all-methods table).", fontsize=8.3, color="#333333")
    out = f"{FIG}/state_ablation_grid_2026-10-12_en.png"; plt.savefig(out, dpi=200, bbox_inches="tight"); print("saved", out)
    json.dump(dump, open(out.replace(".png", ".json"), "w"), default=float, indent=1)
if __name__ == "__main__": main()
