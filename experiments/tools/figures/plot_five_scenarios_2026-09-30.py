"""五场景合一表 (2026-09-30 用户令): 1x1 (剔 Tukey 种子 128, n=9) | 1x3 (10 种子) | Dublin 02:00 (10 种子) | Dublin 11:00 (用户选定: 剔 128/132, 8 种子) | Dublin 18:00 改道第二版 (exp298 vs exp301, 10 种子)。
列: Metric | 8STD | GS-ENUM | Rel. diff (不含配对胜、备注列与保序门行); 块标题只写场景名。每块三组行: 停车时间/visit、停车次数/visit、延误(timeLoss)/visit (2026-10-01 加)。均值 ± 总体标准差, 与各 best10 表/Excel 同源。
输出: experiments/analysis/figures/main_comparison_five_scenarios_2026-09-30_en.{png,json} (不带延误) 与 ..._with_delay_2026-09-30_en.{png,json} (--delay, 带延误行)。"""
import json, glob, os, sys, numpy as np
WITH_DELAY = "--delay" in sys.argv   # 2026-10-01 用户令: 出两张图, 默认不带延误行, --delay 带延误行 (文件名加 _with_delay)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
FIG = "experiments/analysis/figures"; PV = "experiments/analysis/data/pervehicle"
SH18 = {"car": .9568, "bus": .0430, "ambulance": .0002}; WT = {"car": 1, "bus": 3, "ambulance": 5}
SH = {"1x1": {"car": .965, "bus": .033, "ambulance": .0017}, "1x3": {"car": .950, "bus": .050, "ambulance": .0004}, "Dublin 02:00": {"car": .9879, "bus": .0109, "ambulance": .0012},
      "Dublin 11:00": {"car": .9507, "bus": .0485, "ambulance": .0008}, "Dublin 18:00": SH18}
RAW = {"1x1": [(274, 1325), (263, 1370)], "1x3": [(275, 1785), (265, 580)], "Dublin 02:00": [(287, 770), (288, 760)], "Dublin 11:00": [(208, 240), (211, 215)], "Dublin 18:00": [(298, 200), (301, 95)]}

def delay_rows(name, seeds):
    """Delay (SUMO tripinfo timeLoss, 1 s) / visit: 每辆车 timeLoss / k_v (收集器在采样时刻看到它的路口数, k_v>0), 按车类对车辆取均值, 再对评估种子取均值 ± 总体标准差。
    含未完成行程 (timeLoss 截至仿真结束)。与停车指标同一批评估文件。"""
    arms = []
    for e, ep in RAW[name]:
        per_seed = {}
        for f in glob.glob(f"{PV}/exp{e}_ep{ep:05d}_seed*_best_coll.json"):
            seed = int(f.split("_seed")[1].split("_")[0])
            if seed not in seeds: continue
            d = json.load(open(f)); cv = d["collector"]["vehicles"]; acc = {"car": [], "bus": [], "ambulance": [], "all": []}
            for vid, rec in d["vehicles"].items():
                t = rec.get("trip"); k = cv.get(vid, {}).get("k_v", 0)
                if not t or k <= 0 or rec.get("type") not in acc: continue
                x = float(t["timeLoss"]) / k; acc[rec["type"]].append(x); acc["all"].append(x)
            per_seed[seed] = {c: (float(np.mean(v)) if v else np.nan) for c, v in acc.items()}
        arms.append(per_seed)
    ss = sorted(set(arms[0]) & set(arms[1])); rows = []
    vals = {c: [np.array([arm[s][c] for s in ss]) for arm in arms] for c in ("car", "bus", "ambulance", "all")}
    for c in ("car", "bus", "ambulance", "all"):
        rows.append([f"Delay (time loss) / visit (s) - {c}", float(np.nanmean(vals[c][0])), float(np.nanstd(vals[c][0])), float(np.nanmean(vals[c][1])), float(np.nanstd(vals[c][1]))])
    J = [sum(SH[name][c] * WT[c] * np.nan_to_num(vals[c][i]) for c in SH[name]) for i in (0, 1)]
    rows.append(["J(531) weighted delay (per visit)", float(J[0].mean()), float(J[0].std()), float(J[1].mean()), float(J[1].std())])
    return rows
GREEN = "#cfe9cf"; GREY = "#f3f3f3"; HEAD = "#dde3ea"

def gate(am, bu, ca, tol=5.0):
    if am <= bu <= ca: return "pass"
    return "pass (tol. 5 s)" if (am - bu <= tol and bu - ca <= tol) else "fail"

def rows_from_json(path, pick=None):
    d = json.load(open(path)); blocks = {}
    for k, v in d.items():
        if "|" in k and isinstance(v, list) and (pick is None or pick in k): blocks[k.split("|")[1]] = v
    seeds = [x for x in d["seeds"] if not (pick and "Tukey" in pick and x in d.get("tukey_outliers", []))]   # 剔 Tukey 块的 n 要扣掉离群种子
    return blocks["avg_stopped_time_per_visit"], blocks["avg_stop_events_per_visit"], seeds

def rows_from_raw(exps):
    A = []
    for e, ep in exps:
        A.append({int(f.split("_seed")[1].split("_")[0]): json.load(open(f)) for f in glob.glob(f"{PV}/exp{e}_ep{ep:05d}_seed*_best_coll.json")})
    seeds = sorted(set(A[0]) & set(A[1])); out = []
    for m, label, jlabel in (("avg_stopped_time_per_visit", "Stopped time / visit (s)", "J(531) weighted cost (per visit)"), ("avg_stop_events_per_visit", "Stop events / visit", "J(531) weighted stop events (per visit)")):
        vals = {c: [np.array([X[s]["collector"]["summary"].get(f"eval/system/{c}/{m}", np.nan) for s in seeds]) for X in A] for c in ("car", "bus", "ambulance", "all")}
        rows = [[f"{label} - {c}", float(np.nanmean(vals[c][0])), float(np.nanstd(vals[c][0])), float(np.nanmean(vals[c][1])), float(np.nanstd(vals[c][1]))] for c in ("car", "bus", "ambulance", "all")]
        J = [sum(SH18[c] * WT[c] * np.nan_to_num(vals[c][i]) for c in SH18) for i in (0, 1)]
        rows.append([jlabel, float(J[0].mean()), float(J[0].std()), float(J[1].mean()), float(J[1].std())]); out.append(rows)
    return out[0], out[1], seeds

SCEN = [("1x1", lambda: rows_from_json(f"{FIG}/main_comparison_1x1_best10seeds_2026-09-30_en.json", pick="excl. Tukey"), False),
        ("1x3", lambda: rows_from_json(f"{FIG}/main_comparison_1x3_best10seeds_2026-09-30_en.json"), False),
        ("Dublin 02:00", lambda: rows_from_json(f"{FIG}/main_comparison_02h_best10seeds_2026-09-30_en.json"), True),
        ("Dublin 11:00", lambda: rows_from_json(f"{FIG}/main_comparison_11h_best10seeds_SELECTED_excl128_132_2026-09-30_en.json"), True),
        ("Dublin 18:00", lambda: rows_from_raw([(298, 200), (301, 95)]), True)]

def main():
    cells, colors, hdr, titles, dump = [], [], [], [], {}
    for name, loader, amb_na in SCEN:
        rt, re_, seeds = loader(); rd = delay_rows(name, seeds) if WITH_DELAY else []; dump[name] = {"seeds": seeds, "stopped_time": rt, "stop_events": re_, "delay": rd}
        hdr.append(len(cells) + 1); titles.append(name); cells.append(["", "", "", ""]); colors.append([HEAD] * 4)   # 块标题行 (只写场景名, 用户令)
        for block, tag in ((rt, "t"), (re_, "e")) + (((rd, "d"),) if WITH_DELAY else ()):
            d = 2 if tag in ("t", "d") else 3; means = {}
            for row in block:
                lab, am, asd, bm, bsd = row[:5]; c = lab.split(" - ")[-1] if " - " in lab else "J"; means[c] = (am, bm)
                col = [GREY] * 4; col[1 if am <= bm else 2] = GREEN
                rel = f"{100 * (bm / am - 1):+.0f}%" if am > 0 else ""
                cells.append([lab, f"{am:.{d}f} ± {asd:.{d}f}", f"{bm:.{d}f} ± {bsd:.{d}f}", rel]); colors.append(col)
            # 2026-09-30 用户令: 不写保序门行 (means 仍保留在 json 里可复核)
            dump[name][f"gate_{tag}"] = {"8STD": means, "GS": means}
    fig, ax = plt.subplots(figsize=(11.5, 0.30 * len(cells) + 1.6)); ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=["Metric", "8STD", "GS-ENUM", "Rel. diff"], cellColours=colors, colColours=["#c9d3df"] * 4, loc="upper center", cellLoc="center", colWidths=[0.50, 0.18, 0.18, 0.14])
    tbl.auto_set_font_size(False); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        t = cell.get_text(); t.set_fontsize(10.5)
        if r == 0: t.set_weight("bold"); t.set_fontsize(11)
        if c == 0 and r > 0: t.set_ha("left"); cell.PAD = 0.012
        if r in hdr and c > 0: cell.set_edgecolor(HEAD)
    ax.set_title("8STD (template + DQN) vs GS-ENUM (enumerated phases + FRAP), mean ± s.d. per visit", fontsize=12, pad=6)   # 用户令: 去掉 best checkpoints 字样
    fig.canvas.draw(); ren = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for r, title in zip(hdr, titles):
        bb = tbl[r, 0].get_window_extent(ren); x0, y0 = inv.transform((bb.x0, bb.y0)); x1, y1 = inv.transform((bb.x1, bb.y1))
        fig.text(x0 + 0.004, (y0 + y1) / 2, title, ha="left", va="center", fontsize=11, weight="bold", zorder=10)
    tb = tbl.get_window_extent(ren); tt = ax.title.get_window_extent(ren); bb = Bbox.union([tb, tt]); pad = 0.15 * fig.dpi
    crop = Bbox.from_extents((bb.x0 - pad) / fig.dpi, (bb.y0 - pad) / fig.dpi, (bb.x1 + pad) / fig.dpi, (bb.y1 + pad) / fig.dpi)
    out = f"{FIG}/main_comparison_five_scenarios{'_with_delay' if WITH_DELAY else ''}_2026-09-30_en.png"; plt.savefig(out, dpi=200, bbox_inches=crop); print("saved", out)
    json.dump(dump, open(out.replace(".png", ".json"), "w"), default=float, indent=1)

if __name__ == "__main__": main()
