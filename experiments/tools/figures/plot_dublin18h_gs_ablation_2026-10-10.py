"""Dublin 18:00 改道第二版 GS 消融表 (2026-10-10 用户令): 8STD (exp298) 对 GS 三个配置 exp301 / exp309 / exp310, 新图不覆盖旧图。
口径与 plot_five_scenarios_2026-09-30.py 的 Dublin 18:00 块逐式相同 (同一批评估文件、同一公式):
  停车时间/visit、停车次数/visit = 收集器 (EpisodeMetricsCollector, 5 s 采样, 只数受控进口道) 的 eval/system/<class>/... 汇总;
  RL 路口延误/visit = eval_rl_delay.py 的 Σ_RL边 timeLoss_c / Σ_RL进口道 entered_c;
  J(531) = Σ_c share_c × weight_c × 指标_c (share = 18h 车类 visit 份额, weight = 优先级 5/3/1);
  每格 = 评估种子 123-132 上的均值 ± 总体标准差; 括号内 = 相对 8STD 的变化; 每行最小值标绿。
列: 8STD exp298 best ep200 | GS exp301 ep800 (= 主对比图 ..._18hGSep800_2026-10-09 的 301 列, 同一批文件) | GS exp301 best ep95 (冻结快照)
    | GS exp309 best ep280 | GS exp310 best ep445   (best = 训练时 sliding3 判据; 用户令 2026-10-10: 309/310 只评 best)
用法 (仓库根目录): python experiments/tools/figures/plot_dublin18h_gs_ablation_2026-10-10.py [--out-suffix X] [--with-ep800]  (后者多 309/310 ep800 两列, 输出加后缀 _with_ep800)
输出: experiments/analysis/figures/dublin18h_gs_ablation_301_309_310_2026-10-10_en.{png,json}
"""
import glob, json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
OUT_SUFFIX = _arg("--out-suffix", "")
FIG = "experiments/analysis/figures"; PV = "experiments/analysis/data/pervehicle"
SH18 = {"car": .9568, "bus": .0430, "ambulance": .0002}; WT = {"car": 1, "bus": 3, "ambulance": 5}   # 与 plot_five_scenarios 相同
CLASSES = ("car", "bus", "ambulance", "all")
# (列名, exp, 检查点集数, 收集器文件后缀)
ARMS = [("8STD\nexp298 best ep200", 298, 200, "_best_coll"), ("GS exp301\nep800 (main fig.)", 301, 800, "_ckpt_coll"),
        ("GS exp301\nbest ep95", 301, 95, "_best_coll"), ("GS exp309\nbest ep280", 309, 280, "_best_coll"), ("GS exp310\nbest ep445", 310, 445, "_best_coll")]
if "--with-ep800" in sys.argv:   # 2026-10-10 用户令: 309/310 的 ep800 检查点也评了 (合并版工具), 出第二张图, 第一张不动
    ARMS = [ARMS[0], ARMS[1], ("GS exp309\nep800", 309, 800, "_ckpt_coll"), ("GS exp310\nep800", 310, 800, "_ckpt_coll"), ARMS[2], ARMS[3], ARMS[4]]
    OUT_SUFFIX = OUT_SUFFIX or "_with_ep800"
GREEN = "#cfe9cf"; GREY = "#f3f3f3"; HEAD = "#dde3ea"

def load_coll(e, ep, tag):
    return {int(f.split("_seed")[1].split("_")[0]): json.load(open(f)) for f in glob.glob(f"{PV}/exp{e}_ep{ep:05d}_seed*{tag}.json")}

def load_rld(e, ep):
    per = {}
    for f in glob.glob(f"{PV}/exp{e}_ep{ep:05d}_seed*_rldelay.json"):
        s = int(f.split("_seed")[1].split("_")[0]); C = json.load(open(f))["classes"]
        tl = {c: C[c]["rl_timeloss"] for c in C}; en = {c: C[c]["rl_entered"] for c in C}
        per[s] = {c: (tl[c] / en[c] if en[c] else np.nan) for c in C}; per[s]["all"] = sum(tl.values()) / max(sum(en.values()), 1)
    return per

def block_rows(arms):
    """返回 (rows, seeds): rows = [label, [(mean, sd) 每臂], kind]; kind ∈ {t, e, d, Jt, Je, Jd}。"""
    colls = [load_coll(e, ep, tag) for _, e, ep, tag in arms]; rlds = [load_rld(e, ep) for _, e, ep, _ in arms]
    for a, c, r in zip(arms, colls, rlds): print(f"  {a[0].replace(chr(10), ' '):28s} coll seeds={sorted(c)} rldelay seeds={sorted(r)}")
    seeds = sorted(set.intersection(*[set(x) for x in colls + rlds]))
    if not seeds: raise SystemExit("no common seeds")
    rows = []
    for m, label, jlabel, kind in (("avg_stopped_time_per_visit", "Stopped time / visit (s)", "J(531) weighted stopped time (per visit)", "t"),
                                  ("avg_stop_events_per_visit", "Stop events / visit", "J(531) weighted stop events (per visit)", "e")):
        vals = {c: [np.array([X[s]["collector"]["summary"].get(f"eval/system/{c}/{m}", np.nan) for s in seeds]) for X in colls] for c in CLASSES}
        for c in CLASSES:
            rows.append([f"{label} - {c}", [(float(np.nanmean(v)), float(np.nanstd(v))) for v in vals[c]], kind])
        J = [sum(SH18[c] * WT[c] * np.nan_to_num(vals[c][i]) for c in SH18) for i in range(len(arms))]
        rows.append([jlabel, [(float(j.mean()), float(j.std())) for j in J], "J" + kind])
    vals = {c: [np.array([R[s][c] for s in seeds]) for R in rlds] for c in CLASSES}
    for c in CLASSES:
        rows.append([f"RL-junction delay / visit (s) - {c}", [(float(np.nanmean(v)), float(np.nanstd(v))) for v in vals[c]], "d"])
    J = [sum(SH18[c] * WT[c] * np.nan_to_num(vals[c][i]) for c in SH18) for i in range(len(arms))]
    rows.append(["J(531) weighted RL-junction delay (per visit)", [(float(j.mean()), float(j.std())) for j in J], "Jd"])
    return rows, seeds

def main():
    rows, seeds = block_rows(ARMS)
    ncol = 1 + len(ARMS); cells, colors, hdr = [], [], []
    for kind_group, title in (("t", "Stopped time per visit"), ("e", "Stop events per visit"), ("d", "RL-junction delay per visit")):
        hdr.append((len(cells) + 1, title)); cells.append([""] * ncol); colors.append([HEAD] * ncol)
        for lab, ms, kind in rows:
            if kind.replace("J", "") != kind_group: continue
            d = 3 if kind_group == "e" else 2
            means = [m for m, _ in ms]; best = int(np.nanargmin(means)); base = means[0]
            col = [GREY] * ncol; col[1 + best] = GREEN
            txt = [lab]
            for i, (m, sd) in enumerate(ms):
                rel = f" ({100 * (m / base - 1):+.0f}%)" if i > 0 and base > 0 else ""
                txt.append(f"{m:.{d}f} ± {sd:.{d}f}{rel}")
            cells.append(txt); colors.append(col)
    print(f"\n== Dublin 18:00 (seeds n={len(seeds)}: {seeds}) ==")
    for row in cells: print("  " + " | ".join(f"{x:30s}" if i else f"{x:48s}" for i, x in enumerate(row)))
    fig, ax = plt.subplots(figsize=(17.5 + 2.9 * (len(ARMS) - 5), 0.30 * len(cells) + 2.4)); ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=["Metric"] + [a[0] for a in ARMS], cellColours=colors, colColours=["#c9d3df"] * ncol,
                   loc="upper center", cellLoc="center", colWidths=[0.27 * 5 / len(ARMS) if len(ARMS) > 5 else 0.27] + [0.146 * 5 / len(ARMS) if len(ARMS) > 5 else 0.146] * len(ARMS))
    tbl.auto_set_font_size(False); tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        t = cell.get_text(); t.set_fontsize(10)
        if r == 0: t.set_weight("bold"); t.set_fontsize(10.5); cell.set_height(cell.get_height() * 1.6)
        if c == 0 and r > 0: t.set_ha("left"); cell.PAD = 0.012
        if r in [h for h, _ in hdr] and c > 0: cell.set_edgecolor(HEAD)
    ax.set_title("Dublin 18:00: 8STD vs GS-ENUM configurations, mean ± s.d. per visit over evaluation seeds 123-132 (in brackets: change vs 8STD; green = lowest in row)", fontsize=11.5, pad=6)
    fig.canvas.draw(); ren = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for r, title in hdr:
        bb = tbl[r, 0].get_window_extent(ren); x0, y0 = inv.transform((bb.x0, bb.y0)); x1, y1 = inv.transform((bb.x1, bb.y1))
        fig.text(x0 + 0.004, (y0 + y1) / 2, title, ha="left", va="center", fontsize=10.5, weight="bold", zorder=10)
    note = ("exp301 = GS-ENUM + since-green obs + log-scale wait obs + pressure scoring + TD target floor -50.   exp309 = exp301 without pressure scoring (learned scoring).   "
            "exp310 = exp309 without log-scale wait obs.\nbest = training-time sliding-3 best checkpoint (exp301 best = frozen snapshot at ep 95; its final best.pth, ep 135, was not evaluated).\n"
            "Collector metrics: 5 s sampling on RL approaches.   Delay: RL-junction edges only (edgeData timeLoss / entries).   Ambulance = one probe vehicle per episode (underpowered).")
    tb = tbl.get_window_extent(ren); tt = ax.title.get_window_extent(ren)
    fig.text(inv.transform((tb.x0, 0))[0], inv.transform((0, tb.y0))[1] - 0.012, note, ha="left", va="top", fontsize=8.5, color="#333333")
    fig.canvas.draw(); bb = Bbox.union([tb, tt]); pad = 0.15 * fig.dpi; extra = 0.75 * fig.dpi
    crop = Bbox.from_extents((bb.x0 - pad) / fig.dpi, (bb.y0 - pad - extra) / fig.dpi, (bb.x1 + pad) / fig.dpi, (bb.y1 + pad) / fig.dpi)
    out = f"{FIG}/dublin18h_gs_ablation_301_309_310{OUT_SUFFIX}_2026-10-10_en.png"; plt.savefig(out, dpi=200, bbox_inches=crop); print("saved", out)
    dump = {"seeds": seeds, "arms": [(a[0].replace("\n", " "), a[1], a[2], a[3]) for a in ARMS], "rows": [{"label": r[0], "mean_sd": r[1], "kind": r[2]} for r in rows],
            "definitions": {"stopped_time, stop_events": "EpisodeMetricsCollector eval/system/<class>/avg_*_per_visit (5 s sampling, RL approaches)",
                            "delay": "RL-junction edges only: sum edgeData timeLoss / sum approach entries (eval_rl_delay.py)",
                            "J(531)": "sum_c share18_c * weight_c * metric_c, share18 = {car .9568, bus .0430, ambulance .0002}, weight = {5,3,1}"}}
    json.dump(dump, open(out.replace(".png", ".json"), "w"), default=float, indent=1)

if __name__ == "__main__": main()
