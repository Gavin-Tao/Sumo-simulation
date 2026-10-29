"""分批结果的优先级权重对比表图 (2026-10-29 用户令 "我要之前的图表算法的对比表"), 算法与 plot_dublin18h_weights_2026-10-15.py 相同:
停车时间/visit、停车次数/visit = 收集器汇总值 (每车 Σ/k_v 再对车平均, 逐秒采集); 延误/visit = RL 路口进口道 timeLoss / 进入次数 (rldelay, all = 各类之和之比);
每格 均值 ± 总体标准差 (np.nanstd, 除以 n), 评估种子 123–132, best 检查点 (训练中按种子 123 选出)。
样式取 2026-10-15 用户最后定的版本 (--no-j --no-highlight --min-title): 不画 J 行 (J(531) 的 18h 车类份额不适用于配对场景), 不标绿。
--pvalues (2026-10-29 用户令 "把 p 值也算出来, 不要覆盖原来的图, 额外做一张图"): 另存 weights_table_figure_style_pvalues.{png,json,md},
  每对实验 (按命令行顺序两两) 加一列 p 值: Welch 双侧 t 检验, 每臂 10 个评估种子的逐种子值, 不做多重比较校正。
--exclude-all LIST|tukey (2026-10-29 用户令 "去掉异常 seed 的对比表和 p 值, 不要覆盖之前的"): 在所有列同时剔除这些种子 (对称剔除, 不做不对称删数);
  tukey = 与 batch_tables.py 相同的对称 Tukey 规则 (每臂 "全部车辆停车时间/visit" 10 个种子的 1.5×IQR 栅栏, 任一臂越栏的种子在所有臂剔除)。
  输出文件名加 _excl<种子>all, 不覆盖原图。
用法 (仓库根目录): python experiments/tools/figures/plot_batch_weights_2026-10-29.py --tag 第1批_修改后_470-472 --title "Dublin 50+50 paired scenario (fixed timetable)" 111:472 531:470 351:471"""
import argparse, glob, json, os, numpy as np
import itertools
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
PV = "experiments/analysis/data/pervehicle"; OUTROOT = "experiments/analysis/data/batch_tables"; SEEDS = list(range(123, 133)); KEEP = SEEDS; CLS = ("car", "bus", "ambulance", "all")
METRICS = (("avg_stopped_time_per_visit", "Stopped time / visit (s)", 2), ("avg_stop_events_per_visit", "Stop events / visit", 3), ("rl_delay", "Average delay / visit (s)", 2))
GREY = "#f3f3f3"; HEAD = "#dde3ea"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--title", default=""); ap.add_argument("--pvalues", action="store_true"); ap.add_argument("--exclude-all", default="")
ap.add_argument("arms", nargs="+"); a = ap.parse_args()
ARMS = [tuple(x.split(":")) for x in a.arms]


def ep_of(e):
    eps = sorted({int(f.split("_ep")[1].split("_")[0]) for f in glob.glob(f"{PV}/exp{e}_ep*_seed*_best_coll1s.json")}); return eps[-1]


def per_seed(e, metric):
    ep = ep_of(e); res = {c: [] for c in CLS}
    for s in KEEP:
        if metric == "rl_delay":
            C = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_rldelay.json"))["classes"]; tl = {c: C[c]["rl_timeloss"] for c in C}; en = {c: C[c]["rl_entered"] for c in C}
            v = {c: (tl[c] / en[c] if en.get(c) else np.nan) for c in ("car", "bus", "ambulance")}; v["all"] = sum(tl.values()) / max(sum(en.values()), 1)
        else:
            sm = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_best_coll1s.json"))["collector"]["summary"]; v = {c: sm.get(f"eval/system/{c}/{metric}", np.nan) for c in CLS}
        for c in CLS: res[c].append(v[c])
    return {c: np.array(x, float) for c, x in res.items()}


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def tukey_flags():
    out = set()
    for _, e in ARMS:
        ep = ep_of(e); v = np.array([json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_best_coll1s.json"))["collector"]["summary"]["eval/system/all/avg_stopped_time_per_visit"] for s in SEEDS])
        q1, q3 = np.percentile(v, [25, 75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1); out |= {s for s, x in zip(SEEDS, v) if x < lo or x > hi}
    return sorted(out)


def main():
    global KEEP
    EXCL = tukey_flags() if a.exclude_all == "tukey" else sorted(int(x) for x in a.exclude_all.split(",") if x)
    KEEP = [s for s in SEEDS if s not in EXCL]
    PAIRS = list(itertools.combinations(range(len(ARMS)), 2)) if a.pvalues else []
    PL = [f"p ({ARMS[j][0]} vs {ARMS[i][0]})" for i, j in PAIRS]
    n = len(ARMS) + len(PAIRS); cells = [[""] * (n + 1)]; colors = [[HEAD] * (n + 1)]; dump = {"seeds": KEEP, "excluded_all": EXCL, "exclusion_rule": a.exclude_all or None, "sd": "population (ddof=0)", "best_episode": {k: ep_of(e) for k, e in ARMS}, "arms": dict(ARMS)}
    if PAIRS: dump["p_test"] = "Welch two-sided t-test on per-seed values (10 vs 10), no multiplicity correction"
    md = [f"| Metric | " + " | ".join([k for k, _ in ARMS] + PL) + " |", "|---|" + "---|" * n]
    for metric, label, d in METRICS:
        V = {k: per_seed(e, metric) for k, e in ARMS}; dump[metric] = {}
        for c in CLS:
            means = [float(np.nanmean(V[k][c])) for k, _ in ARMS]; sds = [float(np.nanstd(V[k][c])) for k, _ in ARMS]
            ps = [float(stats.ttest_ind(V[ARMS[j][0]][c], V[ARMS[i][0]][c], equal_var=False, nan_policy="omit").pvalue) for i, j in PAIRS]
            row = [f"{label} - {c}"] + [f"{m:.{d}f} ± {s:.{d}f}" for m, s in zip(means, sds)] + [fmt_p(p) for p in ps]; cells.append(row); colors.append([GREY] * (n + 1)); md.append("| " + " | ".join(row) + " |")
            dump[metric][c] = {k: {"mean": m, "sd": s, "per_seed": V[k][c].tolist()} for (k, _), m, s in zip(ARMS, means, sds)}
            if PAIRS: dump[metric][c]["p"] = {lab: p for lab, p in zip(PL, ps)}
    na = len(ARMS); fw = 4.9 + 1.9 * na + 2.1 * len(PAIRS)
    fig, ax = plt.subplots(figsize=(fw, 0.30 * len(cells) + 1.6)); ax.axis("off")
    cw = [4.9 / fw] + [1.9 / fw] * na + [2.1 / fw] * len(PAIRS) if PAIRS else [0.40] + [0.60 / na] * na   # 不加 --pvalues 时与原图逐项相同
    tbl = ax.table(cellText=cells, colLabels=["Metric"] + [k for k, _ in ARMS] + PL, cellColours=colors, colColours=["#c9d3df"] * (n + 1), loc="upper center", cellLoc="center",
                   colWidths=cw)
    tbl.auto_set_font_size(False); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        t = cell.get_text(); t.set_fontsize(10.5)
        if r == 0: t.set_weight("bold"); t.set_fontsize(11)
        if c == 0 and r > 0: t.set_ha("left"); cell.PAD = 0.012
        if r == 1 and c > 0: cell.set_edgecolor(HEAD)
    ax.set_title("8STD, priority weights ambulance/bus/car (111 = no priority)\nmetrics collected every simulation second; mean ± s.d." + ("; p: Welch two-sided t-test" if PAIRS else "") + (f"\n{len(KEEP)} evaluation seeds (seeds {', '.join(map(str, EXCL))} excluded in all columns" + ("; symmetric Tukey rule" if a.exclude_all == "tukey" else "") + ")" if EXCL else ("; 10 evaluation seeds" if PAIRS else "")), fontsize=12, pad=6)
    fig.canvas.draw(); ren = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    bb = tbl[1, 0].get_window_extent(ren); x0, y0 = inv.transform((bb.x0, bb.y0)); _, y1 = inv.transform((bb.x1, bb.y1))
    fig.text(x0 + 0.004, (y0 + y1) / 2, a.title, ha="left", va="center", fontsize=11, weight="bold", zorder=10)
    tb = tbl.get_window_extent(ren); tt = ax.title.get_window_extent(ren); bb = Bbox.union([tb, tt]); pad = 0.15 * fig.dpi
    crop = Bbox.from_extents((bb.x0 - pad) / fig.dpi, (bb.y0 - pad) / fig.dpi, (bb.x1 + pad) / fig.dpi, (bb.y1 + pad) / fig.dpi)
    od = os.path.join(OUTROOT, a.tag); os.makedirs(od, exist_ok=True); out = os.path.join(od, "weights_table_figure_style" + ("_pvalues" if PAIRS else "") + (("_excl" + "-".join(map(str, EXCL)) + "all") if EXCL else "") + ".png")
    plt.savefig(out, dpi=200, bbox_inches=crop); json.dump(dump, open(out.replace(".png", ".json"), "w"), indent=1)
    open(out.replace(".png", ".md"), "w").write("\n".join(md) + "\n"); print("saved", out); print("\n".join(md))


if __name__ == "__main__":
    main()
