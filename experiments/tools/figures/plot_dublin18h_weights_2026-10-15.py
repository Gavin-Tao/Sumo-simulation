"""都柏林 18:00 (改道第二版) 8STD 12 槽 + F1 的优先级权重对比表图 (2026-10-15 用户令: "类似五场景表图, 列是 111 521 531 541, 不用算 rel diff; 用 best 算一个, 其他再算一个, 做两张图")。
臂: 111 = exp458 (无优先级: 全体车辆平均等待, 三类等级全 1), 521 = exp456, 531 = exp449, 541 = exp457; 四臂除优先级权重 (458 另改奖励为全体平均等待) 外逐键相同。
样式与 tools/figures/plot_five_scenarios_2026-09-30.py 相同: 每格 均值 ± 总体标准差 (评估种子 123-132), 每行最低值标绿; 停车时间两位小数, 停车次数三位小数;
J(531) 对所有列都用 5/3/1 权重与 18h 车类份额 (car .9568 / bus .0430 / ambulance .0002) 计算, 作为同一把尺子。
--kind best: 训练中按种子 123 选出的 best 检查点 × 10 种子。
--kind late: 最后 5 个检查点 (第 600/650/700/750/800 集) × 10 种子, 每个种子先对 5 个检查点取平均 (救护车无记录的检查点不计入该种子的救护车平均), 再对种子取均值 ± 标准差。
RL 路口延误 (用户令 "delay 也算上, 是 RL 控制路口的"): 与 plot_five_scenarios_2026-09-30.py 的 rl_delay_rows 同一定义。
输出: experiments/analysis/figures/dublin18h_8std_slot12_f1_weights_{best,late}_2026-10-15_en.{png,json}。用法 (仓库根目录): python experiments/tools/figures/plot_dublin18h_weights_2026-10-15.py --kind best|late"""
import json, glob, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
KIND = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "best"
PER_SECOND = "--per-second" in sys.argv   # 2026-10-15 用户令 "best 10 种子验证要每秒采集": 停车时间/停车次数/访问用逐秒口径 (eval_pervehicle --collector-1s)
EXCL_ALL = [int(x) for x in sys.argv[sys.argv.index("--exclude-all") + 1].split(",")] if "--exclude-all" in sys.argv else []   # 2026-10-15 用户令 "521 去掉那个差的 seed"
EXCL_ARM = {}
if "--exclude" in sys.argv:
    for tok in sys.argv[sys.argv.index("--exclude") + 1].split(","):
        _e, _s = tok.split(":"); EXCL_ARM.setdefault(int(_e), set()).add(int(_s))
def seeds_of(e):
    return [s for s in SEEDS if s not in EXCL_ALL and s not in EXCL_ARM.get(int(e), set())]
NO_J = "--no-j" in sys.argv                 # 2026-10-15 用户令: 去掉 J 行
NO_GREEN = "--no-highlight" in sys.argv     # 2026-10-15 用户令: 不标绿
MIN_TITLE = "--min-title" in sys.argv       # 2026-10-15 用户令: 标题也不写 best 检查点选择与集数 (这些写在 json 与论文图注)
PLAIN_TITLE = "--plain-title" in sys.argv or MIN_TITLE   # 2026-10-15 用户令: 标题不写种子数与剔除说明 (剔除仍记在 json; 论文图注须写明)
DELAY_LABEL = "--old-labels" not in sys.argv   # 2026-10-15 用户令: 图里不写 RL-junction, 写 Average delay / visit (定义不变, 仍只算 RL 路口)
FIG = "experiments/analysis/figures"; PV = "experiments/analysis/data/pervehicle"; PL = "experiments/analysis/data/pervehicle_late_2026-10-14"
SH18 = {"car": .9568, "bus": .0430, "ambulance": .0002}; WT = {"car": 1, "bus": 3, "ambulance": 5}
ARMS = [("111", 458, "exp458_dublin18h_reroute_v2_amb11h_noprio_avgwait_8std_slot12_f1"), ("521", 456, "exp456_dublin18h_reroute_v2_amb11h_521_8std_slot12_f1"),
        ("531", 449, "exp449_dublin18h_reroute_v2_amb11h_531_8std_slot12_f1"), ("541", 457, "exp457_dublin18h_reroute_v2_amb11h_541_8std_slot12_f1")]
SEEDS = list(range(123, 133)); CLS = ("car", "bus", "ambulance", "all")
METRICS = (("avg_stopped_time_per_visit", "Stopped time / visit (s)", "J(531) weighted cost (per visit)", 2),
           ("avg_stop_events_per_visit", "Stop events / visit", "J(531) weighted stop events (per visit)", 3))
GREEN = "#cfe9cf"; GREY = "#f3f3f3"; HEAD = "#dde3ea"


def best_ep(name):
    return int(json.load(open(f"experiments/models/{name}/frozen_best_2026-10-14/best_metrics.json"))["_meta"]["episode"])


def files(e, name):
    """返回 {种子: [评估文件, ...]} (best: 每种子 1 个; late: 每种子 5 个)。"""
    out = {s: [] for s in SEEDS}
    if KIND == "best":
        ep = best_ep(name); fs = glob.glob(f"{PV}/exp{e}_ep{ep:05d}_seed*_best_coll{'1s' if PER_SECOND else ''}.json")
    else:
        fs = glob.glob(f"{PL}/exp{e}_ep00[67][05]0_seed*_late_coll.json") + glob.glob(f"{PV}/exp{e}_ep00800_seed*_ckpt_coll.json")
    for f in fs:
        s = int(f.split("_seed")[1].split("_")[0])
        if s in out: out[s].append(f)
    need = 1 if KIND == "best" else 5
    out = {s: out[s] for s in seeds_of(e)}
    assert all(len(v) == need for v in out.values()), (e, {s: len(v) for s, v in out.items()})
    return out


def per_seed(e, name, metric):
    """{类: 长度 10 的数组}; J 按每个文件先算再在检查点上平均。"""
    F = files(e, name); res = {c: [] for c in CLS + ("J",)}
    for s in seeds_of(e):
        rows = []
        for f in F[s]:
            sm = json.load(open(f))["collector"]["summary"]
            v = {c: sm.get(f"eval/system/{c}/{metric}", np.nan) for c in CLS}
            v["J"] = sum(SH18[c] * WT[c] * np.nan_to_num(v[c]) for c in SH18); rows.append(v)
        for c in CLS + ("J",):
            res[c].append(np.nanmean([r[c] for r in rows]) if any(np.isfinite(r[c]) for r in rows) else np.nan)
    return {c: np.array(v, float) for c, v in res.items()}


def rl_delay_per_seed(e, name):
    """RL 路口延误 / visit (eval_rl_delay.py 的 _rldelay 文件: 18 个 RL 路口进口道边 + 内部边的 timeLoss / 进口道进入次数), 分车类;
    all = 各类 timeLoss 之和 / 进入次数之和; J(531) 用 18h 份额。late: 5 个检查点逐种子平均。"""
    eps = [best_ep(name)] if KIND == "best" else [600, 650, 700, 750, 800]
    res = {c: [] for c in CLS + ("J",)}
    for s in seeds_of(e):
        rows = []
        for ep in eps:
            C = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_rldelay.json"))["classes"]
            tl = {c: C[c]["rl_timeloss"] for c in C}; en = {c: C[c]["rl_entered"] for c in C}
            v = {c: (tl[c] / en[c] if en.get(c) else np.nan) for c in ("car", "bus", "ambulance")}
            v["all"] = sum(tl.values()) / max(sum(en.values()), 1); v["J"] = sum(SH18[c] * WT[c] * np.nan_to_num(v[c]) for c in SH18); rows.append(v)
        for c in CLS + ("J",):
            res[c].append(np.nanmean([r[c] for r in rows]) if any(np.isfinite(r[c]) for r in rows) else np.nan)
    return {c: np.array(v, float) for c, v in res.items()}


def main():
    cells, colors, dump = [], [], {"kind": KIND, "per_second": PER_SECOND, "delay_definition": "RL-junction approaches only (edgeData timeLoss / entries)", "seeds": SEEDS, "excluded_all": EXCL_ALL, "no_j": NO_J, "no_highlight": NO_GREEN, "plain_title": PLAIN_TITLE, "min_title": MIN_TITLE, "excluded_by_arm": {str(k): sorted(v) for k, v in EXCL_ARM.items()}, "arms": {a: e for a, e, _ in ARMS}}
    if KIND == "best":
        dump["best_episode"] = {a: best_ep(n) for a, e, n in ARMS}
    cells.append([""] * 5); colors.append([HEAD] * 5); hdr_rows = [1]
    dl = ("Average delay / visit (s)", "J(531) weighted delay (per visit)") if DELAY_LABEL else ("RL-junction delay / visit (s)", "J(531) weighted RL-junction delay (per visit)")
    for metric, label, jlabel, d in METRICS + (("rl_delay", dl[0], dl[1], 2),):
        V = {a: (rl_delay_per_seed(e, n) if metric == "rl_delay" else per_seed(e, n, metric)) for a, e, n in ARMS}; dump[metric] = {}
        for c in CLS + (("J",) if not NO_J else ()):
            lab = jlabel if c == "J" else f"{label} - {c}"
            means = [float(np.nanmean(V[a][c])) for a, _, _ in ARMS]; sds = [float(np.nanstd(V[a][c])) for a, _, _ in ARMS]
            col = [GREY] * 5
            if not NO_GREEN: col[1 + int(np.nanargmin(means))] = GREEN
            cells.append([lab] + [f"{m:.{d}f} ± {s:.{d}f}" for m, s in zip(means, sds)]); colors.append(col)
            dump[metric][c] = {a: {"mean": m, "sd": s, "per_seed": V[a][c].tolist()} for (a, _, _), m, s in zip(ARMS, means, sds)}
    fig, ax = plt.subplots(figsize=(12.5, 0.30 * len(cells) + 1.6)); ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=["Metric", "111", "521", "531", "541"], cellColours=colors, colColours=["#c9d3df"] * 5,
                   loc="upper center", cellLoc="center", colWidths=[0.40, 0.15, 0.15, 0.15, 0.15])
    tbl.auto_set_font_size(False); tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        t = cell.get_text(); t.set_fontsize(10.5)
        if r == 0: t.set_weight("bold"); t.set_fontsize(11)
        if c == 0 and r > 0: t.set_ha("left"); cell.PAD = 0.012
        if r in hdr_rows and c > 0: cell.set_edgecolor(HEAD)
    block = "Dublin 18:00"
    if KIND == "best":
        be = dump["best_episode"]; sub = "best checkpoint selected on seed 123 (episode " + ", ".join(f"{a}: {be[a]}" for a, _, _ in ARMS) + ")"
    else:
        sub = "last 5 checkpoints (episodes 600/650/700/750/800) averaged per seed"
    tail = "; metrics collected every simulation second" if PER_SECOND else ""
    nlab = "10 evaluation seeds"
    if EXCL_ALL: nlab = f"9 evaluation seeds (seed {','.join(map(str, EXCL_ALL))} excluded in all columns)" if len(EXCL_ALL) == 1 else f"{10 - len(EXCL_ALL)} evaluation seeds (seeds {EXCL_ALL} excluded in all columns)"
    if EXCL_ARM: nlab = "10 evaluation seeds, except " + "; ".join(f"{[a for a, e2, _ in ARMS if e2 == e][0]}: seed {','.join(map(str, sorted(v)))} excluded (n = {len(seeds_of(e))})" for e, v in EXCL_ARM.items())
    if MIN_TITLE: sub, tail = "", "metrics collected every simulation second" if PER_SECOND else ""
    ax.set_title(f"8STD (12-slot state + F1), priority weights ambulance/bus/car (111 = no priority)\n{sub}{tail}; mean ± s.d." + ("" if PLAIN_TITLE else f" over {nlab}") + ("" if MIN_TITLE else ", per visit") + ("" if DELAY_LABEL else "; delay = RL-junction approaches only"),
                 fontsize=12, pad=6)
    fig.canvas.draw(); ren = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    bb = tbl[1, 0].get_window_extent(ren); x0, y0 = inv.transform((bb.x0, bb.y0)); _, y1 = inv.transform((bb.x1, bb.y1))
    fig.text(x0 + 0.004, (y0 + y1) / 2, block, ha="left", va="center", fontsize=11, weight="bold", zorder=10)
    tb = tbl.get_window_extent(ren); tt = ax.title.get_window_extent(ren); bb = Bbox.union([tb, tt]); pad = 0.15 * fig.dpi
    crop = Bbox.from_extents((bb.x0 - pad) / fig.dpi, (bb.y0 - pad) / fig.dpi, (bb.x1 + pad) / fig.dpi, (bb.y1 + pad) / fig.dpi)
    out = f"{FIG}/dublin18h_8std_slot12_f1_weights_{KIND}{'_1s' if PER_SECOND else ''}{('_excl' + '-'.join(map(str, EXCL_ALL)) + 'all') if EXCL_ALL else ''}{''.join(f'_excl{e}s' + '-'.join(map(str, sorted(v))) for e, v in EXCL_ARM.items())}{'_noJ' if NO_J else ''}{'_nohl' if NO_GREEN else ''}{'_mintitle' if MIN_TITLE else ('_plaintitle' if PLAIN_TITLE else '')}_2026-10-15_en.png"; plt.savefig(out, dpi=200, bbox_inches=crop); print("saved", out)
    json.dump(dump, open(out.replace(".png", ".json"), "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
