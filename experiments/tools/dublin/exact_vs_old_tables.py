"""新旧口径八臂对比表 (2026-10-27 用户令 "把新旧的 541 531 521 111 数据对比, 包括 car bus amb all, 做个表格; 一个十个种子, 一个去掉异常值; 看优先级是否反转")。
数据: best 检查点 × 10 评估种子 (123-132), 逐秒收集器 (_best_coll1s) + RL 路口延误 (_rldelay); 指标定义与 plot_dublin18h_weights_2026-10-15.py 相同:
  停车时间/visit、停车次数/visit (收集器 eval/system/<类>/avg_*_per_visit), 平均延误/visit (18 个 RL 路口进口道 timeLoss / 进入次数), J(531) = Σ 份额×权重×指标 (18h 份额 car .9568 / bus .0430 / amb .0002)。
异常种子: 对称 Tukey 规则 —— 对每臂 "全部车辆停车时间/visit" 的 10 个种子值算 1.5×IQR 栅栏, 任一臂越栏的种子在八臂同时剔除 (不做不对称删数)。
输出: analysis/data/exact_vs_old_2026-10-27/tables.json 与 tables.md。用法 (仓库根目录): python experiments/tools/dublin/exact_vs_old_tables.py"""
import json, glob, os, sys, numpy as np
PV = "experiments/analysis/data/pervehicle"; OUT = "experiments/analysis/data/exact_vs_old_2026-10-27"
SH18 = {"car": .9568, "bus": .0430, "ambulance": .0002}; WT = {"car": 1, "bus": 3, "ambulance": 5}
ARMS = [("old", "111", 458), ("old", "521", 456), ("old", "531", 449), ("old", "541", 457), ("new", "111", 460), ("new", "521", 462), ("new", "531", 461), ("new", "541", 459)]
SEEDS = list(range(123, 133)); CLS = ("car", "bus", "ambulance", "all")
METRICS = [("avg_stopped_time_per_visit", "停车时间/visit (s)", 2), ("avg_stop_events_per_visit", "停车次数/visit", 3), ("rl_delay", "平均延误/visit (s)", 2)]
def best_ep(e):
    fs = glob.glob(f"{PV}/exp{e}_ep*_seed*_best_coll1s.json"); assert fs, e
    eps = sorted({int(f.split("_ep")[1].split("_")[0]) for f in fs}); return eps[-1]
def per_seed(e, metric):
    ep = best_ep(e); res = {c: [] for c in CLS + ("J",)}
    for s in SEEDS:
        if metric == "rl_delay":
            C = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_rldelay.json"))["classes"]
            v = {c: (C[c]["rl_timeloss"] / C[c]["rl_entered"] if C.get(c, {}).get("rl_entered") else np.nan) for c in ("car", "bus", "ambulance")}
            v["all"] = sum(C[c]["rl_timeloss"] for c in C) / max(sum(C[c]["rl_entered"] for c in C), 1)
        else:
            sm = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_best_coll1s.json"))["collector"]["summary"]
            v = {c: sm.get(f"eval/system/{c}/{metric}", np.nan) for c in CLS}
        v["J"] = sum(SH18[c] * WT[c] * np.nan_to_num(v[c]) for c in SH18)
        for c in CLS + ("J",): res[c].append(float(v[c]))
    return {c: np.array(x, float) for c, x in res.items()}
def tukey_flags(vals):
    q1, q3 = np.nanpercentile(vals, [25, 75]); iqr = q3 - q1; lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [s for s, v in zip(SEEDS, vals) if np.isfinite(v) and (v < lo or v > hi)], (lo, hi)
def main():
    D = {f"{w}{a}": {m: per_seed(e, m) for m, _, _ in METRICS} for w, a, e in ARMS}
    flags = {}
    for w, a, e in ARMS:
        f, fence = tukey_flags(D[f"{w}{a}"]["avg_stopped_time_per_visit"]["all"]); flags[f"{w}{a}"] = dict(seeds=f, fence=[float(fence[0]), float(fence[1])])
    excl = sorted({s for v in flags.values() for s in v["seeds"]})
    keep = {"all10": SEEDS, "tukey": [s for s in SEEDS if s not in excl]}
    out = {"arms": {f"{w}{a}": {"exp": e, "best_ep": best_ep(e)} for w, a, e in ARMS}, "flags": flags, "excluded_symmetric": excl, "tables": {}}
    md = [f"# 新旧口径八臂对比 (best × 10 种子, 逐秒口径) — 2026-10-27\n", "臂: " + ", ".join(f"{w}{a}=exp{e} (best ep{best_ep(e)})" for w, a, e in ARMS) + "\n",
          f"Tukey 栅栏 (全部车辆停车时间/visit): " + "; ".join(f"{k}: 越栏种子 {v['seeds']} 栅栏 [{v['fence'][0]:.2f}, {v['fence'][1]:.2f}]" for k, v in flags.items()) + f"\n对称剔除种子: {excl} (剔后 n={len(keep['tukey'])})\n"]
    for kname, seeds in keep.items():
        idx = [SEEDS.index(s) for s in seeds]; out["tables"][kname] = {"seeds": seeds, "rows": []}
        md.append(f"\n## {'全部 10 种子' if kname == 'all10' else '剔除异常种子 ' + str(excl) + ' 后 (n=' + str(len(seeds)) + ')'}\n")
        md.append("| 指标 | 旧111 | 旧521 | 旧531 | 旧541 | 新111 | 新521 | 新531 | 新541 | 新/旧 111 | 新/旧 521 | 新/旧 531 | 新/旧 541 |\n|---|" + "---|" * 12)
        for m, label, d in METRICS:
            for c in CLS + ("J",):
                cells, rec = [], {"metric": m, "class": c}
                for w, a, e in ARMS:
                    v = D[f"{w}{a}"][m][c][idx]; mu, sd = float(np.nanmean(v)), float(np.nanstd(v)); rec[f"{w}{a}"] = {"mean": mu, "sd": sd, "n": int(np.isfinite(v).sum())}
                    cells.append(f"{mu:.{d}f} ± {sd:.{d}f}")
                ratios = []
                for a in ("111", "521", "531", "541"):
                    o, n = rec[f"old{a}"]["mean"], rec[f"new{a}"]["mean"]; ratios.append(f"{n / o:.2f}" if o else "—"); rec[f"ratio{a}"] = (n / o if o else None)
                lab = ("J(531) " + label) if c == "J" else f"{label} - {c}"
                md.append(f"| {lab} | " + " | ".join(cells + ratios) + " |"); out["tables"][kname]["rows"].append(rec)
        # 优先级顺序检查 (停车时间/visit): 同臂内 bus<car? amb<bus?; 跨臂 bus 541<531<521<111? car 111 最小? all 111 最小? 配对胜场按种子
        st = "avg_stopped_time_per_visit"; chk = []
        for w in ("old", "new"):
            g = lambda a, c: D[f"{w}{a}"][st][c][idx]
            for a in ("111", "521", "531", "541"):
                chk.append(f"{w}{a}: bus<car {int(np.nansum(g(a,'bus') < g(a,'car')))}/{len(idx)} 种子; amb<bus {int(np.nansum(g(a,'ambulance') < g(a,'bus')))}/{int(np.isfinite(g(a,'ambulance')).sum())}")
            for c in ("bus", "car", "all"):
                pairs = [("541", "531"), ("531", "521"), ("521", "111")]
                chk.append(f"{w} {c}: " + "; ".join(f"{p}<{q} {int(np.nansum(g(p,c) < g(q,c)))}/{len(idx)} (均值 {np.nanmean(g(p,c)):.2f} vs {np.nanmean(g(q,c)):.2f})" for p, q in pairs))
        md.append("\n优先级顺序检查 (停车时间/visit, 配对种子胜场):\n" + "\n".join("- " + x for x in chk)); out["tables"][kname]["ordering"] = chk
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(f"{OUT}/tables.json", "w"), indent=1, ensure_ascii=False, default=float)
    open(f"{OUT}/tables.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
if __name__ == "__main__":
    main()
