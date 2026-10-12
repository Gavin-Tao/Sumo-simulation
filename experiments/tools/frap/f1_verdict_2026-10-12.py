"""F1 判定 (2026-10-12, 判据事先登记于 analysis/GRID1X1_GSG_ANALYSIS_2026-10-12.txt §10.1, 训练开始前写定)。
指标 = 全部车辆停车时间/visit (best 检查点 × 评估种子 123-132, 按种子配对), 口径同 plot_state_ablation_2026-10-12。
  R 严重臂恢复: 371 对 353、372 对 360 各自 均值下降 >= 15% 且 配对胜 >= 8/10 (两臂都满足)
  H 健康臂不变差: 373 对 327、374 对 256 各自 不出现 "均值上升 > 5% 且 配对输 >= 7/10" (两臂都满足)
  采用 F1 <=> R 且 H; 其余一律不采用 (只有一个严重臂恢复 = MIXED, 标注人工复核)。
J(531)、分车类停车时间、停车次数只报告, 不进判定。
用法 (仓库根目录): python experiments/tools/frap/f1_verdict_2026-10-12.py [--apply]
  --apply 且结论为采用: 在 362-370 配置末尾追加 mask_min_green: true (已有该键则跳过); 不采用不动任何配置。
输出: 一行 "F1_VERDICT <ADOPT|REJECT|MIXED_REJECT> ..." + experiments/analysis/data/f1_verdict_2026-10-12.json
"""
import argparse, glob, json, os, sys
import numpy as np

PV = "experiments/analysis/data/pervehicle"
SEEDS = list(range(123, 133))
SEVERE = [(371, 353), (372, 360)]
HEALTHY = [(373, 327), (374, 256)]
REST = list(range(362, 371))
SH = {"car": .965, "bus": .033, "ambulance": .0017}; WT = {"car": 1, "bus": 3, "ambulance": 5}
CL = ("car", "bus", "ambulance", "all")


def best_files(e):
    fs = glob.glob(f"{PV}/exp{e}_ep*_seed*_best_coll.json")
    if not fs:
        return {}
    ep = max(int(f.split("_ep")[1].split("_")[0]) for f in fs)
    return {int(f.split("_seed")[1].split("_")[0]): json.load(open(f)) for f in fs if f"_ep{ep:05d}_" in f}


def metric(d, c, m="avg_stopped_time_per_visit"):
    return float(d["collector"]["summary"].get(f"eval/system/{c}/{m}", np.nan))


def compare(new, base):
    a, b = best_files(new), best_files(base)
    miss = [s for s in SEEDS if s not in a or s not in b]
    if miss:
        return None, f"exp{new}/exp{base} 缺种子 {miss}"
    x = np.array([metric(a[s], "all") for s in SEEDS]); y = np.array([metric(b[s], "all") for s in SEEDS])
    rep = {"new": new, "base": base, "mean_new": float(x.mean()), "mean_base": float(y.mean()),
           "rel_change": float(x.mean() / y.mean() - 1), "wins": int((x < y).sum()), "losses": int((x > y).sum())}
    for c in CL:
        rep[f"stopped_{c}"] = [float(np.nanmean([metric(a[s], c) for s in SEEDS])), float(np.nanmean([metric(b[s], c) for s in SEEDS]))]
        rep[f"stops_{c}"] = [float(np.nanmean([metric(a[s], c, "avg_stop_events_per_visit") for s in SEEDS])),
                             float(np.nanmean([metric(b[s], c, "avg_stop_events_per_visit") for s in SEEDS]))]
    rep["J531"] = [sum(SH[c] * WT[c] * np.nan_to_num(rep[f"stopped_{c}"][i]) for c in SH) for i in (0, 1)]
    return rep, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    reps, errs = {}, []
    for new, base in SEVERE + HEALTHY:
        r, err = compare(new, base)
        if err:
            errs.append(err)
        else:
            reps[new] = r
    if errs:
        print("F1_VERDICT INCOMPLETE " + "; ".join(errs)); sys.exit(2)
    rec = {n: (reps[n]["rel_change"] <= -0.15 and reps[n]["wins"] >= 8) for n, _ in SEVERE}
    worse = {n: (reps[n]["rel_change"] > 0.05 and reps[n]["losses"] >= 7) for n, _ in HEALTHY}
    R = all(rec.values()); H = not any(worse.values())
    if R and H:
        verdict = "ADOPT"
    elif sum(rec.values()) == 1 and H:
        verdict = "MIXED_REJECT"
    else:
        verdict = "REJECT"
    summ = " ".join(f"{n}vs{reps[n]['base']}:{reps[n]['mean_new']:.2f}/{reps[n]['mean_base']:.2f}({100 * reps[n]['rel_change']:+.1f}%,胜{reps[n]['wins']}输{reps[n]['losses']})"
                    for n, _ in SEVERE + HEALTHY)
    applied = []
    if a.apply and verdict == "ADOPT":
        for e in REST:
            fs = [f for f in glob.glob(f"experiments/configs/exp{e}_*.yaml") if "traindry" not in f and "smoke" not in f]
            if len(fs) != 1:
                print(f"F1_VERDICT 配置不唯一 exp{e}: {fs}"); sys.exit(3)
            s = open(fs[0]).read()
            if "mask_min_green" in s:
                continue
            with open(fs[0], "a") as fh:
                fh.write("\nmask_min_green: true          # 2026-10-12 F1 判定结果 = 采用 (§10.1 事先登记的判据, f1_verdict_2026-10-12.py 自动追加)\n")
            applied.append(e)
    out = {"verdict": verdict, "R_severe_recovered": rec, "H_healthy_worse": worse, "reports": reps, "applied_to": applied}
    json.dump(out, open("experiments/analysis/data/f1_verdict_2026-10-12.json", "w"), indent=1, ensure_ascii=False)
    print(f"F1_VERDICT {verdict} R={R} H={H} {summ} 已追加配置={applied}")


if __name__ == "__main__":
    main()
