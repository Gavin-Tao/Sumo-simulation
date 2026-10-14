"""1x1 全开 F1 状态/结构网格的成对比较 (2026-10-14; 判读规则事先写在 analysis/GRID1X1_F1_STATE_GRID_2026-10-13.txt §4)。
每对 (A, B) 在评估种子 123-132 上配对, 两个检查点各算一次: best (训练中用种子 123 选出) 与 第 1500 集检查点。
主指标 = 全部车辆停车时间/visit; 另报分车类停车时间、停车次数/visit、J(531)、训练后诊断 (Q/真实回报, 能切时切换比例)。
判读: |均值差| ≤ 5% → 等价; 差 > 5% 且配对同向 ≥ 7/10 → 一方更好; 其余 → 倾向, 不确定; best 与第 1500 集结论不一致 → 不确定。
用法 (仓库根目录): python experiments/tools/frap/analyze_f1_state_grid_2026-10-14.py <A:B> [<A:B> ...]   例: 374:377 375:378
"""
import glob, json, os, sys
import numpy as np

PV = "experiments/analysis/data/pervehicle"; DG = "experiments/analysis/data/posttrain_diag"
SEEDS = list(range(123, 133))
SH = {"car": .965, "bus": .033, "ambulance": .0017}; WT = {"car": 1, "bus": 3, "ambulance": 5}


def files(e, which):
    if which == "best":
        fs = glob.glob(f"{PV}/exp{e}_ep*_seed*_best_coll.json")
        if not fs:
            return {}
        ep = max(int(f.split("_ep")[1].split("_")[0]) for f in fs)
        fs = [f for f in fs if f"_ep{ep:05d}_" in f]
    else:
        fs = glob.glob(f"{PV}/exp{e}_ep01500_seed*_ckpt_coll.json")
    return {int(f.split("_seed")[1].split("_")[0]): f for f in fs}


def metrics(path):
    s = json.load(open(path))["collector"]["summary"]
    g = lambda c, m: float(s.get(f"eval/system/{c}/{m}", np.nan))
    out = {f"stop_{c}": g(c, "avg_stopped_time_per_visit") for c in ("all", "car", "bus", "ambulance")}
    out.update({f"events_{c}": g(c, "avg_stop_events_per_visit") for c in ("all", "car", "bus")})
    out["J"] = sum(SH[c] * WT[c] * np.nan_to_num(out[f"stop_{c}"]) for c in SH)
    return out


def verdict(a, b):
    rel = a.mean() / b.mean() - 1; wins = int((a < b).sum()); losses = int((a > b).sum())
    if abs(rel) <= 0.05:
        v = "等价"
    elif rel < 0 and wins >= 7:
        v = "A 更好"
    elif rel > 0 and losses >= 7:
        v = "A 更差"
    else:
        v = "倾向, 不确定"
    return rel, wins, losses, v


def diag(e, tag):
    p = f"{DG}/exp{e}_{tag}.json"
    if not os.path.exists(p):
        return None
    m = json.load(open(p))["mean"]; return m


def compare(A, B):
    res = {}
    for which in ("best", "ep1500"):
        fa, fb = files(A, which), files(B, which)
        seeds = [s for s in SEEDS if s in fa and s in fb]
        if len(seeds) < 10:
            res[which] = None; continue
        ma = [metrics(fa[s]) for s in seeds]; mb = [metrics(fb[s]) for s in seeds]
        arr = lambda M, k: np.array([m[k] for m in M])
        res[which] = {k: (arr(ma, k), arr(mb, k)) for k in ma[0]}
    return res


def main():
    pairs = [tuple(int(x) for x in p.split(":")) for p in sys.argv[1:]]
    for A, B in pairs:
        r = compare(A, B)
        print(f"\n===== exp{A} (A) 对 exp{B} (B) =====")
        vs = {}
        for which in ("best", "ep1500"):
            if r[which] is None:
                print(f"  [{which}] 评估不全"); continue
            a, b = r[which]["stop_all"]; rel, w, l, v = verdict(a, b); vs[which] = v
            print(f"  [{which:6s}] 全部车辆停车时间 A {a.mean():6.2f}±{a.std():4.2f}  B {b.mean():6.2f}±{b.std():4.2f}  差 {100*rel:+5.1f}%  A 胜 {w}/10 负 {l}/10  → {v}")
            line = []
            for k, lab in (("stop_car", "小汽车"), ("stop_bus", "公交"), ("stop_ambulance", "救护车"), ("events_all", "停车次数"), ("J", "J")):
                x, y = r[which][k]; line.append(f"{lab} {np.nanmean(x):.2f}/{np.nanmean(y):.2f}")
            print("           " + " | ".join(line) + "   (A/B)")
        final = vs.get("best") if vs.get("best") == vs.get("ep1500") else "不确定 (两个检查点结论不一致)"
        print(f"  结论: {final}")
        for tag in ("best", "ep1500"):
            da, db = diag(A, tag), diag(B, tag)
            if da and db:
                print(f"  诊断[{tag}] Q/真实回报 A {da['q_over_G']:.2f} B {db['q_over_G']:.2f} | 能切时切换 A {100*da['switch_when_free']:.0f}% B {100*db['switch_when_free']:.0f}%")


if __name__ == "__main__":
    main()
