"""救护车 (或其他类) 逐 RL 路口 × 逐种子的停车时间/visit, 并把每个种子相对 10 种子均值的偏差分解到各路口 (只读已有 best×10 逐秒评估结果)。
口径与结果表 (收集器 avg_stopped_time_per_visit, metrics.py _per_visit_mean) 相同: 每辆车 (各路口停车秒之和 / 经过路口数) 再对车平均。
表格单元 = 该路口上该类车平均每次经过的停车秒; 路口 j 的贡献 = (1/V) Σ_v 停车秒_vj / k_v, 各路口贡献之和恰好等于结果表的值。
用法 (仓库根目录): python experiments/tools/dublin/amb_by_junction.py --cls ambulance --out <md> 470 471 472"""
import argparse, collections, glob, json
import numpy as np
PV = "experiments/analysis/data/pervehicle"; SEEDS = list(range(123, 133))
def short(t): return "_".join(t.split("_")[:2]) if t.startswith("cluster") else t
ap = argparse.ArgumentParser(); ap.add_argument("--cls", default="ambulance"); ap.add_argument("--out"); ap.add_argument("exps", nargs="+"); a = ap.parse_args()
L = []
for e in a.exps:
    S = {}; N = {}; C = {}; summ = {}
    for s in SEEDS:
        f = glob.glob(f"{PV}/exp{e}_ep*_seed{s}_best_coll1s.json"); assert len(f) == 1, f; j = json.load(open(f[0]))
        st = collections.Counter(); nv = collections.Counter(); con = collections.Counter(); V = 0
        for vid, v in j["collector"]["vehicles"].items():
            if v.get("type") != a.cls or not v.get("k_v"): continue
            V += 1
            for t in j["vehicles"][vid]["tls"]: nv[t] += 1          # 经过的路口 (逐秒统计器的访问登记)
            for t, sec in v["stopped_sec"].items(): st[t] += float(sec); con[t] += float(sec) / v["k_v"]
        C[s] = {t: x / V for t, x in con.items()}
        S[s], N[s] = st, nv; summ[s] = j["collector"]["summary"][f"eval/system/{a.cls}/avg_stopped_time_per_visit"]
    J = sorted({t for s in SEEDS for t in N[s]}, key=lambda t: -sum(N[s][t] for s in SEEDS))
    tot = {s: sum(C[s].values()) for s in SEEDS}
    chk = max(abs(tot[s] - summ[s]) for s in SEEDS)
    L.append(f"## exp{e}: {a.cls} 停车时间/visit (s), 逐路口 × 逐种子\n\n与结果表核对: 各路口贡献之和与收集器汇总值最大差 {chk:.2g}\n")
    L.append("| 路口 | 访问/集 | " + " | ".join(f"s{s}" for s in SEEDS) + " | 10 种子均值 |"); L.append("|---|---|" + "---|" * (len(SEEDS) + 1))
    for t in J:
        vals = [S[s][t] / N[s][t] if N[s][t] else np.nan for s in SEEDS]
        nvis = sorted({N[s][t] for s in SEEDS}); nvs = str(nvis[0]) if len(nvis) == 1 else f"{nvis[0]}–{nvis[-1]}"
        L.append(f"| {short(t)} | {nvs} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {np.nanmean(vals):.2f} |")
    L.append(f"| **全部 (结果表口径)** | {sum(N[SEEDS[0]].values())} | " + " | ".join(f"**{tot[s]:.2f}**" for s in SEEDS) + f" | **{np.mean(list(tot.values())):.2f}** |\n")
    mu = {t: np.mean([C[s].get(t, 0.0) for s in SEEDS]) for t in J}
    L.append(f"偏差分解: 每个种子比 10 种子均值多/少多少 (s/visit), 及贡献最大的路口\n")
    L.append("| 种子 | 停车时间/visit | 偏差 | 贡献最大的路口 (贡献 s/visit, 占偏差比例) | 第二 | 同向贡献的路口数 |"); L.append("|---|---|---|---|---|---|")
    m_all = np.mean(list(tot.values()))
    for s in sorted(SEEDS, key=lambda s: -tot[s]):
        c = {t: C[s].get(t, 0.0) - mu[t] for t in J}; d = tot[s] - m_all
        sgn = np.sign(d) if d else 1; top = sorted(c.items(), key=lambda kv: -sgn * kv[1])
        same = sum(1 for v in c.values() if np.sign(v) == sgn and abs(v) >= 0.01)
        f = lambda kv: f"{short(kv[0])} ({kv[1]:+.2f}, {kv[1] / d * 100:.0f}%)" if d else "-"
        L.append(f"| s{s} | {tot[s]:.2f} | {d:+.2f} | {f(top[0])} | {f(top[1])} | {same}/{len(J)} |")
    L.append("")
txt = "\n".join(L); print(txt)
if a.out: open(a.out, "w").write(txt + "\n")
