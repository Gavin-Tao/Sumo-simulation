"""分批结果表 (2026-10-29 用户令 "跑完一批就整理结果表, 就是 per visit 那三个指标, 对于 car bus amb all 的, 还要检查是否有优先级反转")。
数据: 每个实验 best 检查点 × 10 评估种子 (123–132), 逐秒收集器 (_best_coll1s) + RL 路口延误 (_rldelay); 指标定义与 exact_vs_old_tables.py 相同:
  停车时间/visit、停车次数/visit (收集器 eval/system/<类>/avg_*_per_visit), 平均延误/visit (RL 路口进口道 timeLoss / 进入次数)。
异常种子: 对称 Tukey 规则 —— 对每个实验 "全部车辆停车时间/visit" 的 10 个种子值算 1.5×IQR 栅栏, 任一实验越栏的种子在本批所有实验同时剔除 (不做不对称删数)。
优先级反转: 按该实验配置的优先级表, 权重更高的类在每个指标上都应更低 (均值); 违反即反转。111 各类权重相同, 不检查只报告。另报每对类在 10 个种子里有几个种子违反。
输出: experiments/analysis/data/batch_tables/<tag>/tables.json 与 tables.md。
用法 (仓库根目录): python experiments/tools/dublin/batch_tables.py --tag 第1批 470 471 472"""
import argparse, glob, json, os, itertools
import numpy as np
import yaml

PV = "experiments/analysis/data/pervehicle"; OUTROOT = "experiments/analysis/data/batch_tables"
SEEDS = list(range(123, 133)); CLS = ("car", "bus", "ambulance", "all"); NAME = {"car": "小汽车", "bus": "公交", "ambulance": "救护车", "all": "全部"}
METRICS = [("avg_stopped_time_per_visit", "停车时间/visit (s)", 2), ("avg_stop_events_per_visit", "停车次数/visit", 3), ("rl_delay", "平均延误/visit (s)", 2)]


def cfg_of(e):
    f = sorted(glob.glob(f"experiments/configs/exp{e}_*.yaml"))[0]
    c = yaml.safe_load(open(f)); ps = c.get("priority_source") or {}
    return c["name"], {t: float(ps.get(t, 1)) for t in ("car", "bus", "ambulance")}


def best_ep(e):
    fs = glob.glob(f"{PV}/exp{e}_ep*_seed*_best_coll1s.json"); assert fs, f"exp{e} 没有评估结果"
    return sorted({int(f.split("_ep")[1].split("_")[0]) for f in fs})[-1]


def per_seed(e, metric):
    ep = best_ep(e); res = {c: [] for c in CLS}
    for s in SEEDS:
        if metric == "rl_delay":
            C = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_rldelay.json"))["classes"]
            v = {c: (C[c]["rl_timeloss"] / C[c]["rl_entered"] if C.get(c, {}).get("rl_entered") else np.nan) for c in ("car", "bus", "ambulance")}
            v["all"] = sum(C[c]["rl_timeloss"] for c in C) / max(sum(C[c]["rl_entered"] for c in C), 1)
        else:
            sm = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_best_coll1s.json"))["collector"]["summary"]
            v = {c: sm.get(f"eval/system/{c}/{metric}", np.nan) for c in CLS}
        for c in CLS: res[c].append(float(v[c]))
    return {c: np.array(x, float) for c, x in res.items()}


def tukey_out(vals):
    q1, q3 = np.nanpercentile(vals, [25, 75]); iqr = q3 - q1; lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [s for s, v in zip(SEEDS, vals) if np.isfinite(v) and (v < lo or v > hi)]


def inversion(D, w, keep):
    """权重高的类应更低。返回 [{metric, hi, lo, mean_hi, mean_lo, inverted, seeds_inverted, n_seeds}]。"""
    idx = [i for i, s in enumerate(SEEDS) if s in keep]; out = []
    for (m, _, _), (a, b) in itertools.product(METRICS, itertools.permutations(("car", "bus", "ambulance"), 2)):
        if not w[a] > w[b]: continue
        xa, xb = D[m][a][idx], D[m][b][idx]; ok = np.isfinite(xa) & np.isfinite(xb)
        ma, mb = float(np.nanmean(xa)), float(np.nanmean(xb))
        out.append(dict(metric=m, hi=a, lo=b, mean_hi=ma, mean_lo=mb, inverted=bool(ma >= mb), seeds_inverted=int(np.sum(xa[ok] >= xb[ok])), n_seeds=int(ok.sum())))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("exps", nargs="+"); a = ap.parse_args()
    arms = []
    for e in a.exps:
        name, w = cfg_of(e); arms.append(dict(exp=e, name=name, w=w, spec="".join(str(int(w[t])) for t in ("ambulance", "bus", "car")), ep=best_ep(e),
                                              D={m: per_seed(e, m) for m, _, _ in METRICS}))
    out_seeds = sorted({s for x in arms for s in tukey_out(x["D"]["avg_stopped_time_per_visit"]["all"])})
    keep_all = SEEDS; keep_t = [s for s in SEEDS if s not in out_seeds]
    res = dict(tag=a.tag, seeds=SEEDS, tukey_excluded=out_seeds, arms=[])
    for x in arms:
        r = dict(exp=x["exp"], name=x["name"], spec=x["spec"], best_episode=x["ep"], weights=x["w"], metrics={})
        for m, _, _ in METRICS:
            r["metrics"][m] = {k: {c: dict(mean=float(np.nanmean(x["D"][m][c][[i for i, s in enumerate(SEEDS) if s in keep]])),
                                          sd=float(np.nanstd(x["D"][m][c][[i for i, s in enumerate(SEEDS) if s in keep]], ddof=1)),
                                          per_seed=[round(v, 4) for v in x["D"][m][c].tolist()]) for c in CLS}
                               for k, keep in (("all_seeds", keep_all), ("tukey", keep_t))}
        r["inversion"] = {k: inversion(x["D"], x["w"], keep) for k, keep in (("all_seeds", keep_all), ("tukey", keep_t))}
        r["any_inversion"] = {k: any(i["inverted"] for i in v) for k, v in r["inversion"].items()}
        res["arms"].append(r)
    od = os.path.join(OUTROOT, a.tag); os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "tables.json"), "w"), indent=1, ensure_ascii=False)
    L = [f"# {a.tag} 结果表\n", f"best 检查点 × 10 评估种子 (123–132), 逐秒收集。Tukey 剔除的种子: {out_seeds or '无'}。数值越低越好。\n"]
    for k, title in (("all_seeds", "10 个种子"), ("tukey", f"剔除异常种子后 ({len(keep_t)} 个种子)")):
        L.append(f"## {title}\n")
        for m, label, dg in METRICS:
            L.append(f"### {label}\n\n| 实验 | 规格 | best 集 | 小汽车 | 公交 | 救护车 | 全部 | 优先级顺序 |\n|---|---|---|---|---|---|---|---|")
            for r in res["arms"]:
                v = r["metrics"][m][k]; inv = [i for i in r["inversion"][k] if i["metric"] == m]
                chk = "不检查 (各类权重相同)" if not inv else ("**反转**: " + "; ".join(f"{NAME[i['hi']]} ≥ {NAME[i['lo']]}" for i in inv if i["inverted"]) if any(i["inverted"] for i in inv) else "正确")
                L.append(f"| exp{r['exp']} | {r['spec']} | {r['best_episode']} | " + " | ".join(f"{v[c]['mean']:.{dg}f} ± {v[c]['sd']:.{dg}f}" for c in CLS) + f" | {chk} |")
            L.append("")
        L.append("逐种子: 每对 (权重高 → 应更低) 在多少个种子上违反\n")
        for r in res["arms"]:
            inv = r["inversion"][k]
            if not inv: L.append(f"- exp{r['exp']} ({r['spec']}): 不检查"); continue
            L.append(f"- exp{r['exp']} ({r['spec']}): " + "; ".join(f"{dict((m_, l_) for m_, l_, _ in METRICS)[i['metric']]} {NAME[i['hi']]}<{NAME[i['lo']]} 违反 {i['seeds_inverted']}/{i['n_seeds']}" for i in inv))
        L.append("")
    open(os.path.join(od, "tables.md"), "w").write("\n".join(L) + "\n")
    print("写出", od); print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
