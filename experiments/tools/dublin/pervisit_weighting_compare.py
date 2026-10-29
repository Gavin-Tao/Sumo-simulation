"""per-visit 两种平均口径对比 (只读已有 best×10 逐秒评估结果):
  按车平均 (结果表口径, 收集器 _per_visit_mean): 每辆车 Σ停车秒(次数) / 经过路口数 k_v, 再对车平均;
  按访问平均: 该类所有车 Σ停车秒(次数) / Σ k_v (停车次数这一项即收集器的 avg_stop_events_per_all_visit, 用来核对)。
延误只有按访问平均一种 (rldelay 只存分车类、分路口的总量, 没有逐车 RL 路口时间损失)。
用法 (仓库根目录): python experiments/tools/dublin/pervisit_weighting_compare.py --tag 第1批_修改后_470-472 470 471 472"""
import argparse, glob, itertools, json, os
import numpy as np
PV = "experiments/analysis/data/pervehicle"; OUTROOT = "experiments/analysis/data/batch_tables"; SEEDS = list(range(123, 133))
CLS = ("car", "bus", "ambulance", "all"); NAME = {"car": "小汽车", "bus": "公交", "ambulance": "救护车", "all": "全部"}
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("exps", nargs="+"); a = ap.parse_args()
def weights(e):
    import yaml; c = yaml.safe_load(open(sorted(glob.glob(f"experiments/configs/exp{e}_*.yaml"))[0])); ps = c.get("priority_source") or {}
    return {t: float(ps.get(t, 1)) for t in ("car", "bus", "ambulance")}
R = {}; chk = 0.0
for e in a.exps:
    D = {k: {c: [] for c in CLS} for k in ("t_veh", "t_vis", "n_veh", "n_vis", "d_vis")}
    for s in SEEDS:
        j = json.load(open(glob.glob(f"{PV}/exp{e}_ep*_seed{s}_best_coll1s.json")[0])); cv = j["collector"]["vehicles"]; sm = j["collector"]["summary"]
        acc = {c: dict(tv=[], ev=[], ts=0.0, es=0.0, k=0) for c in CLS}
        for vid, v in cv.items():
            k = v.get("k_v", 0)
            if k <= 0: continue
            ts, es = sum(v["stopped_sec"].values()), sum(v["stop_events"].values())
            for c in ({v["type"], "all"} if v["type"] in ("car", "bus", "ambulance") else {"all"}):
                A = acc[c]; A["tv"].append(ts / k); A["ev"].append(es / k); A["ts"] += ts; A["es"] += es; A["k"] += k
        C = json.load(open(glob.glob(f"{PV}/exp{e}_ep*_seed{s}_rldelay.json")[0]))["classes"]
        for c in CLS:
            A = acc[c]
            D["t_veh"][c].append(np.mean(A["tv"])); D["t_vis"][c].append(A["ts"] / A["k"])
            D["n_veh"][c].append(np.mean(A["ev"])); D["n_vis"][c].append(A["es"] / A["k"])
            chk = max(chk, abs(np.mean(A["tv"]) - sm[f"eval/system/{c}/avg_stopped_time_per_visit"]), abs(A["es"] / A["k"] - sm[f"eval/system/{c}/avg_stop_events_per_all_visit"]))
            D["d_vis"][c].append(sum(C[x]["rl_timeloss"] for x in C) / sum(C[x]["rl_entered"] for x in C) if c == "all" else C[c]["rl_timeloss"] / C[c]["rl_entered"])
    R[e] = dict(w=weights(e), D={k: {c: np.array(v) for c, v in d.items()} for k, d in D.items()})
M = [("t_veh", "停车时间/visit (s), 按车平均 [结果表口径]", 2), ("t_vis", "停车时间/visit (s), 按访问平均", 2),
     ("n_veh", "停车次数/visit, 按车平均 [结果表口径]", 3), ("n_vis", "停车次数/visit, 按访问平均", 3), ("d_vis", "平均延误/visit (s), 按访问平均 [结果表口径, 只有这一种]", 2)]
L = [f"# {a.tag}: per-visit 两种平均口径对比\n", f"10 个评估种子均值 ± 标准差。核对: 按车平均停车时间与收集器汇总值、按访问平均停车次数与收集器 avg_stop_events_per_all_visit 的最大差 {chk:.1g}。\n"]
L.append("| 指标 | 实验 | 小汽车 | 公交 | 救护车 | 全部 | 优先级顺序 (均值) | 逐种子违反 |"); L.append("|---|---|---|---|---|---|---|---|")
for k, lab, dg in M:
    for e in a.exps:
        w, X = R[e]["w"], R[e]["D"][k]; spec = "".join(str(int(w[t])) for t in ("ambulance", "bus", "car"))
        pairs = [(h, l) for h, l in itertools.permutations(("car", "bus", "ambulance"), 2) if w[h] > w[l]]
        inv = [f"{NAME[h]}≥{NAME[l]}" for h, l in pairs if X[h].mean() >= X[l].mean()]
        order = "不检查" if not pairs else ("**反转**: " + "; ".join(inv) if inv else "正确")
        seedv = "-" if not pairs else ", ".join(f"{NAME[h]}<{NAME[l]} {int((X[h] >= X[l]).sum())}/10" for h, l in pairs)
        L.append(f"| {lab} | exp{e} ({spec}) | " + " | ".join(f"{X[c].mean():.{dg}f} ± {X[c].std(ddof=1):.{dg}f}" for c in CLS) + f" | {order} | {seedv} |")
txt = "\n".join(L) + "\n"; od = os.path.join(OUTROOT, a.tag); os.makedirs(od, exist_ok=True)
open(os.path.join(od, "pervisit_weighting_compare.md"), "w").write(txt); print(txt)
