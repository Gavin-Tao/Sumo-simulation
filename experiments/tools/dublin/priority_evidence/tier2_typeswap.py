"""第二层: 换车型探针对照 (同一辆车、同一出发、同一路线, 分别以 ambulance / bus / car 跑): 每臂 × 种子的 36 辆探针的
零停占比、停车/visit、首绿、遇红占比、行程 timeLoss; 配对差 (amb−car, bus−car) 按 (种子, 探针) 配对。用法: python tier2_typeswap.py"""
import json, glob, collections, numpy as np, re
NAME = {"457": "旧541", "460": "新111", "462": "新521", "461": "新531", "459": "新541"}
D = "analysis/data/pervehicle_probe18h"
rec = collections.defaultdict(dict)    # (exp, seed, var) -> probe -> metrics
for f in glob.glob(f"{D}/exp*_seed*_probe*_coll1s.json"):
    m = re.search(r"exp(\d+)_ep\d+_seed(\d+)_probe(amb|bus|car)_coll1s", f); e, s, var = m.group(1), int(m.group(2)), m.group(3)
    d = json.load(open(f))
    for vid, v in d["vehicles"].items():
        if not vid.startswith("probe_"): continue
        visits = [r for t, rs in v["tls"].items() for r in rs]
        if not visits: continue
        st = np.array([r["stopped"] for r in visits], float); red = np.array([r["entry_state"] in ("r", "R", "y", "Y") for r in visits]); fg = np.array([(r["first_green"] - r["entry"]) if r["first_green"] is not None else np.nan for r in visits])
        rec[(e, s, var)][vid] = dict(visits=len(visits), zero=float((st == 0).mean()), stop=float(st.mean()), red=float(red.mean()), fg=float(np.nanmean(fg)) if np.isfinite(fg).any() else np.nan,
                                     tl=(v["trip"]["timeLoss"] if v.get("trip") else np.nan), dur=(v["trip"]["duration"] if v.get("trip") else np.nan), arrived=bool(v.get("trip")))
    # 其余车辆 (非探针) 的小汽车停车/visit, 看探针车型对别人的代价
    S = N = 0
    for vid, v in d["vehicles"].items():
        if vid.startswith("probe_") or v["type"] != "car": continue
        for t, rs in v["tls"].items():
            for r in rs: S += r["stopped"]; N += 1
    rec[(e, s, var)]["__others_car_stop"] = S / max(N, 1)
md = ["# 第二层: 换车型探针 (36 辆/集, 同车同时同路线, 只换车型; best 检查点 × 10 种子)\n", "| 臂 | 车型 | 集数 | 探针访问/集 | 零停 | 停车/visit (s) | 遇红 | 首绿 (s) | 行程 timeLoss (s) | 到达率 | 其他小汽车 停车/visit |", "|---|---|---|---|---|---|---|---|---|---|---|"]
out = {}
for e in ["460", "462", "461", "459", "457"]:
    for var in ("car", "bus", "amb"):
        eps = [(s, r) for (ee, s, vv), r in rec.items() if ee == e and vv == var]
        if not eps: continue
        P = [p for s, r in eps for k, p in r.items() if k != "__others_car_stop"]
        vis = sum(p["visits"] for p in P); zero = sum(p["zero"] * p["visits"] for p in P) / vis; stop = sum(p["stop"] * p["visits"] for p in P) / vis; red = sum(p["red"] * p["visits"] for p in P) / vis
        fg = np.nanmean([p["fg"] for p in P]); tl = np.nanmean([p["tl"] for p in P]); arr = np.mean([p["arrived"] for p in P]); oth = np.mean([r["__others_car_stop"] for s, r in eps])
        out[f"{NAME[e]}/{var}"] = dict(n_eps=len(eps), visits_per_ep=vis / len(eps), zero=zero, stop=stop, red=red, fg=fg, timeloss=tl, arrived=arr, others_car_stop=oth)
        md.append(f"| {NAME[e]} | {var} | {len(eps)} | {vis/len(eps):.0f} | {zero:.0%} | {stop:.2f} | {red:.0%} | {fg:.2f} | {tl:.1f} | {arr:.0%} | {oth:.3f} |")
md.append("\n## 配对差 (同种子同探针): 车型 − car 孪生\n\n| 臂 | 对比 | 配对数 | Δ停车/visit (s) 均值 ± sd | 停车更少的占比 | Δ行程 timeLoss (s) | Δ首绿 (s) |\n|---|---|---|---|---|---|---|")
for e in ["460", "462", "461", "459", "457"]:
    for var in ("amb", "bus"):
        ds, dt, dfg = [], [], []
        for (ee, s, vv), r in rec.items():
            if ee != e or vv != var: continue
            rc = rec.get((e, s, "car"), {})
            for k, p in r.items():
                if k.startswith("probe_") and k in rc:
                    ds.append(p["stop"] - rc[k]["stop"]); dt.append(p["tl"] - rc[k]["tl"]); dfg.append(p["fg"] - rc[k]["fg"])
        if ds:
            ds = np.array(ds); md.append(f"| {NAME[e]} | {var} − car | {len(ds)} | {ds.mean():+.3f} ± {ds.std(ddof=1):.3f} | {(ds < 0).mean():.0%} (相等 {(ds == 0).mean():.0%}) | {np.nanmean(dt):+.2f} | {np.nanmean(dfg):+.2f} |")
            out[f"{NAME[e]}/{var}-car"] = dict(n=int(len(ds)), d_stop_mean=float(ds.mean()), d_stop_sd=float(ds.std(ddof=1)), share_less=float((ds < 0).mean()), share_equal=float((ds == 0).mean()), d_timeloss=float(np.nanmean(dt)), d_fg=float(np.nanmean(dfg)))
open("analysis/data/priority_evidence_2026-10-27/tier2_typeswap.md", "w").write("\n".join(md) + "\n"); json.dump(out, open("analysis/data/priority_evidence_2026-10-27/tier2_typeswap.json", "w"), indent=1, ensure_ascii=False)
print("\n".join(md))
