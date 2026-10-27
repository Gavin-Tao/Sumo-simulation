"""新旧口径八臂逐路口诊断 (2026-10-27 用户令 "深入 debug 每个路口的效果, 有没有异常 episode")。只读 best × 10 种子的逐秒评估文件。
每路口 × 车类: 停车时间/进入 (收集器逐车按路口的停车秒数之和 / rldelay 的进入次数), 延误/进入 (rldelay per_junction timeloss / entered); 10 种子均值 ± 标准差。
异常访问: 单次访问停车 ≥ 阈值秒 (vehicles[vid].tls[j][k].stopped), 列出 (臂, 种子, 路口, 车类, 停车秒, 进入时刻, 车道, 是否离开)。
输出: analysis/data/exact_vs_old_2026-10-27/per_junction.{json,md}。用法 (仓库根目录): python experiments/tools/dublin/exact_vs_old_junctions.py [--thr 100]"""
import json, glob, sys, os, collections, numpy as np
PV = "experiments/analysis/data/pervehicle"; OUT = "experiments/analysis/data/exact_vs_old_2026-10-27"
ARMS = [("old111", 458), ("old521", 456), ("old531", 449), ("old541", 457), ("new111", 460), ("new521", 462), ("new531", 461), ("new541", 459)]
SEEDS = list(range(123, 133)); THR = int(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 100
FAKE = {"659802", "133307283", "32236377", "135109534"}   # 2026-10-11: 现实无信号、手工加的信号路口
def ep_of(e):
    return sorted({int(f.split("_ep")[1].split("_")[0]) for f in glob.glob(f"{PV}/exp{e}_ep*_seed*_best_coll1s.json")})[-1]
def main():
    J = None; R = {}; anomalies = []; longstop = collections.Counter(); stuck = collections.Counter()
    for a, e in ARMS:
        ep = ep_of(e); R[a] = collections.defaultdict(lambda: collections.defaultdict(list))
        for s in SEEDS:
            C = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_rldelay.json"))["classes"]
            d = json.load(open(f"{PV}/exp{e}_ep{ep:05d}_seed{s}_best_coll1s.json")); cv = d["collector"]["vehicles"]; V = d["vehicles"]
            if J is None: J = sorted(C["car"]["per_junction"])
            st = {c: collections.Counter() for c in ("car", "bus", "ambulance")}
            for vid, c in cv.items():
                for j, x in c.get("stopped_sec", {}).items():
                    if c["type"] in st: st[c["type"]][j] += x
            for j in J:
                for c in ("car", "bus", "ambulance"):
                    pj = C[c]["per_junction"].get(j, {}); en = pj.get("entered", 0)
                    R[a][j][f"{c}_delay"].append(pj["timeloss"] / en if en else np.nan); R[a][j][f"{c}_stop"].append(st[c][j] / en if en else np.nan); R[a][j][f"{c}_entered"].append(en)
                tl = sum(C[c]["per_junction"].get(j, {}).get("timeloss", 0) for c in C); en = sum(C[c]["per_junction"].get(j, {}).get("entered", 0) for c in C)
                R[a][j]["all_delay"].append(tl / en if en else np.nan); R[a][j]["all_stop"].append(sum(st[c][j] for c in st) / en if en else np.nan)
            for vid, rec in V.items():
                for j, visits in rec.get("tls", {}).items():
                    for k in visits:
                        if k.get("exit") is None: stuck[(a, j)] += 1
                        if k.get("stopped", 0) >= THR:
                            longstop[(a, j)] += 1
                            anomalies.append(dict(arm=a, seed=s, junction=j, cls=rec["type"], stopped=k["stopped"], entry=k["entry"], lane=k.get("lane"), first_green=k.get("first_green"), exit=k.get("exit"), vid=vid))
    md = [f"# 逐路口诊断 (best × 10 种子, 逐秒口径) — 2026-10-27\n", "停车 = 停车秒/进入次数, 延误 = timeLoss/进入次数 (RL 路口进口道), 10 种子均值; 路口后 * = 现实无信号的手工信号路口\n"]
    for c in ("car", "bus"):
        md.append(f"\n## {c}: 停车/进入 (s)  — 旧 111/521/531/541 | 新 111/521/531/541 | 新541/旧541 | 公交优先收益 (111→541) 旧/新\n")
        md.append("| 路口 | 进入/集 | 旧111 | 旧521 | 旧531 | 旧541 | 新111 | 新521 | 新531 | 新541 | 新/旧 541 | 111→541 旧 | 111→541 新 |\n|---|" + "---|" * 12)
        for j in J:
            m = {a: float(np.nanmean(R[a][j][f"{c}_stop"])) for a, _ in ARMS}; en = float(np.mean(R["new541"][j][f"{c}_entered"]))
            r = m["new541"] / m["old541"] if m["old541"] else float("nan"); b_old = (m["old111"] - m["old541"]) / m["old111"] if m["old111"] else float("nan"); b_new = (m["new111"] - m["new541"]) / m["new111"] if m["new111"] else float("nan")
            md.append(f"| {j[:28]}{'*' if j in FAKE else ''} | {en:.0f} | " + " | ".join(f"{m[a]:.2f}" for a, _ in ARMS) + f" | {r:.2f} | {b_old * 100:+.0f}% | {b_new * 100:+.0f}% |")
    md.append(f"\n## 全部车辆: 延误/进入 (s)\n\n| 路口 | 旧111 | 旧521 | 旧531 | 旧541 | 新111 | 新521 | 新531 | 新541 | 新/旧 541 |\n|---|" + "---|" * 9)
    for j in J:
        m = {a: float(np.nanmean(R[a][j]["all_delay"])) for a, _ in ARMS}
        md.append(f"| {j[:28]}{'*' if j in FAKE else ''} | " + " | ".join(f"{m[a]:.2f}" for a, _ in ARMS) + f" | {m['new541'] / m['old541']:.2f} |")
    md.append(f"\n## 单次访问停车 ≥ {THR} s 的次数 (10 种子合计), 按臂 × 路口\n")
    js = sorted({j for (_, j) in longstop}, key=lambda j: -sum(longstop[(a, j)] for a, _ in ARMS))
    md.append("| 路口 | " + " | ".join(a for a, _ in ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for j in js:
        md.append(f"| {j[:28]}{'*' if j in FAKE else ''} | " + " | ".join(str(longstop[(a, j)]) for a, _ in ARMS) + " |")
    md.append("| 合计 | " + " | ".join(str(sum(v for (a2, _), v in longstop.items() if a2 == a)) for a, _ in ARMS) + " |")
    md.append(f"\n未离开路口的访问 (exit 缺失): {dict(stuck) if stuck else '无'}\n")
    anomalies.sort(key=lambda x: -x["stopped"]); md.append(f"\n## 最长的 25 次单访问停车\n\n| 臂 | 种子 | 路口 | 车类 | 停车 s | 进入 t | 首绿 t | 离开 t | 车道 |\n|---|" + "---|" * 8)
    for x in anomalies[:25]:
        md.append(f"| {x['arm']} | {x['seed']} | {x['junction'][:28]} | {x['cls']} | {x['stopped']:.0f} | {x['entry']:.0f} | {x['first_green']} | {x['exit']} | {x['lane']} |")
    os.makedirs(OUT, exist_ok=True); open(f"{OUT}/per_junction.md", "w").write("\n".join(md) + "\n")
    json.dump(dict(junctions=J, thr=THR, per_junction={a: {j: {k: [float(x) for x in v] for k, v in R[a][j].items()} for j in J} for a, _ in ARMS}, anomalies=anomalies, longstop={f"{a}|{j}": v for (a, j), v in longstop.items()}),
              open(f"{OUT}/per_junction.json", "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))
if __name__ == "__main__":
    main()
