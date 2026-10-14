# 仓库副本 (2026-10-14): 中间文件在 analysis/data/synthesis_1x1_2026-10-14/ (会话中原在临时目录); 逐集行为记录合并为 behav_out.jsonl。
"""1x1 机制分析 (只读, 2026-10-14): 行为诊断 (behav_out) 按臂汇总 (10 种子均值), 与回报/分车类停车时间做跨臂秩相关; 按结构分组; 基线与零样本干预。"""
import json, glob, os, numpy as np
from scipy.stats import spearmanr
S = "/home/xiaowen/sumo-rl/experiments/analysis/data/synthesis_1x1_2026-10-14"
M = json.load(open(f"{S}/master_1x1.json"))
R = {}
for line in open(f"{S}/behav_out.jsonl"):
    try: d = json.loads(line)
    except Exception: continue
    tag = d["pol"] if d["pol"] != "rl" else ("best" if "best" in os.path.basename(d["ckpt"]) else "final")
    R.setdefault((str(d["exp"]), tag, d["interv"], d["G"]), []).append(d)
K = ["ret", "ret_car", "ret_bus", "ret_amb", "stop_all", "stop_car", "stop_bus", "stop_amb", "switches_per_1000s", "phase_mean", "phase_share_10", "phase_share_ge20",
     "eff_phases", "partial", "extra_share", "gap_mean", "gap_cv", "gap_max", "round_mean", "switch_when_free", "hold_pos", "hold_med", "trunc_mean", "empty_share"]
def agg(rows):
    return {k: float(np.nanmean([r[k] if r[k] is not None else np.nan for r in rows])) for k in K}
A = {key: agg(v) for key, v in R.items() if len(v) == 10}
json.dump({"|".join(map(str, k)): v for k, v in A.items()}, open(f"{S}/mech_1x1.json", "w"), ensure_ascii=False)
rl = {k[0]: v for k, v in A.items() if k[1] == "best" and k[2] == "none" and M[k[0]]["f"]["F1"]}
print(f"开 F1 的臂 (best 检查点, 10 种子) 数: {len(rl)}")
X = ["switches_per_1000s", "phase_mean", "phase_share_10", "phase_share_ge20", "eff_phases", "partial", "extra_share", "gap_cv", "gap_max", "round_mean", "switch_when_free", "hold_pos", "trunc_mean", "empty_share"]
print("\n跨臂秩相关 (Spearman ρ, 括号内 p): 行为指标 与 回报 (越高越好) / 小汽车停车 / 公交停车")
for x in X:
    xs = np.array([rl[e][x] for e in rl]); line = f"  {x:20s}"
    for y in ("ret", "stop_car", "stop_bus"):
        ys = np.array([rl[e][y] for e in rl]); m = ~(np.isnan(xs) | np.isnan(ys))
        rho, p = spearmanr(xs[m], ys[m]); line += f" | {y} {rho:+.2f} ({p:.0e})"
    print(line)
print("\n按结构分组 (开 F1, best, 各臂 10 种子均值再取组中位):")
groups = {}
for e, v in rl.items(): groups.setdefault((M[e]["f"]["arch"] + ("" if M[e]["f"]["layers"] in (None, 1) or M[e]["f"]["arch"] != "枚举条件化" else "2层"), "8" if "8标准" in M[e]["f"]["act"] else "12"), []).append(v)
cols = ["ret", "stop_car", "stop_bus", "switches_per_1000s", "phase_share_10", "phase_mean", "gap_cv", "eff_phases", "partial", "switch_when_free", "hold_pos", "trunc_mean"]
print("  " + f"{'结构':12s}{'相位':4s}{'臂数':4s}" + "".join(f"{c[:10]:>11s}" for c in cols))
for (g, a), vs in sorted(groups.items(), key=lambda kv: -np.median([v["ret"] for v in kv[1]])):
    print("  " + f"{g:12s}{a:4s}{len(vs):4d}" + "".join(f"{np.median([v[c] for v in vs]):11.3f}" for c in cols))
print("\n基线 (378 的 8STD 环境, 10 种子):")
for k, v in sorted(A.items()):
    if k[1] not in ("best", "final"):
        print(f"  {k[1]:10s} G={k[3]}: 回报 {v['ret']:7.1f} (小汽车 {v['ret_car']:6.1f} 公交 {v['ret_bus']:6.1f} 救护车 {v['ret_amb']:6.1f}) | 停车 全部 {v['stop_all']:.2f} 小汽车 {v['stop_car']:.2f} 公交 {v['stop_bus']:.2f} 救护车 {v['stop_amb']:.2f} | 相位均 {v['phase_mean']:.1f} s 10s 占 {v['phase_share_10']:.2f} 间隔 CV {v['gap_cv']:.2f}")
print("\n零样本干预 (best 检查点; 对照 = 无干预):")
for k, v in sorted(A.items()):
    if k[2] != "none" and k[1] == "best":
        b = A.get((k[0], "best", "none", None))
        if b: print(f"  exp{k[0]} {k[2]:10s}: 回报 {v['ret']:7.1f} 对 {b['ret']:7.1f} | 小汽车 {v['stop_car']:.2f} 对 {b['stop_car']:.2f} | 公交 {v['stop_bus']:.2f} 对 {b['stop_bus']:.2f} | 相位均 {v['phase_mean']:.1f} 对 {b['phase_mean']:.1f} s")
