"""wandb 训练曲线审计 (2026-10-27 用户令 "新 config 训练不稳定, q_mean / grad_norm / q_max / loss 突变, 深入分析原因"; 只读本地 CSV)。
对每条 run: (1) train/q_mean, q_abs_max, loss, grad_norm 按 100 集分块; (2) q_abs_max 的 "高台" (连续 ≥3 集高于全程中位数×3) 起止集;
(3) 训练中评估 (种子 123, 每 5 集) 的异常集: 全部车辆停车时间/visit 高于该 run 评估中位数 ×2 的评估, 列出 car/bus/完成率/吞吐与最差的 3 个路口;
(4) 异常评估集与 q_abs_max 高台是否重合。用法: python experiments/tools/dublin/wandb_curve_audit.py <csv> [<csv> ...]"""
import sys, re, numpy as np, pandas as pd
def audit(path):
    df = pd.read_csv(path); tr = df[df["train/episode"].notna()].sort_values("train/episode").reset_index(drop=True); ep = tr["train/episode"].values.astype(int)
    name = path.split("/")[-1].replace(".csv", ""); print(f"\n{'=' * 100}\n{name}: {len(tr)} 集")
    keys = [k for k in ("train/q_mean", "train/q_abs_max", "train/loss", "train/grad_norm") if k in tr]
    print("分块均值 (最小..最大), 每 100 集:")
    for k in keys:
        x = tr[k].values; print(f"  {k:18s} " + " | ".join(f"{np.nanmean(x[(ep > b) & (ep <= b + 100)]):7.3f} ({np.nanmin(x[(ep > b) & (ep <= b + 100)]):.2f}..{np.nanmax(x[(ep > b) & (ep <= b + 100)]):.2f})" for b in range(0, int(ep.max()), 100)))
    qa = tr["train/q_abs_max"].values; med = np.nanmedian(qa[ep > 100]); hi = qa > 3 * med; runs = []; i = 0
    while i < len(hi):
        if hi[i]:
            j = i
            while j + 1 < len(hi) and hi[j + 1]: j += 1
            if j - i + 1 >= 3: runs.append((ep[i], ep[j], float(np.nanmax(qa[i:j + 1]))))
            i = j + 1
        else: i += 1
    print(f"q_abs_max 全程中位数 (>100 集) {med:.2f}; 高台 (≥3 集 > 3×中位数): " + (", ".join(f"ep{a}-{b} (峰 {m:.1f})" for a, b, m in runs) if runs else "无"))
    K = "eval_system/all/avg_stopped_time_per_visit"
    if K not in tr: print("无训练中评估列"); return
    ev = tr[tr[K].notna()].copy(); emed = ev[K].median()
    ers = [c for c in df.columns if re.match(r"eval_[^/]+/reward$", c)]; ev["R"] = ev[ers].sum(axis=1); bad = ev[ev[K] > 2 * emed]
    print(f"训练中评估 {len(ev)} 次; 全部车辆停车时间/visit 中位数 {emed:.2f}; 异常评估 (>2×中位数): {len(bad)} 次")
    jkeys = [c for c in df.columns if re.match(r"eval_(?!system)[^/]+/all/avg_stopped_time$", c)]
    for _, r in bad.iterrows():
        e = int(r["train/episode"]); jv = sorted(((float(r[c]), c.split("/")[0][5:]) for c in jkeys if pd.notna(r[c])), reverse=True)[:3]
        typ = {c.split("/")[0][5:]: float(ev[c].median()) for c in jkeys}
        extra = {k: r.get(f"eval_system/all/{k}", np.nan) for k in ("completion_rate", "throughput", "avg_speed", "p95_wait")}
        inq = any(a <= e <= b for a, b, _ in runs)
        print(f"  ep{e}: all {r[K]:.2f} car {r.get('eval_system/car/avg_stopped_time_per_visit', np.nan):.2f} bus {r.get('eval_system/bus/avg_stopped_time_per_visit', np.nan):.2f} 奖励和 {r['R']:.0f} (中位 {ev['R'].median():.0f}); "
              + ", ".join(f"{k}={v:.2f}" for k, v in extra.items() if pd.notna(v)) + f"; 最差路口 (停车 s, 中位): " + ", ".join(f"{j[:18]} {v:.1f} ({typ[j]:.1f})" for v, j in jv) + f"; 在 q_abs_max 高台内: {inq}")
    late = ev[ev["train/episode"] > 600][K]; mid = ev[(ev["train/episode"] > 300) & (ev["train/episode"] <= 600)][K]
    print(f"评估趋势: 300-600 集 中位 {mid.median():.2f}, 600-800 集 中位 {late.median():.2f}, 最佳单次 {ev[K].min():.2f} @ep{int(ev.loc[ev[K].idxmin(), 'train/episode'])}")
for p in sys.argv[1:]: audit(p)
