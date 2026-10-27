"""wandb 训练曲线八臂汇总表 (2026-10-27; 配合 wandb_curve_audit.py): 每条 run 一行 —
Q 均值 (100 集后中位), q_abs_max (中位 / 最大 / 高台数), loss 与 grad_norm 中位, 训练中评估 (种子 123) 的异常集数 (100 集后, 全部车辆停车时间 > 2× 中位), 最差一次, 300-600 与 600-800 集中位, 最佳单次。
用法: python experiments/tools/dublin/wandb_curve_summary.py <csv> [...] [--md out.md]"""
import sys, re, numpy as np, pandas as pd
def plateaus(ep, qa):
    med = np.nanmedian(qa[ep > 100]); hi = qa > 3 * med; out = []; i = 0
    while i < len(hi):
        if hi[i]:
            j = i
            while j + 1 < len(hi) and hi[j + 1]: j += 1
            if j - i + 1 >= 3 and ep[i] > 100: out.append((int(ep[i]), int(ep[j])))
            i = j + 1
        else: i += 1
    return med, out
def row(path):
    df = pd.read_csv(path); tr = df[df["train/episode"].notna()].sort_values("train/episode").reset_index(drop=True); ep = tr["train/episode"].values.astype(int); late = ep > 100
    qa = tr["train/q_abs_max"].values; med, P = plateaus(ep, qa)
    K = "eval_system/all/avg_stopped_time_per_visit"; ev = tr[tr[K].notna() & (tr["train/episode"] > 100)]; emed = ev[K].median(); bad = ev[ev[K] > 2 * emed]
    jkeys = [c for c in df.columns if re.match(r"eval_(?!system)[^/]+/all/avg_stopped_time$", c)]
    badj = []
    for _, r in bad.iterrows():
        jv = max(((float(r[c]), c.split("/")[0][5:]) for c in jkeys if pd.notna(r[c])), default=(0, ""))
        badj.append(f"ep{int(r['train/episode'])} {r[K]:.1f}s@{jv[1][:16]}")
    name = path.split("/")[-1].split("_")[0]
    return dict(run=name, q_mean=float(np.nanmedian(tr["train/q_mean"].values[late])), qmax_med=float(med), qmax_max=float(np.nanmax(qa[late])), n_plateau=len(P), plateau_len=int(np.mean([b - a + 1 for a, b in P])) if P else 0,
                loss=float(np.nanmedian(tr["train/loss"].values[late])), gnorm=float(np.nanmedian(tr["train/grad_norm"].values[late])), n_eval=len(ev), n_bad=len(bad), bad=badj, worst=float(ev[K].max()),
                mid=float(ev[(ev["train/episode"] > 300) & (ev["train/episode"] <= 600)][K].median()), late_med=float(ev[ev["train/episode"] > 600][K].median()), best=float(ev[K].min()), best_ep=int(ev.loc[ev[K].idxmin(), "train/episode"]))
def main():
    args = [a for a in sys.argv[1:] if a != "--md" and not a.endswith(".md")]; out = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    rows = [row(p) for p in args]
    hdr = "| run | Q 均值 | q_abs_max 中位 / 最大 / 高台数 (均长) | loss 中位 | grad_norm 中位 | 评估次数 (>100 集) | 异常评估 (>2× 中位) | 最差评估 | 300-600 中位 | 600-800 中位 | 最佳 (集) |\n|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [hdr] + [f"| {r['run']} | {r['q_mean']:.3f} | {r['qmax_med']:.2f} / {r['qmax_max']:.1f} / {r['n_plateau']} ({r['plateau_len']}) | {r['loss']:.2e} | {r['gnorm']:.4f} | {r['n_eval']} | {r['n_bad']}: {'; '.join(r['bad']) if r['bad'] else '—'} | {r['worst']:.1f} | {r['mid']:.2f} | {r['late_med']:.2f} | {r['best']:.2f} ({r['best_ep']}) |" for r in rows]
    print("\n".join(lines))
    if out: open(out, "w").write("\n".join(lines) + "\n")
if __name__ == "__main__":
    main()
