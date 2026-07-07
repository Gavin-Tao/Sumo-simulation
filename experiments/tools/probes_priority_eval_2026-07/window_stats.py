"""Last-N-episode trimmed window stats + priority-inversion analysis
from wandb eval series (exp207/211/215)."""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

N = 100
CLASSES = ["car", "bus", "ambulance"]
METRICS = ["avg_stop_events_per_visit", "avg_stopped_time_per_visit"]

def col(c, m): return f"eval_system/{c}/{m}"

for name in ["exp207", "exp211", "exp215"]:
    df = pd.read_pickle(f"{name}_pervisit.pkl").sort_values("train/episode")
    w = df.tail(N).copy()
    ep_lo, ep_hi = int(w["train/episode"].min()), int(w["train/episode"].max())
    # episode-level outlier fence on system-all stopped time per visit (gridlock evals)
    x = w[col("all", "avg_stopped_time_per_visit")]
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    fence = (x >= q1 - 1.5 * (q3 - q1)) & (x <= q3 + 1.5 * (q3 - q1))
    kept = w[fence]
    print(f"\n=== {name}  window ep{ep_lo}-{ep_hi}  n={len(w)}  kept={len(kept)} "
          f"(dropped {len(w)-len(kept)}: ep {sorted(w.loc[~fence,'train/episode'].astype(int).tolist())})")
    print(f"  completion(all) mean={kept[col('all','completion_rate')].mean():.3f}  "
          f"pending max={kept['eval_system/pending_veh'].max():.0f}")
    for m in METRICS:
        row = "  " + m.replace("avg_", "").replace("_per_visit", "/visit") + ": "
        for c in CLASSES:
            v = kept[col(c, m)]
            row += f"{c} {v.mean():.3f}±{v.std():.3f} (med {v.median():.3f})   "
        print(row)
    amb = kept[col("ambulance", "avg_stop_events_per_visit")]
    print(f"  amb bad-rate (stops/visit>0): {(amb > 0).mean():.2f}")
    # priority inversion: higher priority class should NOT be worse (metric higher = worse)
    for m in METRICS:
        for hi, lo in [("bus", "car"), ("ambulance", "car"), ("ambulance", "bus")]:
            a, b = kept[col(hi, m)], kept[col(lo, m)]
            frac = (a > b).mean()
            d = (a - b).dropna()
            try:
                p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
            except ValueError:
                p = float("nan")
            tag = ""
            if frac > 0.5 and p < 0.05: tag = "  <-- INVERSION (systematic)"
            elif frac > 0.5: tag = "  <-- inversion-leaning (n.s.)"
            print(f"    {m.split('avg_')[1]:28s} {hi}>{lo} in {frac:4.0%} of eps, "
                  f"Δmed={d.median():+.3f}, wilcoxon p={p:.3g}{tag}")
