"""Track B part 1: re-pull finetune arms from wandb, recompute tail-10 J.
J(contract) = sum_c w_c * eval avg_stopped_time_per_visit[c], tail = last 10
eval rows. Compare vs FINETUNE_VERDICT_CARD table 二."""
import json
import numpy as np
import pandas as pd
import wandb

api = wandb.Api()
ENT = "taoxw19-"
PROJ = "sumo-rl-1x1"
W421 = {"ambulance": 4, "bus": 2, "car": 1}
W351 = {"ambulance": 3, "bus": 5, "car": 1}
# arm name substring -> (contract weights, doc J claim)
ARMS = {
    "exp227a": (W421, 67.5), "exp228a": (W421, 65.0), "exp228b": (W421, 68.3),
    "exp228c": (W421, 65.9), "exp228e": (W421, 73.0),
    "exp228d": (W351, 75.5), "exp228f": (W351, None),
    "exp229a": (W421, 67.3), "exp229b": (W351, 94.9),
    "exp230a": (W421, 63.3), "exp230b": (W421, 58.7),
}
runs = list(api.runs(f"{ENT}/{PROJ}"))
print(f"project runs: {len(runs)}")
out = {}
dumped_keys = False
for sub, (w, claim) in ARMS.items():
    cand = [r for r in runs if sub in r.name]
    if not cand:
        print(f"{sub}: NO RUN FOUND")
        continue
    run = sorted(cand, key=lambda r: r.created_at)[-1]
    rows = list(run.scan_history(page_size=5000))
    df = pd.DataFrame(rows)
    if not dumped_keys:
        ks = [k for k in df.columns if "per_visit" in k or "stopped" in k]
        print("sample keys:", ks[:12])
        dumped_keys = True
    keys = {c: f"eval_system/{c}/avg_stopped_time_per_visit" for c in w}
    missing = [k for k in keys.values() if k not in df.columns]
    if missing:
        print(f"{sub}: missing {missing}")
        continue
    sel = df[list(keys.values())].apply(pd.to_numeric, errors="coerce").dropna()
    Jrow = sum(wc * sel[keys[c]] for c, wc in w.items())
    tail = Jrow.tail(10)
    out[sub] = {"run": run.name, "id": run.id, "n_evals": int(len(Jrow)),
                "J_tail10_mean": round(float(tail.mean()), 2),
                "J_tail10_std": round(float(tail.std()), 2),
                "J_best": round(float(Jrow.min()), 2), "doc_claim": claim}
    dv = "?" if claim is None else f"{claim} (Δ{tail.mean()-claim:+.1f})"
    print(f"{sub}: J_tail10 {tail.mean():.1f}±{tail.std():.1f}  best {Jrow.min():.1f}"
          f"  n={len(Jrow)}  doc={dv}  [{run.name}]")

json.dump(out, open("/home/xiaowen/sumo-rl/experiments/analysis/kan_data_v2/j_reverify.json", "w"), indent=1)
print("saved j_reverify.json")
