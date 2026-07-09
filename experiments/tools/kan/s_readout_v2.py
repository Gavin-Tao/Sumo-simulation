"""S-channel readout on kan_data_v2 (re-verification of doc table 五).
Runs BOTH methods per arm so data-change vs method-change are separable:
  A "orig" : yesterday's exact recipe — first-60k rows, in-sample R²
  B "fixed": shuffled 60k subsample, 80/20 holdout R²
Output: kan_data_v2/s_readout_v2.json + printed tables."""
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA = "/home/xiaowen/sumo-rl/experiments/analysis/kan_data_v2"
BMAP = {"220": {4: "bus", 5: "amb"}, "228a": {4: "bus", 5: "amb"},
        "229a": {2: "bus", 4: "amb"}, "228d": {3: "amb", 5: "bus"},
        "229b": {3: "amb", 5: "bus"}}

def lvl_idx(l):
    return [1 + 3 * (l - 1), 1 + 3 * (l - 1) + 1, 1 + 3 * (l - 1) + 2]

def fit_read(X, y, Xte=None, yte=None):
    sc = StandardScaler().fit(X)
    r = Ridge(alpha=1.0).fit(sc.transform(X), y)
    w = r.coef_
    adv = [float(np.sum(np.abs(w[lvl_idx(l)]))) for l in range(1, 6)]
    r2 = r.score(sc.transform(Xte), yte) if Xte is not None else r.score(sc.transform(X), y)
    return adv, float(r2)

out = {}
for tag in ["220", "228a", "229a", "228d", "229b"]:
    d = np.load(f"{DATA}/x1_{tag}/frap_targets.npz")
    Xall = np.concatenate([d["s_Xm"], d["s_Xn"]], axis=1)
    yall = d["s_y"]
    # method A: orig (first-60k, in-sample)
    n = min(len(yall), 60000)
    advA, r2A = fit_read(Xall[:n], yall[:n])
    # method B: fixed (shuffled subsample, holdout)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(yall))[:60000]
    ntr = int(0.8 * len(idx))
    advB, r2B = fit_read(Xall[idx[:ntr]], yall[idx[:ntr]],
                         Xall[idx[ntr:]], yall[idx[ntr:]])
    cls = lambda adv: {**{c: adv[l - 1] for l, c in BMAP[tag].items()}, "car": adv[0]}
    out[tag] = {"n_s_total": int(len(yall)),
                "orig": {"adv": advA, "r2_insample": r2A, "cls": cls(advA)},
                "fixed": {"adv": advB, "r2_holdout": r2B, "cls": cls(advB)}}
    pa = " ".join(f"{a:5.2f}" for a in advA)
    pb = " ".join(f"{a:5.2f}" for a in advB)
    print(f"{tag:5s} orig  [{pa}] R2in={r2A:.3f}  cls={ {k: round(v,2) for k,v in cls(advA).items()} }")
    print(f"{tag:5s} fixed [{pb}] R2out={r2B:.3f}  cls={ {k: round(v,2) for k,v in cls(advB).items()} }")

json.dump(out, open(f"{DATA}/s_readout_v2.json", "w"), indent=1)
print("saved", f"{DATA}/s_readout_v2.json")
