"""Dublin S-channel readout (27-dim slots) for dublin211_best + 3 vintages.
Re-verifies doc §八's "s对决_amb 恒0" claim. Winner-side per-level advantage
via Ridge on shuffled 100k subsample with 80/20 holdout; reported overall and
crossing-only / merge-only. amb=l5, bus=l3 (5-3-1 table). Core-φ layout is the
first 16 dims of each 27-dim slot vector (is_green + [cnt,que,awt]×5), same
index math as 1x1; ψ/occ tail (idx 16-26) participates in the fit but is not
part of the per-level readout."""
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA = "/home/xiaowen/sumo-rl/experiments/analysis/kan_data_v2"
ARMS = ["dublin211_best", "tgt_a211ep215", "tgt_a211ep350", "tgt_a211ep800"]

def lvl_idx(l, off=0):
    base = 1 + 3 * (l - 1)
    return [off + base, off + base + 1, off + base + 2]

def read(X, y):
    if len(y) < 200:
        return None
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))[:100000]
    ntr = int(0.8 * len(idx))
    sc = StandardScaler().fit(X[idx[:ntr]])
    Xs = sc.transform(X[idx[:ntr]])
    r = Ridge(alpha=1.0).fit(Xs, y[idx[:ntr]])
    w = r.coef_
    adv = [float(np.sum(np.abs(w[lvl_idx(l)]))) for l in range(1, 6)]  # winner side m
    r2 = float(r.score(sc.transform(X[idx[ntr:]]), y[idx[ntr:]]))
    return {"adv": adv, "r2_holdout": r2, "n": int(len(y))}

out = {}
for arm in ARMS:
    try:
        d = np.load(f"{DATA}/{arm}/frap_targets.npz")
    except FileNotFoundError:
        print(f"{arm}: MISSING, skip")
        continue
    X = np.concatenate([d["s_Xm"], d["s_Xn"]], axis=1)   # 54-dim
    y, rel = d["s_y"], d["s_rel"]
    res = {"all": read(X, y)}
    for name, code in (("crossing", 3), ("merge", 2)):
        m = rel == code
        res[name] = read(X[m], y[m]) if m.sum() >= 200 else None
    out[arm] = res
    a = res["all"]
    if a:
        adv = " ".join(f"{v:5.2f}" for v in a["adv"])
        print(f"{arm:16s} all [{adv}] amb(l5)={a['adv'][4]:.3f} bus(l3)={a['adv'][2]:.3f} "
              f"car(l1)={a['adv'][0]:.3f} R2out={a['r2_holdout']:.3f} n={a['n']}")
        for k in ("crossing", "merge"):
            if res[k]:
                print(f"                 {k:8s} amb={res[k]['adv'][4]:.3f} "
                      f"bus={res[k]['adv'][2]:.3f} R2={res[k]['r2_holdout']:.3f} n={res[k]['n']}")

json.dump(out, open(f"{DATA}/dublin_s_readout.json", "w"), indent=1)
print("saved", f"{DATA}/dublin_s_readout.json")
