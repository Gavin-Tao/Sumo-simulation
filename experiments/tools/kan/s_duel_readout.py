"""S-channel priority readout: does having a level-l vehicle on YOUR side help
you win crossing duels? Ridge fit s(m,n) ≈ b + Σ w^m·φ(m) + Σ w^n·φ(n);
report per-level winner-side advantage (|cnt|+|que|+|awt| coefs), mapped to
class per arm. Analog of the g α reading, for the duel function.
1x1 16-dim slot: is_green + [cnt,que,awt]×5. Read-only."""
import json, os, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA = "experiments/analysis/kan_data"
# per-arm bucket -> class map (from obs priority table)
BMAP = {"220":{4:"bus",5:"amb"}, "228a":{4:"bus",5:"amb"}, "229a":{2:"bus",4:"amb"},
        "228d":{3:"amb",5:"bus"}, "229b":{3:"amb",5:"bus"}}
# feature index within 16-dim slot: 0=is_green, then [cnt,que,awt] per level
def lvl_idx(l): return [1+3*(l-1), 1+3*(l-1)+1, 1+3*(l-1)+2]

print(f"{'arm':6s} {'car(l1)':>8s} {'l2':>6s} {'l3':>6s} {'l4':>6s} {'l5':>6s}   R²    winner-side 每级对决优势 (标准化系数|Σ|)")
for tag in ["220","228a","229a","228d","229b"]:
    d = np.load(f"{DATA}/x1_{tag}/frap_targets.npz")
    Xm, Xn, y = d["s_Xm"], d["s_Xn"], d["s_y"]
    n = min(len(y), 60000)
    X = np.concatenate([Xm[:n], Xn[:n]], axis=1)   # 32-dim: [m(16) | n(16)]
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    r = Ridge(alpha=1.0).fit(Xs, y[:n])
    w = r.coef_                                     # 32
    # winner-side (m) per-level advantage = Σ|coef| over cnt,que,awt of that level
    adv = []
    for l in range(1,6):
        idx = lvl_idx(l)                            # within m-block (0..15)
        adv.append(float(np.sum(np.abs(w[idx]))))
    r2 = r.score(Xs, y[:n])
    row = f"{tag:6s} " + " ".join(f"{a:6.2f}" for a in adv) + f"  {r2:.3f}"
    # class-mapped highlight
    cls = {c: adv[l-1] for l,c in BMAP[tag].items()}
    cls["car"] = adv[0]
    order = " > ".join(f"{k}{v:.2f}" for k,v in sorted(cls.items(), key=lambda x:-x[1]))
    print(row + "   [" + order + "]")
