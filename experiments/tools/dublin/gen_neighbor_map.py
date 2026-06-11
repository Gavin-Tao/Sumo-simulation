"""Dublin neighbor_map generator for CoLight/Coeff coordination trainers.

The grid trainers expect per-TLS [up, down, left, right] neighbor slots.
Dublin's 18 RL junctions are irregular, so: for each TLS take the <=4
nearest OTHER RL TLS (euclidean, capped at MAX_DIST) and assign each to the
compass slot matching its bearing (up=N, down=S, left=W, right=E); on slot
collision keep the nearer junction. Missing slots stay null — the GAT
masks absent neighbors (same mechanism as grid edges).

Output: prints YAML to embed in configs + writes calibration/neighbor_map.yaml
"""
import math, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

MAX_DIST = 350.0  # m — beyond this, junctions are not meaningful neighbors

def main():
    net = common.load_net(os.path.join(common.OUT_DIR, "dublin_8std.net.xml"))
    tls = sorted(t.getID() for t in net.getTrafficLights())
    pos = {t: net.getNode(t).getCoord() for t in tls}
    nbmap = {}
    for t in tls:
        x, y = pos[t]
        cand = sorted(((math.hypot(pos[o][0]-x, pos[o][1]-y), o) for o in tls if o != t))
        slots = [None, None, None, None]  # up(N), down(S), left(W), right(E)
        dist  = [1e18] * 4
        for d, o in cand:
            if d > MAX_DIST:
                break
            b = math.degrees(math.atan2(pos[o][0]-x, pos[o][1]-y)) % 360  # 0=N,90=E
            slot = {0: 0, 1: 3, 2: 1, 3: 2}[int(((b + 45) % 360) // 90)]  # N,E,S,W -> up,right,down,left
            if d < dist[slot]:
                slots[slot], dist[slot] = o, d
        nbmap[t] = slots
    out = os.path.join(common.CALIB_DIR, "neighbor_map.yaml")
    import yaml
    with open(out, "w") as f:
        yaml.safe_dump({"neighbor_map": nbmap}, f, sort_keys=True)
    n_links = sum(1 for v in nbmap.values() for s in v if s)
    print(f"18 TLS, {n_links} directed neighbor links; wrote {out}")
    for t, v in sorted(nbmap.items()):
        print(f"  {t[:36]:38s} U={str(v[0])[:14]:14s} D={str(v[1])[:14]:14s} L={str(v[2])[:14]:14s} R={str(v[3])[:14]:14s}")

if __name__ == "__main__":
    main()
