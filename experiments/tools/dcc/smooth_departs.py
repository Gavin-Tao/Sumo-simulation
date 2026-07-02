"""Depart smoothing + origin dispersal for sampled car routes (spec addendum,
user request 2026-07-02: spawn floods saturate roads regardless of signal
control — cap per-origin insertion intensity at the source).

Policy per origin edge: max CAP veh/min/lane (burst) and CAP*60 veh/h/lane
(sustained). Overflow handling, in order:
  1. shift depart into the least-loaded minute of the same edge (time smoothing)
  2. if the edge is saturated across the hour: advance the vehicle's origin
     FORWARD along its own route (route stays valid by construction) to the
     first edge with spare budget, up to HOPS hops
  3. still nowhere: keep as-is (counted + reported — no silent drops)
Cars also get departPos="random_free" (better multi-lane packing).
Bus/amb files untouched. Writes in place; prints a before/after report.

Usage: python smooth_departs.py <net.xml> <car_rou.xml> [cap_per_min_lane=6]
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import sumolib

HOPS = 3


def main():
    net_path, rou_path = sys.argv[1], sys.argv[2]
    cap = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    net = sumolib.net.readNet(net_path)
    tree = ET.parse(rou_path)
    root = tree.getroot()
    vehs = [el for el in root if el.tag == "vehicle"]

    def lanes(eid):
        return len(net.getEdge(eid).getLanes()) if net.hasEdge(eid) else 1

    # budget[edge][minute] = remaining insertions
    used = defaultdict(lambda: defaultdict(int))
    for v in vehs:
        o = v.find("route").get("edges").split()[0]
        used[o][int(float(v.get("depart")) // 60)] += 1

    def over(edge, minute):
        return used[edge][minute] > cap * lanes(edge)

    n_shift = n_moved = n_stuck = 0
    hot_before = sum(1 for e in used for m in used[e] if used[e][m] > cap * lanes(e))
    for v in sorted(vehs, key=lambda x: float(x.get("depart"))):
        route = v.find("route")
        edges = route.get("edges").split()
        o, dep_min = edges[0], int(float(v.get("depart")) // 60)
        if not over(o, dep_min):
            continue
        # 1. time smoothing: least-loaded minute on same edge with headroom
        cand = min(range(60), key=lambda m: used[o][m])
        if used[o][cand] < cap * lanes(o):
            used[o][dep_min] -= 1
            used[o][cand] += 1
            import random
            v.set("depart", f"{cand * 60 + random.Random(v.get('id')).uniform(0, 60):.2f}")
            n_shift += 1
            continue
        # 2. origin dispersal: advance along own route to an edge with budget
        moved = False
        for hop in range(1, min(HOPS + 1, len(edges) - 1)):
            ne = edges[hop]
            if net.hasEdge(ne) and not over(ne, dep_min) \
                    and used[ne][dep_min] < cap * lanes(ne):
                used[o][dep_min] -= 1
                used[ne][dep_min] += 1
                route.set("edges", " ".join(edges[hop:]))
                n_moved = n_moved + 1
                moved = True
                break
        if not moved:
            n_stuck += 1
    for v in vehs:
        v.set("departPos", "random_free")
    hot_after = sum(1 for e in used for m in used[e] if used[e][m] > cap * lanes(e))
    # CRITICAL: SUMO's incremental route loading requires vehicles sorted by
    # depart — an out-of-order vehicle ABORTS loading (11h dropped 24k->607
    # before this fix). Rebuild the element order after time-shifting.
    for v in vehs:
        root.remove(v)
    for v in sorted(vehs, key=lambda x: float(x.get("depart"))):
        root.append(v)
    tree.write(rou_path, encoding="UTF-8", xml_declaration=True)
    print(f"smooth_departs[{rou_path.split('/')[-1]}]: cap={cap}/min/lane  "
          f"hot-minutes {hot_before}->{hot_after}  time-shifted={n_shift}  "
          f"origin-moved={n_moved}  stuck={n_stuck}")


if __name__ == "__main__":
    main()
