"""SCATS junction-hourly targets -> per-edge counts for routeSampler.

Junction target (minus that TLS's bus crossings, floor 0) is distributed over
its signal-controlled incoming edges proportional to lane counts. Incoming
edges come from the 8std meta links (from_edge); TLS without meta (skipped in
reindex) fall back to net incoming edges. Street-level rows use their
explicit edges directly. Spec: DCC_DEMAND_DESIGN_2026-07-02.txt.

Usage: python scats_to_edgecounts.py <net> <meta> <calib_dir> <hour>
  -> <calib_dir>/edgecounts_{H:02d}h.xml
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "dublin"))
import common  # noqa: E402


def main():
    net_path, meta_path, calib_dir, hour = sys.argv[1:5]
    hour = int(hour)
    net = common.load_net(os.path.abspath(net_path))
    meta = json.load(open(meta_path))["tls"]
    targets = json.load(open(os.path.join(calib_dir, "scats_targets.json")))
    bus_p = os.path.join(calib_dir, f"bus_crossings_{hour:02d}h.json")
    bus = json.load(open(bus_p)) if os.path.exists(bus_p) else {}

    counts = {}

    def add(edge_id, veh):
        counts[edge_id] = counts.get(edge_id, 0.0) + veh

    n_tls = 0
    for tid, t in targets["tls"].items():
        if not t.get("trusted"):
            continue
        tgt = float(t["hourly"][hour])
        tgt = max(0.0, tgt - float(bus.get(tid, 0)))
        if tid in meta:
            edges = sorted({c["from_edge"] for links in meta[tid]["links"].values()
                            for c in links})
        else:
            if not net.hasNode(tid):
                continue
            edges = sorted({e.getID() for e in net.getNode(tid).getIncoming()})
        edges = [e for e in edges if net.hasEdge(e)]
        if not edges:
            continue
        lanes = {e: len(net.getEdge(e).getLanes()) for e in edges}
        tot = sum(lanes.values())
        for e in edges:
            add(e, tgt * lanes[e] / tot)
        n_tls += 1

    n_street = 0
    for sid, s in (targets.get("street_sites") or {}).items():
        hourly = s.get("hourly")
        if not hourly:
            continue
        tgt = float(hourly[hour])
        edges = [e for e in s.get("edges", []) if net.hasEdge(e)]
        if not edges:
            continue
        for e in edges:
            add(e, float(tgt) / len(edges))
        n_street += 1

    out = os.path.join(calib_dir, f"edgecounts_{hour:02d}h.xml")
    with open(out, "w") as f:
        f.write('<data>\n  <interval id="scats" begin="0" end="3600">\n')
        for e, v in sorted(counts.items()):
            if v >= 1:
                f.write(f'    <edge id="{e}" count="{v:.0f}"/>\n')
        f.write("  </interval>\n</data>\n")
    print(f"edgecounts: {n_tls} TLS rows + {n_street} street rows -> "
          f"{sum(1 for v in counts.values() if v >= 1)} edges, "
          f"total count {sum(counts.values()):.0f}; wrote {out}")


if __name__ == "__main__":
    main()
