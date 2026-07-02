"""Prune edges that cannot reach any exit edge via non-U-turn connections —
the calibrate_nnls FATAL guard's condition. Such pockets are OSM-crop
artifacts (e.g. dcc's Royal Canal Bank loop: 8 edges whose outgoing
connections all point back into the loop). Iterates until stable, removes
via netconvert, verifies netOffset survives. Writes IN PLACE (generated
artifact; uploaded source net untouched). New file: dublin scripts unchanged.

Usage: python experiments/tools/dcc/prune_unreachable.py nets/dcc/dcc_8std.net.xml
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "dublin"))
import common  # noqa: E402


def unreachable_to_exit(net_path):
    net = common.load_net(net_path)
    edges = [e for e in net.getEdges() if e.getFunction() != "internal"]
    _, exits = common.boundary_edges(net)
    exit_ids = {e.getID() for e in exits}

    def rev(eid):
        return eid[1:] if eid.startswith("-") else "-" + eid

    # ratio-graph semantics (build_ratios): U-turn excluded UNLESS FORCED —
    # at a dead end the U-turn is the only outgoing and stays usable.
    # (First version banned ALL U-turns and over-pruned 553 cul-de-sac edges.)
    def usable_out(e):
        outs = [o.getID() for o in e.getOutgoing()]
        non_ut = [o for o in outs if o != rev(e.getID())]
        return non_ut if non_ut else outs

    reach = set(exit_ids)
    changed = True
    while changed:
        changed = False
        for e in edges:
            eid = e.getID()
            if eid in reach:
                continue
            if any(oid in reach for oid in usable_out(e)):
                reach.add(eid)
                changed = True
    return [e.getID() for e in edges if e.getID() not in reach]


def main():
    net_path = os.path.abspath(sys.argv[1])
    removed_total = []
    for it in range(10):
        dead = unreachable_to_exit(net_path)
        if not dead:
            break
        print(f"iter {it}: pruning {len(dead)} exit-unreachable edges: {dead[:10]}"
              f"{'...' if len(dead) > 10 else ''}")
        off_a = ET.parse(net_path).getroot().find("location").get("netOffset")
        tmp = net_path + ".pruned.net.xml"
        res = subprocess.run(
            ["netconvert", "-s", net_path, "-o", tmp,
             "--remove-edges.explicit", ",".join(dead),
             "--offset.disable-normalization"],
            capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr[-1500:])
            sys.exit("netconvert failed")
        off_b = ET.parse(tmp).getroot().find("location").get("netOffset")
        assert off_a == off_b, f"netOffset drift: {off_a} -> {off_b}"
        os.replace(tmp, net_path)
        removed_total += dead
    else:
        sys.exit("FATAL: prune did not converge in 10 iterations")
    n_edges = sum(1 for e in common.load_net(net_path).getEdges()
                  if e.getFunction() != "internal")
    print(f"pruned {len(removed_total)} edges total; net now {n_edges} edges; "
          f"exit-unreachable: 0")


if __name__ == "__main__":
    main()
