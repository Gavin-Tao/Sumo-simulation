"""Demote no-competition TLS junctions to priority (yield) junctions.

Rationale (user decision, 2026-06-11): with pedestrians excluded (limitation
L1), a signal whose active vehicle flows never conflict has no remaining
reason to exist — in reality these city-centre signals serve pedestrian
crossings. Keeping them as RL agents only pollutes the shared replay buffer
with constant-action transitions and lets exploration throw red lights at
through-corridors (fake congestion noise for neighbouring agents).

Rule: a TLS is KEPT iff there exist two links i, j from different incoming
edges that are foes (junction <request>) and BOTH carry >= THRESH veh/h in
at least one of the generated scenarios (02h/11h/18h). Demand does not depend
on signals, so there is no circularity.

Outputs nets/dublin/dublin_8action_demoted.net.xml via netconvert node-type
patch (recomputes right-of-way), then VERIFIES that every connection
(from/to/fromLane/toLane) is unchanged — the lane-locked route chains of the
already-generated .rou files must stay valid.

Run AFTER the three scenarios exist; then re-run reindex_8std.py with this
net as source.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

THRESH = 10.0  # veh/h
HOURS = ("02", "11", "18")
DEMOTED_NET = os.path.join(common.OUT_DIR, "dublin_8action_demoted.net.xml")


def edge_usage():
    use = {}
    for H in HOURS:
        cnt = defaultdict(int)
        d = os.path.join(common.OUT_DIR, f"weekday_{H}h")
        for kind in ("car", "bus", "amb"):
            f = os.path.join(d, f"{kind}_weekday_{H}h.rou.xml")
            for v in ET.parse(f).getroot().iter("vehicle"):
                for e in v.find("route").get("edges").split():
                    cnt[e] += 1
        use[H] = cnt
    return use

def main():
    net = common.load_net()
    use = edge_usage()

    demote = []
    for tls in sorted(t.getID() for t in net.getTrafficLights()):
        mov = common.tls_movements(net, tls)
        links = mov["links"]
        competing = False
        for H in HOURS:
            act = [i for i, cl in links.items()
                   if any(use[H].get(c["from_edge"], 0) >= THRESH for c in cl)]
            for ii, i in enumerate(act):
                for j in act[ii + 1:]:
                    a, b = links[i][0], links[j][0]
                    if a["from_edge"] == b["from_edge"] or a["junction"] != b["junction"]:
                        continue
                    if mov["nodes"][a["junction"]].areFoes(
                            a["junction_index"], b["junction_index"]):
                        competing = True
        if not competing:
            demote.append(tls)
    print(f"demoting {len(demote)}/27 no-competition TLS:")
    for d in demote:
        print("  ", d)

    # right_before_left does not exist in Ireland (left-hand traffic uses
    # priority/yield rules) and its symmetric waits DEADLOCK the Grafton
    # mini-gyratory under lane-locked flows (user report: William St S,
    # 7 teleports from 31 vehicles) — retype all such nodes to priority.
    rbl = [n.getID() for n in net.getNodes() if n.getType() == "right_before_left"]
    print(f"retyping {len(rbl)} right_before_left nodes to priority")

    patch = os.path.join(common.OUT_DIR, "calibration", "demote_nodes.nod.xml")
    with open(patch, "w") as f:
        f.write("<nodes>\n")
        for d in demote:
            f.write(f'  <node id="{d}" type="priority"/>\n')
        for d in rbl:
            f.write(f'  <node id="{d}" type="priority"/>\n')
        f.write("</nodes>\n")

    # ---- lane-reachability repair (user-licensed net edit, 2026-06-11):
    # under strict lane-locking a lane is usable only через real connections.
    # netconvert's lane-add/drop mappings leave (a) UNREACHABLE lanes (have
    # outgoing connections but are never fed — e.g. turn pockets gained at an
    # approach widening: 4472590#0 lane 0) and (b) DEAD-END lanes (fed but no
    # outgoing). In reality drivers merge; lane-locked vehicles cannot, so
    # those lanes stay empty forever and their exclusive turns become
    # unroutable. Repair: feed unreachable lanes from the nearest upstream
    # lane (fan-out), give dead-end lanes the target of their nearest
    # neighbour lane (fan-in merge, unambiguous).
    entry_ids = {e.getID() for e in net.getEdges() if not any(True for _ in e.getIncoming())}
    add_conns = []
    for e in net.getEdges():
        fed, out = {}, {}
        for pe in e.getIncoming():
            for c in pe.getToNode().getConnections():
                if c.getTo().getID() == e.getID():
                    fed.setdefault(c.getToLane().getIndex(), []).append(c)
        for to_e, conns in e.getOutgoing().items():
            for c in conns:
                out.setdefault(c.getFromLane().getIndex(), []).append(c)
        for li in range(e.getLaneNumber()):
            if li in out and li not in fed and e.getID() not in entry_ids and fed:
                # nearest fed lane's upstream connection, redirected into li
                src = min(fed, key=lambda k: abs(k - li))
                c = fed[src][0]
                add_conns.append((c.getFrom().getID(), e.getID(),
                                  c.getFromLane().getIndex(), li, "feed-unreachable"))
            if li in fed and li not in out and out:
                # copy nearest neighbour lane's first outgoing target (merge)
                src = min(out, key=lambda k: abs(k - li))
                c = out[src][0]
                add_conns.append((e.getID(), c.getTo().getID(),
                                  li, c.getToLane().getIndex(), "drain-deadend"))
    con_patch = os.path.join(common.OUT_DIR, "calibration", "lane_reach.con.xml")
    with open(con_patch, "w") as f:
        f.write("<connections>\n")
        for fe, te, fl, tl, why in add_conns:
            f.write(f'  <connection from="{fe}" to="{te}" fromLane="{fl}" toLane="{tl}"/>'
                    f'  <!-- {why} -->\n')
        f.write("</connections>\n")
    print(f"lane-reach repair: {len(add_conns)} connections added")
    for fe, te, fl, tl, why in add_conns:
        print(f"  {why:>17s}: {fe}[{fl}] -> {te}[{tl}]")

    res = subprocess.run(
        ["netconvert", "-s", common.SRC_NET, "-n", patch, "-x", con_patch,
         "--offset.disable-normalization", "-o", DEMOTED_NET],
        capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        sys.exit("netconvert failed")

    # ---- invariance check: connections (and therefore lane chains) unchanged
    def conn_set(path):
        out = set()
        for c in ET.parse(path).getroot().findall("connection"):
            if c.get("from", "").startswith(":"):
                continue
            out.add((c.get("from"), c.get("to"), c.get("fromLane"), c.get("toLane")))
        return out

    before, after = conn_set(common.SRC_NET), conn_set(DEMOTED_NET)
    expected_add = {(fe, te, str(fl), str(tl)) for fe, te, fl, tl, _ in add_conns}
    missing = before - after
    extra = (after - before) - expected_add
    if missing or extra:
        sys.exit(f"FATAL: unexpected connection changes: -{len(missing)} +{len(extra)}\n"
                 f"missing: {sorted(missing)[:10]}\nextra: {sorted(extra)[:10]}")
    if expected_add - (after - before):
        sys.exit(f"FATAL: repair connections not applied: "
                 f"{sorted(expected_add - (after - before))}")
    n_tls = len(ET.parse(DEMOTED_NET).getroot().findall("tlLogic"))
    print(f"connections invariant OK; remaining TLS: {n_tls}")
    with open(os.path.join(common.CALIB_DIR, "demoted_tls.json"), "w") as f:
        json.dump({"threshold_veh_h": THRESH, "hours": HOURS, "demoted": demote}, f, indent=1)
    print("wrote", DEMOTED_NET)


if __name__ == "__main__":
    main()
