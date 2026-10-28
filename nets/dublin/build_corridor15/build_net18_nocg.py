"""Restore College Green's v2 local topology on the newer corridor net.

The user explicitly requested the old junction, not the intermediate version
with newly added exits. Keep upstream corridor merges and other signals;
import only the old junction's connection table with merged-edge aliases.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import sumolib

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[2]
NETS = REPO / "nets/dublin"
OUT = WORK / "net18_nocg"
BASE = NETS / "dublin_8std_splitslot19_southclosed_corridor.net.xml"
BASE_META = NETS / "dublin_8std_splitslot19_southclosed_corridor_meta.json"
V2 = NETS / "dublin_8std_splitslot.net.xml"
TAG = "splitslot18_southclosed_corridor_nocg"
NET = NETS / f"dublin_8std_{TAG}.net.xml"
META = NETS / f"dublin_8std_{TAG}_meta.json"
CG = "cluster_1426051358_1426100603_3206752725_3783453672_3783453675"
REMOVED_EDGES = {"source_college_green_N", "129212230#0", "-129212230#1",
                 "sink_college_green_N", "4476041#0"}
REMOVED_NODES = {"source_college_green_N_boundary", "3206753875",
                 "boundary_college_green_N_exit", "boundary_college_green_E_exit"}
ALIASES = {"4476043#0": "304634278#0", "52879957#1": "-304634278#0"}
DISALLOW = "tram rail_urban rail rail_electric rail_fast ship cable_car subway"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    OUT.mkdir(exist_ok=True)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    suffixes = (".nod.xml", ".edg.xml", ".con.xml", ".tll.xml")
    sources = [WORK / "net19" / ("corridor19" + s) for s in suffixes]
    inputs = {str(p.relative_to(REPO)): sha(p) for p in [BASE, BASE_META, V2] + sources}
    trees = {s: ET.parse(p) for s, p in zip(suffixes, sources)}
    nr, er = trees[".nod.xml"].getroot(), trees[".edg.xml"].getroot()
    old_reference = sumolib.net.readNet(str(V2))
    reference_node = old_reference.getNode(CG)
    assert reference_node.getType() == "priority"
    node = next(n for n in nr.findall("node") if n.get("id") == CG)
    node.set("type", "priority")
    for key in ("tl", "tlType", "tlLayout"):
        node.attrib.pop(key, None)
    # Recompute the three-arm boundary from the retained corridor geometry.
    # Do not restore the old segmented/overlapping approach shapes or force
    # lane lengths; report the small endpoint shifts after reconstruction.
    node.attrib.pop("shape", None)
    incoming = {e.get("id") for e in er.findall("edge") if e.get("to") == CG}
    for n in list(nr.findall("node")):
        if n.get("id") in REMOVED_NODES:
            nr.remove(n)
    permission_elements = []
    for edge in list(er.findall("edge")):
        if edge.get("id") in REMOVED_EDGES:
            er.remove(edge)
            continue
        for element in [edge] + edge.findall("lane"):
            if element.get("allow") == "passenger":
                element.attrib.pop("allow")
                element.set("disallow", DISALLOW)
                permission_elements.append((edge.get("id"), element.get("index")))
    for suffix in (".con.xml", ".tll.xml"):
        root = trees[suffix].getroot()
        for c in list(root.findall("connection")):
            if c.get("from") in incoming or c.get("from") in REMOVED_EDGES or c.get("to") in REMOVED_EDGES:
                root.remove(c)
            elif c.get("tl") == CG:
                if suffix == ".tll.xml":
                    root.remove(c)
                else:
                    for key in ("tl", "linkIndex", "linkIndex2"):
                        c.attrib.pop(key, None)
        for logic in list(root.findall("tlLogic")):
            if logic.get("id") == CG:
                root.remove(logic)
    requested = []
    for c in reference_node.getConnections():
        attrs = {"from": ALIASES.get(c.getFrom().getID(), c.getFrom().getID()),
                 "to": ALIASES.get(c.getTo().getID(), c.getTo().getID()),
                 "fromLane": str(c.getFromLane().getIndex()), "toLane": str(c.getToLane().getIndex())}
        ET.SubElement(trees[".con.xml"].getroot(), "connection", attrs)
        requested.append((attrs["from"] + "_" + attrs["fromLane"],
                          attrs["to"] + "_" + attrs["toLane"], c.getState()))
    for suffix, tree in trees.items():
        ET.indent(tree)
        tree.write(OUT / ("corridor18_nocg" + suffix), encoding="UTF-8", xml_declaration=True)
    raw = OUT / "corridor18_nocg.net.xml"
    commands = [
        [sumolib.checkBinary("netconvert"), "-n", OUT / "corridor18_nocg.nod.xml",
         "-e", OUT / "corridor18_nocg.edg.xml", "-x", OUT / "corridor18_nocg.con.xml",
         "-i", OUT / "corridor18_nocg.tll.xml", "--lefthand", "--offset.disable-normalization",
         "--no-turnarounds", "--junctions.corner-detail", "5", "-o", raw],
        [sys.executable, "-B", REPO / "experiments/tools/dublin/reindex_8std.py",
         raw, "--split-slots", "--tag", TAG],
    ]
    logs = []
    for command in commands:
        result = subprocess.run(list(map(str, command)), cwd=REPO, text=True, capture_output=True)
        logs.append(" ".join(map(str, command)) + "\n" + result.stdout + result.stderr)
        (OUT / "build.log").write_text("\n".join(logs))
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)

    old, new = sumolib.net.readNet(str(BASE)), sumolib.net.readNet(str(NET))
    previous, meta = json.loads(BASE_META.read_text())["tls"], json.loads(META.read_text())["tls"]
    assert set(meta) == set(previous) - {CG} and "9101555" in meta
    assert all(previous[t] == meta[t] for t in meta), "Remaining TLS metadata changed"
    assert {e.getID() for e in new.getEdges()} == {e.getID() for e in old.getEdges()} - REMOVED_EDGES
    geometry_changes, widened_lanes = [], []
    for edge in new.getEdges():
        for lane in edge.getLanes():
            before = old.getLane(lane.getID())
            if lane.getLength() != before.getLength() or lane.getShape() != before.getShape():
                assert len(lane.getShape()) == len(before.getShape()), lane.getID()
                shift = max(math.dist(a, b) for a, b in zip(lane.getShape(), before.getShape()))
                assert CG in (edge.getFromNode().getID(), edge.getToNode().getID()), lane.getID()
                assert abs(lane.getLength() - before.getLength()) <= 1.0 and shift <= 1.0, lane.getID()
                geometry_changes.append(dict(lane=lane.getID(), before=before.getLength(), after=lane.getLength(),
                                             max_point_shift_m=shift, reason="v2 three-arm junction boundary restored"))
            if not before.allows("bus") and lane.allows("bus"):
                widened_lanes.append(lane.getID())
            assert lane.allows("passenger") == before.allows("passenger")
            assert lane.allows("bus") and lane.allows("emergency"), lane.getID()
    assert len(widened_lanes) == 21, widened_lanes
    def connections(net):
        return {(c.getFromLane().getID(), c.getToLane().getID()): c.getState()
                for n in net.getNodes() for c in n.getConnections()}
    before, after = connections(old), connections(new)
    expected = {k for k in before if all(old.getLane(l).getEdge().getID() not in REMOVED_EDGES for l in k)
                and old.getLane(k[0]).getEdge().getToNode().getID() != CG}
    expected.update((a, b) for a, b, _ in requested)
    assert set(after) == expected
    changes = []
    for k in after:
        tid = new.getLane(k[0]).getEdge().getToNode().getID()
        if before.get(k) != after[k]:
            changes.append(dict(from_lane=k[0], to_lane=k[1], before=before.get(k), after=after[k]))
            assert tid == CG, (tid, k)
    main_connections = [c for c in new.getNode(CG).getConnections()
                        if c.getFrom().getID() in {"304634278#0", "532097269#1.24"}]
    assert len(main_connections) == 2
    assert all(c.getState() == "M" for c in main_connections), [(c.getFrom().getID(), c.getState()) for c in main_connections]
    actual_cg = {(c.getFromLane().getID(), c.getToLane().getID(), c.getState())
                 for c in new.getNode(CG).getConnections()}
    assert actual_cg == set(requested), (actual_cg, requested)
    assert not new.getEdge("304634278#0").getOutgoing(), "User requested v2's terminal west approach"
    assert inputs == {p: sha(REPO / p) for p in inputs}
    report = dict(inputs=inputs, net=str(NET.relative_to(REPO)), meta=str(META.relative_to(REPO)),
                  tls_count=len(meta), removed_tls=CG, retained_9101555=True,
                  removed_edges=sorted(REMOVED_EDGES), removed_nodes=sorted(REMOVED_NODES),
                  remaining_18_meta_unchanged=True, retained_lane_geometry_changes=geometry_changes,
                  widened_lanes=widened_lanes, permission_elements=permission_elements,
                  college_green_connections_match_v2=True, merged_edge_aliases=ALIASES,
                  college_green_connections=[dict(from_lane=a, to_lane=b, state=s) for a, b, s in requested],
                  physical_connections_elsewhere_unchanged=True,
                  connection_state_changes=changes, college_green_east_inlet_has_priority=True,
                  terminal_edge_restored="304634278#0",
                  demand_note="The west approach terminates as in v2. Future demand must explicitly decide whether to use this study boundary; it must not be mistaken for a through road.")
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs", "permission_elements")}, indent=2))


if __name__ == "__main__":
    build()
