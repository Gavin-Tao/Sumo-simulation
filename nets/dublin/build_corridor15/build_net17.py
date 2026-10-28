"""Add only the two requested DCC junctions to the corridor variant.

No demand, route files, training code, or baseline networks are changed.
The DCC file supplies local geometry and lane connections, not TLS phases.
"""
from pathlib import Path
import copy
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
OUT = WORK / "net17"
REFERENCE = REPO / "nets/dcc/dcc.net.xml"
BASE = NETS / "dublin_8std_splitslot15_southclosed_corridor.net.xml"
BASE_META = NETS / "dublin_8std_splitslot15_southclosed_corridor_meta.json"
TAG = "splitslot17_southclosed_corridor"
NET = NETS / f"dublin_8std_{TAG}.net.xml"
META = NETS / f"dublin_8std_{TAG}_meta.json"
ADDED_TLS = {"659799", "659805"}
NORTH = "532096352#0"
EAST_IN = "-3791034#1"
EAST_OUT = "3791034#0"
BUFFERS = {NORTH: "source_659799_N", EAST_IN: "source_659799_E"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    result = subprocess.run(list(map(str, args)), cwd=REPO, text=True, capture_output=True)
    with (OUT / "build.log").open("a") as f:
        f.write(" ".join(map(str, args)) + "\n" + result.stdout + result.stderr + "\n")
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)


def shape(points):
    return " ".join(f"{x:.8f},{y:.8f}" for x, y in points)


def build():
    OUT.mkdir(exist_ok=True)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    sources = [REFERENCE, BASE, BASE_META] + [WORK / ("corridor" + s) for s in (".nod.xml", ".edg.xml", ".con.xml", ".tll.xml")]
    before = {str(p.relative_to(REPO)): sha(p) for p in sources}
    trees = {s: ET.parse(WORK / ("corridor" + s)) for s in (".nod.xml", ".edg.xml", ".con.xml", ".tll.xml")}
    nr, er = trees[".nod.xml"].getroot(), trees[".edg.xml"].getroot()
    nodes = {x.get("id"): x for x in nr.findall("node")}
    edges = {x.get("id"): x for x in er.findall("edge")}
    ref = sumolib.net.readNet(str(REFERENCE))
    old = sumolib.net.readNet(str(BASE))
    reference_root = ET.parse(REFERENCE).getroot()
    assert reference_root.find("location").get("netOffset") == nr.find("location").get("netOffset")
    assert reference_root.find("location").get("projParameter") == nr.find("location").get("projParameter")
    for tid in ADDED_TLS:
        assert math.dist(ref.getNode(tid).getCoord(), old.getNode(tid).getCoord()) < .001
        nodes[tid].set("type", "traffic_light")
        nodes[tid].set("tl", tid)

    # Remove the old artificial arm at the signal itself. New buffers are
    # upstream of the actual imported approach edges, with one-to-one lanes.
    er.remove(edges["source_4074528#0"])
    nr.remove(nodes["source_node_4074528#0"])

    for eid in (NORTH, EAST_IN, EAST_OUT):
        re = ref.getEdge(eid)
        element = ET.SubElement(er, "edge", {
            "id": eid, "from": re.getFromNode().getID(), "to": re.getToNode().getID(),
            "name": re.getName(), "priority": "11", "numLanes": str(len(re.getLanes())),
            "speed": "8.33", "allow": "passenger", "shape": shape(re.getRawShape()),
        })
        # Retain DCC's centered one-way north arm; two-way east road retains
        # left-hand lane offset on either side of its reference centerline.
        if eid == NORTH:
            element.set("spreadType", "center")
        edges[eid] = element

    buffer_records = []
    for eid, bid in BUFFERS.items():
        re = ref.getEdge(eid)
        points = re.getRawShape()
        p, q = points[:2]
        dx, dy = q[0]-p[0], q[1]-p[1]
        length = math.hypot(dx, dy)
        start = (p[0] - 60*dx/length, p[1] - 60*dy/length)
        join, boundary = re.getFromNode().getID(), bid + "_boundary"
        assert join not in nodes and boundary not in nodes
        ET.SubElement(nr, "node", id=join, x=str(p[0]), y=str(p[1]), type="priority", radius="0")
        ET.SubElement(nr, "node", id=boundary, x=str(start[0]), y=str(start[1]), type="dead_end")
        kwargs = dict(id=bid, name="Boundary buffer for " + re.getName(), priority="11",
                      numLanes=str(len(re.getLanes())), speed="8.33", allow="passenger", shape=shape([start, p]))
        kwargs.update({"from": boundary, "to": join})
        if eid == NORTH:
            kwargs["spreadType"] = "center"
        ET.SubElement(er, "edge", kwargs)
        buffer_records.append(dict(edge=bid, joins=eid, lanes=len(re.getLanes()), planned_length=60))
        # The east outbound road continues to the actual outer boundary;
        # vehicles do not terminate beside a modeled downstream junction.
        if eid == EAST_IN:
            out = edges[EAST_OUT]
            out.set("to", boundary)
            out.set("shape", shape(ref.getEdge(EAST_OUT).getRawShape() + [start]))

    for suffix in (".con.xml", ".tll.xml"):
        root = trees[suffix].getroot()
        for c in list(root.findall("connection")):
            fr = c.get("from")
            if fr == "source_4074528#0" or (fr in {e.getID() for e in old.getEdges()} and old.getEdge(fr).getToNode().getID() in ADDED_TLS):
                root.remove(c)
        for tl in list(root.findall("tlLogic")):
            if tl.get("id") in ADDED_TLS:
                root.remove(tl)
    con = trees[".con.xml"].getroot()
    reference_connections = {}
    for tid in sorted(ADDED_TLS):
        reference_connections[tid] = []
        for c in ref.getNode(tid).getConnections():
            if not c.getFromLane().allows("passenger") or not c.getToLane().allows("passenger"):
                continue
            attrs = {"from": c.getFrom().getID(), "to": c.getTo().getID(),
                     "fromLane": str(c.getFromLane().getIndex()), "toLane": str(c.getToLane().getIndex())}
            assert attrs["from"] in edges and attrs["to"] in edges, attrs
            ET.SubElement(con, "connection", attrs)
            reference_connections[tid].append(attrs)
    for eid, bid in BUFFERS.items():
        for lane in range(len(ref.getEdge(eid).getLanes())):
            ET.SubElement(con, "connection", {"from": bid, "to": eid, "fromLane": str(lane), "toLane": str(lane)})

    for suffix, tree in trees.items():
        ET.indent(tree)
        tree.write(OUT / ("corridor17" + suffix), encoding="UTF-8", xml_declaration=True)
    raw = OUT / "corridor17.net.xml"
    run([sumolib.checkBinary("netconvert"), "-n", OUT/"corridor17.nod.xml", "-e", OUT/"corridor17.edg.xml",
         "-x", OUT/"corridor17.con.xml", "-i", OUT/"corridor17.tll.xml", "--lefthand",
         "--offset.disable-normalization", "--no-turnarounds", "--junctions.corner-detail", "5", "-o", raw])
    run([sys.executable, "-B", REPO/"experiments/tools/dublin/reindex_8std.py", raw, "--split-slots", "--tag", TAG])

    meta = json.loads(META.read_text())["tls"]
    previous = json.loads(BASE_META.read_text())["tls"]
    assert set(meta) == set(previous) | ADDED_TLS
    for tid, item in previous.items():
        for key in ("mask", "canonical", "std_to_green_index", "actions", "links"):
            assert item[key] == meta[tid][key], (tid, key)
    new = sumolib.net.readNet(str(NET))
    for tid in ADDED_TLS:
        actual = [{"from": c.getFrom().getID(), "to": c.getTo().getID(),
                   "fromLane": str(c.getFromLane().getIndex()), "toLane": str(c.getToLane().getIndex())}
                  for c in new.getNode(tid).getConnections()]
        assert sorted(actual, key=str) == sorted(reference_connections[tid], key=str), tid
    changes = []
    for e in old.getEdges():
        if e.getID() == "source_4074528#0":
            continue
        for lane in e.getLanes():
            nl = new.getLane(lane.getID())
            if abs(nl.getLength()-lane.getLength()) > .005:
                changes.append(dict(lane=lane.getID(), old=lane.getLength(), new=nl.getLength(),
                                    old_rl_inlet=e.getToNode().getID() in previous))
                assert e.getFromNode().getID() in ADDED_TLS or e.getToNode().getID() in ADDED_TLS, lane.getID()
    assert before == {str(p.relative_to(REPO)): sha(p) for p in sources}, "input file changed"
    report = dict(inputs=before, net=str(NET.relative_to(REPO)), meta=str(META.relative_to(REPO)),
                  tls_count=len(meta), existing_15_action_tables_unchanged=True,
                  existing_lane_length_changes=changes, reference_connections=reference_connections,
                  buffers=buffer_records, deleted=["source_4074528#0", "source_node_4074528#0"],
                  added_tls={t: dict(mask=meta[t]["mask"], actions=meta[t]["actions"],
                                    approaches={e.getID(): [l.getLength() for l in e.getLanes()]
                                                for e in new.getNode(t).getIncoming()}) for t in sorted(ADDED_TLS)})
    (OUT/"build_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k,v in report.items() if k not in ("reference_connections", "inputs")}, indent=2))


if __name__ == "__main__":
    build()
