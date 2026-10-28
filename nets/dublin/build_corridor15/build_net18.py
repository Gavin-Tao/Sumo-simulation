"""Restore College Green's missing local connections on the 17-TLS variant.

Keep the merged 389332 corridor. DCC supplies the local movement layout;
its short eastern circulation loop and neighboring signals are not imported.
Foster Place's former turnaround is replaced with an artificial boundary
entry and exit, with no merge between the two streams.
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
OUT = WORK / "net18"
REFERENCE = REPO / "nets/dcc/dcc.net.xml"
BASE = NETS / "dublin_8std_splitslot17_southclosed_corridor.net.xml"
BASE_META = NETS / "dublin_8std_splitslot17_southclosed_corridor_meta.json"
TAG = "splitslot18_southclosed_corridor"
NET = NETS / f"dublin_8std_{TAG}.net.xml"
META = NETS / f"dublin_8std_{TAG}_meta.json"
TLS = "cluster_1426051358_1426100603_3206752725_3783453672_3783453675"
NORTH_JOIN = "3206753875"
NORTH_SOURCE = "source_college_green_N"
NORTH_SINK = "sink_college_green_N"
NORTH_BOUNDARY_LENGTH = 100.0
EAST_SOURCE = "source_college_green_E_outer"
WEST_SOURCE = "source_1448436_W"
WEST_JOIN = "cluster_1426051354_2705523737"
RENAMES = {"4476043#0": "304634278#0", "52879957#1": "-304634278#0"}
APPROACH_MERGES = (
    ("source_668344588", "668344588", "source_join_668344588"),
    ("source_college_green_E", "532097269#1.24", "gneJ70"),
)


def merge_unbranched_approaches(trees):
    """Make the two complete physical approaches visible to RL (user request).

    Only remove degree-two segmentation nodes. In particular, the real west
    junction's Sycamore Street exit is retained. No lane length is hand-set.
    """
    nr=trees[".nod.xml"].getroot();er=trees[".edg.xml"].getroot()
    records=[]
    for head,tail,join in APPROACH_MERGES:
        edges={e.get("id"):e for e in er.findall("edge")}
        first,last=edges[head],edges[tail]
        assert first.get("to")==join==last.get("from")
        assert first.get("numLanes")==last.get("numLanes")
        incident={e.get("id") for e in er.findall("edge") if join in (e.get("from"),e.get("to"))}
        assert incident=={head,tail}, (join,incident)
        points=[]
        for e in (first,last):
            for token in e.get("shape").split():
                point=tuple(map(float,token.split(",")))
                if not points or math.dist(points[-1],point)>1e-5:points.append(point)
        last.set("from",first.get("from"));last.set("shape",shape(points))
        for attr in ("length","endOffset"):last.attrib.pop(attr,None)
        for lane in last.findall("lane"):
            for attr in ("length","endOffset","shape"):lane.attrib.pop(attr,None)
        er.remove(first)
        nr.remove(next(n for n in nr.findall("node") if n.get("id")==join))
        for suffix in (".con.xml",".tll.xml"):
            cr=trees[suffix].getroot()
            for c in list(cr.findall("connection")):
                if c.get("from")==head and c.get("to")==tail:
                    cr.remove(c);continue
                for attr in ("from","to"):
                    if c.get(attr)==head:c.set(attr,tail)
        records.append(dict(old_edges=[head,tail],new_edge=tail,removed_node=join,
                            from_node=last.get("from"),to_node=last.get("to")))
    return records


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def shape(ps): return " ".join(f"{x:.8f},{y:.8f}" for x,y in ps)


def add_north_boundary(trees, ref):
    """Independent, opposing through paths at the former Foster Place dead end.

    The 100 m source/exit extensions are artificial study boundaries, not a
    claim about the real street. Keep the existing RL inlet and outlet shapes;
    derive the extensions from their end tangents, without hand-set lengths.
    """
    nr=trees[".nod.xml"].getroot();er=trees[".edg.xml"].getroot()
    con=trees[".con.xml"].getroot()
    inlet=ref.getEdge("129212230#0").getRawShape()
    outlet=ref.getEdge("-129212230#1").getRawShape()
    a,b=inlet[:2];norm=math.dist(a,b)
    source_end=(a[0]-NORTH_BOUNDARY_LENGTH*(b[0]-a[0])/norm,
                a[1]-NORTH_BOUNDARY_LENGTH*(b[1]-a[1])/norm)
    c,d=outlet[-2:];norm=math.dist(c,d)
    sink_end=(d[0]+NORTH_BOUNDARY_LENGTH*(d[0]-c[0])/norm,
              d[1]+NORTH_BOUNDARY_LENGTH*(d[1]-c[1])/norm)
    for eid,end,frm,to,points in (
        (NORTH_SOURCE,source_end,"source_college_green_N_boundary",NORTH_JOIN,[source_end,a]),
        (NORTH_SINK,sink_end,NORTH_JOIN,"boundary_college_green_N_exit",[d,sink_end]),
    ):
        boundary=frm if eid==NORTH_SOURCE else to
        ET.SubElement(nr,"node",id=boundary,x=str(end[0]),y=str(end[1]),type="dead_end")
        ET.SubElement(er,"edge",{"id":eid,"from":frm,"to":to,"priority":"11",
                                  "numLanes":"1","speed":"8.33","allow":"passenger",
                                  "shape":shape(points)})
    for frm,to in ((NORTH_SOURCE,"129212230#0"),("-129212230#1",NORTH_SINK)):
        ET.SubElement(con,"connection",{"from":frm,"to":to,"fromLane":"0","toLane":"0"})


def add_external_approach_buffers(trees):
    """Keep full approaches; put synthetic departures on upstream roads.

    College Green E extends straight. At 1448436 W, a straight western source
    would cross the retained northbound exit and turn its M connection into m.
    Instead add a parallel inbound road beside that northbound exit. Its turn
    into the west approach is conflict-free; netconvert computes the enlarged
    junction boundary and the resulting (slightly shorter) approach lengths.
    """
    nr=trees[".nod.xml"].getroot();er=trees[".edg.xml"].getroot()
    con=trees[".con.xml"].getroot()
    nodes={n.get("id"):n for n in nr.findall("node")}
    edges={e.get("id"):e for e in er.findall("edge")}
    east=edges["532097269#1.24"]
    a,b=[tuple(map(float,p.split(","))) for p in east.get("shape").split()[:2]]
    norm=math.dist(a,b)
    start=(a[0]-100*(b[0]-a[0])/norm,a[1]-100*(b[1]-a[1])/norm)
    join=east.get("from")
    nodes[join].set("type","priority");nodes[join].set("radius","0")
    specs=[(EAST_SOURCE,"source_college_green_E_outer_boundary",join,[start,a],"532097269#1.24","center")]

    # The existing one-way northbound exit uses a centered single lane.
    # Offset its centerline by half its width, then reverse the new two-lane
    # edge; left-hand lane offsets place it beside, not on, the existing lane.
    north=[tuple(map(float,p.split(","))) for p in edges["129212233#0"].get("shape").split()]
    incoming=[]
    for i,(x,y) in enumerate(north):
        p,q=north[max(0,i-1)],north[min(len(north)-1,i+1)]
        dx,dy=q[0]-p[0],q[1]-p[1];norm=math.hypot(dx,dy)
        incoming.append((x+1.6*dy/norm,y-1.6*dx/norm))
    incoming.reverse()
    specs.append((WEST_SOURCE,"source_1448436_W_boundary",WEST_JOIN,incoming,"668344588",None))
    records=[]
    for eid,boundary,join,points,target,spread in specs:
        ET.SubElement(nr,"node",id=boundary,x=str(points[0][0]),y=str(points[0][1]),type="dead_end")
        attrs={"id":eid,"from":boundary,"to":join,"priority":"11","numLanes":"2",
               "speed":"8.33","allow":"passenger","shape":shape(points)}
        if spread:attrs["spreadType"]=spread
        ET.SubElement(er,"edge",attrs)
        for lane in range(2):
            ET.SubElement(con,"connection",{"from":eid,"to":target,"fromLane":str(lane),"toLane":str(lane)})
        records.append(dict(source=eid,join=join,controlled_approach=target,artificial=True))
    return records


def run(args):
    result = subprocess.run(list(map(str,args)),cwd=REPO,text=True,capture_output=True)
    with (OUT/"build.log").open("a") as f:
        f.write(" ".join(map(str,args))+"\n"+result.stdout+result.stderr+"\n")
    if result.returncode: raise RuntimeError(result.stderr or result.stdout)


def build():
    OUT.mkdir(exist_ok=True)
    os.environ["PATH"] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH", "")
    suffixes=(".nod.xml", ".edg.xml", ".con.xml", ".tll.xml")
    inputs=[BASE,BASE_META,REFERENCE]+[WORK/"net17"/("corridor17"+s) for s in suffixes]
    digests={str(p.relative_to(REPO)):sha(p) for p in inputs}
    trees={s:ET.parse(WORK/"net17"/("corridor17"+s)) for s in suffixes}
    nr,er=trees[".nod.xml"].getroot(),trees[".edg.xml"].getroot()
    nodes={x.get("id"):x for x in nr.findall("node")}
    edges={x.get("id"):x for x in er.findall("edge")}
    old=sumolib.net.readNet(str(BASE));ref=sumolib.net.readNet(str(REFERENCE))
    nodes[TLS].set("type","traffic_light");nodes[TLS].set("tl",TLS)
    # Keep the reference street up to its end, then use separate artificial
    # boundary streams. No turnaround and no added traffic light at this node.
    end=ref.getNode(NORTH_JOIN)
    ET.SubElement(nr,"node",id=end.getID(),x=str(end.getCoord()[0]),y=str(end.getCoord()[1]),type="priority",radius="0")
    for eid in ("129212230#0","-129212230#1","4476041#0"):
        e=ref.getEdge(eid)
        kwargs={"id":eid,"from":e.getFromNode().getID(),"to":e.getToNode().getID(),
                "name":e.getName(),"priority":"11","numLanes":str(len(e.getLanes())),
                "speed":"8.33","allow":"passenger","shape":shape(e.getRawShape())}
        if eid=="4476041#0":kwargs["spreadType"]="center"
        edges[eid]=ET.SubElement(er,"edge",kwargs)
    # Separate eastbound boundary exit from the two-lane westbound source.
    # No gneE66/gneE68 3--7 m loop and no new downstream signal.
    eo=ref.getEdge("4476041#0").getRawShape()
    a,b=eo[-2:];norm=math.dist(a,b)
    endpoint=(b[0]+60*(b[0]-a[0])/norm,b[1]+60*(b[1]-a[1])/norm)
    exit_node="boundary_college_green_E_exit"
    ET.SubElement(nr,"node",id=exit_node,x=str(endpoint[0]),y=str(endpoint[1]),type="dead_end")
    edges["4476041#0"].set("to",exit_node)
    edges["4476041#0"].set("shape",shape(eo+[endpoint]))

    # The existing eastern boundary becomes an upstream source buffer.
    # The controlled approach itself retains its original road geometry.
    inlet=old.getEdge("532097269#1.24")
    points=inlet.getRawShape();a,b=points[:2];norm=math.dist(a,b)
    start=(a[0]-60*(b[0]-a[0])/norm,a[1]-60*(b[1]-a[1])/norm)
    source_node="source_college_green_E_boundary";source="source_college_green_E"
    nodes["gneJ70"].set("type","priority");nodes["gneJ70"].set("radius","0")
    ET.SubElement(nr,"node",id=source_node,x=str(start[0]),y=str(start[1]),type="dead_end")
    ET.SubElement(er,"edge",{"id":source,"from":source_node,"to":"gneJ70","priority":"11",
                             "numLanes":"2","speed":"8.33","allow":"passenger",
                             "spreadType":"center","shape":shape([start,a])})

    incoming={e.getID() for e in old.getNode(TLS).getIncoming()}
    for suffix in (".con.xml",".tll.xml"):
        root=trees[suffix].getroot()
        for c in list(root.findall("connection")):
            if c.get("from") in incoming:root.remove(c)
    con=trees[".con.xml"].getroot();requested=[]
    for c in ref.getNode(TLS).getConnections():
        attrs={"from":RENAMES.get(c.getFrom().getID(),c.getFrom().getID()),
               "to":RENAMES.get(c.getTo().getID(),c.getTo().getID()),
               "fromLane":str(c.getFromLane().getIndex()),"toLane":str(c.getToLane().getIndex())}
        assert attrs["from"] in edges and attrs["to"] in edges
        ET.SubElement(con,"connection",attrs);requested.append(attrs)
    add_north_boundary(trees,ref)
    for i in range(2):ET.SubElement(con,"connection",{"from":source,"to":"532097269#1.24","fromLane":str(i),"toLane":str(i)})
    merged=merge_unbranched_approaches(trees)
    entry_buffers=add_external_approach_buffers(trees)
    for suffix,tree in trees.items():
        ET.indent(tree);tree.write(OUT/("corridor18"+suffix),encoding="UTF-8",xml_declaration=True)
    raw=OUT/"corridor18.net.xml"
    run([sumolib.checkBinary("netconvert"),"-n",OUT/"corridor18.nod.xml","-e",OUT/"corridor18.edg.xml",
         "-x",OUT/"corridor18.con.xml","-i",OUT/"corridor18.tll.xml","--lefthand",
         "--offset.disable-normalization","--no-turnarounds","--junctions.corner-detail","5","-o",raw])
    run([sys.executable,"-B",REPO/"experiments/tools/dublin/reindex_8std.py",raw,"--split-slots","--tag",TAG])
    meta=json.loads(META.read_text())["tls"];previous=json.loads(BASE_META.read_text())["tls"]
    assert set(meta)==set(previous)|{TLS}
    for tid,item in previous.items():
        for key in ("mask","canonical","std_to_green_index","actions","links"):
            assert item[key]==meta[tid][key],(tid,key)
    new=sumolib.net.readNet(str(NET));changes=[]
    for e in old.getEdges():
        if e.getID() in {head for head,_,_ in APPROACH_MERGES}:continue
        for l in e.getLanes():
            nl=new.getLane(l.getID())
            if abs(l.getLength()-nl.getLength())>.005:
                assert e.getFromNode().getID() in (TLS,WEST_JOIN) or e.getToNode().getID() in (TLS,WEST_JOIN) or e.getID()=="668344588", l.getID()
                changes.append(dict(lane=l.getID(),old=l.getLength(),new=nl.getLength(),old_rl_inlet=e.getToNode().getID() in previous))
    actual=[{"from":c.getFrom().getID(),"to":c.getTo().getID(),"fromLane":str(c.getFromLane().getIndex()),"toLane":str(c.getToLane().getIndex())}
            for c in new.getNode(TLS).getConnections()]
    assert sorted(requested,key=str)==sorted(actual,key=str)
    north=new.getNode(NORTH_JOIN)
    boundary_connections=north.getConnections()
    assert {(c.getFrom().getID(),c.getTo().getID()) for c in boundary_connections}=={
        (NORTH_SOURCE,"129212230#0"),("-129212230#1",NORTH_SINK)}
    assert all(c.getState()=="M" for c in boundary_connections)
    assert all(not north.areFoes(i,j) for i in range(2) for j in range(2))
    for record in entry_buffers:
        node=new.getNode(record["join"])
        connections=node.getConnections()
        assert all(c.getState()=="M" for c in connections),record
        assert all(not node.areFoes(i,j) for i in range(len(connections)) for j in range(len(connections))),record
        record["source_lane_lengths"]=[l.getLength() for l in new.getEdge(record["source"]).getLanes()]
        record["approach_lane_lengths"]=[l.getLength() for l in new.getEdge(record["controlled_approach"]).getLanes()]
    assert not [e.getID() for e in new.getEdges() if not e.getIncoming() and e.getToNode().getID() in meta]
    assert new.getNode("2397311044").getType()=="dead_end"
    assert not new.getNode("2397311044").getOutgoing()
    assert not new.getNode("2397311044").getConnections()
    assert digests=={str(p.relative_to(REPO)):sha(p) for p in inputs}
    report=dict(inputs=digests,net=str(NET.relative_to(REPO)),meta=str(META.relative_to(REPO)),
                tls_count=len(meta),existing_17_action_tables_unchanged=True,reference_edge_mapping=RENAMES,
                approach_merges=merged,
                external_entry_buffers=entry_buffers,
                north_boundary=dict(join=NORTH_JOIN,source=NORTH_SOURCE,sink=NORTH_SINK,
                                    artificial=True,turnaround_removed=True,conflicts=0,
                                    lane_lengths={eid:[l.getLength() for l in new.getEdge(eid).getLanes()]
                                                  for eid in (NORTH_SOURCE,"129212230#0","-129212230#1",NORTH_SINK)}),
                existing_lane_length_changes=changes,reference_connections=requested,
                new_tls=dict(id=TLS,mask=meta[TLS]["mask"],actions=meta[TLS]["actions"],
                             approaches={e.getID():[l.getLength() for l in e.getLanes()] for e in new.getNode(TLS).getIncoming()}))
    (OUT/"build_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("inputs","reference_connections")},indent=2))


if __name__=="__main__":build()
