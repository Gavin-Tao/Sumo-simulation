"""Restore College Green's missing local connections on the 17-TLS variant.

Keep the merged 389332 corridor. DCC supplies the local movement layout;
its short eastern circulation loop and neighboring signals are not imported.
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
RENAMES = {"4476043#0": "304634278#0", "52879957#1": "-304634278#0"}


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def shape(ps): return " ".join(f"{x:.8f},{y:.8f}" for x,y in ps)


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
    # Explicit real cul-de-sac; retain its local turn-around connection.
    end=ref.getNode("3206753875")
    ET.SubElement(nr,"node",id=end.getID(),x=str(end.getCoord()[0]),y=str(end.getCoord()[1]),type="priority")
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
    ET.SubElement(con,"connection",{"from":"-129212230#1","to":"129212230#0","fromLane":"0","toLane":"0"})
    for i in range(2):ET.SubElement(con,"connection",{"from":source,"to":"532097269#1.24","fromLane":str(i),"toLane":str(i)})
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
        for l in e.getLanes():
            nl=new.getLane(l.getID())
            if abs(l.getLength()-nl.getLength())>.005:
                assert e.getFromNode().getID()==TLS or e.getToNode().getID()==TLS, l.getID()
                changes.append(dict(lane=l.getID(),old=l.getLength(),new=nl.getLength(),old_rl_inlet=e.getToNode().getID() in previous))
    actual=[{"from":c.getFrom().getID(),"to":c.getTo().getID(),"fromLane":str(c.getFromLane().getIndex()),"toLane":str(c.getToLane().getIndex())}
            for c in new.getNode(TLS).getConnections()]
    assert sorted(requested,key=str)==sorted(actual,key=str)
    assert digests=={str(p.relative_to(REPO)):sha(p) for p in inputs}
    report=dict(inputs=digests,net=str(NET.relative_to(REPO)),meta=str(META.relative_to(REPO)),
                tls_count=len(meta),existing_17_action_tables_unchanged=True,reference_edge_mapping=RENAMES,
                existing_lane_length_changes=changes,reference_connections=requested,
                new_tls=dict(id=TLS,mask=meta[TLS]["mask"],actions=meta[TLS]["actions"],
                             approaches={e.getID():[l.getLength() for l in e.getLanes()] for e in new.getNode(TLS).getIncoming()}))
    (OUT/"build_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("inputs","reference_connections")},indent=2))


if __name__=="__main__":build()
