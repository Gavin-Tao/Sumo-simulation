"""Add DCC signal 9101555 as the 19th RL node; preserve the completed 18."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import sumolib
from build_net18 import WORK,REPO,NETS,REFERENCE

BASE=NETS/"dublin_8std_splitslot18_southclosed_corridor.net.xml"
BASE_META=NETS/"dublin_8std_splitslot18_southclosed_corridor_meta.json"
OUT=WORK/"net19"
TAG="splitslot19_southclosed_corridor"
NET=NETS/f"dublin_8std_{TAG}.net.xml"
META=NETS/f"dublin_8std_{TAG}_meta.json"
ADDED="9101555"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    OUT.mkdir(exist_ok=True)
    os.environ["PATH"]=str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH","")
    suffixes=(".nod.xml",".edg.xml",".con.xml",".tll.xml")
    paths=[WORK/"net18"/("corridor18"+s) for s in suffixes]
    inputs={str(p.relative_to(REPO)):sha(p) for p in [BASE,BASE_META,REFERENCE]+paths}
    trees={s:ET.parse(p) for s,p in zip(suffixes,paths)}
    nodes=trees[".nod.xml"].getroot()
    node=next(n for n in nodes.findall("node") if n.get("id")==ADDED)
    assert node.get("type")=="priority"
    node.set("type","traffic_light");node.set("tl",ADDED)
    for s,tr in trees.items():tr.write(OUT/("corridor19"+s),encoding="UTF-8",xml_declaration=True)
    raw=OUT/"corridor19.net.xml"
    commands=[
        [sumolib.checkBinary("netconvert"),"-n",OUT/"corridor19.nod.xml","-e",OUT/"corridor19.edg.xml",
         "-x",OUT/"corridor19.con.xml","-i",OUT/"corridor19.tll.xml","--lefthand",
         "--offset.disable-normalization","--no-turnarounds","--junctions.corner-detail","5","-o",raw],
        [sys.executable,"-B",REPO/"experiments/tools/dublin/reindex_8std.py",raw,"--split-slots","--tag",TAG],
    ]
    for args in commands:
        p=subprocess.run(list(map(str,args)),cwd=REPO,text=True,capture_output=True)
        with (OUT/"build.log").open("a") as f:f.write(" ".join(map(str,args))+"\n"+p.stdout+p.stderr+"\n")
        if p.returncode:raise RuntimeError(p.stderr or p.stdout)
    old=sumolib.net.readNet(str(BASE));new=sumolib.net.readNet(str(NET));ref=sumolib.net.readNet(str(REFERENCE))
    previous=json.loads(BASE_META.read_text())["tls"];meta=json.loads(META.read_text())["tls"]
    assert ref.getNode(ADDED).getType()=="traffic_light"
    assert set(meta)==set(previous)|{ADDED}
    assert all(previous[t]==meta[t] for t in previous)
    lengths=[]
    for e in old.getEdges():
        for lane in e.getLanes():
            after=new.getLane(lane.getID())
            if lane.getLength()!=after.getLength() or lane.getShape()!=after.getShape():
                lengths.append(dict(lane=lane.getID(),old=lane.getLength(),new=after.getLength()))
    assert not lengths,lengths
    def connections(net):
        return {(c.getFromLane().getID(),c.getToLane().getID()):c.getState()
                for node in net.getNodes() for c in node.getConnections()}
    before=connections(old);after=connections(new)
    assert set(before)==set(after)
    assert all(before[k]==after[k] for k in before if old.getLane(k[0]).getEdge().getToNode().getID()!=ADDED)
    assert inputs=={p:sha(REPO/p) for p in inputs}
    record=dict(inputs=inputs,net=str(NET.relative_to(REPO)),meta=str(META.relative_to(REPO)),
                added_tls=ADDED,tls_count=len(meta),original_18_meta_unchanged=True,
                existing_lane_shapes_and_lengths_unchanged=True,physical_connections_unchanged=True,
                non_rl_connections_elsewhere_unchanged=True,
                new_tls=meta[ADDED],downstream_yield_9101554_retained=True)
    (OUT/"build_report.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k not in ("inputs","new_tls")},indent=2))
    print('new mask',meta[ADDED]['mask'],'actions',meta[ADDED]['actions'])


if __name__=="__main__":build()
