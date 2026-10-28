"""Low-volume end-to-end diagnostics; not either scene's formal demand."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import networkx as nx
import validate_net18 as diagnostic
from routing import LaneRoutes

OUT=diagnostic.WORK/"net18/validation/full_routes"
OUT.mkdir(exist_ok=True)
diagnostic.OUT=OUT


def make_routes(label):
    router=LaneRoutes(diagnostic.NET,diagnostic.META)
    graph=router.g
    forward=nx.descendants(graph,"__SOURCE__");backward=nx.ancestors(graph,"__SINK__")
    root=ET.Element("routes");records=[]
    for i,((a,b),movement) in enumerate(router.movements.items()):
        assert a in forward and b in backward,(movement,a,b)
        pre=nx.shortest_path(graph,"__SOURCE__",a,weight="weight")
        post=nx.shortest_path(graph,b,"__SINK__",weight="weight")
        desc=router.describe(pre+post)
        first=router.net.getLane(desc["lanes"][0]);last=router.net.getLane(desc["lanes"][-1])
        assert first.getID() in router.sources and last.getID() in router.sinks
        assert first.getEdge().getToNode().getID() not in router.meta
        records.append(dict(connection=[a,b],movement=movement,**desc))
        for q,ty in enumerate(("car","bus","ambulance")):
            # Spread diagnostic demand uniformly; this is not priority-pair training traffic.
            v=ET.SubElement(root,"vehicle",id=f"full_probe_{i}_{q}",type=ty,
                            depart=str(10+6*(3*i+q)),departLane=str(first.getIndex()),
                            departPos="0",departSpeed="0",arrivalPos="max",arrivalLane=str(last.getIndex()))
            ET.SubElement(v,"route",edges=" ".join(desc["edges"]))
    ET.indent(root);path=OUT/f"full_routes_{label}.rou.xml"
    ET.ElementTree(root).write(path,encoding="UTF-8",xml_declaration=True)
    net_hash=hashlib.sha256(diagnostic.NET.read_bytes()).hexdigest()
    (OUT/f"static_routes_{label}.json").write_text(json.dumps(dict(
        net_sha256=net_hash,connections=len(records),source_lanes=len(router.sources),
        sink_lanes=len(router.sinks),unreachable_connections=[],
        route_selection="weighted shortest legal lane paths; actual intermediate lane choice remains SUMO's",
        routes=records),indent=2)+"\n")
    return path,3*len(records)


if __name__=="__main__":
    strict="--strict-no-lc" in sys.argv
    legacy="--legacy-types" in sys.argv
    assert not (strict and legacy)
    if legacy:
        diagnostic.vtypes=diagnostic.REPO/"nets/dublin/weekday_18h_15_southclosed/vtypes_same.add.xml"
        OUT=OUT/"legacy_types";OUT.mkdir(exist_ok=True);diagnostic.OUT=OUT
    if strict:
        OUT=OUT/"strict_no_lanechange";OUT.mkdir(exist_ok=True);diagnostic.OUT=OUT
        types=ET.parse(diagnostic.vtypes)
        for vtype in types.getroot().findall("vType"):
            vtype.set("lcStrategic","-1");vtype.set("lcCooperative","-1")
        diagnostic.vtypes=OUT/"vtypes_strict_diagnostic.add.xml"
        types.write(diagnostic.vtypes,encoding="UTF-8",xml_declaration=True)
    diagnostic.main(sys.argv[1] if len(sys.argv)>1 else "fixed",
                    int(sys.argv[2]) if len(sys.argv)>2 else 123,
                    steps=720,route_factory=make_routes,require_no_lane_changes=not legacy)
