"""Read-only checks of entry topology, geometry, and preserved source scenes."""
from pathlib import Path
import hashlib
import json
import math
import sumolib
from build_net18 import NET,META,OUT,NORTH_JOIN,NORTH_SOURCE,NORTH_SINK,WEST_JOIN,EAST_SOURCE,WEST_SOURCE


def lane_distance(point, shape):
    answer=float("inf")
    for a,b in zip(shape,shape[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        den=dx*dx+dy*dy
        t=max(0,min(1,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/den)) if den else 0
        answer=min(answer,math.dist(point,(a[0]+t*dx,a[1]+t*dy)))
    return answer


def samples(shape):
    points=[]
    for a,b in zip(shape,shape[1:]):
        length=math.dist(a,b);steps=max(1,math.ceil(length/.5));delta=length/steps
        points.extend(((a[0]+(i+.5)/steps*(b[0]-a[0]),a[1]+(i+.5)/steps*(b[1]-a[1])),delta)
                      for i in range(steps))
    return points


def main():
    repo=Path(__file__).resolve().parents[3]
    before=OUT/"before_north_boundary"
    old=sumolib.net.readNet(str(before/NET.name));new=sumolib.net.readNet(str(NET))
    old_meta=json.loads((before/META.name).read_text())["tls"]
    meta=json.loads(META.read_text())["tls"]
    assert old_meta==meta
    expected_new={NORTH_SOURCE,NORTH_SINK,EAST_SOURCE,WEST_SOURCE}
    assert {e.getID() for e in new.getEdges()}-{e.getID() for e in old.getEdges()}==expected_new
    changes=[]
    for e in old.getEdges():
        for lane in e.getLanes():
            after=new.getLane(lane.getID())
            if lane.getShape()!=after.getShape() or lane.getLength()!=after.getLength():
                changes.append(dict(lane=lane.getID(),old=lane.getLength(),new=after.getLength()))
    assert {x["lane"] for x in changes}=={"668344588_0","668344588_1","-668344588_0","129212233#0_0"}
    join_reports=[]
    for nid in (NORTH_JOIN,WEST_JOIN,"source_college_green_E_boundary"):
        node=new.getNode(nid);cons=node.getConnections()
        assert all(c.getState()=="M" for c in cons)
        assert not any(node.areFoes(i,j) for i in range(len(cons)) for j in range(len(cons)))
        join_reports.append(dict(node=nid,conflicts=0,connections=[dict(
            from_lane=c.getFromLane().getID(),to_lane=c.getToLane().getID(),state=c.getState()) for c in cons]))
    before_connections={(c.getFromLane().getID(),c.getToLane().getID()):c.getState()
                        for node in old.getNodes() for c in node.getConnections()}
    after_connections={(c.getFromLane().getID(),c.getToLane().getID()):c.getState()
                       for node in new.getNodes() for c in node.getConnections()}
    removed=set(before_connections)-set(after_connections)
    assert removed=={("-129212230#1_0","129212230#0_0")}
    downgraded=[pair for pair,state in before_connections.items()
                if state=="M" and pair in after_connections and after_connections[pair]!="M"]
    assert not downgraded
    assert not new.getNode("2397311044").getOutgoing()
    assert new.getEdge("4074531#0").getLength()==old.getEdge("4074531#0").getLength()
    entries=[]
    for e in new.getEdges():
        if not e.getIncoming():
            assert e.getToNode().getID() not in meta,e.getID()
            entries.append(dict(edge=e.getID(),lanes=len(e.getLanes()),length=e.getLength(),
                                downstream=e.getToNode().getID(),direct_rl=False))
    # Geometry screen, not a continuous-time collision proof: at 0.5 m spacing,
    # report long close runs between new lanes and lanes on different edges.
    all_lanes=[l for e in new.getEdges() for l in e.getLanes()]
    close=[]
    for eid in sorted(expected_new):
        for lane in new.getEdge(eid).getLanes():
            points=samples(lane.getShape())
            for other in all_lanes:
                if other.getEdge().getID()==eid:continue
                run=maximum=0.0
                for point,delta in points:
                    run=run+delta if lane_distance(point,other.getShape())<2 else 0
                    maximum=max(maximum,run)
                if maximum>5:close.append(dict(lane=lane.getID(),other=other.getID(),close_run=maximum))
    assert not close,close
    hashes=json.loads((before/"original_scene_hashes.json").read_text())
    changed=[p for p,h in hashes.items() if hashlib.sha256((repo/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    report=dict(tls_count=len(meta),complete_tls_meta_unchanged=True,new_edges=sorted(expected_new),
                original_scene_files_checked=len(hashes),original_scene_changes=changed,
                existing_lane_changes=changes,other_existing_lane_shapes_and_lengths_unchanged=True,
                removed_connections=sorted(removed),existing_M_connections_downgraded=downgraded,
                joins=join_reports,exit_2397311044_unchanged=True,entry_count=len(entries),entries=entries,
                direct_external_rl_entries=[],geometry_close_runs=close,
                geometry_screen=dict(sample_spacing_max_m=.5,distance_threshold_m=2,continuous_run_threshold_m=5))
    (OUT/"boundary_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("joins","entries")},indent=2))


if __name__=="__main__":main()
