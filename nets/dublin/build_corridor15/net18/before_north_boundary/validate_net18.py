"""Net-only diagnostic vehicles: connectivity, 8STD, state and reward checks.

This does not generate scenario demand, calibrate traffic, or train an agent.
All diagnostic routes and outputs remain under net18/validation/.
"""
from pathlib import Path
from collections import defaultdict, Counter
import functools
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET

WORK=Path(__file__).resolve().parent
REPO=WORK.parents[2]
OUT=WORK/"net18/validation"
OUT.mkdir(exist_ok=True)
os.environ["PATH"]=str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH","")
os.environ["SUMO_RL_LIBSUMO"]="1"
os.environ["OMP_NUM_THREADS"]="1"
sys.path[:0]=[str(REPO),str(REPO/"experiments")]
import numpy as np
import yaml
import sumolib
from exp_params import obs_kwargs_from_cfg, env_extra_kwargs, reward_fn_from_cfg, turnmap_from_meta
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.observations import PriorityMovementObservationFunction
from build_net18 import NET,META,TLS

meta=json.loads(META.read_text())["tls"]
net=sumolib.net.readNet(str(NET))
NEW={"659799","659805",TLS}
cfg=yaml.safe_load(next((REPO/"experiments/configs").glob("exp461_*.yaml")).read_text())
vtypes=REPO/"nets/dublin/weekday_18h_15_southclosed/vtypes_same.add.xml"
LABELS=[f"{a}-{t}" for a in "NESW" for t in "LTR"]
PRIO=cfg["priority_source"]
EXPANDED={"668344588":("1448436",47.69),"532097269#1.24":(TLS,65.79)}


def make_probes(label):
    vehicles=[]
    # Every actual connection, including all members of a shared link index.
    for tid,t in meta.items():
        connections=[c for group in t["links"].values() for c in group]
        for k,c in enumerate(connections):
            for q,ty in enumerate(("car","bus","ambulance")):
                route=[c["from_edge"],c["to_edge"]]
                first=net.getEdge(route[0]);pos=max(0,first.getLength()-20)
                # Also exercise the new boundary buffers with unchanged lanes.
                prefix={"532096352#0":"source_659799_N","-3791034#1":"source_659799_E",
                        "532097269#1.24":"source_college_green_E"}.get(route[0])
                if prefix and net.hasEdge(prefix):
                    route.insert(0,prefix);pos=max(0,net.getEdge(prefix).getLength()-20)
                vehicles.append((10+q*180+k*6,tid,c,ty,route,pos))
    root=ET.Element("routes")
    for i,(depart,tid,c,ty,route,pos) in enumerate(sorted(vehicles,key=lambda x:(x[0],x[1],x[2]["from_lane"]))):
        v=ET.SubElement(root,"vehicle",id=f"net_probe_{i}",type=ty,depart=str(depart),
                        departLane=str(c["from_lane"]),departPos=str(pos),departSpeed="0",
                        arrivalPos=str(min(15,net.getEdge(c["to_edge"]).getLength())),
                        arrivalLane=str(c["to_lane"]))
        ET.SubElement(v,"route",edges=" ".join(route))
    # Exercise the retained Foster Place cul-de-sac and a second visit to the
    # same TLS from a different arm; this path is not scenario demand.
    for q,ty in enumerate(("car","bus","ambulance")):
        v=ET.SubElement(root,"vehicle",id=f"culdesac_probe_{q}",type=ty,depart=str(600+q*10),
                        departLane="0",departPos=str(net.getEdge("304634278#0").getLength()-20),
                        departSpeed="0",arrivalPos="15",arrivalLane="0")
        ET.SubElement(v,"route",edges="304634278#0 -129212230#1 129212230#0 4476041#0")
    # Former buffers now belong to the controlled inlet. Start diagnostic
    # vehicles there and hold them via setSpeed (not a planned SUMO stop), so
    # count, queue and visit waiting must include this previously hidden area.
    for k,(edge,(tid,_)) in enumerate(EXPANDED.items()):
        connection=next(c for group in meta[tid]["links"].values() for c in group
                        if c["from_edge"]==edge and c["from_lane"]==0)
        for q,ty in enumerate(("car","bus","ambulance")):
            v=ET.SubElement(root,"vehicle",id=f"upstream_probe_{k}_{q}",type=ty,
                            depart=str(700+60*k+20*q),departLane="0",departPos="10",departSpeed="0",
                            arrivalPos="15",arrivalLane=str(connection["to_lane"]))
            ET.SubElement(v,"route",edges=edge+" "+connection["to_edge"])
    ET.indent(root);path=OUT/f"connectivity_probes_{label}.rou.xml"
    ET.ElementTree(root).write(path,encoding="UTF-8",xml_declaration=True)
    return path,len(vehicles)+3+3*len(EXPANDED)


def polygon(s,v):
    x,y=s.vehicle.getPosition(v);ang=math.radians(s.vehicle.getAngle(v));dx,dy=math.sin(ang),math.cos(ang)
    width=s.vehicle.getWidth(v)/2;length=s.vehicle.getLength(v)
    return [(x+dy*width,y-dx*width),(x-dy*width,y+dx*width),
            (x-length*dx-dy*width,y-length*dy+dx*width),(x-length*dx+dy*width,y-length*dy-dx*width)]


def overlap(a,b):
    for pts in (a,b):
        for p,q in zip(pts,pts[1:]+pts[:1]):
            ax,ay=p[1]-q[1],q[0]-p[0];norm=math.hypot(ax,ay)
            aa=[(x*ax+y*ay)/norm for x,y in a];bb=[(x*ax+y*ay)/norm for x,y in b]
            if min(max(aa),max(bb))-max(min(aa),min(bb))<=.02:return False
    return True


def main(policy,seed):
    routes,planned=make_probes(f"{policy}_{seed}")
    prefix=OUT/f"{policy}_{seed}"
    config=OUT/f"connectivity_probes_{policy}_{seed}.sumocfg"
    config.write_text('<configuration><report><no-step-log value="true"/><duration-log.disable value="true"/></report></configuration>\n')
    env=SumoEnvironment(net_file=str(NET),route_file=str(routes),cfg_file=str(config),out_csv_name=None,use_gui=False,num_seconds=1200,
                        min_green=5,max_green=50,use_max_green=True,single_agent=False,yellow_time=2,delta_time=5,
                        reward_fn=reward_fn_from_cfg(cfg),observation_class=functools.partial(PriorityMovementObservationFunction,**obs_kwargs_from_cfg(cfg)),
                        sumo_seed=seed,sumo_warnings=True,waiting_time_memory=100000,time_to_teleport=-1,
                        additional_sumo_cmd=f"--additional-files {vtypes} --collision.check-junctions true --collision.action warn --error-log {prefix}_sumo.log",
                        **env_extra_kwargs(cfg))
    states=env.reset(seed);s=env.sumo;ids=list(env.ts_ids)
    assert set(ids)==set(meta) and len(ids)==18
    tracker=env.visit_tracker;tracker.strict=True
    tables={}
    for tid in ids:
        ts=env.traffic_signals[tid];ts.observation_fn.rebind_movements(turnmap_from_meta(meta[tid]))
        assert [p.state for p in ts.green_phases]==[meta[tid]["actions"][f"a{a}"]["state"] for a in range(8) if meta[tid]["mask"][a]]
        ins=defaultdict(set);outs=defaultdict(set);edges={};index_slot={}
        for index,group in meta[tid]["links"].items():
            names={c["approach"]+"-"+c["turn"] for c in group};assert len(names)==1
            name=next(iter(names));index_slot[int(index)]=name
            for c in group:
                ins[name].add(c["from_edge"]+"_"+str(c["from_lane"]))
                outs[name].add(c["to_edge"]+"_"+str(c["to_lane"]))
                edges[(c["from_edge"],c["to_edge"])]=name
        tables[tid]=(ins,outs,edges,index_slot)
    reference={};previous_accum={};arrived=set();seen_turns=defaultdict(set);last_lanes={}
    freeze_until={}
    audit=dict(planned=planned,arrived=0,decisions=0,state_checks=0,reward_checks=0,tracker_checks=0,
               collisions=0,teleports=0,lane_changes=0,body_overlap_samples=[],tracker_missing=0,nonzero_wait_checks=0,
               expanded_upstream_wait_samples={edge:0 for edge in EXPANDED},
               farthest_observed_distance={edge:0.0 for edge in EXPANDED})
    watch=NEW|{"cluster_2498303858_3624079904_6141082777_6141082778_9936781"}
    centers={tid:net.getNode(tid).getCoord() for tid in watch}
    original_step=env._sumo_step
    def step_hook():
        original_step()
        now=s.simulation.getTime()
        for v in s.simulation.getDepartedIDList():
            if v.startswith("upstream_probe_"):
                freeze_until[v]=now+10;s.vehicle.setSpeed(v,0)
        for v,until in list(freeze_until.items()):
            if now>=until:
                s.vehicle.setSpeed(v,-1);del freeze_until[v]
        audit["collisions"]+=s.simulation.getCollidingVehiclesNumber()
        audit["teleports"]+=s.simulation.getStartingTeleportNumber()
        arrived.update(s.simulation.getArrivedIDList())
        current={};accum={}
        for tid in ids:
            for lane in env.traffic_signals[tid].lanes:
                edge=lane.rsplit("_",1)[0]
                for v in s.lane.getLastStepVehicleIDs(lane):
                    key=(tid,v,edge);w=s.vehicle.getAccumulatedWaitingTime(v)
                    inc=w-previous_accum[key] if key in reference else min(s.vehicle.getWaitingTime(v),tracker.dt)
                    assert -1e-8<=inc<=tracker.dt+1e-8
                    current[key]=reference.get(key,0)+inc;accum[key]=w
                    assert abs(current[key]-tracker.get(tid,v,lane))<1e-8,(s.simulation.getTime(),key)
                    audit["tracker_checks"]+=1;audit["nonzero_wait_checks"]+=current[key]>0
                    if edge in EXPANDED:
                        distance=s.lane.getLength(lane)-s.vehicle.getLanePosition(v)
                        audit["farthest_observed_distance"][edge]=max(audit["farthest_observed_distance"][edge],distance)
                        if distance>EXPANDED[edge][1]+5 and current[key]>0:
                            audit["expanded_upstream_wait_samples"][edge]+=1
                    rt=s.vehicle.getRoute(v);ri=s.vehicle.getRouteIndex(v)
                    if ri+1<len(rt):seen_turns[tid].add((edge,rt[ri+1],int(lane.rsplit("_",1)[1])))
        reference.clear();reference.update(current);previous_accum.clear();previous_accum.update(accum)
        active=s.vehicle.getIDList();positions={v:s.vehicle.getPosition(v) for v in active}
        for v in active:
            lane=s.vehicle.getLaneID(v);prev=last_lanes.get(v)
            if prev and not lane.startswith(":") and prev.rsplit("_",1)[0]==lane.rsplit("_",1)[0] and prev!=lane:audit["lane_changes"]+=1
            last_lanes[v]=lane
        for tid,center in centers.items():
            local=[v for v in active if math.dist(positions[v],center)<40]
            for i,v in enumerate(local):
                for u in local[i+1:]:
                    if math.dist(positions[v],positions[u])<8 and overlap(polygon(s,v),polygon(s,u)):
                        if len(audit["body_overlap_samples"])<20:audit["body_overlap_samples"].append(dict(time=s.simulation.getTime(),tls=tid,vehicles=[v,u],lanes=[s.vehicle.getLaneID(v),s.vehicle.getLaneID(u)]))
    env._sumo_step=step_hook
    rng=random.Random(seed)
    try:
        for step in range(240):
            actions={tid:(rng.randrange(len(env.traffic_signals[tid].green_phases)) if policy=="random" else (step//5)%len(env.traffic_signals[tid].green_phases)) for tid in ids}
            states,rewards,done,info=env.step(actions);audit["decisions"]+=1
            for tid in ids:
                ts=env.traffic_signals[tid];ins,outs,by_edges,index_slot=tables[tid]
                totals=defaultdict(list);stats=defaultdict(list);counts={}
                for lane in ts.lanes:
                    vs=s.lane.getLastStepVehicleIDs(lane);counts[lane]=len(vs);edge=lane.rsplit("_",1)[0]
                    for v in vs:
                        p=PRIO.get(s.vehicle.getTypeID(v),1);w=reference[(tid,v,edge)];totals[p].append(w)
                        rt=s.vehicle.getRoute(v);ri=s.vehicle.getRouteIndex(v)
                        slot=by_edges.get((edge,rt[ri+1])) if ri+1<len(rt) else None
                        if slot:stats[(slot,p)].append((s.vehicle.getSpeed(v)<.1,w))
                wanted_reward=-sum(p*sum(ws)/len(ws) for p,ws in totals.items())/100
                assert abs(wanted_reward-rewards[tid])<1e-8,(tid,wanted_reward,rewards[tid])
                audit["reward_checks"]+=1
                served={name for i,name in index_slot.items() if ts.green_phases[ts.green_phase].state[i] in "Gg"}
                mg=float(ts.time_since_last_phase_change>=ts.min_green+ts.yellow_time);expected=[]
                for name in LABELS:
                    cap=max(sum(s.lane.getLength(l)/7.5 for l in ins[name]),1)
                    expected.append(float(name in served))
                    for p in range(1,6):
                        rows=stats[(name,p)];mean=sum(w for q,w in rows)/len(rows) if rows else 0
                        expected.extend([len(rows)/cap,sum(q for q,w in rows)/cap,np.log1p(mean)/np.log1p(3600)])
                    down=defaultdict(list)
                    for lane in outs[name]:
                        for v in s.lane.getLastStepVehicleIDs(lane):down[PRIO.get(s.vehicle.getTypeID(v),1)].append(s.vehicle.getSpeed(v)<.1)
                    outcap=max(sum(s.lane.getLength(l)/7.5 for l in outs[name]),1)
                    expected.extend(len(down[p])/outcap for p in range(1,6));expected.extend(sum(down[p])/outcap for p in range(1,6))
                    expected.append(max((counts[l]/max(s.lane.getLength(l)/7.5,1) for l in ins[name]),default=0))
                    expected.append(mg if name in served else 0)
                assert len(states[tid])==336
                np.testing.assert_allclose(states[tid],np.asarray(expected,dtype=np.float32),rtol=1e-6,atol=1e-7,err_msg=f"{tid} t={s.simulation.getTime()}")
                audit["state_checks"]+=1
        audit["arrived"]=len(arrived);audit["remaining"]=list(s.vehicle.getIDList());audit["pending"]=list(s.simulation.getPendingVehicles())
        audit["tracker_missing"]=tracker.missing
        audit["observed_turn_lane_pairs"]={k:len(v) for k,v in seen_turns.items()}
    finally:
        env.close()
        (OUT/f"{policy}_{seed}_report.json").write_text(json.dumps(audit,indent=2)+"\n")
    assert audit["arrived"]==planned,audit
    assert not audit["collisions"] and not audit["teleports"] and not audit["tracker_missing"] and not audit["lane_changes"],audit
    assert not audit["body_overlap_samples"],audit["body_overlap_samples"]
    assert audit["nonzero_wait_checks"]>0
    assert all(v>0 for v in audit["expanded_upstream_wait_samples"].values()),audit
    print(json.dumps(audit,indent=2))


if __name__=="__main__":main(sys.argv[1] if len(sys.argv)>1 else "fixed",int(sys.argv[2]) if len(sys.argv)>2 else 123)
