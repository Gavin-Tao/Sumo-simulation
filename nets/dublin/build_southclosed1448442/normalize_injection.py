"""Normalize BOTH new scenario copies to non-RL upstream injection.

The pre-injection snapshot is immutable input; reruns only replace these new
variants. Main/background demand caps are explicit and deterministic. No model
or training code is changed and no training is started.
"""
from pathlib import Path
from collections import defaultdict, Counter
import heapq
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import sumolib

from build import REPO, WORK, NETS, NEW_NET, NEW_META, SCENES, digest, run
SNAP = WORK / 'before_injection'
NAMES = tuple(s+'_southclosed' for s in SCENES)
CAPS = {'weekday_pairs50_15_southclosed': {'317003251#0': 196, '4919535#1': 200},
        'weekday_18h_15_southclosed': {'-77616869#2': 500, '-4074527#1': 400,
                                      '-4475730': 480, '22962979': 240}}
SOURCE_LANE_CAPS = {'weekday_18h_15_southclosed': {'532097269#1.24_1': 300}}

def route_data(folder):
    trees = {p.name:ET.parse(p) for p in folder.glob('*.rou.xml')}
    named = {r.get('id'):r for t in trees.values() for r in t.getroot().findall('route')}
    lanes = defaultdict(set)
    for tree in trees.values():
        for v in tree.getroot().findall('vehicle'):
            if v.get('route'):lanes[v.get('route')].add(int(v.get('departLane','0')))
    assert all(len(x)==1 for x in lanes.values())
    return trees,named,lanes

def predecessors(net):
    result=defaultdict(list)
    for edge in net.getEdges():
        for lane in edge.getLanes():
            for c in lane.getOutgoing():
                if lane.allows('passenger') and c.getToLane().allows('passenger'):
                    result[(c.getTo().getID(),c.getToLane().getIndex())].append((edge.getID(),lane.getIndex(),c.getState()))
    return result

def upstream(net,prev,start):
    heap=[((0,0,0.0),start,())]; seen=set()
    while heap:
        cost,key,tail=heapq.heappop(heap)
        if key in seen:continue
        seen.add(key); edge=net.getEdge(key[0])
        if edge.getToNode().getType()!='traffic_light' and edge.getLength()>=40:
            return [key]+list(tail),cost
        for eid,lane,state in prev[key]:
            if eid in {x[0] for x in tail}:continue
            e=net.getEdge(eid)
            c=(cost[0]+int(e.getToNode().getType()=='traffic_light'), cost[1]+int(state=='m'),cost[2]+e.getLength())
            heapq.heappush(heap,(c,(eid,lane),(key,)+tail))
    return None

def main():
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    if not SNAP.exists():
        SNAP.mkdir()
        for p in (NEW_NET,NEW_META):shutil.copy2(p,SNAP/p.name)
        for name in NAMES:shutil.copytree(NETS/name,SNAP/name)
    oldnet=sumolib.net.readNet(str(SNAP/NEW_NET.name)); prev=predecessors(oldnet)
    starts=set()
    for name in NAMES:
        trees,named,nlanes=route_data(SNAP/name)
        for t in trees.values():
            for v in t.getroot().findall('vehicle'):
                r=v.find('route');r=r if r is not None else named[v.get('route')]
                starts.add((r.get('edges').split()[0],int(v.get('departLane','0'))))
    missing={e for e,l in starts if upstream(oldnet,prev,(e,l)) is None}
    assert len(missing)==6, missing
    with tempfile.TemporaryDirectory(prefix='injection_plain_',dir=WORK) as td:
        prefix=Path(td)/'base'
        run([sumolib.checkBinary('netconvert'),'-s',SNAP/NEW_NET.name,'--plain-output-prefix',prefix,'--offset.disable-normalization'])
        trees={s:ET.parse(str(prefix)+s) for s in ('.nod.xml','.edg.xml','.con.xml','.tll.xml')}
        for node in trees['.nod.xml'].getroot():
            if node.get('id') == '659804':
                node.set('radius','0')
        for eid in sorted(missing):
            edge=oldnet.getEdge(eid); node=edge.getFromNode(); x,y=node.getCoord(); shape=edge.getShape()
            dx,dy=shape[1][0]-shape[0][0],shape[1][1]-shape[0][1];scale=150/math.hypot(dx,dy)
            if eid == '-4074527#1':
                # This boundary uses an offset one-way centerline. Continue it
                # collinearly so rebuilding the connector does not trim 1.5 m.
                x,y=shape[0]
            sx,sy=x-dx*scale,y-dy*scale
            nid='source_node_'+eid; sourceid='source_'+eid
            ET.SubElement(trees['.nod.xml'].getroot(),'node',id=nid,x=str(sx),y=str(sy),type='priority')
            attrs={'id':sourceid,'from':nid,'to':node.getID(),'numLanes':str(edge.getLaneNumber()),
                   'speed':str(edge.getSpeed()),'priority':'10','spreadType':'center','allow':'passenger'}
            if eid == '-4074527#1':attrs['shape']=f'{sx},{sy} {x},{y}'
            ET.SubElement(trees['.edg.xml'].getroot(),'edge',attrs)
            for i in range(edge.getLaneNumber()):
                ET.SubElement(trees['.con.xml'].getroot(),'connection',{'from':sourceid,'to':eid,'fromLane':str(i),'toLane':str(i)})
        for suffix,t in trees.items():t.write(WORK/('injection'+suffix),encoding='UTF-8',xml_declaration=True)
    run([sumolib.checkBinary('netconvert'),'-n',WORK/'injection.nod.xml','-e',WORK/'injection.edg.xml',
         '-x',WORK/'injection.con.xml','-i',WORK/'injection.tll.xml','--lefthand','--offset.disable-normalization',
         '--no-turnarounds','--junctions.corner-detail','5','-o',WORK/'injection.net.xml'])
    run([sys.executable,'-B',REPO/'experiments/tools/dublin/reindex_8std.py',WORK/'injection.net.xml',
         '--split-slots','--tag','splitslot15_southclosed'])
    oldmeta=json.loads((SNAP/NEW_META.name).read_text())['tls']; meta=json.loads(NEW_META.read_text())['tls']
    assert set(meta)==set(oldmeta)
    for tid in meta:
        for k in ('mask','canonical','std_to_green_index','actions','links'):
            assert meta[tid][k]==oldmeta[tid][k],(tid,k)
    net=sumolib.net.readNet(str(NEW_NET));prev=predecessors(net)
    length_changes=[]
    for tid in meta:
        for e in oldnet.getNode(tid).getIncoming():
            for l in e.getLanes():
                new=net.getLane(l.getID()).getLength()
                if new!=l.getLength():length_changes.append(dict(lane=l.getID(),old=l.getLength(),new=new))
    print('RL length changes:',length_changes,flush=True)
    # Stop before route writes if adding source stubs shortened any controlled inlet substantially.
    assert all(abs(x['new']-x['old'])<.05 for x in length_changes),length_changes
    report=dict(source_buffers=sorted('source_'+e for e in missing),rl_length_changes=length_changes,
                source_net_sha256=digest(SNAP/NEW_NET.name),net_sha256=digest(NEW_NET),scenarios={})
    for name in NAMES:
        target=NETS/name;trees,named,nlanes=route_data(SNAP/name)
        old_named={key:r.get('edges').split() for key,r in named.items()}
        mapping={}; record=[]
        def transform(edges,dl):
            cache=(tuple(edges),dl)
            if cache not in mapping:
                found=upstream(net,prev,(edges[0],dl));assert found is not None,(edges[0],dl)
                chain,cost=found
                result=[e for e,l in chain]+edges[1:]
                assert net.getEdge(result[0]).getToNode().getType()!='traffic_light'
                mapping[cache]=(result,chain[0][1],cost)
            return mapping[cache]
        for key,r in named.items():
            es,dl,cost=transform(old_named[key],next(iter(nlanes[key])))
            r.set('edges',' '.join(es))
        # Count all main-route traffic through capped inlets, protecting all probes/priority vehicles.
        cfg=ET.parse(next((SNAP/name).glob('*.sumocfg'))).getroot()
        mainfiles=set(cfg.find('input/route-files').get('value').split(','))
        removable=defaultdict(list); protected=Counter()
        for filename,t in trees.items():
            for v in t.getroot().findall('vehicle'):
                r=v.find('route');old=r.get('edges').split() if r is not None else old_named[v.get('route')]
                dl=int(v.get('departLane','0'));new,nd,cost=transform(old,dl)
                if filename in mainfiles:
                    for eid,cap in CAPS.get(name,{}).items():
                        if eid in new:
                            if v.get('type')=='car' and filename in ('background_cars.rou.xml','car_weekday_18h.rou.xml','added_flows.rou.xml'):
                                removable[eid].append((float(v.get('depart')),filename,v.get('id')))
                            else:protected[eid]+=1
                if new!=old:
                    record.append(dict(file=filename,id=v.get('id'),old_origin=old[0],new_origin=new[0],
                                       old_depart_lane=dl,new_depart_lane=nd,prefix=new[:len(new)-len(old)],extra_rl=cost[0],yield_connections=cost[1]))
                    v.set('departLane',str(nd));v.attrib.pop('departPos',None)
                if r is not None:r.set('edges',' '.join(new))
        removed=set();counts={}
        for eid,cap in CAPS.get(name,{}).items():
            candidates=sorted(removable[eid]); keep=max(0,min(len(candidates),cap-protected[eid]))
            chosen={int((k+.5)*len(candidates)/keep) for k in range(keep)} if keep else set()
            removed.update((fn,vid) for i,(_,fn,vid) in enumerate(candidates) if i not in chosen)
            counts[eid]=dict(total_before=len(candidates)+protected[eid],protected=protected[eid],kept_background=keep,total_after=keep+protected[eid])
        source_counts={}
        for source_lane,cap in SOURCE_LANE_CAPS.get(name,{}).items():
            candidates=[]; protected_count=0
            for filename,t in trees.items():
                if filename not in mainfiles:continue
                for v in t.getroot().findall('vehicle'):
                    if (filename,v.get('id')) in removed:continue
                    r=v.find('route'); r=r if r is not None else named[v.get('route')]
                    source=r.get('edges').split()[0]+'_'+v.get('departLane','0')
                    if source!=source_lane:continue
                    if v.get('type')=='car' and filename in ('background_cars.rou.xml','car_weekday_18h.rou.xml','added_flows.rou.xml'):
                        candidates.append((float(v.get('depart')),filename,v.get('id')))
                    else:protected_count+=1
            candidates.sort();keep=max(0,min(len(candidates),cap-protected_count))
            chosen={int((k+.5)*len(candidates)/keep) for k in range(keep)} if keep else set()
            removed.update((fn,vid) for i,(_,fn,vid) in enumerate(candidates) if i not in chosen)
            source_counts[source_lane]=dict(total_before=len(candidates)+protected_count,protected=protected_count,total_after=keep+protected_count)
        for filename,t in trees.items():
            for v in list(t.getroot().findall('vehicle')):
                if (filename,v.get('id')) in removed:t.getroot().remove(v)
            t.write(target/filename,encoding='UTF-8',xml_declaration=True)
        result=dict(injection_rule='departure edge ends at a non-RL node; >=40m source edge',
                    relocated_vehicles=record,removed_background_vehicles=sorted(removed),caps=counts,source_lane_caps=source_counts)
        (target/'injection_changes.json').write_text(json.dumps(result,indent=2)+'\n')
        report['scenarios'][name]=dict(relocated=len(record),removed_background=len(removed),caps=counts,source_lane_caps=source_counts)
    (WORK/'injection_build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
