"""Validate final topology, injection rule, traffic preservation and demand."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import xml.etree.ElementTree as ET
import sumolib
from build import REPO,WORK,NETS,NEW_NET,NEW_META,digest,connection_set,lane_path
from normalize_injection import SNAP,NAMES,route_data,CAPS,SOURCE_LANE_CAPS

build=json.loads((WORK/'build_report.json').read_text())
assert all(digest(REPO/p)==h for p,h in build['source_hashes'].items()),'Original scenario changed'
net=sumolib.net.readNet(str(NEW_NET));oldnet=sumolib.net.readNet(str(SNAP/NEW_NET.name))
meta=json.loads(NEW_META.read_text())['tls'];oldmeta=json.loads((SNAP/NEW_META.name).read_text())['tls']
assert set(meta)==set(oldmeta)
for tid in meta:
    for key in ('links','mask','canonical','std_to_green_index','actions'):
        assert meta[tid][key]==oldmeta[tid][key],(tid,key)
    for edge in oldnet.getNode(tid).getIncoming():
        for lane in edge.getLanes():assert lane.getLength()==net.getLane(lane.getID()).getLength(),lane.getID()
before=connection_set(ET.parse(SNAP/NEW_NET.name).getroot())
after=connection_set(ET.parse(NEW_NET).getroot())
assert before<=after
assert all(c[0].startswith('source_') and c[0]=='source_'+c[1] and c[2]==c[3] for c in after-before)
slot_by_edges={}
for tid,t in meta.items():
    for group in t['links'].values():
        for c in group:slot_by_edges[(c['from_edge'],c['to_edge'])]=(tid,c['approach']+'-'+c['turn'])

def read(folder):
    trees,named,_=route_data(folder);result={}
    for name,t in trees.items():
        for v in t.getroot().findall('vehicle'):
            r=v.find('route');r=r if r is not None else named[v.get('route')]
            result[(name,v.get('id'))]=(v.attrib.copy(),r.get('edges').split())
    return result
summary={}
for name in NAMES:
    folder=NETS/name;old=read(SNAP/name);new=read(folder)
    changes=json.loads((folder/'injection_changes.json').read_text())
    removed={tuple(k) for k in changes['removed_background_vehicles']}
    assert set(new)==set(old)-removed
    assert all(old[k][0].get('type')=='car' for k in removed)
    cfg=ET.parse(next(folder.glob('*.sumocfg'))).getroot()
    mainfiles=set(cfg.find('input/route-files').get('value').split(','))
    counts=Counter();sources=Counter();source_lanes=Counter();core=Counter();demand=defaultdict(Counter);pairs={};end_rl=[]
    for key,(attr,edges) in new.items():
        beforeattr,beforeedges=old[key]
        assert {k:v for k,v in attr.items() if k not in ('departPos','departLane')}=={k:v for k,v in beforeattr.items() if k not in ('departPos','departLane')}
        assert edges[-len(beforeedges):]==beforeedges,(key,edges,beforeedges)
        assert net.getEdge(edges[0]).getToNode().getType()!='traffic_light',key
        assert net.getEdge(edges[0]).getLength()>=40,key
        assert lane_path(net,edges,int(attr.get('departLane','0'))),key
        if key[0] in mainfiles:
            counts[attr.get('type')]+=1;sources[edges[0]]+=1
            source_lanes[edges[0]+'_'+attr.get('departLane','0')]+=1
            for e in set(edges):core[e]+=1
            for pair in zip(edges,edges[1:]):
                if pair in slot_by_edges:
                    tid,slot=slot_by_edges[pair];demand[tid][slot]+=1
            if net.getEdge(edges[-1]).getToNode().getType()=='traffic_light':end_rl.append(key)
        if attr['id'].startswith('pair_'):
            pairkey=(key[0],attr['id'].rsplit('_',1)[0])
            if pairkey in pairs:assert pairs[pairkey]==edges
            pairs[pairkey]=edges
    for edge,cap in CAPS.get(name,{}).items():assert core[edge]<=cap,(edge,core[edge],cap)
    for lane,cap in SOURCE_LANE_CAPS.get(name,{}).items():assert source_lanes[lane]<=cap,(lane,source_lanes[lane],cap)
    oldpriority=Counter(a.get('type') for (f,vid),(a,r) in old.items() if f in mainfiles and a.get('type')!='car')
    assert Counter({p:n for p,n in counts.items() if p!='car'})==oldpriority
    assert (folder/'vtypes_same.add.xml').read_bytes()==(SNAP/name/'vtypes_same.add.xml').read_bytes()
    result=dict(main_vehicles=sum(counts.values()),main_vehicles_by_type=dict(counts),all_files_vehicles=len(new),
                removed_background=len(removed),vehicles_starting_on_rl_inlets=0,all_lane_connected=True,
                priority_vehicles_preserved=True,paired_routes_identical=True,sources=dict(sources),
                capped_inlet_demand={e:core[e] for e in CAPS.get(name,{})},
                capped_source_demand={l:source_lanes[l] for l in SOURCE_LANE_CAPS.get(name,{})},existing_terminal_rl_routes=end_rl)
    (folder/'final_static_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (folder/'movement_demand.json').write_text(json.dumps(dict(scope='final main route files; all vehicle types',counts={t:dict(c) for t,c in demand.items()}),indent=2)+'\n')
    summary[name]={k:v for k,v in result.items() if k not in ('sources','existing_terminal_rl_routes')}
report=dict(originals_unchanged=True,rl_inlet_lengths_unchanged_after_injection=True,all_15_tls_action_semantics_unchanged=True,
            only_six_external_source_connections_added=len(after-before)==6,scenarios=summary)
(WORK/'final_static_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
