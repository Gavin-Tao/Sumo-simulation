"""Static provenance, connectivity, lane reachability and turn-demand checks."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import xml.etree.ElementTree as ET
from build import REPO, WORK, NETS, NEW_NET, NEW_META, SOURCE_NET, SOURCE_META, SCENES, RENAMES, CLOSED, INTERNAL_PAIRS, REMOVED_NODES, remap_links, digest, connection_set, lane_path
import sumolib

report = json.loads((WORK / 'build_report.json').read_text())
assert all(digest(REPO / p) == h for p,h in report['source_hashes'].items())
oldnet = sumolib.net.readNet(str(SOURCE_NET))
net = sumolib.net.readNet(str(NEW_NET))
meta = json.loads(NEW_META.read_text())['tls']
oldmeta = json.loads(SOURCE_META.read_text())['tls']
assert set(meta) == set(oldmeta)
for tid in meta:
    for key in ('mask','canonical','std_to_green_index','actions'):
        assert meta[tid][key] == oldmeta[tid][key], (tid,key)
    assert meta[tid]['links'] == remap_links(oldmeta[tid]['links'])
expected = {(RENAMES.get(fr,fr),RENAMES.get(to,to),fl,tl)
            for fr,to,fl,tl in connection_set(ET.parse(SOURCE_NET).getroot())
            if fr not in CLOSED and to not in CLOSED and (fr,to) not in INTERNAL_PAIRS}
assert connection_set(ET.parse(NEW_NET).getroot()) == expected
assert not any(net.hasNode(n) for n in REMOVED_NODES)
assert not any(net.hasEdge(e) for e in CLOSED | set(RENAMES))
by_pair = {}; length_changes = []
for tid,m in meta.items():
    for group in m['links'].values():
        for c in group:
            by_pair[(c['from_edge'],c['to_edge'])] = (tid,c['approach']+'-'+c['turn'])
    for edge in oldnet.getNode(tid).getIncoming():
        for lane in edge.getLanes():
            new_lane = net.getEdge(RENAMES.get(edge.getID(),edge.getID())).getLane(lane.getIndex())
            if new_lane.getLength() != lane.getLength():
                length_changes.append(dict(tls=tid,lane=lane.getID(),old=lane.getLength(),new=new_lane.getLength()))

def vehicles(folder):
    files = {p.name:ET.parse(p).getroot() for p in folder.glob('*.rou.xml')}
    named = {r.get('id'):r.get('edges').split() for root in files.values() for r in root.findall('route')}
    result = {}
    for name,root in files.items():
        for v in root.findall('vehicle'):
            key = (name,v.get('id')); r = v.find('route')
            assert key not in result
            result[key] = (dict(v.attrib), r.get('edges').split() if r is not None else named[v.get('route')])
    return result

summaries = {}
for scene in SCENES:
    source = NETS / scene; target = NETS / (scene+'_southclosed')
    original = vehicles(source); revised = vehicles(target)
    extended = {(r['file'],r['id']) for r in json.loads((target/'modification_report.json').read_text())['extended_terminals']}
    assert set(original) == set(revised)
    changed_od = 0; demand = defaultdict(Counter)
    mainfiles = set(ET.parse(next(target.glob('*.sumocfg'))).getroot().find('input/route-files').get('value').split(','))
    defs = {}
    for key,(attr,edges) in revised.items():
        oldattr,oldedges = original[key]
        assert {k:v for k,v in attr.items() if k not in ('departPos','arrivalPos')} == oldattr
        assert edges[0] == RENAMES.get(oldedges[0],oldedges[0])
        assert edges[-1] == RENAMES.get(oldedges[-1],oldedges[-1]) or key in extended
        if key in extended:
            assert net.getEdge(edges[-1]).getToNode().getType() != 'traffic_light'
        assert not set(edges) & CLOSED
        assert all(net.hasEdge(e) for e in edges)
        assert lane_path(net,edges,int(attr.get('departLane','0'))), key
        for poskey,index in [('departPos',0),('arrivalPos',-1)]:
            if poskey in attr:
                assert 0 <= float(attr[poskey]) <= net.getEdge(edges[index]).getLength()
        if key[0] in mainfiles:
            for pair in zip(edges,edges[1:]):
                if pair in by_pair:
                    tid,slot = by_pair[pair]; demand[tid][slot] += 1
        if attr['id'].startswith('pair_'):
            pairkey = (key[0],attr['id'].rsplit('_',1)[0])
            if pairkey in defs: assert defs[pairkey] == edges, pairkey
            defs[pairkey] = edges
    demanded = {tid:dict(sorted(counts.items())) for tid,counts in demand.items()}
    (target/'movement_demand.json').write_text(json.dumps(dict(scope='loaded main route files, all vehicle types',counts=demanded),indent=2)+'\n')
    summaries[scene] = dict(checked_vehicle_definitions=len(revised),main_vehicles=sum(k[0] in mainfiles for k in revised),
                            paired_routes_identical=True,all_lane_connected=True,original_vehicle_attributes_preserved=True)
out = dict(source_hashes_unchanged=True,external_connections_exact=True,action_semantics_unchanged=True,
           tls_count=len(meta),inlet_length_changes=length_changes,scenarios=summaries)
(WORK/'static_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
