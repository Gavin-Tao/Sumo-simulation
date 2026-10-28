"""Independent connection/foes/8STD closure checks for the new 18-signal net."""
from pathlib import Path
import json,xml.etree.ElementTree as ET,itertools,hashlib
import sumolib,networkx as nx
from build_net18_nocg import NET, OUT
p=NET
r=ET.parse(p).getroot();n=sumolib.net.readNet(str(p),withInternal=True)
m=json.loads(p.with_name(p.name.replace('.net.xml','_meta.json')).read_text())['tls']
requests={j.get('id'):{int(q.get('index')):q.get('foes') for q in j.findall('request')} for j in r.findall('junction')}
report={'hash':hashlib.sha256(p.read_bytes()).hexdigest(),'tls':{},'missing_extensions':[],'unsafe':[],'partial_slots':[],'metadata_errors':[]}
for tid,t in m.items():
 groups={}
 for c in n.getNode(tid).getConnections():
  k=c.getTLLinkIndex()
  if k>=0:groups.setdefault(k,[]).append(c)
 assert set(groups)==set(map(int,t['links']))
 def identity(c):return(c.getFrom().getID(),c.getFromLane().getIndex(),c.getTo().getID(),c.getToLane().getIndex())
 for k,cs in groups.items():
  decl={(c['from_edge'],c['from_lane'],c['to_edge'],c['to_lane']) for c in t['links'][str(k)]}
  assert {identity(c) for c in cs}==decl,(tid,k)
  assert {c.getJunctionIndex() for c in cs}=={c['junction_index'] for c in t['links'][str(k)]},(tid,k)
 def foe(a,b):
  ia,ib=a.getJunctionIndex(),b.getJunctionIndex()
  return requests[tid][ia][-1-ib]=='1' or requests[tid][ib][-1-ia]=='1'
 def reasons(i,j):
  found=set()
  for a in groups[i]:
   for b in groups[j]:
    if a.getToLane().getID()==b.getToLane().getID():found.add('same-target-lane')
    if a.getFrom()!=b.getFrom() and foe(a,b):found.add('foes')
  return sorted(found)
 def internal_bad(i):
  for a,b in itertools.combinations(groups[i],2):
   if a.getToLane()==b.getToLane() or (a.getFrom()!=b.getFrom() and foe(a,b)):return True
  return False
 links=sorted(groups);slots={}
 for i in links:
  c=t['links'][str(i)][0];slots.setdefault(c['approach']+'-'+c['turn'],set()).add(i)
 program=next(x for x in r.findall('tlLogic') if x.get('id')==tid)
 greens=[x.get('state') for x in program.findall('phase') if 'y' not in x.get('state') and 'G' in x.get('state')]
 declared=[v['state'] for v in t['actions'].values() if v.get('valid')]
 assert greens==declared,(tid,'phase-meta')
 menus=[];perphase=[]
 for name,a in t['actions'].items():
  if not a.get('valid'):continue
  state=a['state'];gs={i for i in links if state[i]=='G'};menus.append(gs)
  assert all(state[i] in 'rG' for i in links)
  unsafe=[(i,j,reasons(i,j)) for i,j in itertools.combinations(gs,2) if reasons(i,j)]
  self_bad=[i for i in gs if internal_bad(i)]
  if unsafe or self_bad:report['unsafe'].append(dict(tls=tid,action=name,unsafe=unsafe,self_bad=self_bad))
  can_add=[i for i in links if i not in gs and not internal_bad(i) and all(not reasons(i,j) for j in gs)]
  movement_add=[slot for slot,idxs in slots.items() if not idxs&gs and not any(internal_bad(i) for i in idxs)
                and all(not reasons(i,j) for i,j in itertools.combinations(idxs|gs,2))]
  partial=[(slot,sorted(idxs&gs),sorted(idxs-gs)) for slot,idxs in slots.items() if idxs&gs and not idxs<=gs]
  if partial:report['partial_slots'].append(dict(tls=tid,action=name,partial=partial))
  row=dict(action=name,state=state,green_links=sorted(gs),green_movements=[s for s,k in slots.items() if k&gs],can_add_links=can_add,can_add_whole_movements=movement_add)
  perphase.append(row)
  if can_add:report['missing_extensions'].append(dict(tls=tid,**row,additions=[t['links'][str(i)] for i in can_add]))
 orphan=[i for i in links if not any(i in g for g in menus)];assert not orphan,(tid,orphan)
 graph=nx.Graph();graph.add_nodes_from(links)
 graph.add_edges_from((i,j) for i,j in itertools.combinations(links,2) if not reasons(i,j))
 maximal=[set(x) for x in nx.find_cliques(graph)]
 absent=[sorted(x) for x in maximal if x not in menus]
 report['tls'][tid]=dict(links=len(links),connections=sum(map(len,groups.values())),phases=perphase,
                         maximal_sets=len(maximal),missing_maximal_sets=absent,
                         always_green_movements=[s for s,idxs in slots.items() if all(idxs<=g for g in menus)])
(OUT/'phase_closure_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'tls':len(m),'phases':sum(len(x['phases']) for x in report['tls'].values()),'unsafe':report['unsafe'],'missing_extensions':report['missing_extensions'],'partial_slots':report['partial_slots'],'missing_alternative_phase_counts':{tid:len(x['missing_maximal_sets']) for tid,x in report['tls'].items() if x['missing_maximal_sets']}},indent=2))

assert not any(report[k] for k in ("unsafe", "missing_extensions", "partial_slots", "metadata_errors"))
