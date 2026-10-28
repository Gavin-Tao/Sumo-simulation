"""Generate explicit-source/exit demand; fit movement demand without extending every old trip."""
from pathlib import Path
from collections import Counter,defaultdict
import copy,json,math,random,shutil,sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix,csr_matrix,hstack,eye,vstack
from build_net import NETS,WORK,NET,META,SCENES,RENAMES,CLOSED,GEOMETRY_TLS,compress
from routing import LaneRoutes

MUST=[('1430847303','W-T'),('659802','S-R|S-T'),('1448436','S-R'),('26165126','W-R'),
      ('133307283','N-R'),('389281','W-R'),('389332','W-R'),('cluster_1448439_2464758361','S-R'),
      ('cluster_1998326407_2876012598_4781977262_658429','N-T'),(GEOMETRY_TLS,'S-R'),
      ('cluster_5166592137_5166592140_557198554_57744042','W-R'),('5166592132','S-R'),
      ('1448442','S-L|N-R|E-T|N-T'),('cluster_21620852_2383668110_2383668118_4009885977','S-L|S-R|W-T|S-T'),
      ('26165090','E-R|N-T')]

def write_xml(path,root):
    ET.indent(root,space='    ');ET.ElementTree(root).write(path,encoding='UTF-8',xml_declaration=True)

def main():
    scale=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
    router=LaneRoutes(NET,META,zonal_sources=True)
    pool=json.loads((WORK/'route_candidates.json').read_text())
    for r in pool:r['traversals']=[tuple(x) for x in r['traversals']]
    always=set()
    for tid,t in router.meta.items():
        greens=[a['state'] for a in t['actions'].values() if a.get('valid')]
        slots=defaultdict(set)
        for i,g in t['links'].items():
            for c in g:slots[c['approach']+'-'+c['turn']].add(int(i))
        for mv,indices in slots.items():
            if all(any(s[i] in 'Gg' for i in indices) for s in greens):always.add((tid,mv))

    # Supplement the candidate pool with the earlier long experimental routes.
    src=WORK/'before/weekday_pairs50_15_southclosed'
    route_defs={r.get('id'):r.get('edges').split() for r in ET.parse(src/'pairs50_routes.rou.xml').getroot()}
    lane_defs={v.get('route'):int(v.get('departLane')) for v in ET.parse(src/'pairs50_tt1.rou.xml').getroot()}
    for rid,es in route_defs.items():
        try:r=router.extend(compress(es),lane_defs[rid]);r['source_route']=rid;pool.append(r)
        except ValueError:pass
    uniq={}
    for r in pool:
        r['traversals']=[tuple(x) for x in r['traversals']]
        uniq[(tuple(r['edges']),r['lane'])]=r
    pool=list(uniq.values())
    exp_pool=[r for r in pool if 3<=len(r['traversals'])<=8 and len({t for t,m in r['traversals']})==len(r['traversals'])]
    selected=[];unmet=set(MUST);junctions=Counter();seen_moves=set()
    def covers(r,m):return any((m[0],mv) in r['traversals'] for mv in m[1].split('|'))
    for k in range(18):
        choices=[r for r in exp_pool if r not in selected]
        def score(r):
            trav=set(r['traversals']);ts={t for t,m in trav}
            return (100*sum(covers(r,m) for m in unmet)+20*sum(junctions[t]<2 for t in ts)
                    +3*len((trav-always)-seen_moves)+.3*len(trav)-2*r['minor']-.0002*r['length'])
        best=max(choices,key=score);selected.append(best)
        unmet={m for m in unmet if not covers(best,m)}
        junctions.update({t for t,m in best['traversals']});seen_moves.update(set(best['traversals'])-always)
        if k>=13 and not unmet and all(junctions[t]>=2 for t in router.meta):break
    assert not unmet,unmet
    assert all(junctions[t]>=2 for t in router.meta),junctions
    print('Experimental routes',len(selected),'junction coverage',dict(junctions),flush=True)

    def timetable(routes,seed):
        rng=random.Random(seed);indices=[i%len(routes) for i in range(50)];rng.shuffle(indices)
        vehicles=[]
        for i,ri in enumerate(indices):
            # Balanced order and small independent jitter, identical under 111/531/351.
            first=60+50*i+(0 if seed==1 else rng.uniform(-12,12))
            for typ,suffix,offset in [('bus','bus',0 if i%2==0 else 90),('ambulance','amb',90 if i%2==0 else 0)]:
                vehicles.append(dict(id=f'pair_{i+1:02d}_{suffix}',type=typ,depart=round(first+offset,1),
                                     route=f'r{ri+1:02d}',lane=routes[ri]['lane'],desc=routes[ri]))
        return sorted(vehicles,key=lambda x:x['depart'])

    pair_vehicles=timetable(selected,1)
    fixed_18=[]; source_18=WORK/'before/weekday_18h_15_southclosed'
    for fn in ['bus_weekday_18h.rou.xml','amb_weekday_18h_probe11h.rou.xml']:
        for v in ET.parse(source_18/fn).getroot().findall('vehicle'):
            es=compress(v.find('route').get('edges').split())
            assert not CLOSED & set(es)
            desc=router.extend(es,int(v.get('departLane','0')))
            fixed_18.append(dict(id=v.get('id'),type=v.get('type'),depart=float(v.get('depart')),lane=desc['lane'],desc=desc))

    # Initial background demands: designed competitive traffic vs 18h-derived shape.
    pair_targets=json.loads((src/'demand_targets.json').read_text())['targets']
    pair_targets[GEOMETRY_TLS]['S-L']=pair_targets[GEOMETRY_TLS].pop('S-T')
    pair_targets['26165090']['E-L']=0;pair_targets['389332']['E-L']=0
    # Short shared south and north approaches were the previous bottlenecks.
    pair_targets['cluster_21620852_2383668110_2383668118_4009885977'].update({'S-L':100,'S-R':40,'S-T':30})
    pair_targets['1448442'].update({'N-R':80,'N-T':60})
    oldmeta=json.load(open(WORK/'before/dublin_8std_splitslot15_southclosed_meta.json'))['tls']
    by_edge={}
    for tid,t in oldmeta.items():
        for grp in t['links'].values():
            for c in grp:
                mv=c['approach']+'-'+c['turn']
                if tid==GEOMETRY_TLS and mv=='S-T':mv='S-L'
                by_edge[(c['from_edge'],c['to_edge'])]=(tid,mv)
    counts18=Counter()
    for fn in ['car_weekday_18h.rou.xml','added_flows.rou.xml']:
        for v in ET.parse(source_18/fn).getroot().findall('vehicle'):
            es=v.find('route').get('edges').split()
            for a,b in zip(es,es[1:]):
                if (a,b) in by_edge:counts18[by_edge[a,b]]+=1
    counts18[('26165090','E-L')]=0;counts18[('389332','E-L')]=0
    targets18=defaultdict(dict)
    for (t,m),n in counts18.items():targets18[t][m]=n

    # Both scenes are regenerated. Fitting route counts keeps through-demand
    # from growing automatically when formerly short routes become longer.
    for source_name,fixed,targets in [(SCENES[0],fixed_18,targets18),(SCENES[1],pair_vehicles,pair_targets)]:
        folder=NETS/(source_name+'_corridor');folder.mkdir(exist_ok=True)
        shutil.copy2(WORK/'before'/source_name/'vtypes_same.add.xml',folder/'vtypes_same.add.xml')
        target={(t,m):float(n)*scale for t,d in targets.items() for m,n in d.items()}
        movements=sorted(set(router.movements.values())|set(target));mi={m:i for i,m in enumerate(movements)}
        routes=[r for r in pool if r['traversals'] and len(r['traversals'])<=8 and len(r['edges'])==len(set(r['edges']))]
        # Prefer unimpeded alternatives; retain minor links only where necessary.
        safe_cover={m for r in routes if r['minor']==0 for m in r['traversals']}
        routes=[r for r in routes if r['minor']==0 or any(m not in safe_cover and target.get(m,0)>0 for m in r['traversals'])]
        n=len(routes);nm=len(movements);A=lil_matrix((nm,n))
        for j,r in enumerate(routes):
            for m,num in Counter(r['traversals']).items():A[mi[m],j]=num
        A=A.tocsr();goal=np.array([target.get(m,0) for m in movements])
        # L1 demand error plus mild penalties for non-RL yielding and long trips.
        weights=np.array([1.0/max(40,goal[i]) if m not in always else .3/max(40,goal[i]) for i,m in enumerate(movements)])
        route_cost=np.array([.00001*r['length']/1000+.0004*r['minor'] for r in routes])
        cost=np.r_[route_cost,weights,weights]
        eq=hstack([A,-eye(nm),eye(nm)],format='csr')
        lane_use=defaultdict(Counter);fixed_lane=Counter()
        rl_inlets={l.getID() for t in router.meta for e in router.net.getNode(t).getIncoming() for l in e.getLanes()}
        for j,r in enumerate(routes):
            for lane in r['lanes']:
                if lane in rl_inlets:lane_use[lane][j]+=1
        for v in fixed:
            fixed_lane.update(l for l in v['desc']['lanes'] if l in rl_inlets)
        caps={'317003251#0_0':210,'4919535#1_0':220,'22962979_0':240,
              '-77616869#2_0':480,'-4074527#1_0':400}
        # Protect all single-lane approaches serving 5+ mutually exclusive phases.
        for tid,t in router.meta.items():
            if sum(t['mask'])>=5:
                for e in router.net.getNode(tid).getIncoming():
                    if len(e.getLanes())==1:caps.setdefault(e.getLanes()[0].getID(),240)
        ub=[];limits=[]
        for lane,cap in caps.items():
            if lane not in lane_use:continue
            row=lil_matrix((1,n+2*nm))
            for j,num in lane_use[lane].items():row[0,j]=num
            ub.append(row.tocsr());limits.append(max(0,cap-fixed_lane[lane]))
        source_use=defaultdict(list)
        for j,r in enumerate(routes):source_use[r['lanes'][0]].append(j)
        fixed_source=Counter(v['desc']['lanes'][0] for v in fixed)
        for lane,js in source_use.items():
            cap=300 if router.net.getLane(lane).getLength()<70 else 450
            row=lil_matrix((1,n+2*nm))
            for j in js:row[0,j]=1
            ub.append(row.tocsr());limits.append(max(0,cap-fixed_source[lane]))
        sol=linprog(cost,A_ub=vstack(ub),b_ub=limits,A_eq=eq,b_eq=goal,bounds=(0,None),method='highs')
        assert sol.success,sol.message
        amounts=np.floor(sol.x[:n]+.5).astype(int);real=A@amounts
        rng=random.Random(1028);background=[]
        for j,(route,num) in enumerate(zip(routes,amounts)):
            for k in range(int(num)):
                # Stratification avoids large accidental injection bursts.
                depart=(k+rng.random())*3600/max(num,1)
                background.append(dict(id=f'bg_{j:04d}_{k:04d}',type='car',depart=round(depart,1),lane=route['lane'],desc=route))
        def vehicles_root(vehicles,named=False):
            root=ET.Element('routes')
            for v in sorted(vehicles,key=lambda x:x['depart']):
                attrs=dict(id=v['id'],type=v['type'],depart=f"{v['depart']:.1f}",departLane=str(v['lane']),departSpeed='max')
                if named:attrs['route']=v['route']
                child=ET.SubElement(root,'vehicle',attrs)
                if not named:ET.SubElement(child,'route',edges=' '.join(v['desc']['edges']))
            return root
        write_xml(folder/'background_cars.rou.xml',vehicles_root(background))
        if 'pairs50' in source_name:
            rr=ET.Element('routes')
            for i,r in enumerate(selected,1):ET.SubElement(rr,'route',id=f'r{i:02d}',edges=' '.join(r['edges']))
            write_xml(folder/'pairs50_routes.rou.xml',rr)
            for seed in range(1,7):write_xml(folder/f'pairs50_tt{seed}.rou.xml',vehicles_root(timetable(selected,seed),True))
            route_files='background_cars.rou.xml,pairs50_routes.rou.xml,pairs50_tt1.rou.xml'
        else:
            write_xml(folder/'bus_weekday_18h.rou.xml',vehicles_root([v for v in fixed if v['type']=='bus']))
            write_xml(folder/'amb_weekday_18h_probe11h.rou.xml',vehicles_root([v for v in fixed if v['type']=='ambulance']))
            route_files='background_cars.rou.xml,bus_weekday_18h.rou.xml,amb_weekday_18h_probe11h.rou.xml'
        cfg=ET.Element('configuration');inp=ET.SubElement(cfg,'input')
        ET.SubElement(inp,'net-file',value='../'+NET.name);ET.SubElement(inp,'additional-files',value='vtypes_same.add.xml');ET.SubElement(inp,'route-files',value=route_files)
        tm=ET.SubElement(cfg,'time');ET.SubElement(tm,'begin',value='0');ET.SubElement(tm,'end',value='3600')
        proc=ET.SubElement(cfg,'processing');ET.SubElement(proc,'ignore-junction-blocker',value='60');ET.SubElement(proc,'time-to-teleport',value='-1')
        rep=ET.SubElement(cfg,'report');ET.SubElement(rep,'no-step-log',value='true');ET.SubElement(rep,'duration-log.statistics',value='true')
        write_xml(folder/'scenario.sumocfg',cfg)
        by_fixed=Counter(m for v in fixed for m in v['desc']['traversals'])
        report=dict(version=1,scale=scale,background_vehicles=len(background),fixed_by_type=dict(Counter(v['type'] for v in fixed)),
                    source_rule='boundary or non-RL-to-non-RL edge >=40m; never an RL inlet or outlet',
                    sink_rule='edge with no outgoing connection; no mid-network route termination',
                    source_lanes=dict(Counter(v['desc']['lanes'][0] for v in background+fixed)),
                    movement_demand={f'{m[0]}|{m[1]}':dict(target=target.get(m,0),background=int(real[i]),priority=by_fixed[m]) for i,m in enumerate(movements)},
                    required_pairs=[list(m) for m in MUST],pair_junction_routes=dict(junctions),
                    pair_routes=[{k:v for k,v in r.items() if k!='lanes'} for r in selected] if 'pairs50' in source_name else None,
                    nnls_note='L1 linear programming with explicit lane/source caps; no greedy top-up',
                    fixed_route_changes=[dict(id=v['id'],edges=v['desc']['edges']) for v in fixed])
        (folder/'generation_report.json').write_text(json.dumps(report,indent=2)+'\n')
        (folder/'demand_targets.json').write_text(json.dumps({'targets':targets,'scale':scale},indent=2)+'\n')
        print(folder.name,'background',len(background),'fixed',dict(Counter(v['type'] for v in fixed)),
              'target_visits',sum(goal),'realized_visits',sum(real),'large_deviation',sum(abs(real-goal)>np.maximum(10,.2*goal)),flush=True)
    (WORK/'pair_route_selection.json').write_text(json.dumps({'routes':selected,'junction_coverage':dict(junctions)},indent=2)+'\n')

if __name__=='__main__':main()
