"""Read-only one-hour native-GUI-program diagnostic for selected injection lanes."""
from pathlib import Path
from collections import Counter
import bisect
import json
import sys
import xml.etree.ElementTree as ET
import libsumo as sumo

scene = sys.argv[1]
root = Path(__file__).resolve().parents[3]
folder = root/'nets/dublin'/scene
assert folder.name in ('weekday_pairs50_15_southclosed','weekday_18h_15_southclosed')
cfg = next(folder.glob('*.sumocfg'))
routefiles = ET.parse(cfg).getroot().find('input/route-files').get('value').split(',')
roots = [ET.parse(folder / f).getroot() for f in routefiles]
defs = {r.get('id'):r.get('edges').split() for doc in roots for r in doc.findall('route')}
lanes = ('-77616869#2_0', '317003251#0_0')
planned = {lane:{} for lane in lanes}
for doc in roots:
    for v in doc.findall('vehicle'):
        route = v.find('route')
        es = route.get('edges').split() if route is not None else defs[v.get('route')]
        lane = es[0]+'_'+v.get('departLane','0')
        if lane in planned:
            planned[lane][v.get('id')] = float(v.get('depart'))
sumo.start(['sumo','-c',str(cfg),'--seed','123','--no-warnings','true','--duration-log.statistics','false'])
data = {}; rows = {}; arrived = set()
for lane in lanes:
    tls = next(tid for tid in sumo.trafficlight.getIDList() if lane in sumo.trafficlight.getControlledLanes(tid))
    links = {}
    for i, grp in enumerate(sumo.trafficlight.getControlledLinks(tls)):
        for fr, to, via in grp:
            if fr == lane: links.setdefault(to.rsplit('_',1)[0], []).append(i)
    data[lane] = dict(tls=tls, length=sumo.lane.getLength(lane), planned=len(planned[lane]), inserted=0,
                     injection_delay=[], discharged=0, ever_seen=set(), previous=set(), queue_streak=0,
                     longest_continuous_queue=0, queue_seconds=0, empty_queue_seconds=0,
                     queue_to_entry_seconds=0, head_green_seconds=0, head_green_stopped_seconds=0,
                     head_red_seconds=0, max_stopped=0, max_vehicles=0, max_pending=0, links=links,
                     planned_times=sorted(planned[lane].values()), windows=[], pending_samples=[])
    rows[lane] = []
for now in range(1,3601):
    sumo.simulationStep()
    for v in sumo.simulation.getDepartedIDList():
        for lane in lanes:
            if v in planned[lane]:
                data[lane]['inserted'] += 1
                data[lane]['injection_delay'].append(now - planned[lane][v])
    for lane in lanes:
        d = data[lane]; vids = set(sumo.lane.getLastStepVehicleIDs(lane)); d['ever_seen'].update(vids)
        d['discharged'] += len(d['previous'] - vids); d['previous'] = vids
        stopped = [v for v in vids if sumo.vehicle.getSpeed(v) < .1]
        d['max_stopped'] = max(d['max_stopped'],len(stopped)); d['max_vehicles'] = max(d['max_vehicles'],len(vids))
        d['queue_seconds'] += bool(stopped); d['empty_queue_seconds'] += not stopped
        d['queue_streak'] = d['queue_streak'] + 1 if stopped else 0
        d['longest_continuous_queue'] = max(d['longest_continuous_queue'],d['queue_streak'])
        d['queue_to_entry_seconds'] += bool(stopped) and min(sumo.vehicle.getLanePosition(v) for v in stopped) <= 10
        pending = bisect.bisect_right(d['planned_times'],now) - d['inserted']
        d['max_pending'] = max(d['max_pending'],pending)
        if now in (600,1200,1800,2400,3000,3300,3600):d['pending_samples'].append([now,pending])
        headgreen = False; blockedgreen = False
        if vids:
            head = max(vids,key=sumo.vehicle.getLanePosition)
            route = sumo.vehicle.getRoute(head); ri = sumo.vehicle.getRouteIndex(head)
            nxt = route[ri+1] if ri+1 < len(route) else None
            light = sumo.trafficlight.getRedYellowGreenState(d['tls'])
            headgreen = any(light[i] in 'Gg' for i in d['links'].get(nxt,[]))
            # Count only an actual stopped head near its stop line, not travel in a green approach.
            blockedgreen = headgreen and sumo.vehicle.getSpeed(head)<.1 and d['length']-sumo.vehicle.getLanePosition(head)<15
            d['head_green_seconds'] += headgreen; d['head_green_stopped_seconds'] += blockedgreen
            d['head_red_seconds'] += not headgreen
        rows[lane].append((len(vids),len(stopped),pending,int(headgreen),int(blockedgreen)))
sumo.close()
for lane,d in data.items():
    for k in range(6):
        rr=rows[lane][k*600:(k+1)*600]
        d['windows'].append(dict(time=[k*600,(k+1)*600],mean_vehicles=sum(r[0] for r in rr)/600,
                                mean_queued=sum(r[1] for r in rr)/600,end_pending=rr[-1][2]))
    d['seen_vehicles']=len(d.pop('ever_seen'))
    d['delay_mean']=sum(d['injection_delay'])/max(len(d['injection_delay']),1)
    d['delay_max']=max(d['injection_delay'],default=0)
    for k in ('previous','queue_streak','links','planned_times','injection_delay'):d.pop(k)
out = dict(scenario=scene,controller='native net TLS program, as sumo-gui loading the sumocfg',seed=123,seconds=3600,lanes=data)
(folder/'validation/injection_native_s123.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
