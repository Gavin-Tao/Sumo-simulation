"""Per-second, whole-network lane/source diagnostics; no control changes."""
import bisect
from collections import defaultdict
import xml.etree.ElementTree as ET


class LaneDebug:
    def __init__(self, sumo, route_files):
        self.sumo=sumo
        roots=[ET.parse(f).getroot() for f in route_files]
        defs={r.get('id'):r.get('edges').split() for rt in roots for r in rt.findall('route')}
        self.vehicle_origin={}; self.schedule=defaultdict(list); self.insertion=defaultdict(int)
        self.depart_delays=defaultdict(list)
        for rt in roots:
            for v in rt.findall('vehicle'):
                r=v.find('route');es=r.get('edges').split() if r is not None else defs[v.get('route')]
                lane=es[0]+'_'+v.get('departLane','0');tm=float(v.get('depart'))
                self.vehicle_origin[v.get('id')]=(lane,tm)
                self.schedule[lane].append(tm)
        for times in self.schedule.values():times.sort()
        self.control={}; self.links={}
        for tid in sumo.trafficlight.getIDList():
            for i,grp in enumerate(sumo.trafficlight.getControlledLinks(tid)):
                for fr,to,via in grp:
                    self.control[fr]=tid
                    self.links.setdefault((fr,to.rsplit('_',1)[0]),[]).append(i)
        self.data={};self.previous={};self.streak=defaultdict(int);self.tail_streak=defaultdict(int)
        self.green_streak=defaultdict(int);self.windows=defaultdict(lambda:defaultdict(lambda:[0,0]))
        self.pending_history=defaultdict(list);self.max_pending=defaultdict(int)
        self.collisions=0;self.teleports=0;self.seconds=0
        self.lanes=[l for l in sumo.lane.getIDList() if not l.startswith(':')]
        self.length={l:sumo.lane.getLength(l) for l in self.lanes}
    def step(self):
        s=self.sumo;now=s.simulation.getTime();self.seconds+=1
        self.collisions+=s.simulation.getCollidingVehiclesNumber()
        self.teleports+=s.simulation.getStartingTeleportNumber()
        for vid in s.simulation.getDepartedIDList():
            if vid in self.vehicle_origin:
                lane,planned=self.vehicle_origin[vid]
                self.insertion[lane]+=1;self.depart_delays[lane].append(now-planned)
        for lane in self.lanes:
            vids=set(s.lane.getLastStepVehicleIDs(lane)); old=self.previous.get(lane,set())
            if not vids and not old and lane not in self.schedule:continue
            d=self.data.setdefault(lane,dict(length=self.length[lane],tls=self.control.get(lane),source=lane in self.schedule,
                   queue_seconds=0,tail_near_entry_seconds=0,max_vehicles=0,max_queued=0,longest_queue=0,
                   longest_tail_near_entry=0,head_green_seconds=0,head_green_stopped=0,longest_head_green_stopped=0,
                   arrivals=0,departures=0,end_vehicles=0))
            d['arrivals']+=len(vids-old);d['departures']+=len(old-vids);self.previous[lane]=vids;d['end_vehicles']=len(vids)
            speeds={v:s.vehicle.getSpeed(v) for v in vids};stopped=[v for v in vids if speeds[v]<.1]
            positions={v:s.vehicle.getLanePosition(v) for v in vids}
            tail=bool(stopped) and min(positions[v] for v in stopped)<=10
            d['queue_seconds']+=bool(stopped);d['tail_near_entry_seconds']+=tail
            d['max_vehicles']=max(d['max_vehicles'],len(vids));d['max_queued']=max(d['max_queued'],len(stopped))
            self.streak[lane]=self.streak[lane]+1 if stopped else 0
            self.tail_streak[lane]=self.tail_streak[lane]+1 if tail else 0
            d['longest_queue']=max(d['longest_queue'],self.streak[lane]);d['longest_tail_near_entry']=max(d['longest_tail_near_entry'],self.tail_streak[lane])
            blocked=False
            if vids and lane in self.control:
                head=max(vids,key=positions.get);r=s.vehicle.getRoute(head);i=s.vehicle.getRouteIndex(head)
                nxt=r[i+1] if i+1<len(r) else None;state=s.trafficlight.getRedYellowGreenState(self.control[lane])
                green=any(state[k] in 'Gg' for k in self.links.get((lane,nxt),[]))
                blocked=green and speeds[head]<.1 and self.length[lane]-positions[head]<15
                d['head_green_seconds']+=green;d['head_green_stopped']+=blocked
            self.green_streak[lane]=self.green_streak[lane]+1 if blocked else 0
            d['longest_head_green_stopped']=max(d['longest_head_green_stopped'],self.green_streak[lane])
            win=int((now-1)//600);self.windows[lane][win][0]+=len(vids);self.windows[lane][win][1]+=len(stopped)
        for lane,times in self.schedule.items():
            pending=bisect.bisect_right(times,now)-self.insertion[lane];self.max_pending[lane]=max(self.max_pending[lane],pending)
            if now%600==0 or now==3300:self.pending_history[lane].append([now,pending])
    def result(self):
        for lane,d in self.data.items():
            d['mean_vehicles_10min']=[self.windows[lane][k][0]/600 for k in range(6)]
            d['mean_queued_10min']=[self.windows[lane][k][1]/600 for k in range(6)]
            if lane in self.schedule:
                ds=self.depart_delays[lane]
                d.update(planned=len(self.schedule[lane]),inserted=self.insertion[lane],max_pending=self.max_pending[lane],
                         pending_samples=self.pending_history[lane],depart_delay_mean=sum(ds)/max(len(ds),1),depart_delay_max=max(ds,default=0))
        return dict(seconds=self.seconds,collisions=self.collisions,teleports=self.teleports,lanes=self.data)

