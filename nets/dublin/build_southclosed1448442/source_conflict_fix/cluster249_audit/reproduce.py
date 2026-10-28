import xml.etree.ElementTree as E
from pathlib import Path
import json,math,os,sys
import libsumo as s
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
base=Path('/home/xiaowen/sumo-rl/nets/dublin');tid='cluster_2498303858_3624079904_6141082777_6141082778_9936781';internal=':'+tid+'_3_0'
def poly(v):
 x,y=s.vehicle.getPosition(v);a=math.radians(s.vehicle.getAngle(v));ux,uy=math.sin(a),math.cos(a);w=s.vehicle.getWidth(v)/2;l=s.vehicle.getLength(v)
 return [(x+uy*w,y-ux*w),(x-uy*w,y+ux*w),(x-l*ux-uy*w,y-l*uy+ux*w),(x-l*ux+uy*w,y-l*uy-ux*w)]
def overlaps(a,b):
 for pts in [a,b]:
  for p,q in zip(pts,pts[1:]+pts[:1]):
   axis=(p[1]-q[1],q[0]-p[0]);aa=[x*axis[0]+y*axis[1] for x,y in a];bb=[x*axis[0]+y*axis[1] for x,y in b]
   if max(aa)<=min(bb) or max(bb)<=min(aa):return False
 return True
scene=sys.argv[1];stress=scene=='stress';cfg=next((base/'weekday_18h_15_southclosed').glob('*.sumocfg')) if stress else next((base/scene).glob('*.sumocfg'))
if not stress:
 rr=E.parse(cfg).getroot();fs=[E.parse(cfg.parent/f).getroot() for f in rr.find('input/route-files').get('value').split(',')]; routes={x.get('id'):x.get('edges').split() for r in fs for x in r.findall('route')}; counts={}
 for r in fs:
  for v in r.findall('vehicle'):
   rt=v.find('route');es=rt.get('edges').split() if rt is not None else routes[v.get('route')]
   for a,b in zip(es,es[1:]):
    if a=='52879956':counts[b]=counts.get(b,0)+1
 print('DEMAND',scene,counts,flush=True)
args=['sumo','-n',str(base/'dublin_8std_splitslot15_southclosed.net.xml'),'-a',str(cfg.parent/'vtypes_same.add.xml')] if stress else ['sumo','-c',str(cfg)]
args += ['-n', str(base/'build_southclosed1448442/source_conflict_fix/before/dublin_8std_splitslot15_southclosed.net.xml')]
s.start(args+['--seed','123','--no-step-log','true','--no-warnings','true','--collision.check-junctions','true','--collision.action','warn','--time-to-teleport','-1','--duration-log.statistics','false'])
if stress:
 s.route.add('test_straight',['52879956','357012332#0'])
 for i in range(100):s.vehicle.add('test_'+str(i),'test_straight',typeID='car',depart=str(i*2),departLane='0')
result={'scene':scene,'internal_visits':set(),'stopped_on_internal_seconds':0,'overlap_pair_seconds':0,'examples':[],'SUMO_collision_vehicles':0,'teleports':0,'nonG':0}
for t in range(500 if stress else 3600):
 s.simulationStep();result['SUMO_collision_vehicles']+=s.simulation.getCollidingVehiclesNumber();result['teleports']+=s.simulation.getStartingTeleportNumber()
 if s.trafficlight.getRedYellowGreenState(tid)[3]!='G':result['nonG']+=1
 a=s.lane.getLastStepVehicleIDs(internal);b=s.lane.getLastStepVehicleIDs('52879956_0');result['internal_visits'].update(a)
 for v in a:
  result['stopped_on_internal_seconds']+=s.vehicle.getSpeed(v)<.1
  for u in b:
   if v==u:continue
   if overlaps(poly(v),poly(u)):
    result['overlap_pair_seconds']+=1
    if len(result['examples'])<3:result['examples'].append({'time':s.simulation.getTime(),'vehicles':[v,u],'positions':[s.vehicle.getPosition(v),s.vehicle.getPosition(u)],'angles':[s.vehicle.getAngle(v),s.vehicle.getAngle(u)],'speeds':[s.vehicle.getSpeed(v),s.vehicle.getSpeed(u)]})
result['internal_visits']=len(result['internal_visits']);s.close();print(json.dumps(result),flush=True);Path('/tmp/cluster249_'+scene+'.json').write_text(json.dumps(result,indent=2))
