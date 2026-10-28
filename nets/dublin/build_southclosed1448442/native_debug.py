"""Run the complete scenario with native SUMO-GUI timing and all-lane diagnostics."""
from pathlib import Path
import sys,json
import xml.etree.ElementTree as ET
import libsumo as s
from lane_debug import LaneDebug
root=Path(__file__).resolve().parents[3];scene=sys.argv[1];seed=int(sys.argv[2]) if len(sys.argv)>2 else 123
folder=root/'nets/dublin'/scene;cfg=next(folder.glob('*.sumocfg'))
files=[str(folder/f) for f in ET.parse(cfg).getroot().find('input/route-files').get('value').split(',')]
s.start(['sumo','-c',str(cfg),'--seed',str(seed),'--no-warnings','true','--duration-log.statistics','false'])
d=LaneDebug(s,files)
for t in range(3600):s.simulationStep();d.step()
result=d.result()
if '--drain' in sys.argv[3:]:
    collisions=0;teleports=0
    while s.simulation.getMinExpectedNumber()>0 and s.simulation.getTime()<7200:
        s.simulationStep()
        collisions+=s.simulation.getCollidingVehiclesNumber()
        teleports+=s.simulation.getStartingTeleportNumber()
    result['drain']=dict(cleared=s.simulation.getMinExpectedNumber()==0,time=s.simulation.getTime(),
                         remaining=s.vehicle.getIDCount(),pending=len(s.simulation.getPendingVehicles()),
                         collisions=collisions,teleports=teleports)
    assert result['drain']['cleared'] and collisions==0 and teleports==0,result['drain']
s.close()
assert result['collisions']==result['teleports']==0
out=folder/'validation';out.mkdir(exist_ok=True)
(out/f'native_s{seed}_lanes.json').write_text(json.dumps(result,indent=2)+'\n')
print(scene,seed,'sources pending at end',sum(x.get('planned',0)-x.get('inserted',0) for x in result['lanes'].values()))

