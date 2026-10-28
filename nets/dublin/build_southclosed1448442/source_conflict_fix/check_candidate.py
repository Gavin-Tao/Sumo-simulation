"""Exercise current scene inputs on the candidate, keeping live files untouched."""
from pathlib import Path
import sys, os, json, xml.etree.ElementTree as ET

WORK = Path(__file__).resolve().parent
BASE = WORK.parent
REPO = WORK.parents[3]
NET = WORK / 'dublin_8std_splitslot15_southclosed.net.xml'
META = WORK / 'dublin_8std_splitslot15_southclosed_meta.json'
scene, policy, seed = sys.argv[1:4]
folder = REPO / 'nets/dublin' / scene
out = WORK / 'validation' / scene
if policy == 'baseline':
    NET = REPO / 'nets/dublin/dublin_8std_splitslot15_southclosed.net.xml'
    out = WORK / 'before' / 'validation' / scene
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO))
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')

if policy not in ('native', 'baseline'):
    code = (BASE / 'validate.py').read_text()
    code = code.replace('out = folder / "validation"', f'out = Path({str(out)!r})')
    code = code.replace('REPO / "nets/dublin/dublin_8std_splitslot15_southclosed.net.xml"', f'Path({str(NET)!r})')
    code = code.replace('REPO / "nets/dublin/dublin_8std_splitslot15_southclosed_meta.json"', f'Path({str(META)!r})')
    exec(compile(code, str(BASE / 'validate.py'), 'exec'), {'__name__': '__main__', '__file__': str(BASE / 'validate.py')})
else:
    import libsumo as s
    from lane_debug import LaneDebug
    cfg = next(folder.glob('*.sumocfg'))
    files = [str(folder / f) for f in ET.parse(cfg).getroot().find('input/route-files').get('value').split(',')]
    s.start(['sumo', '-c', str(cfg), '-n', str(NET), '--seed', seed,
             '--no-warnings', 'true', '--duration-log.statistics', 'false',
             '--collision.check-junctions', 'true', '--collision.action', 'warn'])
    debug = LaneDebug(s, files)
    turn = dict(stopped_vehicle_seconds=0, stopped_ids=set(), passed_ids=set())
    for t in range(3600):
        s.simulationStep(); debug.step()
        for lane in ('77616869#0_0', '77616869#0_1'):
            for v in s.lane.getLastStepVehicleIDs(lane):
                route = s.vehicle.getRoute(v); i = s.vehicle.getRouteIndex(v)
                if i + 1 < len(route) and route[i + 1] == '8080823#0':
                    turn['passed_ids'].add(v)
                    if s.vehicle.getSpeed(v) < .1:
                        turn['stopped_vehicle_seconds'] += 1
                        turn['stopped_ids'].add(v)
    result = debug.result()
    result['outlet_turn'] = {k: len(v) if isinstance(v, set) else v for k, v in turn.items()}
    collisions = teleports = 0
    while s.simulation.getMinExpectedNumber() and s.simulation.getTime() < 7200:
        s.simulationStep()
        collisions += s.simulation.getCollidingVehiclesNumber()
        teleports += s.simulation.getStartingTeleportNumber()
    result['drain'] = dict(cleared=s.simulation.getMinExpectedNumber() == 0,
                           time=s.simulation.getTime(), collisions=collisions, teleports=teleports)
    s.close()
    assert result['collisions'] == result['teleports'] == collisions == teleports == 0
    assert result['drain']['cleared']
    (out / f'native_s{seed}_lanes.json').write_text(json.dumps(result, indent=2) + '\n')
    print(scene, seed, result['outlet_turn'], result['drain'])
