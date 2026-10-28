"""One-hour, non-learning checks for the copied scenarios.

Run: python -B validate.py weekday_pairs50_15_southclosed random [123]
Uses the existing calibration script in memory, with additional assertions and
all output redirected into this variant's validation directory.
"""
from pathlib import Path
import sys
import os

REPO = Path(__file__).resolve().parents[3]
scene, policy = sys.argv[1:3]
seed = sys.argv[3] if len(sys.argv) > 3 else "123"
assert scene in ("weekday_pairs50_15_southclosed", "weekday_18h_15_southclosed")
assert policy in ("random", "fixed")
folder = REPO / "nets/dublin" / scene
out = folder / "validation"
out.mkdir(exist_ok=True)
import xml.etree.ElementTree as ET
cfgpath = next(folder.glob("*.sumocfg"))
cfgxml = ET.parse(cfgpath).getroot()
routes = ",".join(str(folder / f) for f in cfgxml.find("input/route-files").get("value").split(","))
prefix = out / f"{policy}_s{seed}"
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
sys.argv = ["scenario_check.py", "--net", str(REPO / "nets/dublin/dublin_8std_splitslot15_southclosed.net.xml"),
            "--meta", str(REPO / "nets/dublin/dublin_8std_splitslot15_southclosed_meta.json"),
            "--sumocfg", str(cfgpath), "--routes", routes, "--policy", policy,
            "--seed", seed, "--out", str(prefix), "--exp-types", "bus,ambulance"]
source_path = REPO / "experiments/tools/dublin/scenario/scenario_check.py"
source = source_path.read_text()
old_output = 'TRIP = f"{REPO}/experiments/logs/tripinfo_scen_{os.getpid()}.xml"; LC = f"{REPO}/experiments/logs/lanechange_scen_{os.getpid()}.xml"'
assert old_output in source
source = source.replace(old_output, f"TRIP = {str(prefix) + '_trip.xml'!r}; LC = {str(prefix) + '_lanechange.xml'!r}")
source = source.replace('additional_sumo_cmd=f"--tripinfo-output', 'waiting_time_memory=100000, additional_sumo_cmd=f"--tripinfo-output')
checks = r'''
# Independent long-window accumulated-wait increments; no tracker data used as
# the reference. The reference is reset when a vehicle leaves an inlet edge.
tracker.strict = True
from lane_debug import LaneDebug
lane_debug = LaneDebug(sumo, ROUTES.split(','))
reference = {}; previous_accum = {}
audit = dict(seconds=0, decisions=0, state_checks=0, reward_checks=0,
             collisions=0, teleports=0, tracker_checks=0, tracker_missing=0,
             south_farthest_from_stopline=0.0, south_upstream_wait_seen=False,
             south_capacity={}, arrived_detours=[])
for lane in ('532427444#7_0', '532427444#7_1'):
    audit['south_capacity'][lane] = sumo.lane.getLength(lane) / 7.5
assert abs(sumo.lane.getLength('532427444#7_0') - 110.24) < 1e-8
assert abs(sumo.lane.getLength('-532427444#3_0') - 110.04) < 1e-8
assert abs(sumo.lane.getLength('-304634278#0_0') - 88.24) < 1e-8
route_detours = json.load(open(os.path.join(os.path.dirname(CFG), 'modification_report.json')))['detours']
detour_ids = {d['id'] for d in route_detours if d['file'] == 'car_weekday_18h.rou.xml'}
def audit_second():
    lane_debug.step()
    audit['seconds'] += 1
    audit['collisions'] += sumo.simulation.getCollidingVehiclesNumber()
    audit['teleports'] += sumo.simulation.getStartingTeleportNumber()
    audit['arrived_detours'].extend(set(sumo.simulation.getArrivedIDList()) & detour_ids)
    current = {}; accum = {}
    for tid in ids:
        for lane in T[tid]['lanes']:
            edge = lane.rsplit('_', 1)[0]
            for v in sumo.lane.getLastStepVehicleIDs(lane):
                key = (tid, v, edge)
                value = sumo.vehicle.getAccumulatedWaitingTime(v)
                if key in reference:
                    delta = value - previous_accum[key]
                    assert -1e-8 <= delta <= tracker.dt + 1e-8, (key, delta)
                else:
                    delta = min(sumo.vehicle.getWaitingTime(v), tracker.dt)
                current[key] = reference.get(key, 0.0) + delta
                accum[key] = value
                assert abs(tracker.get(tid, v, lane) - current[key]) < 1e-8, key
                audit['tracker_checks'] += 1
                if tid == '1448442' and edge == '532427444#7':
                    distance = sumo.lane.getLength(lane) - sumo.vehicle.getLanePosition(v)
                    audit['south_farthest_from_stopline'] = max(audit['south_farthest_from_stopline'], distance)
                    if distance > 20 and current[key] > 0:
                        audit['south_upstream_wait_seen'] = True
    reference.clear(); reference.update(current)
    previous_accum.clear(); previous_accum.update(accum)

labels = [f'{approach}-{turn}' for approach in ('N','E','S','W') for turn in ('L','T','R')]
prio_table = cfg['priority_source']
def audit_decision(states, rewards):
    audit['decisions'] += 1
    for tid in ids:
        ts = env.traffic_signals[tid]
        totals = collections.defaultdict(list)
        stats = collections.defaultdict(list)
        lane_counts = {}
        for lane in T[tid]['lanes']:
            vids = sumo.lane.getLastStepVehicleIDs(lane); lane_counts[lane] = len(vids)
            for v in vids:
                p = prio_table.get(sumo.vehicle.getTypeID(v), 1)
                w = reference[(tid, v, lane.rsplit('_', 1)[0])]
                totals[p].append(w)
                slot = slot_of(tid, v, lane)
                if slot is not None:
                    stats[(slot,p)].append((sumo.vehicle.getSpeed(v) < .1, w))
        expected_reward = -sum(p * sum(ws)/len(ws) for p,ws in totals.items()) / 100
        assert abs(expected_reward - rewards[tid]) < 1e-8, (tid, expected_reward, rewards[tid])
        audit['reward_checks'] += 1
        inlanes = collections.defaultdict(set); outlanes = collections.defaultdict(set)
        for grp in meta[tid]['links'].values():
            for c in grp:
                slot = name(c)
                inlanes[slot].add(c['from_edge']+'_'+str(c['from_lane']))
                outlanes[slot].add(c['to_edge']+'_'+str(c['to_lane']))
        expected = []
        mg = float(ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time)
        served = T[tid]['serves'][ts.green_phase]
        for slot in labels:
            cap = max(sum(sumo.lane.getLength(l)/7.5 for l in inlanes[slot]), 1)
            expected.append(float(slot in served))
            for p in range(1,6):
                rows = stats[(slot,p)]
                mean = sum(w for q,w in rows)/len(rows) if rows else 0
                expected.extend([len(rows)/cap, sum(q for q,w in rows)/cap, np.log1p(mean)/np.log1p(3600)])
            downstream = collections.defaultdict(list)
            for lane in outlanes[slot]:
                for v in sumo.lane.getLastStepVehicleIDs(lane):
                    p = prio_table.get(sumo.vehicle.getTypeID(v),1)
                    downstream[p].append(sumo.vehicle.getSpeed(v) < .1)
            outcap = max(sum(sumo.lane.getLength(l)/7.5 for l in outlanes[slot]), 1)
            expected.extend(len(downstream[p])/outcap for p in range(1,6))
            expected.extend(sum(downstream[p])/outcap for p in range(1,6))
            expected.append(max((lane_counts[l]/max(sumo.lane.getLength(l)/7.5,1) for l in inlanes[slot]), default=0))
            expected.append(mg if slot in served else 0)
        np.testing.assert_allclose(states[tid], np.asarray(expected,dtype=np.float32), rtol=1e-6, atol=1e-7,
                                   err_msg=f'{tid}, t={sumo.simulation.getTime()}')
        assert len(states[tid]) == 336
        audit['state_checks'] += 1
'''
assert 'tracker = env.visit_tracker' in source
source = source.replace('tracker = env.visit_tracker', 'tracker = env.visit_tracker\n' + checks)
source = source.replace('r = _orig_step(*args, **kwargs); per_second_hook(); return r',
                        'r = _orig_step(*args, **kwargs); audit_second(); per_second_hook(); return r')
source = source.replace('states, r, done, _ = env.step(action=acts)',
                        'states, r, done, _ = env.step(action=acts)\n    audit_decision(states, r)')
source = source.replace('env.close()', '''audit['tracker_missing'] = tracker.missing
json.dump(lane_debug.result(), open(a.out + '_lanes.json', 'w'), indent=2)
audit['unfinished_detours'] = {}
active = set(sumo.vehicle.getIDList())
pending = set(sumo.simulation.getPendingVehicles())
for vid in detour_ids - set(audit['arrived_detours']):
    if vid in active:
        audit['unfinished_detours'][vid] = dict(status='in_network', lane=sumo.vehicle.getLaneID(vid),
                                               route_index=sumo.vehicle.getRouteIndex(vid),
                                               route_edges=len(sumo.vehicle.getRoute(vid)))
    else:
        audit['unfinished_detours'][vid] = dict(status='pending' if vid in pending else 'not_present')
assert tracker.missing == 0
assert audit['collisions'] == 0, audit
assert audit['teleports'] == 0, audit
assert audit['seconds'] == 3600 and audit['decisions'] == 720, audit
assert audit['south_farthest_from_stopline'] > 80, audit
json.dump(audit, open(a.out + '_audit.json', 'w'), indent=2)
env.close()''')
exec(compile(source, str(source_path), 'exec'), {'__name__': '__main__', '__file__': str(source_path)})
