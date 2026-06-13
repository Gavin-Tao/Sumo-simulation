"""Fixed-time baseline over N seeds, same EpisodeMetricsCollector口径 as RL eval.
Usage: python3 tools/fixed_baseline.py <config> <N> <seed_base>
Dumps /tmp/fixed_baseline.json. Pure analysis."""
import sys, os, json
import numpy as np
os.environ.setdefault('SUMO_RL_LIBSUMO', '1')
sys.path.insert(0, '/home/xiaowen/sumo-rl')
import yaml
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.metrics import EpisodeMetricsCollector

cfg_path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
seed_base = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
c = yaml.safe_load(open(cfg_path))

CLASSES = ["all", "car", "bus", "ambulance"]
METRICS = ["avg_speed", "avg_stopped_time", "avg_stopped_time_per_visit",
           "avg_stop_events_per_visit", "avg_waiting_inc_per_visit", "completion_rate"]
rows = []
for i in range(N):
    seed = seed_base + i
    env = SumoEnvironment(net_file=c['net_file'], route_file=c['route_file'], cfg_file=c['cfg_file'],
        out_csv_name=None, use_gui=False, num_seconds=c.get('num_seconds', 3600),
        delta_time=5, yellow_time=2, min_green=5, max_green=50, use_max_green=True,
        single_agent=False, sumo_seed=seed, fixed_ts=True)
    env.reset(seed)
    exc = set().union(*(getattr(env.traffic_signals[ts], 'always_green_lanes', set()) for ts in env.ts_ids))
    mc = EpisodeMetricsCollector({ts: env.traffic_signals[ts].signal_controlled_lanes for ts in env.ts_ids},
                                 delta_time=5, excluded_lanes=exc)
    done = {"__all__": False}; steps = 0
    while not done["__all__"]:
        if steps >= 20:
            mc.collect_step(env.sumo)
        out = env.step({}); done = out[2]; steps += 1
    mc.collect_step(env.sumo); mc.finalize(env.sumo)
    rows.append(mc.to_flat_dict())
    env.close()
    print(f"[fixed seed {seed}] done")

agg = {}
for cls in CLASSES:
    for m in METRICS:
        a = np.array([r.get(f"eval_system/{cls}/{m}", np.nan) for r in rows], float)
        a = a[~np.isnan(a)]
        agg[f"{cls}/{m}"] = [float(a.mean()) if len(a) else None,
                             float(a.std()) if len(a) else None]
json.dump(agg, open("/tmp/fixed_baseline.json", "w"))
print("saved /tmp/fixed_baseline.json")
for cls in CLASSES:
    pv = agg[f"{cls}/avg_stopped_time_per_visit"][0]
    sp = agg[f"{cls}/avg_speed"][0]
    print(f"  {cls:10} speed={sp:.2f} stop/visit={pv:.2f}")
