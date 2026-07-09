"""Multi-seed J robustness probe for enum-frap 1x1 arms (read-only analysis).
Usage: probe_arm_eval.py <config> <variant:base|w421b|swap351> <seed1,seed2,...>
Per seed: greedy 1-episode eval, per-class per-visit stopped time + J421/J351.
Generalizes probe_1x1.py's 220 branch to any enum-frap arm config."""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml

cfg_path, variant, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
cfg = yaml.safe_load(open(cfg_path))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from frap_glue import load_enum_tables, build_frap_agent

obs_kwargs = dict(fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"])
if variant == "w421b":
    obs_kwargs["priority_source"] = {"ambulance": 4, "bus": 2, "car": 1}
elif variant == "swap351":
    obs_kwargs["priority_source"] = {"ambulance": 4, "bus": 5, "truck": 2, "car": 1}
else:
    src = cfg.get("obs_priority_source", cfg.get("priority_source"))
    if src: obs_kwargs["priority_source"] = src
for s_, d_ in [("obs_awt_cap", "awt_cap"), ("obs_awt_basis", "awt_basis")]:
    if s_ in cfg: obs_kwargs[d_] = cfg[s_]
obs_class = functools.partial(obsmod.PriorityMovementObservationFunction, **obs_kwargs)
rf = cfg["reward_fn"]
if rf == "priority-avg-waiting":
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    from sumo_rl.environment.priority_map import load_priority_table
    rf = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))

env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
    cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
    num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
    max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
    single_agent=False, yellow_time=cfg["yellow_time"], delta_time=cfg["delta_time"],
    reward_fn=rf, observation_class=obs_class, sumo_seed=cfg["seed"], sumo_warnings=False)

rows = []
for sd in seeds:
    env.reset(sd)
    tid = env.ts_ids[0]
    ts = env.traffic_signals[tid]
    tables = load_enum_tables(cfg["enum_meta_file"])
    ts.observation_fn.rebind_movements(tables["turnmap"][tid])
    ts.observation_space = ts.observation_fn.observation_space()
    states = {tid: ts.observation_fn()}
    agent = build_frap_agent(cfg, tables, env, "cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval(); agent.epsilon = 0.0
    ts_lane_map = {tid: ts.signal_controlled_lanes}
    always_green = set(ts.always_green_lanes)
    mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time,
                                 excluded_lanes=always_green)
    done = {"__all__": False}
    while not done["__all__"]:
        mc.collect_step(env.sumo)
        a = int(agent.take_action(states[tid], tid))
        states, r, done, _ = env.step(action={tid: a})
    mc.collect_step(env.sumo); mc.finalize(env.sumo)
    s = mc.summary()["system"]
    a5 = s["ambulance"]["avg_stopped_time_per_visit"]
    b5 = s["bus"]["avg_stopped_time_per_visit"]
    c5 = s["car"]["avg_stopped_time_per_visit"]
    nA = s["ambulance"].get("n_visits", s["ambulance"].get("n", -1))
    rows.append({"seed": sd, "amb": round(a5, 2), "bus": round(b5, 2), "car": round(c5, 2),
                 "J421": round(4 * a5 + 2 * b5 + c5, 1), "J351": round(3 * a5 + 5 * b5 + c5, 1),
                 "n_amb": nA})
    print("SEEDROW", json.dumps(rows[-1]))
env.close()
J421 = [r["J421"] for r in rows]; J351 = [r["J351"] for r in rows]
print("SUMMARY", json.dumps({"cfg": os.path.basename(cfg_path), "variant": variant,
    "J421_mean": round(float(np.mean(J421)), 1), "J421_min": min(J421), "J421_max": max(J421),
    "J351_mean": round(float(np.mean(J351)), 1), "J351_min": min(J351), "J351_max": max(J351)}))
