"""exp219 multihead: decision-time weight sweep on ONE trained policy.
Each weight vector -> greedy eval episode -> (car,bus,amb) per-visit profile."""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
cfg = yaml.safe_load(open("configs/exp219_1x1_531_NS20bus_multihead_bnf.yaml"))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from multihead_glue import build_multihead_agent
obs_class = functools.partial(obsmod.PriorityMovementObservationFunction,
    fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
    priority_source=cfg["priority_source"])
SWEEP = [("w421_gated", [1,2,2,4,4]),
         ("w351_gated", [1,2,5,4,3])]
GATED = True
for tag, w in SWEEP:
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
        num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
        max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
        single_agent=False, yellow_time=cfg["yellow_time"], delta_time=cfg["delta_time"],
        reward_fn=make_priority_avg_waiting_reward(load_priority_table(cfg["priority_source"])),
        observation_class=obs_class, sumo_seed=cfg["seed"], sumo_warnings=False)
    env.reset(int(cfg["eval_seed"]))
    tid = env.ts_ids[0]
    states = {tid: env.traffic_signals[tid].observation_fn()}
    od = len(states[tid])
    agent = build_multihead_agent(cfg, starting_state=tuple([0.0]*od),
                                  state_space=od, action_space=env.action_space.n, device="cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval(); agent.epsilon = 0.0
    agent.weights = np.asarray(w, dtype=np.float64)
    agent._w_t = torch.tensor(agent.weights, dtype=torch.float, device='cpu')
    mc = EpisodeMetricsCollector({tid: env.traffic_signals[tid].signal_controlled_lanes},
        delta_time=env.delta_time, excluded_lanes=set(env.traffic_signals[tid].always_green_lanes))
    LV = load_priority_table(cfg["priority_source"])
    lanes = env.traffic_signals[tid].signal_controlled_lanes
    base_w = np.asarray(w, dtype=np.float64)
    done = {"__all__": False}; rew = []
    while not done["__all__"]:
        mc.collect_step(env.sumo)
        present = {LV.get(env.sumo.vehicle.getTypeID(v), 1)
                   for l in lanes for v in env.sumo.lane.getLastStepVehicleIDs(l)}
        gw = np.array([base_w[k] if (k+1) in present else 0.0 for k in range(5)])
        if not gw.any(): gw = base_w
        agent._w_t = torch.tensor(gw, dtype=torch.float, device='cpu')
        a = int(agent.take_action(states[tid]))
        states, r, done, _ = env.step(action={tid: a})
        rew.append(float(r[tid]))
    mc.collect_step(env.sumo); mc.finalize(env.sumo)
    s = mc.summary()["system"]
    env.close()
    out = {"w": tag, "ep_return_531": round(float(np.sum(rew)),2)}
    for cls in ("car","bus","ambulance"):
        out[cls] = [round(s[cls]["avg_stop_events_per_visit"],3),
                    round(s[cls]["avg_stopped_time_per_visit"],3)]
    print("SWEEP", json.dumps(out), flush=True)
