"""Per-class metrics (wandb口径) for 207/211 with/without the empty-hold rule.
Usage: probe_guard_metrics.py {207|211} {base|guard}"""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
which, variant = sys.argv[1], sys.argv[2]
CFGS = {"207": "configs/exp207_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095_rfloor.yaml",
        "211": "configs/exp211_dublin11h_531_enumfrap.yaml"}
cfg = yaml.safe_load(open(CFGS[which]))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.agents.dqn_agent_txw import DQN
obs_class = functools.partial(obsmod.PriorityMovementObservationFunction,
    fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
    priority_source={"ambulance": 1, "bus": 3, "car": 1}, awt_cap=float(cfg["obs_awt_cap"]),
    awt_basis=cfg["obs_awt_basis"],
    include_downstream=bool(cfg.get("obs_downstream", False)),
    downstream_fields=tuple(cfg.get("obs_downstream_fields", ())),
    include_lane_occ=bool(cfg.get("obs_lane_occ", False)),
    slot_stats=str(cfg.get("obs_slot_stats", "intent")))
env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
    cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
    num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
    max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
    single_agent=False, yellow_time=cfg["yellow_time"], delta_time=cfg["delta_time"],
    reward_fn=make_priority_avg_waiting_reward(load_priority_table(cfg["priority_source"])),
    observation_class=obs_class, sumo_seed=cfg["seed"], sumo_warnings=False)
env.reset(int(cfg["eval_seed"]))
if which == "207":
    sys.path.insert(0, os.path.join(REPO, "experiments", "tools", "kan"))
    from extract_dqn8std_targets import load_meta_tables
    ts_mask, std2green, green2std, turnmap, A = load_meta_tables(cfg["action_meta_file"])
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]
        ts.std_action_map = green2std[tid]
        ts.observation_fn.rebind_movements(turnmap[tid])
        ts.observation_space = ts.observation_fn.observation_space()
    states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
    od = len(next(iter(states.values())))
    agent = DQN(starting_state=tuple([0.0]*od), state_space=od, hidden_dim=cfg["hidden_dim"],
        action_space=8, learning_rate=1e-3, gamma=0.95, epsilon=0.0, target_update=10,
        capacity=100, mini_size=10**9, batch_size=1, eps_start=0, eps_end=0, eps_decay=1, device="cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval()
    def rl_act(t):
        x = torch.tensor(np.asarray(states[t], dtype=np.float32)).unsqueeze(0)
        with torch.no_grad(): q = agent.q_net(x)[0].numpy()
        a = int(np.where(ts_mask[t], q, -np.inf).argmax())
        return int(std2green[t][a])
else:
    from frap_glue import load_enum_tables, build_frap_agent
    enum_tables = load_enum_tables(cfg["enum_meta_file"])
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]
        ts.observation_fn.rebind_movements(enum_tables["turnmap"][tid])
        ts.observation_space = ts.observation_fn.observation_space()
    states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
    agent = build_frap_agent(cfg, enum_tables, env, "cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval(); agent.epsilon = 0.0
    def rl_act(t):
        return int(agent.take_action(states[t], t))
ts_lane_map = {t: env.traffic_signals[t].signal_controlled_lanes for t in env.ts_ids}
always_green = set().union(*(env.traffic_signals[t].always_green_lanes for t in env.ts_ids))
approach_lanes = {t: [l for l in ts_lane_map[t]] for t in env.ts_ids}
mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green)
done = {"__all__": False}
while not done["__all__"]:
    mc.collect_step(env.sumo)
    acts = {}
    for t in env.ts_ids:
        if variant == "guard":
            n = sum(env.sumo.lane.getLastStepVehicleNumber(l) for l in approach_lanes[t])
            acts[t] = env.traffic_signals[t].green_phase if n == 0 else rl_act(t)
        else:
            acts[t] = rl_act(t)
    states, r, done, _ = env.step(action=acts)
mc.collect_step(env.sumo); mc.finalize(env.sumo)
s = mc.summary()["system"]
out = {"arm": f"{which}-{variant}-AMBBLIND"}
for cls in ("car", "bus", "ambulance"):
    out[cls] = {"stops/visit": round(s[cls]["avg_stop_events_per_visit"], 3),
                "stopped_s/visit": round(s[cls]["avg_stopped_time_per_visit"], 3)}
print("PROBE", json.dumps(out, ensure_ascii=False))
