"""Behavior fingerprint probe: greedy eval, phase-hold/switch stats per controller.
Usage: probe_behavior.py dqn8std|moe"""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
mode = sys.argv[1]
CFGS = {"dqn8std": "configs/exp207_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095_rfloor.yaml",
        "moe": "configs/exp215_dublin11h_531_moe_enum.yaml",
        "frap": "configs/exp211_dublin11h_531_enumfrap.yaml"}
CFG = CFGS[mode]
cfg = yaml.safe_load(open(CFG))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.agents.dqn_agent_txw import DQN
obs_class = functools.partial(obsmod.PriorityMovementObservationFunction,
    fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
    priority_source=cfg["priority_source"], awt_cap=float(cfg["obs_awt_cap"]),
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

if mode == "dqn8std":
    sys.path.insert(0, os.path.join(REPO, "experiments", "tools", "kan"))
    from extract_dqn8std_targets import load_meta_tables
    ts_mask, std2green, green2std, turnmap, A = load_meta_tables(cfg["action_meta_file"])
    env.reset(int(cfg["eval_seed"]))
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
    def act(states):
        out = {}
        for t in env.ts_ids:
            x = torch.tensor(np.asarray(states[t], dtype=np.float32)).unsqueeze(0)
            with torch.no_grad(): q = agent.q_net(x)[0].numpy()
            a = int(np.where(ts_mask[t], q, -np.inf).argmax())
            out[t] = int(std2green[t][a])
        return out
elif mode == "frap":
    from frap_glue import load_enum_tables, build_frap_agent
    enum_tables = load_enum_tables(cfg["enum_meta_file"])
    env.reset(int(cfg["eval_seed"]))
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]
        ts.observation_fn.rebind_movements(enum_tables["turnmap"][tid])
        ts.observation_space = ts.observation_fn.observation_space()
    states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
    agent = build_frap_agent(cfg, enum_tables, env, "cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval(); agent.epsilon = 0.0
    def act(states):
        return {t: int(agent.take_action(states[t], t)) for t in env.ts_ids}
else:
    import moe_glue
    moe = moe_glue.load_moe_tables(cfg["moe_meta_file"])
    env.reset(int(cfg["eval_seed"]))
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]
        ts.observation_fn.rebind_movements(moe["turnmap"][tid])
        ts.observation_space = ts.observation_fn.observation_space()
    states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
    experts = moe_glue.build_experts(cfg, moe, env)
    od = len(next(iter(states.values())))
    agent = DQN(starting_state=tuple([0.0]*od), state_space=od, hidden_dim=cfg["hidden_dim"],
        action_space=6, learning_rate=1e-3, gamma=0.95, epsilon=0.0, target_update=10,
        capacity=100, mini_size=10**9, batch_size=1, eps_start=0, eps_end=0, eps_decay=1, device="cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/2026-07-03T19-29-06/ckpt_ep*.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval()
    klog = []
    def act(states):
        out = {}
        for t in env.ts_ids:
            ts = env.traffic_signals[t]
            props, lv = experts.propose(t, ts.sumo, ts.green_phase)
            m = moe_glue.gate_mask(lv, bool(cfg.get("moe_lexicographic", False)),
                                   presence=bool(cfg.get("moe_presence_mask", False)))
            k = int(agent.take_action(states[t], mask=m))
            klog.append(k)
            out[t] = int(props[k])
        return out

prev = {t: env.traffic_signals[t].green_phase for t in env.ts_ids}
hold = {t: 1 for t in env.ts_ids}
holds = {t: [] for t in env.ts_ids}
switches = {t: 0 for t in env.ts_ids}
rew = []
done = {"__all__": False}; step = 0
while not done["__all__"]:
    actions = act(states)
    states, r, done, _ = env.step(action=actions)
    for t in env.ts_ids:
        g = env.traffic_signals[t].green_phase
        if g != prev[t]:
            holds[t].append(hold[t]); switches[t] += 1; hold[t] = 1
        else:
            hold[t] += 1
        prev[t] = g
    rew.append(float(np.mean([r[t] for t in env.ts_ids]))); step += 1
tele = int(env.sumo.simulation.getStartingTeleportNumber())
env.close()
allh = [h*5 for hs in holds.values() for h in hs]
out = {"mode": mode, "ckpt": os.path.basename(os.path.dirname(ck))+"/"+os.path.basename(ck),
       "teleports": tele, "steps": step, "ep_return": round(float(np.sum(rew)),2),
       "switch_rate": round(float(np.mean([s/(step-1) for s in switches.values()])),3),
       "green_hold_s": {"mean": round(float(np.mean(allh)),1), "p50": float(np.percentile(allh,50)),
                        "p90": float(np.percentile(allh,90)), "max": float(np.max(allh))}}
if mode == "moe":
    out["gate_k_hist"] = np.bincount(klog, minlength=6).tolist()
print("PROBE", json.dumps(out))
