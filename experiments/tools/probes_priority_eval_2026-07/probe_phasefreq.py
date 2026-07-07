"""Phase-selection frequency at convergence: {217|220|207|211}, best.pth,
greedy, eval seed. Counts the CHOSEN target green phase per decision step per
TLS; reports share vector, normalized entropy, top1 share, dead phases."""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
which = sys.argv[1]
CFGS = {"217": "configs/exp217_1x1_531_NS20bus_vanilla_B_movement_legacy_cqm.yaml",
        "220": "configs/exp220_1x1_531_NS20bus_enumfrap_cqm.yaml",
        "207": "configs/exp207_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095_rfloor.yaml",
        "211": "configs/exp211_dublin11h_531_enumfrap.yaml"}
cfg = yaml.safe_load(open(CFGS[which]))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.agents.dqn_agent_txw import DQN
obs_kwargs = dict(fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"])
if "priority_source" in cfg: obs_kwargs["priority_source"] = cfg["priority_source"]
for src, dst in [("obs_awt_cap","awt_cap"), ("obs_awt_basis","awt_basis"),
                 ("obs_downstream","include_downstream"), ("obs_lane_occ","include_lane_occ")]:
    if src in cfg: obs_kwargs[dst] = cfg[src]
if "obs_downstream_fields" in cfg: obs_kwargs["downstream_fields"] = tuple(cfg["obs_downstream_fields"])
if "obs_slot_stats" in cfg: obs_kwargs["slot_stats"] = str(cfg["obs_slot_stats"])
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
    K = {t: int(ts_mask[t].sum()) for t in env.ts_ids}   # valid std phases per TLS
    def act(t):
        x = torch.tensor(np.asarray(states[t], dtype=np.float32)).unsqueeze(0)
        with torch.no_grad(): q = agent.q_net(x)[0].numpy()
        a = int(np.where(ts_mask[t], q, -np.inf).argmax())
        return a   # count in std-action space (8std semantics)
    def to_green(t, a): return int(std2green[t][a])
elif which in ("211", "220"):
    from frap_glue import load_enum_tables, build_frap_agent
    tables = load_enum_tables(cfg["enum_meta_file"])
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]
        ts.observation_fn.rebind_movements(tables["turnmap"][tid])
        ts.observation_space = ts.observation_fn.observation_space()
    states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
    agent = build_frap_agent(cfg, tables, env, "cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval(); agent.epsilon = 0.0
    K = {t: int(tables["tls"][t]["mask"].sum()) for t in env.ts_ids}
    def act(t): return int(agent.take_action(states[t], t))
    def to_green(t, a): return a
else:  # 217 native 4-phase DQN
    states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
    od = len(next(iter(states.values())))
    A = env.action_space.n
    agent = DQN(starting_state=tuple([0.0]*od), state_space=od, hidden_dim=cfg["hidden_dim"],
        action_space=A, learning_rate=1e-3, gamma=0.99, epsilon=0.0, target_update=10,
        capacity=100, mini_size=10**9, batch_size=1, eps_start=0, eps_end=0, eps_decay=1, device="cpu")
    ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
    agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval()
    K = {t: env.action_space.n for t in env.ts_ids}
    def act(t): return int(agent.take_action(states[t]))
    def to_green(t, a): return a

counts = {t: {} for t in env.ts_ids}
done = {"__all__": False}; step = 0
while not done["__all__"]:
    acts = {}
    for t in env.ts_ids:
        a = act(t)
        counts[t][a] = counts[t].get(a, 0) + 1
        acts[t] = to_green(t, a)
    states, r, done, _ = env.step(action=acts)
    step += 1
env.close()

out = {"arm": which, "steps": step, "tls": {}}
for t in env.ts_ids:
    k = K[t]
    vec = np.zeros(k)
    for a, c in counts[t].items():
        if a < k: vec[a] = c
    share = vec / vec.sum()
    nz = share[share > 0]
    H = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    Hmax = float(np.log(k)) if k > 1 else 1.0
    out["tls"][t] = {"K": int(k), "share": [round(float(x), 3) for x in share],
                     "H_norm": round(H / Hmax, 3),
                     "top1": round(float(share.max()), 3),
                     "dead": int((vec == 0).sum()),
                     "low5pct": int(((share > 0) & (share < 0.05)).sum())}
print("PHASEFREQ", json.dumps(out, ensure_ascii=False))
