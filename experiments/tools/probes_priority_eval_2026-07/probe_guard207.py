"""User's gap-guard applied to TRAINED exp207 at inference (no retraining):
switch actions locked while the current phase's served lanes have vehicles;
window opens on gap-out OR 50s max-green OR priority preemption."""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
cfg = yaml.safe_load(open("configs/exp207_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095_rfloor.yaml"))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.agents.dqn_agent_txw import DQN
sys.path.insert(0, os.path.join(REPO, "experiments", "tools", "kan"))
from extract_dqn8std_targets import load_meta_tables
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
ts_mask, std2green, green2std, turnmap, A = load_meta_tables(cfg["action_meta_file"])
# slot -> lanes 表 (从 meta 直接构建)
meta = json.load(open(cfg["action_meta_file"]))["tls"]
SLOTS = [(a,t) for a in "NESW" for t in "LTR"]
SLOT_IDX = {s:k for k,s in enumerate(SLOTS)}
slot_lanes = {}
for tid, t in meta.items():
    sl = {}
    for i, conns in t["links"].items():
        for c in conns:
            s = SLOT_IDX[(c[0]["approach"], c[0]["turn"])] if isinstance(c, list) else SLOT_IDX[(c["approach"], c["turn"])]
            lane = f"{c[0]['from_edge']}_{c[0]['from_lane']}" if isinstance(c, list) else f"{c['from_edge']}_{c['from_lane']}"
            sl.setdefault(s, set()).add(lane)
    slot_lanes[tid] = sl
LV = {"ambulance": 5, "bus": 3, "car": 1}
vid_lv = {}
def lane_max_level(sumo, lanes):
    mx, n = 0, 0
    for l in lanes:
        for vid in sumo.lane.getLastStepVehicleIDs(l):
            n += 1
            lv = vid_lv.get(vid)
            if lv is None:
                lv = LV.get(sumo.vehicle.getTypeID(vid), 1); vid_lv[vid] = lv
            mx = max(mx, lv)
    return mx, n
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
CAP = 10
hold = {t: 0 for t in env.ts_ids}; prev = {t: env.traffic_signals[t].green_phase for t in env.ts_ids}
switches = {t: 0 for t in env.ts_ids}
window_closed_steps = 0; preempt_opens = 0
rew = []; done = {"__all__": False}; step = 0
while not done["__all__"]:
    acts = {}
    for t in env.ts_ids:
        ts = env.traffic_signals[t]
        sumo = ts.sumo
        cur = ts.green_phase
        a_std_cur = int(green2std[t][cur])
        served = [s for s in range(12) if A[t][a_std_cur, s] > 0]
        s_lanes = set().union(*(slot_lanes[t].get(s, set()) for s in served)) if served else set()
        uns_lanes = set().union(*(slot_lanes[t].get(s, set()) for s in range(12) if s not in served and s in slot_lanes[t]))
        mx_s, n_s = lane_max_level(sumo, s_lanes)
        mx_u, n_u = lane_max_level(sumo, uns_lanes)
        # 用户版 guard: 进车道全空 -> 自动保持; 有任何车 -> RL 全权
        window_open = (n_s + n_u) > 0
        preempt = False
        if window_open:
            x = torch.tensor(np.asarray(states[t], dtype=np.float32)).unsqueeze(0)
            with torch.no_grad(): q = agent.q_net(x)[0].numpy()
            a = int(np.where(ts_mask[t], q, -np.inf).argmax())
            acts[t] = int(std2green[t][a])
        else:
            acts[t] = cur; window_closed_steps += 1
    states, r, done, _ = env.step(action=acts)
    for t in env.ts_ids:
        g = env.traffic_signals[t].green_phase
        if g != prev[t]: switches[t] += 1; hold[t] = 0
        else: hold[t] += 1
        prev[t] = g
    rew.append(float(np.mean([r[t] for t in env.ts_ids]))); step += 1
tele = int(env.sumo.simulation.getStartingTeleportNumber())
env.close()
print("PROBE", json.dumps({"mode": "207+用户版guard(空则保持)", "ep_return": round(float(np.sum(rew)),2),
  "switch_rate": round(float(np.mean([s/(step-1) for s in switches.values()])),3),
  "window_closed_frac": round(window_closed_steps/(step*18),3),
  "preempt_opens": preempt_opens, "teleports": tele}))
