"""Amb stop-mechanism forensics on the eval episode (seed=123, best.pth, greedy).
For every ambulance, every 5s decision step: if stopped, classify
  denied_green  — on a TLS-controlled lane whose links are all red
  green_blocked — link green but >=1 vehicle ahead in lane
  green_head    — link green, amb is queue head (startup/decel loss)
  mid_queue     — stopped on a non-controlled lane (upstream spillback)
Usage: probe_amb_mech.py {207|211}"""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
which = sys.argv[1]
CFGS = {"207": "configs/exp207_dublin11h_531_B_movement_legacy_cqm_outcq_mask_nocoord_stab_obsfix_clamp_eps05_g095_rfloor.yaml",
        "211": "configs/exp211_dublin11h_531_enumfrap.yaml"}
cfg = yaml.safe_load(open(CFGS[which]))
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

# lane -> (tls, [link indices]) for all controlled lanes
lane_tls = {}
for t in env.ts_ids:
    for i, links in enumerate(env.sumo.trafficlight.getControlledLinks(t)):
        for (inlane, _out, _via) in links:
            lane_tls.setdefault(inlane, (t, []))
            if lane_tls[inlane][0] == t:
                lane_tls[inlane][1].append(i)

amb = {}   # vid -> per-step records
done = {"__all__": False}; step = 0
while not done["__all__"]:
    sumo = env.sumo
    for vid in sumo.vehicle.getIDList():
        if sumo.vehicle.getTypeID(vid) != "ambulance": continue
        v = sumo.vehicle.getSpeed(vid)
        lane = sumo.vehicle.getLaneID(vid)
        rec = {"step": step, "v": round(v, 1), "lane": lane,
               "wait": int(sumo.vehicle.getWaitingTime(vid)), "cls": "moving"}
        if v < 0.1:
            if lane in lane_tls:
                t, idxs = lane_tls[lane]
                st = sumo.trafficlight.getRedYellowGreenState(t)
                green = any(st[i] in "Gg" for i in idxs)
                pos = sumo.vehicle.getLanePosition(vid)
                ahead = sum(1 for o in sumo.lane.getLastStepVehicleIDs(lane)
                            if o != vid and sumo.vehicle.getLanePosition(o) > pos)
                rec["tls"] = t; rec["ahead"] = ahead
                rec["cls"] = ("green_blocked" if (green and ahead > 0)
                              else "green_head" if green
                              else "denied_green")
                if rec["cls"] == "denied_green":
                    links = sumo.trafficlight.getControlledLinks(t)
                    srv_lanes = {links[i][0][0] for i in range(len(st))
                                 if st[i] in "Gg" and links[i]}
                    n_srv = n_mov = 0
                    for sl in srv_lanes:
                        for ov in sumo.lane.getLastStepVehicleIDs(sl):
                            n_srv += 1
                            if sumo.vehicle.getSpeed(ov) > 0.1:
                                n_mov += 1
                    rec["green_lanes_veh"] = n_srv
                    rec["green_lanes_moving"] = n_mov
            else:
                rec["cls"] = "mid_queue"
        amb.setdefault(vid, []).append(rec)
    acts = {t: rl_act(t) for t in env.ts_ids}
    states, r, done, _ = env.step(action=acts)
    step += 1
env.close()
print(f"\n===== ARM {which} ckpt={ck}")
for vid, es in amb.items():
    stopped = [e for e in es if e["cls"] != "moving"]
    from collections import Counter
    c = Counter(e["cls"] for e in stopped)
    print(f"AMB {vid}: 在网 {len(es)} 步(5s/步), 停车步 {len(stopped)}, 分类 {dict(c)}, "
          f"max_wait={max((e['wait'] for e in es), default=0)}s")
    for e in stopped:
        print(f"  s{e['step']:4d} {e['cls']:13s} lane={e['lane']} wait={e['wait']}s "
              f"tls={e.get('tls','-')} ahead={e.get('ahead','-')} "
              f"greenlanes_veh={e.get('green_lanes_veh','-')} moving={e.get('green_lanes_moving','-')}")
