"""Contract-conditioned MPC probe (RESEARCH_DIRECTIONS §2.8c, I1):
zero-training re-contract via FIFO clearance-cost re-ranking on exp220.
  mode top2: Q proposes top-2 phases, FIFO cost (target contract) picks
  mode all : pure planner, argmin FIFO cost over all legal phases (Q unused)
Usage: probe_mpc_recontract.py {421|351} {top2|all}
Cost(p) = sum_lv lv * mass_v2[lv-1, p]  with experts' priority table = target
table (bucket == weight), matching the reward semantics."""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
table_tag, mode = sys.argv[1], sys.argv[2]
TABLES = {"421": {"ambulance": 4, "bus": 2, "car": 1},
          "351": {"ambulance": 3, "bus": 5, "car": 1},
          "531": {"ambulance": 5, "bus": 3, "car": 1}}
W = TABLES[table_tag]
cfg = yaml.safe_load(open("configs/exp220_1x1_531_NS20bus_enumfrap_cqm.yaml"))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.metrics import EpisodeMetricsCollector
import moe_glue
from frap_glue import load_enum_tables, build_frap_agent

# obs table stays at the TRAINED default (Q is only a proposer; its obs
# semantics must match training). The CONTRACT enters via the planner cost.
obs_class = functools.partial(obsmod.PriorityMovementObservationFunction,
    fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"])
env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
    cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
    num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
    max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
    single_agent=False, yellow_time=cfg["yellow_time"], delta_time=cfg["delta_time"],
    reward_fn=cfg["reward_fn"], observation_class=obs_class,
    sumo_seed=cfg["seed"], sumo_warnings=False)
env.reset(int(cfg["eval_seed"]))
tid = env.ts_ids[0]

tables = load_enum_tables(cfg["enum_meta_file"])
ts = env.traffic_signals[tid]
ts.observation_fn.rebind_movements(tables["turnmap"][tid])
ts.observation_space = ts.observation_fn.observation_space()
states = {tid: ts.observation_fn()}
agent = build_frap_agent(cfg, tables, env, "cpu")
ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
agent.q_net.eval(); agent.epsilon = 0.0

# planner: MoE FIFO clearance model with priority table = TARGET contract
mcfg = dict(cfg); mcfg["priority_source"] = W
moe = moe_glue.load_moe_tables(cfg["enum_meta_file"])
experts = moe_glue.build_experts(mcfg, moe, env)
tab = moe["tables"][tid]
n_phases = len(tab["phase_slots"])
LV_W = np.zeros(5)
for lv in set(W.values()):
    LV_W[lv-1] = lv          # bucket == weight (reward semantics)

def q_values():
    x = torch.tensor([np.asarray(states[tid], dtype=np.float32)])
    i = agent._ids.index(tid)
    with torch.no_grad():
        q = agent.q_net(x, agent.PM[i:i+1], agent.REL[i:i+1], agent.EXIST[i:i+1])[0].numpy()
    mask = tables["tls"][tid]["mask"]
    q[~mask] = -np.inf
    return q

def plan_cost(cur):
    _, lane_q, lane_arr, _, _ = experts._scan(tab, ts.sumo)
    mass = experts._mass_v2(tab, lane_q, lane_arr, cur)     # (5, n_phases)
    return LV_W @ mass                                       # (n_phases,)

ts_lane_map = {tid: ts.signal_controlled_lanes}
mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time,
                             excluded_lanes=set(ts.always_green_lanes))
switch = 0; prev = ts.green_phase
done = {"__all__": False}; step = 0; agree = 0
while not done["__all__"]:
    mc.collect_step(env.sumo)
    cur = ts.green_phase
    cost = plan_cost(cur)
    if mode == "top2":
        q = q_values()
        top2 = np.argsort(q)[-2:]
        a = int(top2[np.argmin(cost[top2])])
        agree += int(a == int(np.argmax(q)))
    else:
        mask = tables["tls"][tid]["mask"]
        cost_m = np.where(mask, cost, np.inf)
        a = int(np.argmin(cost_m))
    states, r, done, _ = env.step(action={tid: a})
    g = ts.green_phase
    if g != prev: switch += 1
    prev = g; step += 1
mc.collect_step(env.sumo); mc.finalize(env.sumo)
s = mc.summary()["system"]
env.close()
out = {"arm": f"mpc-{mode}-{table_tag}", "steps": step,
       "switch_rate": round(switch/(step-1), 3)}
J = 0.0
for cls, w in [("car", W["car"]), ("bus", W["bus"]), ("ambulance", W["ambulance"])]:
    st = s[cls]["avg_stopped_time_per_visit"]
    out[cls] = [round(s[cls]["avg_stop_events_per_visit"], 3), round(st, 3)]
    J += w * st
out["J"] = round(J, 1)
if mode == "top2": out["q_top1_kept"] = round(agree/step, 3)
print("MPC", json.dumps(out, ensure_ascii=False))
