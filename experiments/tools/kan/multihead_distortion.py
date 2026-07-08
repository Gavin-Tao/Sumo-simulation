"""Per-head KAN/linear distortion readout for the multihead agent (exp219,
1x1). FRAP bakes distortion into the internalized WEIGHT (KAN g-alpha);
multihead applies the BNF weight exactly at argmax, so distortion moves to
the per-head VALUE QUALITY: is each head l trained enough to (a) read its own
priority level's features (diagonal alpha_{l,l}) and (b) actually swing
decisions in proportion to its nominal weight?

For each head l we distil Q_l(s, a*) ~ per-level aggregated phi features via a
ridge (the linear/shared-levels analog of the KAN alpha), reading the 5x5
alpha_{l,p} matrix (own-level = diagonal, cross-level = leakage). We also
measure each head's decision swing and the effective decision weight vs the
nominal 5-3-1. Read-only. Usage: multihead_distortion.py [seeds]
"""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
os.chdir(os.path.join(REPO, "experiments"))
import numpy as np, torch, yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CFG = "configs/exp219_1x1_531_NS20bus_multihead_bnf.yaml"
seeds = sys.argv[1] if len(sys.argv) > 1 else "123,2000,2001,2002,2003,2004,2005,2006"
cfg = yaml.safe_load(open(CFG))
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from multihead_glue import build_multihead_agent

obs_class = functools.partial(obsmod.PriorityMovementObservationFunction,
    fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
    priority_source=cfg["priority_source"])
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
A = env.action_space.n
agent = build_multihead_agent(cfg, starting_state=tuple([0.0]*od),
                              state_space=od, action_space=A, device="cpu")
ck = sorted(glob.glob(f"models/{cfg['name']}/*/best.pth"))[-1]
agent.q_net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["policy_state_dict"])
agent.q_net.eval(); agent.epsilon = 0.0
W = agent.weights.copy()                                  # nominal BNF weights
print("checkpoint:", ck, "| BNF weights:", W.tolist(), flush=True)

# obs layout: header + 12 slots x (5 levels x 3 fields). header = od - 180.
SLOT_DIM = 15                                             # 5 x [cnt,que,awt]
HDR = od - 12 * SLOT_DIM
assert HDR >= 0, (od, HDR)

def per_level_feats(x):
    """aggregate 12 slots -> 15 features: for each level, sum(cnt,que,awt)."""
    sl = np.asarray(x[HDR:], dtype=np.float32).reshape(12, SLOT_DIM)
    agg = sl.sum(0)                                       # (15,) = 5 lvl x 3
    return agg

# rollout: record per-level feats + each head's Q at greedy action + swing
FE, QH = [], []                       # feats (n,15), head Q at greedy (n,5)
swing = np.zeros(5); nsteps = 0
for sd in seeds.split(","):
    env.reset(int(sd))
    st = {tid: env.traffic_signals[tid].observation_fn()}
    done = {"__all__": False}
    while not done["__all__"]:
        x = np.asarray(st[tid], dtype=np.float32)
        with torch.no_grad():
            q = agent.q_net(torch.tensor([x]))[0].numpy()  # (5, A)
        score = W @ q                                      # (A,)
        a = int(score.argmax())
        FE.append(per_level_feats(x)); QH.append(q[:, a])
        # decision swing of level l = w_l * (max-min of Q_l over actions)
        swing += W * (q.max(1) - q.min(1)); nsteps += 1
        st, r, done, _ = env.step(action={tid: a})
env.close()
FE = np.array(FE); QH = np.array(QH)                        # (n,15), (n,5)
print(f"rollout: {nsteps} steps, {len(FE)} rows", flush=True)

# per-head shared-levels alpha: Q_l ~ per-level feats; group 3 fields/level
def level_group(coef):                                     # |coef| summed per level
    return np.array([np.abs(coef[3*p:3*p+3]).sum() for p in range(5)])

scaler = StandardScaler().fit(FE); FEs = scaler.transform(FE)
alpha = np.zeros((5, 5)); r2 = np.zeros(5); scale = np.zeros(5)
for l in range(5):
    r = Ridge(alpha=1.0).fit(FEs, QH[:, l])
    alpha[l] = level_group(r.coef_)
    r2[l] = r.score(FEs, QH[:, l]); scale[l] = QH[:, l].std()

# effective decision weight per level (normalized to level-1=car)
eff = swing / max(swing[0], 1e-9)
nominal = W / max(W[0], 1e-9)

# CONDITIONAL-ON-PRESENT (disentangles "head under-trained" from "class rarely
# present"): for each level, restrict to rows where that level's cnt>0 and
# re-read the own-level alpha + head value scale. cnt is field 0 of 3/level.
cond = {}
for l in range(5):
    present = FE[:, 3 * l] > 0
    n_pres = int(present.sum())
    if n_pres >= 30:
        Xp = StandardScaler().fit_transform(FE[present])
        rr = Ridge(alpha=1.0).fit(Xp, QH[present, l])
        cond[f"l{l+1}"] = {"n_present": n_pres,
                           "own_alpha_present": round(float(level_group(rr.coef_)[l]), 3),
                           "head_scale_present": round(float(QH[present, l].std()), 3),
                           "r2_present": round(float(rr.score(Xp, QH[present, l])), 3)}
    else:
        cond[f"l{l+1}"] = {"n_present": n_pres, "note": "too few present rows"}

out = {"ckpt": os.path.basename(os.path.dirname(ck)), "steps": int(nsteps),
       "bnf_weights": W.tolist(),
       "alpha_matrix_l_by_p": [[round(float(a), 3) for a in row] for row in alpha],
       "diag_own_level": [round(float(alpha[l, l]), 3) for l in range(5)],
       "head_value_scale": [round(float(s), 3) for s in scale],
       "head_r2": [round(float(x), 3) for x in r2],
       "eff_decision_weight_norm": [round(float(x), 3) for x in eff],
       "nominal_weight_norm": [round(float(x), 3) for x in nominal],
       "conditional_on_present": cond}
os.makedirs(f"{REPO}/experiments/analysis/kan_data/mh219", exist_ok=True)
json.dump(out, open(f"{REPO}/experiments/analysis/kan_data/mh219/distortion.json", "w"), indent=1)
print("MHDISTORTION", json.dumps(out, ensure_ascii=False))
