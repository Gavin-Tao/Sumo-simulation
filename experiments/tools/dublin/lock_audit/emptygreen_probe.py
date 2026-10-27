"""零重训探针: 决策时屏蔽 "所服务的槽全部没有车, 而本路口另有槽在排队" 的相位 (空槽绿灯掩码); 其余同 mingreen_probe。"""
import argparse, os, sys, json, collections, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env, torch
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--mask", action="store_true"); ap.add_argument("--out", required=True); a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed); env, act, states, cfg, agent = B["env"], B["act"], B["states"], B["cfg"], B["agent"]; sumo = env.sumo
meta = json.load(open(cfg["action_meta_file"]))["tls"]
fields = list(cfg.get("obs_fields", ["count", "queue", "mean_awt"])); ds = list(cfg.get("obs_downstream_fields", [])) if cfg.get("obs_downstream") else []
D = 1 + 3 * 0  # computed below
per_slot = 1 + 5 * len(fields) + 5 * len(ds) + (1 if cfg.get("obs_lane_occ") else 0) + (1 if cfg.get("obs_slot_min_green") else 0)
assert per_slot * 12 == len(states[env.ts_ids[0]]), (per_slot, len(states[env.ts_ids[0]]))
ci = fields.index("count"); qi = fields.index("queue")
def slot_stats(s):
    s = np.asarray(s); cnt = np.zeros(12); q = np.zeros(12)
    for k in range(12):
        base = k * per_slot + 1
        for p in range(5): cnt[k] += s[base + p * len(fields) + ci]; q[k] += s[base + p * len(fields) + qi]
    return cnt, q
serves = {}; std_of = {}
for t in env.ts_ids:
    ts = env.traffic_signals[t]; ls = {int(i): None for i in meta[t]["links"]}
    APP = ("N", "E", "S", "W"); TURN = ("L", "T", "R")
    for i, grp in meta[t]["links"].items(): ls[int(i)] = APP.index(grp[0]["approach"]) * 3 + TURN.index(grp[0]["turn"])
    serves[t] = {int(ts.std_action_map[g]): {ls[i] for i in ls if i < len(ph.state) and ph.state[i] in "Gg"} for g, ph in enumerate(ts.green_phases)}
    std_of[t] = {int(ts.std_action_map[g]): g for g in range(len(ts.green_phases))}
masked_steps = collections.Counter()
def act2(t, s):
    a_env, k, q = act(t, s)
    if not a.mask: return a_env
    cnt, qq = slot_stats(s)
    if qq.sum() <= 0: return a_env
    ok = {std: not (sum(cnt[x] for x in sl) <= 0) for std, sl in serves[t].items()}
    if all(not v for v in ok.values()): return a_env
    qm = np.where([ok.get(i, False) for i in range(8)], q, np.nan)
    k2 = int(np.nanargmax(qm))
    if k2 != k: masked_steps[t] += 1
    return std_of[t][k2]
tr = env.visit_tracker; seen = set(); halt = collections.Counter(); mx = collections.Counter()
done = {"__all__": False}
while not done["__all__"]:
    acts = {t: act2(t, states[t]) for t in env.ts_ids}
    states, r, done, _ = env.step(action=acts)
    for (t, vid, e), w in tr.W.items():
        seen.add((t, vid, e)); mx[t] = max(mx[t], w)
    for t in env.ts_ids:
        for l in env.traffic_signals[t].lanes: halt[t] += sumo.lane.getLastStepHaltingNumber(l) * 5
remaining = sumo.vehicle.getIDCount(); env.close()
visits = collections.Counter(k[0] for k in seen); tot_v = sum(visits.values()); tot_s = sum(halt.values())
lock = {t[:20]: round(w) for t, w in mx.items() if w >= 150}
print(f"exp{a.exp} ep{B['ckpt_episode']} seed{a.seed} mask={a.mask}: 全网 {tot_s/tot_v:.3f} s/visit, 剩余 {remaining}, 锁死 {lock}, 掩码改变选择的步数 {sum(masked_steps.values())} {dict(masked_steps.most_common(3))}")
json.dump(dict(exp=a.exp, seed=a.seed, mask=a.mask, remaining=remaining, stop_per_visit=tot_s / tot_v, lock=lock, masked=dict(masked_steps)), open(a.out, "w"))
