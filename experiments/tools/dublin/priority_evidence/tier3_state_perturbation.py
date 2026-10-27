"""第三层: 策略决策时是否对车型有反应 (状态扰动测试, 离线)。用 best 检查点贪心跑一集 (种子 123) 收集全部决策状态;
对每个状态、每个当前红灯且该类车计数为 0 的真实槽, 往该槽的某一优先级 (car=1 / bus=4 / ambulance=5) 注入 1 辆等待 w 秒的车
(count/queue 各 +1/容量, mean_awt = log1p(w)/log1p(3600), lane_occ +1/容量), 看贪心动作是否翻到放行该槽的相位。
111 臂的观测把所有车都记为优先级 1 (对车型盲), 三类都注入优先级 1。
用法: python tier3_state_perturbation.py <exp> --ckpt P --seed 123 --out prefix"""
import argparse, os, sys, json, math, collections, numpy as np, torch
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); ap.add_argument("--with-car-queue", action="store_true", help="只取该槽已有小汽车排队 (car count>0 且 queue>0) 的状态再注入; 注入 car 时 = 再加一辆"); a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed); env, act, states, cfg, agent = B["env"], B["act"], B["states"], B["cfg"], B["agent"]
meta = json.load(open(cfg["action_meta_file"]))["tls"]; LOG = math.log1p(3600.0)
fields = list(cfg.get("obs_fields")); ds = list(cfg.get("obs_downstream_fields", [])) if cfg.get("obs_downstream") else []
D = 1 + 5 * len(fields) + 5 * len(ds) + (1 if cfg.get("obs_lane_occ") else 0) + (1 if cfg.get("obs_slot_min_green") else 0); assert D * 12 == len(states[env.ts_ids[0]])
ci, qi, ai = fields.index("count"), fields.index("queue"), fields.index("mean_awt"); occ_i = 1 + 5 * len(fields) + 5 * len(ds); mg_i = D - 1
prio = dict(cfg.get("obs_priority_source", cfg.get("priority_source")) or {"ambulance": 5, "bus": 4, "car": 1})
blind = len(set(prio.values())) == 1
APP = ("N", "E", "S", "W"); TURN = ("L", "T", "R")
serves, mask, cap, real = {}, {}, {}, {}
for t in env.ts_ids:
    ts = env.traffic_signals[t]; ls = {int(i): APP.index(grp[0]["approach"]) * 3 + TURN.index(grp[0]["turn"]) for i, grp in meta[t]["links"].items()}
    serves[t] = {int(ts.std_action_map[g]): {ls[i] for i in ls if i < len(ph.state) and ph.state[i] in "Gg"} for g, ph in enumerate(ts.green_phases)}
    mask[t] = np.array(meta[t]["mask"], bool); cap[t] = dict(ts.observation_fn._slot_cap); real[t] = set().union(*serves[t].values())
S, A, J = [], [], []
done = {"__all__": False}
while not done["__all__"]:
    acts = {}
    for t in env.ts_ids:
        a_env, k, q = act(t, states[t]); acts[t] = a_env; S.append(np.asarray(states[t], np.float32)); A.append(int(k)); J.append(t)
    states, r, done, _ = env.step(action=acts)
env.close()
S = np.stack(S); net = agent.q_net
def greedy(X, t):
    with torch.no_grad(): q = net(torch.tensor(X)).numpy()
    q[:, ~mask[t]] = -1e9; return q.argmax(1)
res = collections.defaultdict(lambda: dict(n=0, flip=0, changed=0)); per_j = collections.defaultdict(lambda: dict(n=0, flip=0))
CLS = {"car": prio.get("car", 1), "bus": prio.get("bus", 4), "ambulance": prio.get("ambulance", 5)}
for t in env.ts_ids:
    idx = np.flatnonzero(np.array(J) == t); X0 = S[idx]; A0 = np.array(A)[idx]
    mg_ok = X0[:, [k * D + mg_i for k in range(12)]].max(1) >= 0.5
    for k in sorted(real[t]):
        red = X0[:, k * D] < 0.5; base_serves = np.array([k in serves[t].get(int(a_), set()) for a_ in A0])
        for cname, p in CLS.items():
            lvl = (1 if blind else p) - 1; cnt_col = k * D + 1 + lvl * len(fields) + ci
            car_cnt = k * D + 1 + (0 if blind else prio.get("car", 1) - 1) * len(fields) + ci; car_q = car_cnt - ci + qi
            if a.with_car_queue:
                cond = (X0[:, car_cnt] > 0) & (X0[:, car_q] > 0) & ((X0[:, cnt_col] <= 0) if cname != "car" else True)
            else:
                cond = X0[:, cnt_col] <= 0
            sel = np.flatnonzero(mg_ok & red & ~base_serves & cond)
            if len(sel) == 0: continue
            for w in (10.0, 30.0):
                X = X0[sel].copy(); c = 1.0 / cap[t].get(k, 1.0)
                X[:, cnt_col] += c; X[:, k * D + 1 + lvl * len(fields) + qi] += c
                a_col = k * D + 1 + lvl * len(fields) + ai
                if a.with_car_queue and cname == "car":      # 再加一辆等 w 秒的小汽车: 均值按计数加权更新
                    n0 = X0[sel, cnt_col] / c; old = np.expm1(X0[sel, a_col] * LOG); X[:, a_col] = np.log1p((old * n0 + w) / (n0 + 1)) / LOG
                else:
                    X[:, a_col] = math.log1p(w) / LOG
                if cfg.get("obs_lane_occ"): X[:, k * D + occ_i] = np.minimum(X[:, k * D + occ_i] + c, 1.5)
                A1 = greedy(X, t); flip = np.array([k in serves[t].get(int(a_), set()) for a_ in A1]); ch = A1 != A0[sel]
                r_ = res[(cname, w)]; r_["n"] += len(sel); r_["flip"] += int(flip.sum()); r_["changed"] += int(ch.sum())
                if w == 10.0: pj = per_j[(cname, t)]; pj["n"] += len(sel); pj["flip"] += int(flip.sum())
short = lambda t: t.split("_")[0] + ("_" + t.split("_")[1] if t.startswith("cluster") else "")
out = dict(exp=a.exp, ckpt_episode=B["ckpt_episode"], seed=a.seed, class_blind=blind, with_car_queue=a.with_car_queue, n_states=int(len(S)),
           summary={f"{c}@{int(w)}s": dict(n=v["n"], flip_rate=v["flip"] / max(v["n"], 1), change_rate=v["changed"] / max(v["n"], 1)) for (c, w), v in res.items()},
           per_junction={f"{c}@{short(t)}": dict(n=v["n"], flip_rate=v["flip"] / max(v["n"], 1)) for (c, t), v in per_j.items()})
json.dump(out, open(a.out + ".json", "w"), indent=1)
print(f"exp{a.exp} ep{B['ckpt_episode']} 状态 {len(S)} 车型盲={blind}: " + " | ".join(f"{k}: 翻转 {v['flip_rate']:.1%} (改动 {v['change_rate']:.1%}, n={v['n']})" for k, v in sorted(out["summary"].items())))
