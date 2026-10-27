"""ψ 使用探针 (只读): 用检查点贪心跑一集, 每决策步对每个路口:
  flip_psi0:  把状态里 12 槽的 ψ (下游 count/queue × 5 级) 全清零后贪心动作是否改变 (只对 obs_downstream=true 的臂有意义);
  qgap:       贪心 Q 的最大值与次大值之差 (按有排队/无排队分别统计均值与中位), 衡量 "Q 并列" 程度;
  extend:     当前相位已满最短绿 (slot_min_green=1) 时, 贪心是 "延长" (动作 = 当前相位) 还是 "切换", 并记录被放行槽的 ψ count 之和 (按 ψ count 之和分桶: 0 / (0,0.2] / >0.2), 看延长概率是否随 ψ 变化。
ψ 维位置按 observation_fn 的属性推算: 每槽 1 + φ_dim + ψ_dim + lane_occ + slot_min_green, ψ 紧跟 φ。
用法: python psi_use_probe.py <exp> --ckpt P --seed 123 --out prefix"""
import argparse, collections, json, os, sys
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]; sumo = env.sumo
OBS = {tid: env.traffic_signals[tid].observation_fn for tid in env.ts_ids}
o0 = next(iter(OBS.values()))
hdr = 2 if getattr(o0, "include_header", True) else 0
phi = int(o0._phi_dim); psi = 5 * len(o0.downstream_fields) if o0.include_downstream else 0
slot_dim = 1 + phi + psi + (1 if o0.include_lane_occ else 0) + (1 if getattr(o0, "include_since_green", False) else 0) + (1 if getattr(o0, "include_green_elapsed", False) else 0) + (1 if getattr(o0, "include_slot_min_green", False) else 0)
dim = hdr + 12 * slot_dim; assert dim == len(next(iter(states.values()))), (dim, len(next(iter(states.values()))))
psi_idx = np.array([hdr + s * slot_dim + 1 + phi + k for s in range(12) for k in range(psi)], int)
mg_off = slot_dim - 1 if getattr(o0, "include_slot_min_green", False) else None
nf = len(o0.fields); q_idx = [1 + 3 * 0 + p * nf + o0.fields.index("queue") for p in range(5)] if nf == 3 else [1 + p * nf + o0.fields.index("queue") for p in range(5)]   # 槽内按优先级为主序: is_green, (fields…)_p1, (fields…)_p2, …
assert "queue" in o0.fields, o0.fields
print(f"dim={dim} hdr={hdr} phi={phi} psi={psi} slot_dim={slot_dim}")
st = {tid: dict(steps=0, flips=0, flips_q=0, qgap_q=[], qgap_nq=[], ext=collections.Counter(), ext_n=collections.Counter(), flip_by_phase=collections.Counter(), steps_by_phase=collections.Counter()) for tid in env.ts_ids}
done = {"__all__": False}
while not done["__all__"]:
    acts = {}
    for t in env.ts_ids:
        s = np.asarray(states[t], dtype=np.float32); ea, k, q = act(t, s); acts[t] = ea; s_ = st[t]; s_["steps"] += 1
        ts = env.traffic_signals[t]; cur = ts.green_phase
        sl = s[hdr:].reshape(12, slot_dim)
        has_q = bool((sl[:, q_idx].sum()) > 0)
        qs = np.sort(q[np.isfinite(q)])[::-1]; gap = float(qs[0] - qs[1]) if len(qs) > 1 else np.nan
        (s_["qgap_q"] if has_q else s_["qgap_nq"]).append(gap)
        s_["steps_by_phase"][int(cur)] += 1
        if psi:
            s2 = s.copy(); s2[psi_idx] = 0.0; ea2, k2, q2 = act(t, s2)
            if ea2 != ea:
                s_["flips"] += 1; s_["flip_by_phase"][int(cur)] += 1
                if has_q: s_["flips_q"] += 1
        if mg_off is not None:
            served = [i for i in range(12) if sl[i, 0] > 0.5]
            if served and sl[served[0], mg_off] > 0.5:       # 已满最短绿, 可切
                psi_cnt = float(sum(sl[i, 1 + phi:1 + phi + 5].sum() for i in served)) if psi else -1.0
                bucket = "noψ" if not psi else ("0" if psi_cnt <= 0 else ("(0,.2]" if psi_cnt <= 0.2 else ">.2"))
                key = (bucket, "q" if has_q else "nq")
                s_["ext_n"][key] += 1
                if ea == cur: s_["ext"][key] += 1
    states, r, done, _ = env.step(action=acts)
env.close()
out = {}
for t, s_ in st.items():
    out[t] = dict(steps=s_["steps"], flips=s_["flips"], flips_with_queue=s_["flips_q"], steps_with_queue=len(s_["qgap_q"]),
                  qgap_queue_mean=float(np.nanmean(s_["qgap_q"])) if s_["qgap_q"] else None, qgap_queue_median=float(np.nanmedian(s_["qgap_q"])) if s_["qgap_q"] else None,
                  qgap_noqueue_mean=float(np.nanmean(s_["qgap_nq"])) if s_["qgap_nq"] else None, qgap_noqueue_median=float(np.nanmedian(s_["qgap_nq"])) if s_["qgap_nq"] else None,
                  extend={f"{b}|{qq}": [s_["ext"][(b, qq)], s_["ext_n"][(b, qq)]] for (b, qq) in s_["ext_n"]},
                  flip_by_phase={str(k): v for k, v in s_["flip_by_phase"].items()}, steps_by_phase={str(k): v for k, v in s_["steps_by_phase"].items()})
json.dump(dict(exp=a.exp, ckpt_episode=B["ckpt_episode"], seed=a.seed, dim=dim, psi=psi, junctions=out), open(a.out + ".json", "w"), indent=1)
print(f"exp{a.exp} ep{B['ckpt_episode']} seed{a.seed} 完成")
