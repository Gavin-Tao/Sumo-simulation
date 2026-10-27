"""等待特征取值分布 (只读): 用检查点贪心跑一集, 从状态里读每槽小汽车级 (优先级 1) 的平均等待特征, 还原成原始秒数
(awt_log 开: s = expm1(f·log(3601)); 关: s = f·awt_scale), 只统计该槽小汽车 count>0 的槽步。输出: 零占比、分位数、最大值,
以及按线性刻度 (÷100) 落在 <0.1 / <0.3 区间的占比, 用来比较 1x1 与都柏林、新旧口径下 "线性特征被压扁" 的程度。
用法: python wait_feature_dist.py <exp> --ckpt P --seed 123 --out prefix"""
import argparse, json, os, sys
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]
o0 = next(iter(env.traffic_signals.values())).observation_fn
hdr = 2 if getattr(o0, "include_header", True) else 0
fields = list(o0.fields); phi = int(o0._phi_dim); psi = 5 * len(o0.downstream_fields) if o0.include_downstream else 0
slot_dim = (len(next(iter(states.values()))) - hdr) // 12
i_cnt = 1 + fields.index("count"); i_awt = 1 + fields.index("mean_awt")   # 槽内按优先级为主序: is_green, (count,queue,awt)_p1, (…)_p2, …; 这里取 p1 (小汽车)
log_on = bool(getattr(o0, "awt_log", False)); scale = float(o0.awt_scale); cap = o0.awt_cap
print(f"exp{a.exp} dim={hdr + 12 * slot_dim} fields={fields} awt_log={log_on} awt_scale={scale} awt_cap={cap} awt_basis={getattr(o0, 'awt_basis', '?')}")
secs = []; n_slotsteps = 0
done = {"__all__": False}
while not done["__all__"]:
    acts = {}
    for t in env.ts_ids:
        s = np.asarray(states[t], dtype=np.float32); acts[t] = act(t, s)[0]
        sl = s[hdr:].reshape(12, slot_dim)
        for k in range(12):
            if sl[k, i_cnt] > 0:            # 小汽车级 count > 0
                f = float(sl[k, i_awt]); n_slotsteps += 1
                secs.append(float(np.expm1(f * np.log1p(3600.0))) if log_on else f * scale)
    states, r, done, _ = env.step(action=acts)
env.close()
x = np.array(secs); nz = x[x > 0]
q = lambda arr, p: float(np.percentile(arr, p)) if len(arr) else None
out = dict(exp=a.exp, ckpt_episode=B["ckpt_episode"], seed=a.seed, awt_log=log_on, awt_scale=scale, awt_cap=cap, awt_basis=getattr(o0, "awt_basis", None),
           slot_steps_with_cars=int(n_slotsteps), zero_share=float((x == 0).mean()) if len(x) else None,
           all=dict(q50=q(x, 50), q90=q(x, 90), q99=q(x, 99), max=float(x.max()) if len(x) else None, mean=float(x.mean()) if len(x) else None),
           nonzero=dict(n=int(len(nz)), q10=q(nz, 10), q50=q(nz, 50), q90=q(nz, 90), q99=q(nz, 99), max=float(nz.max()) if len(nz) else None),
           linear_feature_lt_0p1_share_nonzero=float((nz / 100.0 < 0.1).mean()) if len(nz) else None,
           linear_feature_lt_0p3_share_nonzero=float((nz / 100.0 < 0.3).mean()) if len(nz) else None,
           share_nonzero_gt_100s=float((nz > 100).mean()) if len(nz) else None, share_nonzero_gt_500s=float((nz > 500).mean()) if len(nz) else None)
json.dump(out, open(a.out + ".json", "w"), indent=1); print(json.dumps(out, ensure_ascii=False))
