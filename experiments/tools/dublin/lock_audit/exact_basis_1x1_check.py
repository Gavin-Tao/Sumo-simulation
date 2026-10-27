"""精确版口径在 1x1 上改变了什么 (只读, 两段式; libsumo 一个进程只能有一个仿真, 所以分两个进程):
  --mode record: 按原配置用检查点贪心跑一集, 记录每步各路口的观测向量、动作、奖励 → <out>_A.npz
  --mode replay: 按 "461 精确版五键" (obs_awt_basis/reward_wait_basis = visit, obs_all_connections/all_connections = true, obs_slot_memo = false)
                 重建环境, 同一种子, 逐步回放 record 的动作 (车流轨迹因此完全一致), 记录观测与奖励 → <out>_B.npz, 并与 A 比较 → <out>.json:
                 观测最大绝对差、有差的 (路口,步) 数、曾有差的维数、若按 B 观测贪心会改动作的步数、奖励差。差为 0 → 精确版在该场景上与旧口径逐位等价。
用法: python exact_basis_1x1_check.py <exp> --ckpt P --seed 123 --out prefix --mode record ; 再 --mode replay"""
import argparse, json, os, sys
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); ap.add_argument("--mode", choices=["record", "replay"], required=True); a = ap.parse_args()
EXACT = dict(obs_awt_basis="visit", reward_wait_basis="visit", obs_all_connections=True, all_connections=True, obs_slot_memo=False)
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed, cfg_override=(EXACT if a.mode == "replay" else None))
env, act, states = B["env"], B["act"], B["states"]; ids = list(env.ts_ids)
o0 = next(iter(env.traffic_signals.values())).observation_fn
print(f"mode={a.mode} basis={o0.awt_basis} all_conn={getattr(o0, 'all_connections', None)} memo={getattr(o0, 'slot_memo', None)} tracker={env.visit_tracker is not None} dim={len(next(iter(states.values())))}")
rec_actions = np.load(a.out + "_A.npz")["actions"] if a.mode == "replay" else None
OBS, ACT, REW, GREEDY = [], [], [], []
done = {"__all__": False}; k = 0
while not done["__all__"]:
    obs_k = np.stack([np.asarray(states[t], np.float32) for t in ids]); OBS.append(obs_k)
    greedy = [act(t, obs_k[i])[0] for i, t in enumerate(ids)]; GREEDY.append(greedy)
    acts = {t: (int(rec_actions[k][i]) if rec_actions is not None else greedy[i]) for i, t in enumerate(ids)}
    ACT.append([acts[t] for t in ids])
    states, r, done, _ = env.step(action=acts); REW.append([float(r[t]) for t in ids]); k += 1
env.close()
tag = "_A" if a.mode == "record" else "_B"
np.savez(a.out + tag + ".npz", obs=np.stack(OBS), actions=np.array(ACT), rewards=np.array(REW), greedy=np.array(GREEDY), ids=np.array(ids))
print(f"{a.mode} 完成: {k} 步")
if a.mode == "replay":
    A = np.load(a.out + "_A.npz"); Bn = np.load(a.out + "_B.npz")
    n = min(len(A["obs"]), len(Bn["obs"])); oa, ob = A["obs"][:n], Bn["obs"][:n]; d = np.abs(oa - ob)
    out = dict(exp=a.exp, ckpt_episode=B["ckpt_episode"], seed=a.seed, steps_A=int(len(A["obs"])), steps_B=int(len(Bn["obs"])), dim=int(oa.shape[-1]),
               actions_replayed_identical=bool(np.array_equal(A["actions"][:n], Bn["actions"][:n])),
               max_obs_diff=float(d.max()), ts_steps_with_obs_diff=int((d.max(axis=-1) > 1e-6).sum()), ts_steps_total=int(n * oa.shape[1]),
               dims_ever_diff=int(((d > 1e-6).sum(axis=(0, 1)) > 0).sum()), dims_ever_diff_idx=[int(i) for i in np.where((d > 1e-6).sum(axis=(0, 1)) > 0)[0]][:40],
               greedy_action_would_differ=int((A["greedy"][:n] != Bn["greedy"][:n]).sum()),
               reward_sum_A=float(A["rewards"][:n].sum()), reward_sum_B=float(Bn["rewards"][:n].sum()), reward_max_abs_diff=float(np.abs(A["rewards"][:n] - Bn["rewards"][:n]).max()),
               reward_steps_diff=int((np.abs(A["rewards"][:n] - Bn["rewards"][:n]) > 1e-9).sum()))
    json.dump(out, open(a.out + ".json", "w"), indent=1); print(json.dumps(out, ensure_ascii=False))
