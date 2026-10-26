"""旧行为流快照 (2026-10-26, 状态/奖励修正实施计划任务 0): 用 ckpt_env.build 构造环境与策略, 跑 N 个决策步,
把每步每路口的 state 向量与 reward 存成 npz; --compare 逐位比较两份快照。用途: 证明不写新配置键的旧配置在修改前后逐位不变。
用法:
  python experiments/tools/dublin/obs_reward_dump.py <exp> --seed 123 --steps 200 --out F.npz
  python experiments/tools/dublin/obs_reward_dump.py --compare A.npz B.npz
只读分析, 不训练。"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def dump(exp, seed, steps, out, routes=None, ckpt=None):
    out = os.path.abspath(out)                      # ckpt_env.build 会 chdir 到 experiments/, 先取绝对路径
    import ckpt_env
    B = ckpt_env.build(exp, ckpt=ckpt, seed=seed, route_override=routes)
    env, act, states = B["env"], B["act"], B["states"]
    ts_ids = list(env.ts_ids)
    S = {t: [np.asarray(states[t], dtype=np.float32)] for t in ts_ids}
    R = {t: [] for t in ts_ids}
    for _ in range(steps):
        states, r, done, _ = env.step(action={t: act(t, states[t])[0] for t in ts_ids})
        for t in ts_ids:
            S[t].append(np.asarray(states[t], dtype=np.float32)); R[t].append(float(r[t]))
        if done["__all__"]:
            break
    env.close()
    arrays = {}
    for t in ts_ids:
        arrays[f"state/{t}"] = np.stack(S[t]); arrays[f"reward/{t}"] = np.asarray(R[t], dtype=np.float64)
    arrays["meta/exp"] = np.asarray(str(exp)); arrays["meta/seed"] = np.asarray(int(seed)); arrays["meta/ckpt"] = np.asarray(str(B["ckpt"]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"saved {out} | exp{exp} seed{seed} ckpt {B['ckpt']} | 路口 {len(ts_ids)} | 步 {len(R[ts_ids[0]])} | state 维 {S[ts_ids[0]][0].shape[0]}")


def compare(a, b):
    A, Bz = np.load(a), np.load(b)
    ka, kb = set(A.files), set(Bz.files)
    if ka != kb:
        print("键不同:", sorted(ka ^ kb)); return 1
    worst = 0.0; bad = []
    for k in sorted(ka):
        if k.startswith("meta/"):
            continue
        x, y = A[k], Bz[k]
        if x.shape != y.shape:
            bad.append((k, "shape", x.shape, y.shape)); continue
        d = float(np.max(np.abs(x - y))) if x.size else 0.0
        worst = max(worst, d)
        if d > 0:
            bad.append((k, "maxdiff", d, int(np.sum(x != y))))
    print(f"compare {os.path.basename(a)} vs {os.path.basename(b)}: 键 {len(ka)}, 最大绝对差 {worst:.3e}, 有差的键 {len(bad)}")
    for row in bad[:10]:
        print("   ", row)
    return 0 if not bad else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="?")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--routes")
    ap.add_argument("--ckpt", help="检查点路径 (默认 ckpt_env 取最后一个 ckpt)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    if not a.exp or not a.out:
        ap.error("需要 <exp> 与 --out, 或 --compare A B")
    dump(a.exp, a.seed, a.steps, a.out, a.routes, a.ckpt)


if __name__ == "__main__":
    main()
