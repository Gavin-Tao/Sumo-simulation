"""同一轨迹双算 (2026-10-26, 状态/奖励修正实施计划任务 10)。只读分析, 不训练。

用旧口径 (配置原样: legacy 等待、首连接建表、运动缓存) 驱动贪心策略跑一集; 同一条轨迹上, 每个决策步按新口径
(visit 等待 + 全连接 + 无缓存) 独立算出 state 与 reward (复用 state_reward_debug.Ref), 与环境给出的旧口径逐维比对。
动作不变, 所以差异只来自定义的变化, 回答 "旧配置改到新口径, 输入与训练信号变多少"。只能在同一张网上做 (分槽网改动作集, 轨迹不可比)。

用法: python experiments/tools/dublin/dual_run_compare.py <exp> [--seed 123] [--steps 720] [--out report.json]"""
import argparse, collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out"); a = ap.parse_args()
    import ckpt_env
    from state_reward_debug import Ref, group_layout, TOL
    from sumo_rl.environment.visit_wait import VisitWaitTracker
    B = ckpt_env.build(a.exp, seed=a.seed); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]
    if cfg.get("obs_awt_basis") == "visit" or cfg.get("reward_wait_basis", "legacy") == "visit":
        sys.exit("配置已是 visit 口径, 双算无意义")
    env.visit_tracker = VisitWaitTracker(env.sumo, {t: env.traffic_signals[t].lanes for t in env.ts_ids}, strict=True)   # 旧口径环境上挂统计器, 只给新口径重算读
    meta = json.load(open(cfg["action_meta_file"]))["tls"]
    cfg_new = {**cfg, "obs_awt_basis": "visit", "reward_wait_basis": "visit", "obs_all_connections": True, "obs_slot_memo": False}
    ref = Ref(env, cfg_new, meta); groups = group_layout(ref); G = sorted(set(groups)); gidx = {g: [i for i, x in enumerate(groups) if x == g] for g in G}
    acc = {g: dict(n=0, nd=0, sum=0.0, max=0.0) for g in G}; r_old = collections.Counter(); r_new = collections.Counter(); r_absdiff = 0.0; n_steps = 0
    for step in range(a.steps):
        states, r, done, _ = env.step(action={t: act(t, states[t])[0] for t in env.ts_ids}); n_steps += 1
        for t in env.ts_ids:
            old = np.asarray(states[t], dtype=np.float64); new = ref.expected_state(t); d = np.abs(old - new)
            for g in G:
                x = d[gidx[g]]; acc[g]["n"] += x.size; acc[g]["nd"] += int((x > 1e-6).sum()); acc[g]["sum"] += float(x.sum()); acc[g]["max"] = max(acc[g]["max"], float(x.max()))
            rn = ref.expected_reward(t); r_old[t] += float(r[t]); r_new[t] += rn; r_absdiff = max(r_absdiff, abs(float(r[t]) - rn))
        if done["__all__"]:
            break
    env.close()
    rows = {g: dict(max=acc[g]["max"], mean_abs=acc[g]["sum"] / acc[g]["n"], frac_diff=acc[g]["nd"] / acc[g]["n"]) for g in G}
    tot_old, tot_new = sum(r_old.values()), sum(r_new.values())
    print(f"exp{a.exp} seed{a.seed} 步 {n_steps} | 旧口径驱动, 新口径 (visit + 全连接 + 无缓存) 同轨迹重算")
    print(f"{'特征组':<16} {'最大差':>9} {'平均绝对差':>11} {'有差维占比':>10}")
    for g in G:
        print(f"{g:<16} {rows[g]['max']:9.4f} {rows[g]['mean_abs']:11.6f} {rows[g]['frac_diff']:10.4f}")
    print(f"奖励: 整集累计 旧 {tot_old:.3f} 新 {tot_new:.3f} (新/旧 {tot_new / tot_old if tot_old else float('nan'):.3f}); 单步最大绝对差 {r_absdiff:.4f}")
    for t in sorted(r_old, key=lambda k: r_old[k])[:8]:
        print(f"   {t[:30]:<30} 旧 {r_old[t]:8.3f} 新 {r_new[t]:8.3f}")
    print("统计:", dict(ref.stats))
    if a.out:
        json.dump(dict(exp=a.exp, seed=a.seed, steps=n_steps, groups=rows, reward_old={t: r_old[t] for t in r_old}, reward_new={t: r_new[t] for t in r_new},
                       reward_step_max_absdiff=r_absdiff, stats=dict(ref.stats)), open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
