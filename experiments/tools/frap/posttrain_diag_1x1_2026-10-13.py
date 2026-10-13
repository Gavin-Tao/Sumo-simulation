"""1x1 训练后诊断 (2026-10-13, 用户批准; 只读, 不训练)。对一个检查点在评估种子 123-125 各跑一集贪心, 记录:
  · 决策点中 "最短绿已满足" 的占比 (未满足时环境只会保持当前相位); 已满足时选择切换的比例
  · 最短绿未满足的状态里: Q 最高的是不可执行相位的比例, (不可执行相位最高 Q − 当前相位 Q) 的中位
      (开 F1 时这些相位不参与 TD 目标, 此项只作参考; 不开 F1 时它就是最短绿缺陷的直接测量)
  · 执行动作的 Q 与蒙特卡洛折扣回报 G 的比 (Q/G): 只用每集前 70% 步; 1x1 一集 200 步、γ=0.99, 截断使 G 偏向 0,
      健康臂 Q/G 约 1.3-1.6, 明显 < 1 表示高估 (Q 比真实回报更接近 0)
支持 8STD (action_meta_file) 与 GS (enum_frap)。依据: analysis/GRID1X1_GSG_ANALYSIS_2026-10-12.txt §8-§10。
用法 (仓库根目录): python experiments/tools/frap/posttrain_diag_1x1_2026-10-13.py <exp> --ckpt <pth> --tag <best|ep1500>
输出: experiments/analysis/data/posttrain_diag/exp<exp>_<tag>.json, 并打印一行摘要 (以 DIAG 开头)
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "dublin"))
import ckpt_env  # noqa: E402

SEEDS = (123, 124, 125)


def run_episode(exp, ckpt, seed):
    B = ckpt_env.build(exp, ckpt, seed)
    env, act, states = B["env"], B["act"], B["states"]
    t = list(env.ts_ids)[0]; ts = env.traffic_signals[t]
    gamma = float(B["cfg"].get("gamma", 0.99))
    rows = []; done = {"__all__": False}
    while not done["__all__"]:
        x = np.asarray(states[t], np.float32); g, k, q = act(t, x); q = np.array(q, float)
        mg = int(ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time)
        cur = int(ts.std_action_map[ts.green_phase]) if B["kind"] == "dqn_masked" else int(ts.green_phase)
        other = np.delete(q, cur); other = other[~np.isnan(other)]
        states, r, done, _ = env.step(action={t: g})
        ex = int(ts.std_action_map[ts.green_phase]) if B["kind"] == "dqn_masked" else int(ts.green_phase)
        rows.append(dict(mg=mg, choose_switch=int(k != cur), q_exec=float(q[ex]), r=float(r[t]),
                         argmax_not_cur=int(np.nanargmax(q) != cur), gap=float(np.max(other) - q[cur]) if len(other) else float("nan")))
    env.close()
    T = len(rows); G = np.zeros(T); acc = 0.0
    for i in range(T - 1, -1, -1):
        acc = rows[i]["r"] + gamma * acc; G[i] = acc
    keep = int(0.7 * T)
    q = np.array([x["q_exec"] for x in rows])
    mg = np.array([x["mg"] for x in rows]); sw = np.array([x["choose_switch"] for x in rows])
    am = np.array([x["argmax_not_cur"] for x in rows]); gap = np.array([x["gap"] for x in rows])
    return dict(seed=seed, ckpt_episode=B["ckpt_episode"], kind=B["kind"], gamma=gamma, n_decisions=T,
                mg_ok_share=float(mg.mean()), switch_when_free=float(sw[mg == 1].mean()) if (mg == 1).any() else float("nan"),
                mg0_argmax_not_cur=float(am[mg == 0].mean()) if (mg == 0).any() else float("nan"),
                mg0_gap_median=float(np.nanmedian(gap[mg == 0])) if (mg == 0).any() else float("nan"),
                mean_q=float(q[:keep].mean()), mean_G=float(G[:keep].mean()),
                q_minus_G_median=float(np.median(q[:keep] - G[:keep])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ck = os.path.abspath(a.ckpt)
    per = [run_episode(a.exp, ck, s) for s in SEEDS]
    agg = {k: float(np.mean([p[k] for p in per])) for k in ("mg_ok_share", "switch_when_free", "mg0_argmax_not_cur", "mg0_gap_median", "mean_q", "mean_G", "q_minus_G_median")}
    agg["q_over_G"] = agg["mean_q"] / agg["mean_G"] if agg["mean_G"] != 0 else float("nan")
    out = dict(exp=a.exp, ckpt=ck, tag=a.tag, ckpt_episode=per[0]["ckpt_episode"], kind=per[0]["kind"], seeds=list(SEEDS), per_seed=per, mean=agg)
    root = os.path.abspath(os.path.join(_HERE, "..", "..", "analysis", "data", "posttrain_diag")); os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"exp{a.exp}_{a.tag}.json"); json.dump(out, open(path, "w"), indent=1)
    print(f"DIAG exp{a.exp} {a.tag} (ep{out['ckpt_episode']}): 最短绿已满足 {100*agg['mg_ok_share']:.1f}% | 满足时切换 {100*agg['switch_when_free']:.1f}% | "
          f"未满足时 Q 最高为不可执行相位 {100*agg['mg0_argmax_not_cur']:.0f}% (差值中位 {agg['mg0_gap_median']:+.3f}) | "
          f"Q/G {agg['q_over_G']:.2f} (Q {agg['mean_q']:.2f}, G {agg['mean_G']:.2f}) | 存 {path}")


if __name__ == "__main__":
    main()
