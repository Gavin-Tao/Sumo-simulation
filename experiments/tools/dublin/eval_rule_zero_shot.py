"""(2026-10-11, 只读诊断; 副本自 scratchpad/sparse/eval_rule.py) 只读诊断 (2026-10-11): 在 8STD 检查点上零样本加决策规则, 其它一切同 eval_pervehicle.py。
规则 gate: 需求门控 —— 只有 "将获得绿灯的运动上有车 (φ count>0)" 的相位可选, 否则保持当前相位 (无需求不动)。
规则 hyst: 滞回 —— 只有 Q(new) > Q(current) + δ 才切换, δ = max(0.05, 0.02·|Q(current)|)。
用法: python eval_rule.py <exp> <ckpt> <seed> <gate|hyst|both>"""
import sys, os, runpy, numpy as np, yaml, glob
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
exp, ckpt, seed, rule = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
_build = ckpt_env.build
def build(*a, **k):
    B = _build(*a, **k); env, act0, cfg = B["env"], B["act"], B["cfg"]
    kind = B["kind"]
    legacy = cfg.get("obs_phase_state", "legacy") == "legacy"; head = 9 if legacy else 2
    nf = len(cfg.get("obs_fields", ["count", "queue", "mean_awt"])); nlev = 5
    def demand_by_green(t, s):
        ts = env.traffic_signals[t]; of = ts.observation_fn; slot_dim = (len(s) - head) // 12; off_phi = 0 if legacy else 1
        cnt = np.array([sum(s[head + sl * slot_dim + off_phi + p * nf] for p in range(nlev)) for sl in range(12)])
        return {g: float(sum(cnt[sl] for sl in of._phase_serves.get(g, ()))) for g in range(ts.num_green_phases)}
    def act_enum(t, s):     # enum_frap: 动作 = 绿相位索引本身, q 长度 K_max, 无效位 NaN
        g_sel, k_sel, q = act0(t, s); ts = env.traffic_signals[t]; cur = ts.green_phase; q = q.copy()
        if rule in ("gate", "both"):
            dem = demand_by_green(t, s)
            for k in range(len(q)):
                if np.isnan(q[k]) or k == cur: continue
                if dem.get(k, 0.0) <= 0.0: q[k] = np.nan
        if rule in ("hyst", "both"):
            qc = q[cur]; delta = max(0.05, 0.02 * abs(qc))
            for k in range(len(q)):
                if k != cur and not np.isnan(q[k]) and q[k] <= qc + delta: q[k] = np.nan
        if np.all(np.isnan(q)): return cur, cur, q
        k = int(np.nanargmax(q)); return k, k, q
    def act(t, s):
        g_sel, k_sel, q = act0(t, s); ts = env.traffic_signals[t]; cur_g = ts.green_phase; g2s = ts.std_action_map; cur_k = int(g2s[cur_g])
        q = q.copy()
        if rule in ("gate", "both"):
            dem = demand_by_green(t, s)
            for k in range(8):
                if np.isnan(q[k]) or k == cur_k: continue
                g = int(np.where(g2s == k)[0][0]) if (g2s == k).any() else None
                if g is None or dem.get(g, 0.0) <= 0.0: q[k] = np.nan        # 无车的相位不可选
        if rule in ("hyst", "both"):
            qc = q[cur_k]; delta = max(0.05, 0.02 * abs(qc))
            for k in range(8):
                if k != cur_k and not np.isnan(q[k]) and q[k] <= qc + delta: q[k] = np.nan
        if np.all(np.isnan(q)): return cur_g, cur_k, q
        k = int(np.nanargmax(q)); g = int(np.where(g2s == k)[0][0]); return g, k, q
    B["act"] = act_enum if kind == "enum_frap" else act; return B
ckpt_env.build = build
sys.argv = ["eval_pervehicle.py", exp, "--ckpt", ckpt, "--seed", seed, "--collector", "--tag", f"_{rule}_coll", "--rl-delay", "--rl-delay-tag", f"_{rule}_rldelay"]
runpy.run_path("/home/xiaowen/sumo-rl/experiments/tools/dublin/eval_pervehicle.py", run_name="__main__")
