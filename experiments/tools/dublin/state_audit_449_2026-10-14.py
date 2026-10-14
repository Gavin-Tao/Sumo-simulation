"""exp449 状态审计 (只读, 不训练): 按 train.py / ckpt_env 的方式构造都柏林 8STD 环境 (按元数据重绑运动槽、挂 std_action_map), 跑若干决策步, 核对:
  (1) 18 个路口观测维数都为 336, 无头部; (2) 全部有限; (3) 槽内 is_green 与 SUMO 实际灯色一致 (非黄灯时), 也与 _phase_serves 一致;
  (4) 槽内 min_green_ok 只在绿灯槽为非零, 且等于 (距上次切换 ≥ min_green + yellow); (5) 各段数值范围。动作: 每个路口在自己的可选标准相位中每 4 步换一个。"""
import os, sys, functools, json, yaml, glob, numpy as np
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
REPO = "/home/xiaowen/sumo-rl"; sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/experiments"); os.chdir(f"{REPO}/experiments")
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
from sumo_rl.environment.priority_map import load_priority_table
from sumo_rl.environment.observations import PRIORITY_LEVELS
cfg = yaml.safe_load(open(glob.glob("configs/exp449_*.yaml")[0])); steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
kw = dict(fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"], priority_source=cfg.get("obs_priority_source", cfg.get("priority_source")),
          include_downstream=bool(cfg["obs_downstream"]), downstream_fields=tuple(cfg["obs_downstream_fields"]), include_lane_occ=bool(cfg["obs_lane_occ"]),
          awt_cap=float(cfg["obs_awt_cap"]), awt_basis=str(cfg["obs_awt_basis"]), awt_log=bool(cfg["obs_awt_log"]),
          include_header=bool(cfg["obs_header"]), include_slot_min_green=bool(cfg["obs_slot_min_green"]))
print("观测参数:", kw)
env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
                      num_seconds=cfg["num_seconds"], min_green=cfg["min_green"], max_green=cfg["max_green"], use_max_green=cfg["use_max_green"], single_agent=False,
                      yellow_time=cfg["yellow_time"], delta_time=cfg["delta_time"], reward_fn=make_priority_avg_waiting_reward(load_priority_table(cfg["priority_source"])),
                      observation_class=functools.partial(obsmod.PriorityMovementObservationFunction, **kw), sumo_seed=cfg["seed"], sumo_warnings=False)
env.reset(123)
meta = json.load(open(cfg["action_meta_file"]))["tls"]; s2g, slot_links = {}, {}
for tid, t in meta.items():
    ts = env.traffic_signals[tid]
    turnmap = {int(i): (c[0]["approach"], c[0]["turn"]) for i, c in t["links"].items()}
    m = np.full(8, -1, int)
    for k, gi in t["std_to_green_index"].items(): m[int(k)] = gi
    g2s = np.full(int(m.max()) + 1, -1, int)
    for k in range(8):
        if m[k] >= 0: g2s[m[k]] = k
    ts.std_action_map = g2s; ts.observation_fn.rebind_movements(turnmap); ts.observation_space = ts.observation_fn.observation_space()
    s2g[tid] = [int(m[k]) for k in range(8) if t["mask"][k]]
states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
PHI = len(kw["fields"]) * len(PRIORITY_LEVELS); PSI = len(PRIORITY_LEVELS) * len(kw["downstream_fields"]); SD = 1 + PHI + PSI + 1 + 1
print(f"18 路口观测维数: {sorted({len(v) for v in states.values()})} (期望 {12 * SD} = 12 × {SD}, 无头部); 路口数 {len(states)}")
P = {"dim": 0, "nonfinite": 0, "green_vs_serves": 0, "green_vs_signal": 0, "mg_on_red": 0, "mg_wrong": 0, "checked_green": 0, "checked_mg": 0}
bad = []; rng = {"is_green": [], "count": [], "queue": [], "wait": [], "psi": [], "occ": [], "mg": []}
for step in range(steps):
    act = {t: s2g[t][(step // 4) % len(s2g[t])] for t in env.ts_ids}
    states, _, done, _ = env.step(action=act)
    for t in env.ts_ids:
        v = np.asarray(states[t]); ts = env.traffic_signals[t]; of = ts.observation_fn
        if len(v) != 12 * SD: P["dim"] += 1; continue
        if not np.all(np.isfinite(v)): P["nonfinite"] += 1
        S = v.reshape(12, SD); served = of._phase_serves.get(ts.green_phase, set())
        mg_ok = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0
        state_str = env.sumo.trafficlight.getRedYellowGreenState(t)
        for s in range(12):
            ig = S[s, 0]; P["green_vs_serves"] += int((ig > 0.5) != (s in served))
            mg = S[s, -1]; P["checked_mg"] += 1
            if s not in served and mg != 0: P["mg_on_red"] += 1
            if s in served and mg != mg_ok: P["mg_wrong"] += 1
            rng["is_green"].append(ig); rng["mg"].append(mg); rng["occ"].append(S[s, 1 + PHI + PSI])
            phi = S[s, 1:1 + PHI].reshape(len(PRIORITY_LEVELS), len(kw["fields"]))
            rng["count"] += list(phi[:, 0]); rng["queue"] += list(phi[:, 1]); rng["wait"] += list(phi[:, 2]); rng["psi"] += list(S[s, 1 + PHI:1 + PHI + PSI])
        # 槽内 is_green 与实际灯色: 非黄灯时, 绿灯槽的受控连接中至少有一条为 G/g
        if not ts.is_yellow:
            for s in range(12):
                idx = [i for i, sl in of._link_slot.items() if sl == s]
                if not idx: continue
                P["checked_green"] += 1
                actual = any(state_str[i] in "Gg" for i in idx)
                if actual != (S[s, 0] > 0.5): P["green_vs_signal"] += 1; bad.append((t, s, state_str, [state_str[i] for i in idx], S[s, 0]))
env.close()
print("问题计数:", P); print("不一致示例:", bad[:5])
for k, x in rng.items():
    x = np.array(x, float); print(f"  {k:8s} 最小 {x.min():.3f} 最大 {x.max():.3f} 非零比例 {np.mean(x != 0):.2f}")
