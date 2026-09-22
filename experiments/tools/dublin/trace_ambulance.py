"""逐秒追踪评估集里的救护车 (2026-09-22, 用户令: "逐步运行仿真看救护车出现的时候的状态, 从出现到消失").

用法 (仓库根目录):
  python experiments/tools/dublin/trace_ambulance.py 283            # exp283 最终 ckpt, 评估种子 123
  python experiments/tools/dublin/trace_ambulance.py 284 --ckpt <path> --seed 123
只读分析: 不训练、不写 wandb。环境/智能体构造逐键复刻 train.py 的评估分支 (贪心, epsilon=0)。
输出: experiments/analysis/data/amb_trace_exp<exp>_ep<ckpt集数>_seed<seed>.json + 终端摘要。
记录内容 (每仿真秒, 每辆救护车): 所在边/车道/位置/速度/等待时间, 下一个信号灯 (id, 连接序号, 距离, 该连接当前灯色),
前车 (id, 车型, 间距), 该路口当前绿灯相位序号/相位串/距上次切换秒数/是否黄灯, 救护车车道与同路口各进口边的排队数;
每个决策步 (5 s) 另记该路口的动作与各相位 Q 值 (救护车距离 <= 200 m 时)。
"""
import argparse, functools, glob, json, os, sys, collections
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
import numpy as np, torch, yaml

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--near", type=float, default=200.0); a = ap.parse_args()
    if a.ckpt: a.ckpt = os.path.abspath(a.ckpt)   # 先转绝对路径, 再 chdir
    os.chdir(os.path.join(_REPO, "experiments"))
    cfg_path = sorted(glob.glob(f"configs/exp{a.exp}_*.yaml"))[0]; cfg = yaml.safe_load(open(cfg_path))
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    from sumo_rl.environment.priority_map import load_priority_table
    # ── obs class: 与 train.py 同键 ──
    kw = {}
    if "obs_fields" in cfg: kw["fields"] = tuple(cfg["obs_fields"])
    if "obs_phase_state" in cfg: kw["phase_state"] = cfg["obs_phase_state"]
    kw["priority_source"] = cfg.get("obs_priority_source", cfg.get("priority_source"))
    if "obs_downstream" in cfg: kw["include_downstream"] = bool(cfg["obs_downstream"])
    if "obs_downstream_fields" in cfg: kw["downstream_fields"] = tuple(cfg["obs_downstream_fields"])
    if "obs_lane_occ" in cfg: kw["include_lane_occ"] = bool(cfg["obs_lane_occ"])
    if "obs_awt_cap" in cfg: kw["awt_cap"] = float(cfg["obs_awt_cap"])
    if "obs_awt_basis" in cfg: kw["awt_basis"] = str(cfg["obs_awt_basis"])
    if "obs_slot_stats" in cfg: kw["slot_stats"] = str(cfg["obs_slot_stats"])
    obs_class = functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
    reward_fn = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None,
        use_gui=False, num_seconds=cfg.get("num_seconds", 1000), min_green=cfg.get("min_green", 5), max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False), single_agent=False, yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5), reward_fn=reward_fn, observation_class=obs_class, sumo_seed=cfg.get("seed", 0), sumo_warnings=False)
    seed = a.seed if a.seed is not None else int(cfg.get("eval_seed", 123))
    ckpt = a.ckpt or sorted(glob.glob(f"models/{cfg['name']}/*/ckpt_ep*.pth"))[-1]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta_file, scheme = cfg.get("action_meta_file"), cfg.get("action_scheme")
    states = env.reset(seed)
    if meta_file:   # 8std 掩码 DQN (train.py M3 分支)
        from sumo_rl.agents.dqn_agent_txw import DQN
        meta = json.load(open(meta_file))["tls"]; ts_mask, std2green, green2std, turnmap = {}, {}, {}, {}
        for tid, t in meta.items():
            ts_mask[tid] = np.array(t["mask"], dtype=bool)
            turnmap[tid] = {int(i): (c[0]["approach"], c[0]["turn"]) for i, c in t["links"].items()}
            s2g = np.full(8, -1, dtype=int)
            for k, gi in t["std_to_green_index"].items(): s2g[int(k)] = gi
            std2green[tid] = s2g; g2s = np.full(int(s2g.max()) + 1, -1, dtype=int)
            for k in range(8):
                if s2g[k] >= 0: g2s[s2g[k]] = k
            green2std[tid] = g2s
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]; ts.std_action_map = green2std[tid]
            ts.observation_fn.rebind_movements(turnmap[tid]); ts.observation_space = ts.observation_fn.observation_space()
        states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
        od = len(next(iter(states.values())))
        agent = DQN(starting_state=tuple([0.0] * od), state_space=od, hidden_dim=cfg.get("hidden_dim", 64), action_space=8,
                    learning_rate=1e-3, gamma=0.95, epsilon=0.0, target_update=10, capacity=100, mini_size=10**9, batch_size=1,
                    eps_start=0, eps_end=0, eps_decay=1, device="cpu")
        agent.q_net.load_state_dict(ck["policy_state_dict"]); agent.q_net.eval()
        def qvals(t, s):
            with torch.no_grad(): q = agent.q_net(torch.tensor(np.asarray(s, dtype=np.float32)).unsqueeze(0))[0].numpy()
            return np.where(ts_mask[t], q, np.nan)
        def act(t, s):
            q = qvals(t, s); k = int(np.nanargmax(q)); return int(std2green[t][k]), k, q
        phase_label = lambda t, k: f"std{k}"
    elif scheme == "enum_frap":
        from frap_glue import load_enum_tables, build_frap_agent
        tables = load_enum_tables(cfg["enum_meta_file"])
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]; ts.observation_fn.rebind_movements(tables["turnmap"][tid]); ts.observation_space = ts.observation_fn.observation_space()
        states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
        agent = build_frap_agent(cfg, tables, env, "cpu"); agent.q_net.load_state_dict(ck["policy_state_dict"]); agent.q_net.eval(); agent.epsilon = 0.0
        def qvals(t, s):
            i = agent._idx[t]; ii = torch.tensor([i]); pm, rel, exist, mask = agent._tensors(ii)
            with torch.no_grad(): q = agent.q_net(torch.tensor(np.asarray(s, dtype=np.float32)).unsqueeze(0), pm, rel, exist)[0].numpy()
            return np.where(mask[0].numpy(), q, np.nan)
        def act(t, s):
            q = qvals(t, s); k = int(np.nanargmax(q)); return k, k, q
        phase_label = lambda t, k: f"enum{k}"
    else:
        sys.exit("unsupported config (需要 action_meta_file 或 action_scheme=enum_frap)")
    print(f"[trace] exp{a.exp} cfg={cfg_path} ckpt={ckpt} (ep {ck.get('episode')}) seed={seed} n_ts={len(env.ts_ids)}")
    sumo = env.sumo
    # 连接表: (tls, link_index) -> (fromLane, toLane); 进口边集合
    links = {t: sumo.trafficlight.getControlledLinks(t) for t in env.ts_ids}
    in_edges = {t: sorted(set(l[0][0].rsplit("_", 1)[0] for l in links[t] if l)) for t in env.ts_ids}
    in_lanes = {t: sorted(set(l[0][0] for l in links[t] if l)) for t in env.ts_ids}
    trace = collections.defaultdict(list); tls_state_log = collections.defaultdict(list); decisions = []
    seen = set()
    def record():
        now = sumo.simulation.getTime()
        for vid in sumo.vehicle.getIDList():
            if not vid.startswith("amb"): continue
            seen.add(vid)
            nxt = sumo.vehicle.getNextTLS(vid); lead = sumo.vehicle.getLeader(vid, 300.0)
            row = dict(t=now, edge=sumo.vehicle.getRoadID(vid), lane=sumo.vehicle.getLaneID(vid), pos=round(sumo.vehicle.getLanePosition(vid), 1),
                       speed=round(sumo.vehicle.getSpeed(vid), 2), wait=sumo.vehicle.getWaitingTime(vid), acc_wait=sumo.vehicle.getAccumulatedWaitingTime(vid),
                       lane_halt=sumo.lane.getLastStepHaltingNumber(sumo.vehicle.getLaneID(vid)) if not sumo.vehicle.getLaneID(vid).startswith(":") else None)
            if lead and lead[0]:
                row["leader"] = lead[0]; row["leader_gap"] = round(lead[1], 1); row["leader_type"] = sumo.vehicle.getTypeID(lead[0]); row["leader_speed"] = round(sumo.vehicle.getSpeed(lead[0]), 2)
            if nxt:
                tid, li, dist, st = nxt[0]; row.update(tls=tid, link=li, dist=round(dist, 1), link_state=st)
                if tid in env.traffic_signals:
                    ts = env.traffic_signals[tid]
                    row.update(green_phase=ts.green_phase, phase_label=phase_label(tid, int(ts.std_action_map[ts.green_phase]) if meta_file else ts.green_phase),
                               ryg=sumo.trafficlight.getRedYellowGreenState(tid), since_change=ts.time_since_last_phase_change, yellow=ts.is_yellow,
                               halt_by_edge={e: sum(sumo.lane.getLastStepHaltingNumber(l) for l in in_lanes[tid] if l.rsplit("_", 1)[0] == e) for e in in_edges[tid]},
                               veh_by_edge={e: sum(sumo.lane.getLastStepVehicleNumber(l) for l in in_lanes[tid] if l.rsplit("_", 1)[0] == e) for e in in_edges[tid]})
            trace[vid].append(row)
        for t in env.ts_ids: tls_state_log[t].append(sumo.trafficlight.getRedYellowGreenState(t))
    _orig = env._sumo_step
    def hooked():
        _orig(); record()
    env._sumo_step = hooked
    record()
    done = {"__all__": False}
    while not done["__all__"]:
        actions = {}
        amb_near = {}
        for vid in sumo.vehicle.getIDList():
            if vid.startswith("amb"):
                nxt = sumo.vehicle.getNextTLS(vid)
                if nxt and nxt[0][2] <= a.near: amb_near[nxt[0][0]] = (vid, round(nxt[0][2], 1))
        for t in env.ts_ids:
            g, k, q = act(t, states[t]); actions[t] = g
            if t in amb_near:
                ts = env.traffic_signals[t]
                decisions.append(dict(t=sumo.simulation.getTime(), tls=t, amb=amb_near[t][0], dist=amb_near[t][1], chosen=k, chosen_label=phase_label(t, k),
                                      current=int(ts.std_action_map[ts.green_phase]) if meta_file else ts.green_phase, since_change=ts.time_since_last_phase_change,
                                      q=[None if np.isnan(x) else round(float(x), 3) for x in q], obs=[round(float(x), 3) for x in np.asarray(states[t]).ravel()]))
        states, _, done, _ = env.step(action=actions)
    out = dict(exp=a.exp, ckpt=ckpt, seed=seed, in_edges=in_edges, links={t: [(l[0][0], l[0][1]) if l else None for l in links[t]] for t in env.ts_ids},
               trace=dict(trace), decisions=decisions, phase_share={t: dict(collections.Counter(tls_state_log[t])) for t in env.ts_ids})
    os.makedirs("analysis/data", exist_ok=True); path = f"analysis/data/amb_trace_exp{a.exp}_ep{int(ck.get('episode', 0)):04d}_seed{seed}.json"
    json.dump(out, open(path, "w")); print("saved", path)
    env.close()
    # ── 终端摘要 ──
    for vid, rows in trace.items():
        stops = []; cur = None
        for r in rows:
            if r["speed"] < 0.1:
                if cur is None: cur = dict(start=r["t"], edge=r["edge"], lane=r["lane"], tls=r.get("tls"), dist=r.get("dist"), link_state=r.get("link_state"), phase=r.get("phase_label"),
                                          since=r.get("since_change"), leader=r.get("leader_type"), gap=r.get("leader_gap"), halt=r.get("lane_halt"), halt_by_edge=r.get("halt_by_edge"))
                cur["end"] = r["t"]
            elif cur is not None: stops.append(cur); cur = None
        if cur is not None: stops.append(cur)
        tls_seq = []
        for r in rows:
            if r.get("tls") and (not tls_seq or tls_seq[-1] != r["tls"]): tls_seq.append(r["tls"])
        print(f"\n== {vid}: {rows[0]['t']:.0f}s → {rows[-1]['t']:.0f}s ({rows[-1]['t']-rows[0]['t']:.0f}s 在网), 途经信号灯 {len(tls_seq)}: {[t[:18] for t in tls_seq]}")
        print(f"   停车段 {len(stops)}: 总停车 {sum(s['end']-s['start']+1 for s in stops):.0f}s")
        for s in stops:
            print(f"   - {s['start']:.0f}-{s['end']:.0f}s ({s['end']-s['start']+1:.0f}s) 边 {s['edge']} 距灯 {s['dist']}m 灯 {str(s['tls'])[:18]} 连接灯色 {s['link_state']} 相位 {s['phase']} 已持续 {s['since']}s | 前车 {s['leader']} 间距 {s['gap']} | 本车道停车数 {s['halt']} | 各进口排队 {s['halt_by_edge']}")
if __name__ == "__main__": main()
