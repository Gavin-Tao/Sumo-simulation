"""只读分析用: 按训练配置复刻评估环境 + 贪心智能体 (与 train.py 评估分支同键), 供 eval_pervehicle.py 等工具调用。
不修改任何训练代码; 支持 action_meta_file (掩码 DQN, 含 1x1/1x3 完整8 与 Dublin 8std) 与 action_scheme=enum_frap (FRAP)。
"""
import functools, glob, json, os, sys
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
import numpy as np, torch, yaml

def find_cfg(exp):
    os.chdir(os.path.join(_REPO, "experiments"))
    c = sorted(glob.glob(f"configs/exp{exp}_*.yaml"))
    c = [x for x in c if "traindry" not in x and "smoke" not in x]
    return c[0]

def list_ckpts(cfg):
    """所有 ckpt_epNNNNN.pth (跨时间戳目录), 按集数排序; 同集数取最新目录。"""
    best = {}
    for p in glob.glob(f"models/{cfg['name']}/*/ckpt_ep*.pth"):
        ep = int(os.path.basename(p)[7:12])
        if ep not in best or os.path.getmtime(p) > os.path.getmtime(best[ep]): best[ep] = p
    return [best[k] for k in sorted(best)]

def build(exp, ckpt=None, seed=None, extra_sumo=None):
    """返回 dict(env, act, states, cfg, ckpt, kind, agent, phase_label)。act(ts, state) -> (env 动作, 内部动作序号, Q 向量)。"""
    if ckpt: ckpt = os.path.abspath(ckpt)
    cfg_path = find_cfg(exp); cfg = yaml.safe_load(open(cfg_path))
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    from sumo_rl.environment.priority_map import load_priority_table
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
    assert cfg["observation_class"] == "PriorityMovement", cfg["observation_class"]
    obs_class = functools.partial(obsmod.PriorityMovementObservationFunction, **kw)
    reward_fn = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"], cfg_file=cfg["cfg_file"], out_csv_name=None,
        use_gui=False, num_seconds=cfg.get("num_seconds", 1000), min_green=cfg.get("min_green", 5), max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False), single_agent=False, yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5), reward_fn=reward_fn, observation_class=obs_class, sumo_seed=cfg.get("seed", 0),
        sumo_warnings=False, additional_sumo_cmd=extra_sumo)
    seed = seed if seed is not None else int(cfg.get("eval_seed", 123))
    ckpt = ckpt or list_ckpts(cfg)[-1]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta_file, scheme = cfg.get("action_meta_file"), cfg.get("action_scheme")
    states = env.reset(seed)
    if meta_file:
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
        def act(t, s):
            with torch.no_grad(): q = agent.q_net(torch.tensor(np.asarray(s, dtype=np.float32)).unsqueeze(0))[0].numpy()
            q = np.where(ts_mask[t], q, np.nan); k = int(np.nanargmax(q)); return int(std2green[t][k]), k, q
        kind = "dqn_masked"
    elif scheme == "enum_frap":
        from frap_glue import load_enum_tables, build_frap_agent
        tables = load_enum_tables(cfg["enum_meta_file"])
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]; ts.observation_fn.rebind_movements(tables["turnmap"][tid]); ts.observation_space = ts.observation_fn.observation_space()
        states = {t: env.traffic_signals[t].observation_fn() for t in env.ts_ids}
        agent = build_frap_agent(cfg, tables, env, "cpu"); agent.q_net.load_state_dict(ck["policy_state_dict"]); agent.q_net.eval(); agent.epsilon = 0.0
        def act(t, s):
            i = agent._idx[t]; pm, rel, exist, mask = agent._tensors(torch.tensor([i]))
            with torch.no_grad(): q = agent.q_net(torch.tensor(np.asarray(s, dtype=np.float32)).unsqueeze(0), pm, rel, exist)[0].numpy()
            q = np.where(mask[0].numpy(), q, np.nan); k = int(np.nanargmax(q)); return k, k, q
        kind = "enum_frap"
    else:
        raise SystemExit("unsupported config: 需要 action_meta_file 或 action_scheme=enum_frap")
    return dict(env=env, act=act, states=states, cfg=cfg, cfg_path=cfg_path, ckpt=ckpt, ckpt_episode=int(ck.get("episode", 0)), seed=seed, kind=kind, agent=agent)
