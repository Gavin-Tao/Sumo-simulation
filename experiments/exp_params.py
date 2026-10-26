"""配置 → 环境 / 观测 / 奖励参数的共用构造 (2026-10-26, 状态/奖励修正实施计划任务 1)。

train.py、tools/dublin/ckpt_env.py、eval_unified.py 三个入口原来各自从配置读观测参数, eval_unified.py 漏了
obs_awt_log / obs_header / obs_slot_min_green 三个键。改为三处共用本模块, 以后加键只改这里。
键的转换规则逐条照搬 train.py 原来的写法 (旧配置逐位不变)。"""


def resolve_obs_priority_table(cfg: dict):
    """观测分桶表: obs_priority_source (exp227 微调线, 观测与奖励定价解耦) 优先于共享的 priority_source;
    两者都没有 → None, 观测类用内置默认表。"""
    if "obs_priority_source" in cfg:
        return cfg["obs_priority_source"]
    return cfg.get("priority_source")


# (配置键, 观测类构造参数名, 转换函数) —— 顺序与 train.py 原块一致
_OBS_KEYS = (
    ("obs_fields", "fields", tuple),
    ("obs_phase_state", "phase_state", lambda v: v),
    ("obs_downstream", "include_downstream", bool),            # ψ 下游块
    ("obs_downstream_fields", "downstream_fields", tuple),
    ("obs_lane_occ", "include_lane_occ", bool),                # I2 共享车道物理占用
    ("obs_awt_cap", "awt_cap", float),                         # F2 等待截断 (秒)
    ("obs_awt_basis", "awt_basis", str),                       # global | local | visit
    ("obs_slot_stats", "slot_stats", str),                     # intent | lane_copy
    ("obs_since_green", "include_since_green", bool),
    ("obs_awt_log", "awt_log", bool),                          # 等待特征对数刻度
    ("obs_green_elapsed", "include_green_elapsed", bool),
    ("obs_header", "include_header", bool),                    # false → 不输出头部两维
    ("obs_green_elapsed_reset", "green_elapsed_reset", bool),
    ("obs_green_elapsed_exact", "green_elapsed_exact", bool),
    ("obs_slot_min_green", "include_slot_min_green", bool),    # 每槽一位 min_green_ok
    ("obs_green_elapsed_include_yellow", "green_elapsed_include_yellow", bool),
    ("obs_green_elapsed_cap", "green_elapsed_cap", float),
    ("obs_phase_service", "include_phase_service", bool),      # PriorityLaneToken
    ("obs_remote_slots", "remote_slots", int),                 # PriorityLaneTokenNb
    ("obs_pad_tokens", "pad_tokens", int),                     # T 异构网
    ("obs_all_connections", "all_connections", bool),          # 2026-10-26: 建表遍历索引下全部连接 (默认关)
    ("obs_slot_memo", "slot_memo", bool),                      # 2026-10-26: false → 每步重查车辆运动 (默认 true)
)


def obs_kwargs_from_cfg(cfg: dict) -> dict:
    """观测类构造参数: 只放配置里出现的键, 不写默认值 (默认值由观测类自己定, 旧配置逐位不变)。"""
    kw = {}
    for ck, name, conv in _OBS_KEYS:
        if ck in cfg:
            kw[name] = conv(cfg[ck])
        if ck == "obs_phase_state":                    # 优先级表紧跟 phase_state 之后 (与原块顺序一致)
            prio = resolve_obs_priority_table(cfg)
            if prio is not None:
                kw["priority_source"] = prio
    return kw


def reward_wait_basis(cfg: dict) -> str:
    """奖励的等待口径: legacy (现行跨车道记账) | visit (逐秒访问统计器)。"""
    v = str(cfg.get("reward_wait_basis", "legacy"))
    if v not in ("legacy", "visit"):
        raise ValueError(f"reward_wait_basis 必须是 legacy 或 visit, 得到 {v!r}")
    return v


def env_kwargs_from_cfg(cfg: dict) -> dict:
    """SumoEnvironment 的新增参数 (全部有默认值 = 旧行为)。
    wait_tracker: 观测或奖励任一选 visit 口径就建逐秒访问统计器 (两键逻辑或);
    all_connections: TrafficSignal.out_lanes 收齐每个信号索引下全部连接的出口车道。"""
    return {
        "wait_tracker": (str(cfg.get("obs_awt_basis", "")) == "visit") or (reward_wait_basis(cfg) == "visit"),
        "all_connections": bool(cfg.get("all_connections", False)),
    }


def env_extra_kwargs(cfg: dict) -> dict:
    """传给 SumoEnvironment 的新增参数, 只含非默认值 —— 旧配置得到空字典, 构造调用逐字不变。"""
    return {k: v for k, v in env_kwargs_from_cfg(cfg).items() if v}


def turnmap_from_meta(tls_meta: dict) -> dict:
    """8std / enum meta 的 links → {信号索引: (approach, turn)}。同一索引下的多条连接 (SUMO 把同一进口不同车道、
    或分流到两条出口车道的连接合在一个索引下) 必须贴同一个槽标签, 否则观测的槽表无法定义 → 直接报错。"""
    out = {}
    for i, conns in tls_meta["links"].items():
        labels = {(c["approach"], c["turn"]) for c in conns}
        if len(labels) != 1:
            raise ValueError(f"信号索引 {i} 下各连接的槽标签不一致: {sorted(labels)}")
        out[int(i)] = next(iter(labels))
    return out
