"""Glue between train.py and the enum_frap action scheme.

Imported ONLY from train.py's `action_scheme: enum_frap` branch — importing
this module has no side effects and old configs never reach it.
Spec: experiments/analysis/FRAP_ENUM_DESIGN_2026-07-02.txt §4.
"""
import json

import numpy as np

SLOTS = [(a, t) for a in ("N", "E", "S", "W") for t in ("L", "T", "R")]


def load_enum_tables(meta_path):
    """dublin_enum_meta.json -> per-TLS tensors + obs turnmap.

    Returns {"k_max": int,
             "tls": {tid: {"pm": (K_max,12)f32, "rel": (12,12)i64,
                            "exist": (12,)f32, "mask": (K_max,)bool}},
             "turnmap": {tid: {link_index: (approach, turn)}}}
    """
    meta = json.load(open(meta_path))
    assert meta.get("action_scheme") == "enum_frap", meta_path
    k_max = int(meta["n_actions"])
    tls_tensors, turnmap = {}, {}
    for tid, t in meta["tls"].items():
        pm = np.zeros((k_max, 12), dtype=np.float32)
        pm[: t["n_phases"]] = np.array(t["phase_movements"], dtype=np.float32)
        rel = np.array(t["movement_rel"], dtype=np.int64)
        exist = (rel.diagonal() >= 0).astype(np.float32)
        tls_tensors[tid] = {"pm": pm, "rel": rel, "exist": exist,
                            "mask": np.array(t["mask"], dtype=bool)}
        turnmap[tid] = {int(i): (c[0]["approach"], c[0]["turn"])
                        for i, c in t["links"].items()}
    return {"k_max": k_max, "tls": tls_tensors, "turnmap": turnmap}


def build_frap_agent(cfg, tables, env, device):
    """Construct FRAPAgent from cfg hyperparams + enum tables.

    Requires obs_phase_state: perphase (junction-independent obs dim:
    header [min_green_ok, elapsed/100] + 12 slots)."""
    from sumo_rl.agents.frap_agent import FRAPAgent
    obs_dim = env.observation_space.shape[0]
    header_dim = 2
    assert cfg.get("obs_phase_state") == "perphase", \
        "enum_frap requires obs_phase_state: perphase"
    slot_dim, rem = divmod(obs_dim - header_dim, 12)
    assert rem == 0, f"obs dim {obs_dim} is not header(2) + 12*slot_dim"
    fp = cfg.get("frap", {}) or {}
    # frap.score (2026-09-30): learned (默认) | pressure | pressure_only。压力模式需要槽位内 queue / 下游 queue 的位置:
    # perphase 槽位布局 = [is_green] + φ(5 级 × fields) + ψ(downstream_fields × 5 级, 若开) + lane_occ(若开) + since_green(若开)。
    score_mode = str(fp.get("score", "learned")); layout = None
    if score_mode != "learned":
        from sumo_rl.environment.observations import PRIORITY_LEVELS
        fields = tuple(cfg.get("obs_fields", ("count", "queue", "mean_awt", "max_awt")))
        if "queue" not in fields:
            raise SystemExit("frap.score pressure* requires 'queue' in obs_fields")
        nl = len(PRIORITY_LEVELS)
        queue_idx = [1 + (k) * len(fields) + fields.index("queue") for k in range(nl)]
        down_idx = None
        if cfg.get("obs_downstream"):
            dfs = tuple(cfg.get("obs_downstream_fields", ("count", "queue")))
            if "queue" in dfs:
                down_idx = [1 + nl * len(fields) + dfs.index("queue") * nl + k for k in range(nl)]
        layout = dict(queue_idx=queue_idx, down_queue_idx=down_idx, levels=[float(l) for l in PRIORITY_LEVELS])
        if down_idx is None:
            print("  → frap.score: 无下游 queue 特征, 压力退化为进口需求 (无出口项)")
    return FRAPAgent(
        obs_dim=obs_dim, header_dim=header_dim, slot_dim=slot_dim,
        tls_tensors=tables["tls"],
        lr=cfg.get("lr", 1e-3), gamma=cfg.get("gamma", 0.95),
        epsilon=cfg.get("epsilon", 0.1), target_update=cfg.get("target_update", 10),
        capacity=cfg.get("capacity", 10000), mini_size=cfg.get("mini_size", 500),
        batch_size=cfg.get("batch_size", 256), eps_start=cfg.get("eps_start", 0.5),
        eps_end=cfg.get("eps_end", 0.01), eps_decay=cfg.get("eps_decay", 1000),
        device=device, embed_dim=int(fp.get("embed_dim", 16)),
        pair_dim=int(fp.get("pair_dim", 16)), k_max=tables["k_max"],
        use_double=cfg.get("use_double", True), loss_fn=cfg.get("loss_fn", "huber"),
        grad_clip=cfg.get("grad_clip", 1.0),
        target_clip_max=cfg.get("target_clip_max", None),
        target_clip_min=cfg.get("target_clip_min", None),
        arch=str(fp.get("arch", "frap")),        # "frap" (default) | "mtt"
        mtt_heads=int(fp.get("mtt_heads", 4)),
        mtt_layers=int(fp.get("mtt_layers", 2)),
        hold_bias=bool(fp.get("hold_bias", False)),   # 2026-09-02: 当前相位保持偏置, 默认关
        score_mode=score_mode, layout=layout,         # 2026-09-30: 压力打分, 默认 learned
        header_in=bool(fp.get("header_in", False)),   # 2026-10-10: 观测头部拼入编码器 (与 8STD 信息对等), 默认关
        menu_hidden=int(fp.get("menu_hidden", 128)),  # 2026-10-10: 仅 arch menuq 用 (菜单条件化 Q 的 MLP 宽度/层数)
        menu_layers=int(fp.get("menu_layers", 2)))


def load_neighbor_map(path, n_neighbors=4):
    """Load a {ts_id: [nb1..nbN]} map (null = missing). Pads/truncates each
    junction's neighbour list to exactly n_neighbors. Used only by the
    mtt_colight arch; absent -> the caller stays on the plain per-junction path."""
    import yaml
    raw = yaml.safe_load(open(path))
    nbm = raw.get("neighbor_map", raw) if isinstance(raw, dict) else {}
    out = {}
    for t, nbs in nbm.items():
        nbs = list(nbs or [])[:n_neighbors]
        nbs += [None] * (n_neighbors - len(nbs))
        out[str(t)] = [None if n in (None, "null", "") else str(n) for n in nbs]
    return out


def build_mtt_colight_agent(cfg, tables, env, device, n_neighbors=4):
    """MTTCoLightAgent from the same enum tables as build_frap_agent, plus the
    neighbour count. Neighbour lookup is done in train.py (via neighbor_map);
    the agent only needs n_neighbors for its qnet."""
    from sumo_rl.agents.frap_agent import MTTCoLightAgent
    obs_dim = env.observation_space.shape[0]
    header_dim = 2
    assert cfg.get("obs_phase_state") == "perphase", \
        "mtt_colight requires obs_phase_state: perphase"
    slot_dim, rem = divmod(obs_dim - header_dim, 12)
    assert rem == 0, f"obs dim {obs_dim} is not header(2) + 12*slot_dim"
    fp = cfg.get("frap", {}) or {}
    return MTTCoLightAgent(
        obs_dim=obs_dim, header_dim=header_dim, slot_dim=slot_dim,
        tls_tensors=tables["tls"],
        lr=cfg.get("lr", 1e-3), gamma=cfg.get("gamma", 0.95),
        epsilon=cfg.get("epsilon", 0.1), target_update=cfg.get("target_update", 10),
        capacity=cfg.get("capacity", 10000), mini_size=cfg.get("mini_size", 500),
        batch_size=cfg.get("batch_size", 256), eps_start=cfg.get("eps_start", 0.5),
        eps_end=cfg.get("eps_end", 0.01), eps_decay=cfg.get("eps_decay", 1000),
        device=device, embed_dim=int(fp.get("embed_dim", 16)),
        pair_dim=int(fp.get("pair_dim", 16)), k_max=tables["k_max"],
        use_double=cfg.get("use_double", True), loss_fn=cfg.get("loss_fn", "huber"),
        grad_clip=cfg.get("grad_clip", 1.0),
        target_clip_max=cfg.get("target_clip_max", None),
        target_clip_min=cfg.get("target_clip_min", None),
        mtt_heads=int(fp.get("mtt_heads", 4)), mtt_layers=int(fp.get("mtt_layers", 2)),
        n_neighbors=n_neighbors)
