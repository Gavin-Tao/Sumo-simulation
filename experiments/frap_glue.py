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
        arch=str(fp.get("arch", "frap")),        # "frap" (default) | "mtt"
        mtt_heads=int(fp.get("mtt_heads", 4)),
        mtt_layers=int(fp.get("mtt_layers", 2)))
