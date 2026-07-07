"""Unified DQN training script for SUMO-RL.

Usage:
    python experiments/train.py --config experiments/configs/exp18_presslight.yaml

Each experiment is fully described by its YAML config file.
Run name in wandb: {exp_name}_{timestamp}  — distinguishes both
  different experiments (by exp_name) and repeated runs (by timestamp).
Checkpoints: ./models/{exp_name}/{timestamp}/ckpt_ep{episode:05d}.pth
"""

import argparse
import functools
import json
import math
import os
import sys
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb

# ── Project root ──────────────────────────────────────────────────────────────
os.environ["SUMO_RL_LIBSUMO"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
probe = PROJECT_ROOT
while not (probe / "sumo_rl").exists() and probe.parent != probe:
    probe = probe.parent
PROJECT_ROOT = probe
sys.path.insert(0, str(PROJECT_ROOT))

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.dqn_agent_txw import DQN
from sumo_rl.environment.metrics import EpisodeMetricsCollector
from sumo_rl.environment.observations import (
    DefaultObservationFunction,
    PressLightObservationFunction,
    PressLightNormObservationFunction,
    QueuePrioObservationFunction,
    CTBPriorityObservationFunction,
    PriorityObservationFunction,
    PriorityCtrlObservationFunction,
    PriorityNormObservationFunction,
    DiffWaitingObservationFunction,
    PriorityDiffWaitingObservationFunction,
    PriorityBCAObservationFunction,
    PriorityCtrlBCAObservationFunction,
    PriorityWaitingBCAObservationFunction,
    PriorityPhaseObservationFunction,
    PriorityMovementObservationFunction,
    PriorityLaneTokenObservationFunction,
    PriorityLaneTokenNbObservationFunction,
)

# ── Registries ────────────────────────────────────────────────────────────────
OBS_REGISTRY = {
    "Default":             DefaultObservationFunction,
    "PressLight":          PressLightObservationFunction,
    "PressLightNorm":      PressLightNormObservationFunction,
    "QueuePrio":           QueuePrioObservationFunction,
    "CTBPriority":         CTBPriorityObservationFunction,
    "Priority":            PriorityObservationFunction,
    "PriorityCtrl":        PriorityCtrlObservationFunction,
    "PriorityNorm":        PriorityNormObservationFunction,
    "DiffWaiting":         DiffWaitingObservationFunction,
    "PriorityDiffWaiting": PriorityDiffWaitingObservationFunction,
    "PriorityBCA":            PriorityBCAObservationFunction,
    "PriorityCtrlBCA":        PriorityCtrlBCAObservationFunction,
    "PriorityWaitingBCA":     PriorityWaitingBCAObservationFunction,
    # A/B/T priority-bucket families (configure via obs_phase_state + obs_fields)
    "PriorityPhase":          PriorityPhaseObservationFunction,      # A: per phase
    "PriorityMovement":       PriorityMovementObservationFunction,   # B: per turning movement
    "PriorityLaneToken":      PriorityLaneTokenObservationFunction,  # T (simplified): per lane
    "PriorityLaneTokenNb":    PriorityLaneTokenNbObservationFunction,  # T + boundary tokens (obs-level coord)
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def resolve_obs_priority_table(cfg: dict):
    """obs bucket table: obs_priority_source (exp227 fine-tune line, decouples
    obs wiring from reward re-pricing) wins over the legacy shared key;
    None -> observation class built-in default table."""
    if "obs_priority_source" in cfg:
        return cfg["obs_priority_source"]
    return cfg.get("priority_source")


def save_checkpoint(agent: DQN, episode: int, model_dir: str, filename: str = None) -> str:
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = {
        "policy_state_dict":   agent.q_net.state_dict(),
        "target_state_dict":   agent.target_q_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "episode":             episode,
    }
    path = os.path.join(model_dir, filename or f"ckpt_ep{episode:05d}.pth")
    torch.save(checkpoint, path)
    return path


# ── Main training loop ────────────────────────────────────────────────────────
def train(cfg: dict, timestamp: str):
    exp_name = cfg["name"]
    run_id   = f"{exp_name}_{timestamp}"

    # ── Output paths ──────────────────────────────────────────────────────────
    model_dir = os.path.join("./models", exp_name, timestamp)
    os.makedirs("./tmux", exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(cfg["cfg_file"]), "output"), exist_ok=True)

    # ── Seed ──────────────────────────────────────────────────────────────────
    set_seed(cfg.get("seed", 0))
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (no CUDA GPU available)")

    # logging_mode: "full" | "basic" | "simple" | "none"
    logging_mode = cfg.get("logging_mode", "basic")

    # ── wandb ─────────────────────────────────────────────────────────────────
    if logging_mode != "none":
        wandb.init(
            project=cfg.get("wandb_project", "sumo-rl"),
            group=exp_name,
            name=run_id,
            config=cfg,
            dir="./logs/wandb",
            reinit=True,
        )

    # ── Environment ───────────────────────────────────────────────────────────
    obs_class = OBS_REGISTRY[cfg["observation_class"]]
    # Optional obs params (A/B/T families): bind via partial so env can call obs_class(ts).
    obs_kwargs = {}
    if "obs_fields" in cfg:
        obs_kwargs["fields"] = tuple(cfg["obs_fields"])
    if "obs_phase_state" in cfg:
        obs_kwargs["phase_state"] = cfg["obs_phase_state"]
    _obs_prio = resolve_obs_priority_table(cfg)
    if _obs_prio is not None:
        obs_kwargs["priority_source"] = _obs_prio
    if "obs_downstream" in cfg:      # ψ downstream block (E1b) — B family
        obs_kwargs["include_downstream"] = bool(cfg["obs_downstream"])
    if "obs_downstream_fields" in cfg:   # ψ field ablation: subset of (count, queue)
        obs_kwargs["downstream_fields"] = tuple(cfg["obs_downstream_fields"])
    if "obs_lane_occ" in cfg:        # I2 (B only): shared-lane physical occupancy, +1/slot
        obs_kwargs["include_lane_occ"] = bool(cfg["obs_lane_occ"])
    if "obs_awt_cap" in cfg:         # F2 (B only): bound mean/max_awt at N raw seconds
        obs_kwargs["awt_cap"] = float(cfg["obs_awt_cap"])
    if "obs_awt_basis" in cfg:       # A/B/T: mean/max_awt accounting basis (global|local)
        obs_kwargs["awt_basis"] = str(cfg["obs_awt_basis"])
    if "obs_slot_stats" in cfg:      # B only: intent (default) | lane_copy ablation
        obs_kwargs["slot_stats"] = str(cfg["obs_slot_stats"])
    if "obs_phase_service" in cfg:   # PriorityLaneToken only: lane-identity multi-hot (R1)
        obs_kwargs["include_phase_service"] = bool(cfg["obs_phase_service"])
    if "obs_remote_slots" in cfg:    # PriorityLaneTokenNb only: override auto slot count
        obs_kwargs["remote_slots"] = int(cfg["obs_remote_slots"])
    if "obs_pad_tokens" in cfg:      # T on heterogeneous nets: uniform own-token count
        obs_kwargs["pad_tokens"] = int(cfg["obs_pad_tokens"])
    if obs_kwargs:
        obs_class = functools.partial(obs_class, **obs_kwargs)
    # Table-driven reward (reward-side dual of φ): resolve to a callable here —
    # it needs this experiment's priority table, so it can't live in the registry.
    reward_fn = cfg["reward_fn"]
    # ── agent_arch: multihead — B-line per-level value heads with decision-
    # time BNF weighting (MULTIHEAD_BNF_DESIGN doc). The vector reward glue
    # returns the IDENTICAL scalar (env/logging unchanged) and caches the
    # per-level decomposition for the replay buffer. Mutually exclusive with
    # enum_frap/moe (their guards require agent_arch mlp) and transformer.
    multihead = cfg.get("agent_arch", "mlp") == "multihead"
    mh_reward_cache = None
    if multihead:
        if cfg["reward_fn"] != "priority-avg-waiting":
            sys.exit("agent_arch=multihead requires reward_fn: "
                     "priority-avg-waiting (exact per-level decomposition)")
        if cfg.get("use_per"):
            sys.exit("multihead + use_per not supported")
        if float(cfg.get("reward_scale", 1.0)) != 1.0 \
                or cfg.get("reward_floor") is not None:
            sys.exit("multihead v1 excludes reward_scale/reward_floor "
                     "(scalar-only wrappers would desync the stored vector)")
        from multihead_glue import make_priority_avg_waiting_reward_vec
        from sumo_rl.environment.priority_map import load_priority_table
        reward_fn, mh_reward_cache = make_priority_avg_waiting_reward_vec(
            load_priority_table(cfg.get("priority_source")))
        print("  → agent_arch=multihead: 5 per-level heads, "
              "decision-time BNF weights")
    if reward_fn == "priority-avg-waiting":
        from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
        from sumo_rl.environment.priority_map import load_priority_table
        reward_fn = make_priority_avg_waiting_reward(
            load_priority_table(cfg.get("priority_source")))
    # reward_scale (Dublin survival package, 2026-06-12): multiply the reward
    # by a constant to bring Dublin's per-step magnitudes (~35x the 1x3 ones
    # in congestion) back into the regime where the standard DQN package is
    # PROVEN stable. Default 1.0 = no effect. NOTE: eval_mean_reward and all
    # logged rewards scale with it — only compare runs sharing the same scale.
    _rs = float(cfg.get("reward_scale", 1.0))
    if _rs != 1.0 and callable(reward_fn):
        _base_reward_fn = reward_fn
        def reward_fn(ts, _f=_base_reward_fn, _s=_rs):  # noqa: F811
            return _f(ts) * _s
        print(f"[reward_scale] active: x{_rs}")
    # reward_floor (R2, QMAX_DIVERGENCE_FORENSICS Part 7): clip the training
    # reward at a lower bound (Atari-style, local form). Bites only in
    # gridlock (r < floor ⇔ weighted avg wait > |floor|*100 s) — bounds the
    # TD-target scale explosion. NOTE: the in-training eval episodes share
    # this wrapper, so eval_system/mean_reward is floor-compressed for runs
    # using it; cross-arm comparisons should use waiting metrics. Default
    # absent = original behaviour.
    _rfloor = cfg.get("reward_floor", None)
    if _rfloor is not None and callable(reward_fn):
        _base_reward_fn2 = reward_fn
        def reward_fn(ts, _f=_base_reward_fn2, _c=float(_rfloor)):
            return max(_f(ts), _c)
        print(f"[reward_floor] active: {_rfloor}")
    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"],
        out_csv_name=None,
        use_gui=cfg.get("use_gui", False),
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5),
        max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=cfg.get("single_agent", False),
        yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5),
        reward_fn=reward_fn,
        observation_class=obs_class,
        sumo_seed=cfg.get("seed", 0),
        sumo_warnings=True,
    )

    episodes            = cfg.get("episodes", 5000)
    checkpoint_interval = cfg.get("checkpoint_interval", 5)
    eval_interval       = cfg.get("eval_interval", 0)      # 0 = disabled
    training_seed       = int(cfg.get("seed", 0))
    eval_seed           = cfg.get("eval_seed", 42)         # different from training seed (0) by default
    # ── M3 (Dublin): static per-junction action masks from the reindex meta ──
    # cfg action_meta_file -> dublin_8std_meta.json. Fixes action_dim=8;
    # take_action samples/argmaxes only valid std actions; env receives the
    # mapped green-phase index; the buffer stores the executed STD action and
    # the junction's mask (for the masked target max).
    action_meta_file = cfg.get("action_meta_file")
    if action_meta_file and cfg.get("use_per"):
        sys.exit("action_meta_file + use_per: masked PER not implemented yet")
    ts_mask, std2green, green2std = {}, {}, {}
    if action_meta_file:
        with open(action_meta_file) as _f:
            _meta = json.load(_f)["tls"]
        ts_turnmap = {}   # per ts: link index -> (approach, turn) — the
        # reindexer's arbitrated mapping (compass tie-break), shared with obs
        for _tid, _t in _meta.items():
            ts_mask[_tid] = np.array(_t["mask"], dtype=bool)
            ts_turnmap[_tid] = {int(_i): (_c[0]["approach"], _c[0]["turn"])
                                for _i, _c in _t["links"].items()}
            _s2g = np.full(8, -1, dtype=int)
            for _a, _gi in _t["std_to_green_index"].items():
                _s2g[int(_a)] = _gi
            std2green[_tid] = _s2g
            _g2s = np.full(int(_s2g.max()) + 1, -1, dtype=int)
            for _a in range(8):
                if _s2g[_a] >= 0:
                    _g2s[_s2g[_a]] = _a
            green2std[_tid] = _g2s

    # ── enum_frap (spec FRAP_ENUM_DESIGN_2026-07-02): enumerated protected
    # phases + movement-duel agent. Old configs lack action_scheme -> None ->
    # every enum line below is skipped; mutually exclusive with action_meta_file.
    action_scheme = cfg.get("action_scheme")
    enum_tables = None
    moe = None
    if action_scheme not in (None, "enum_frap", "moe"):
        sys.exit(f"unknown action_scheme: {action_scheme}")
    if action_scheme == "moe":
        # MoE-Gcμ (RESEARCH_ROADMAP §A1): 6 rule experts + DQN gate.
        # Menu (L1) is configurable: moe_meta_file may be an enum meta OR an
        # 8std meta — moe_glue handles both schemas. Old configs lack the
        # key and never reach this branch.
        if action_meta_file:
            sys.exit("action_scheme=moe is mutually exclusive with action_meta_file")
        if cfg.get("use_per"):
            sys.exit("moe + use_per not supported")
        if cfg.get("agent_arch", "mlp") != "mlp":
            sys.exit("moe gate uses the plain DQN; agent_arch must be absent/mlp")
        from moe_glue import load_moe_tables
        moe = load_moe_tables(cfg["moe_meta_file"])
        moe_lex = bool(cfg.get("moe_lexicographic", False))
        # structural validity mask: absent-level experts are not selectable
        # (undefined delegation; prevents the free-hold exploit, doc §6)
        moe_presence = bool(cfg.get("moe_presence_mask", False))
        print(f"  → action_scheme=moe: {len(moe['tables'])} TLS, "
              f"menus={sorted(len(t['phase_slots']) for t in moe['tables'].values())}, "
              f"lexicographic={moe_lex}")
    if action_scheme == "enum_frap":
        if action_meta_file:
            sys.exit("action_scheme=enum_frap is mutually exclusive with action_meta_file")
        if cfg.get("use_per"):
            sys.exit("enum_frap + use_per not supported (uniform replay by design)")
        if cfg.get("agent_arch", "mlp") != "mlp":
            sys.exit("enum_frap defines its own network; agent_arch must be absent/mlp")
        from frap_glue import load_enum_tables
        enum_tables = load_enum_tables(cfg["enum_meta_file"])
        print(f"  → action_scheme=enum_frap: K_max={enum_tables['k_max']}, "
              f"{len(enum_tables['tls'])} TLS")

    def _masked_reset(_env, _seed):
        """env.reset + (masked mode) stash green->std maps on the freshly
        recreated TrafficSignal objects and recompute obs so the legacy
        one-hot is emitted in the fixed 8-dim std space from step one."""
        _states = _env.reset(int(_seed))
        if action_meta_file:
            for _tid in _env.ts_ids:
                _ts = _env.traffic_signals[_tid]
                _ts.std_action_map = green2std[_tid]
                # obs/action use the SAME movement mapping (C2): rebind the
                # obs slot tables to the reindexer's arbitrated link->
                # (approach, turn) map — fixes the compass-sector collision
                # at the 4-arm junction where raw rounding merges two arms
                if hasattr(_ts.observation_fn, "rebind_movements"):
                    _ts.observation_fn.rebind_movements(ts_turnmap[_tid])
                # TrafficSignal caches observation_space at __init__ (pre-
                # stash, K-dim one-hot) — refresh to the 8-dim std space
                _ts.observation_space = _ts.observation_fn.observation_space()
            _states = {_tid: _env.traffic_signals[_tid].observation_fn()
                       for _tid in _env.ts_ids}
        if enum_tables is not None:
            for _tid in _env.ts_ids:
                _ts = _env.traffic_signals[_tid]
                if hasattr(_ts.observation_fn, "rebind_movements"):
                    _ts.observation_fn.rebind_movements(enum_tables["turnmap"][_tid])
                _ts.observation_space = _ts.observation_fn.observation_space()
            _states = {_tid: _env.traffic_signals[_tid].observation_fn()
                       for _tid in _env.ts_ids}
        if moe is not None:
            for _tid in _env.ts_ids:
                _ts = _env.traffic_signals[_tid]
                if hasattr(_ts.observation_fn, "rebind_movements"):
                    _ts.observation_fn.rebind_movements(moe["turnmap"][_tid])
                _ts.observation_space = _ts.observation_fn.observation_space()
            _states = {_tid: _env.traffic_signals[_tid].observation_fn()
                       for _tid in _env.ts_ids}
        return _states

    # ── Training ──────────────────────────────────────────────────────────────
    for _ in range(1, cfg.get("runs", 1) + 1):
        initial_states = _masked_reset(env, training_seed)
        last_ts_id = list(env.ts_ids)[-1]

        # Build lane map once (lane structure is fixed across episodes)
        ts_lane_map  = {ts: env.traffic_signals[ts].signal_controlled_lanes for ts in env.ts_ids}
        always_green = set().union(*(env.traffic_signals[ts].always_green_lanes for ts in env.ts_ids))

        # ── Optional Q-net architecture override (agent_arch: "transformer") ──────
        # Lane-token Transformer (Scheme T full version minus static geometry):
        # requires the PriorityLaneToken obs family (token_layout() defines the split).
        # Action space unchanged — CLS readout → K fixed Q-values.
        q_net_factory = None
        if cfg.get("agent_arch", "mlp") == "transformer":
            from sumo_rl.agents.qnet_lane_transformer import LaneTokenTransformerQNet
            obs_fn0 = env.traffic_signals[last_ts_id].observation_fn
            if not hasattr(obs_fn0, "token_layout"):
                raise ValueError(
                    f"agent_arch=transformer needs a token obs (PriorityLaneToken*), "
                    f"got {cfg['observation_class']}")
            layout = obs_fn0.token_layout()
            tf_cfg = cfg.get("transformer", {}) or {}
            assert layout["header_dim"] + layout["n_tokens"] * layout["token_dim"] \
                == env.observation_space.shape[0], "token_layout inconsistent with obs dim"
            q_net_factory = lambda: LaneTokenTransformerQNet(
                header_dim=layout["header_dim"], n_tokens=layout["n_tokens"],
                token_dim=layout["token_dim"], action_dim=(8 if action_meta_file else env.action_space.n),
                d_model=int(tf_cfg.get("d_model", 128)), nhead=int(tf_cfg.get("nhead", 4)),
                num_layers=int(tf_cfg.get("num_layers", 2)), dim_ff=int(tf_cfg.get("dim_ff", 256)),
            )
            print(f"  → Q-net arch: LaneTokenTransformer  layout={layout}  "
                  f"d_model={tf_cfg.get('d_model',128)} heads={tf_cfg.get('nhead',4)} "
                  f"layers={tf_cfg.get('num_layers',2)} ff={tf_cfg.get('dim_ff',256)}")

        # ── Transformer eval diagnostics: CLS attention over lane tokens ──────────
        # eval_diag/cls_attn_entropy: low = CLS focuses few lanes; eval_diag/cls_attn_max:
        # peak attention; eval_diag/cls_attn_on_amb: attention mass on tokens whose p5
        # (ambulance) count > 0 — "does the net look at the ambulance lane?"
        attn_diag = None
        if q_net_factory is not None and hasattr(obs_fn0, "token_layout"):
            nf = len(obs_fn0.fields)
            K_ts = env.traffic_signals[last_ts_id].num_green_phases
            tok_base = layout.get("phi_off",
                (0 if obs_fn0._is_legacy else 1)
                + (K_ts if obs_fn0.include_phase_service else 0))
            amb_off = (tok_base + 4 * nf + obs_fn0.fields.index("count")) \
                      if "count" in obs_fn0.fields else None
            attn_diag = {"layout": layout, "amb_off": amb_off}

        moe_experts = None
        if moe is not None:
            import moe_glue
            moe_experts = moe_glue.build_experts(cfg, moe, env)
            print(f"  → agent: DQN gate over {6} rule experts (MoE-Gcμ)")
        if multihead:
            from multihead_glue import build_multihead_agent
            agent = build_multihead_agent(
                cfg,
                starting_state=tuple(initial_states[last_ts_id]),
                state_space=env.observation_space.shape[0],
                action_space=(8 if action_meta_file else env.action_space.n),
                device=device)
            print(f"  → agent: DQNMultiHead (5 heads × {agent.action_space}), "
                  f"decision weights={agent.weights.tolist()}")
        elif enum_tables is not None:
            from frap_glue import build_frap_agent
            agent = build_frap_agent(cfg, enum_tables, env, device)
            print(f"  → agent: FRAPAgent (movement-duel), obs_dim="
                  f"{env.observation_space.shape[0]}")
        else:
            agent = DQN(
            starting_state=tuple(initial_states[last_ts_id]),
            state_space=env.observation_space.shape[0],
            hidden_dim=cfg.get("hidden_dim", 64),
            # moe: gate over 6 experts; masked-8std: fixed std space; else native
            action_space=(6 if moe is not None
                          else (8 if action_meta_file else env.action_space.n)),
            learning_rate=cfg.get("lr", 0.01),
            gamma=cfg.get("gamma", 0.99),
            epsilon=cfg.get("epsilon", 0.1),
            target_update=cfg.get("target_update", 10),
            capacity=cfg.get("capacity", 10000),
            mini_size=cfg.get("mini_size", 500),
            batch_size=cfg.get("batch_size", 256),
            eps_start=cfg.get("eps_start", 0.5),
            eps_end=cfg.get("eps_end", 0.01),
            eps_decay=cfg.get("eps_decay", 1000),
            device=device,
            use_double=cfg.get("use_double", False),
            loss_fn=cfg.get("loss_fn", "mse"),
            target_clip_max=cfg.get("target_clip_max", None),
            use_per=cfg.get("use_per", False),
            per_alpha=cfg.get("per_alpha", 0.6),
            per_beta_start=cfg.get("per_beta_start", 0.4),
            per_beta_end=cfg.get("per_beta_end", 1.0),
            per_beta_steps=cfg.get("per_beta_steps", 100_000),
            per_eps=cfg.get("per_eps", 1e-6),
            q_net_factory=q_net_factory,
            grad_clip=cfg.get("grad_clip", None),
        )

        # ── warm-start (exp227 fine-tune line): policy weights only; target
        # synced to policy; optimizer/replay/epsilon start fresh under THIS
        # config's schedule. Key absent -> cold start, bit-identical legacy.
        init_ckpt = cfg.get("init_checkpoint")
        if init_ckpt:
            if not os.path.isfile(init_ckpt):
                sys.exit(f"init_checkpoint not found: {init_ckpt}")
            _ck = torch.load(init_ckpt, map_location=device, weights_only=False)
            agent.q_net.load_state_dict(_ck["policy_state_dict"])
            agent.target_q_net.load_state_dict(agent.q_net.state_dict())
            print(f"  → warm-start: policy loaded from {init_ckpt} "
                  f"(source ep {_ck.get('episode', '?')}); "
                  f"replay/epsilon/optimizer fresh")

        step_counter = 0
        best_eval_reward = -float("inf")
        best_eval_hist: list = []   # sliding window for the best-ckpt criterion (B2 fix)

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = _masked_reset(env, training_seed)

            done = {"__all__": False}
            phase_counts = {ts_id: {} for ts_id in env.ts_ids}
            ep_losses: list = []
            ep_gnorms: list = []
            ep_qmeans: list = []
            ep_qmaxs:  list = []

            try:
                while not done["__all__"]:
                    # ── Act ───────────────────────────────────────────────────────
                    if moe is not None:
                        # MoE: gate picks WHICH expert; expert's argmin phase
                        # (a dense green index) goes to the env.
                        actions, moe_k = {}, {}
                        import moe_glue as _mg
                        for ts in env.ts_ids:
                            _sumo = env.traffic_signals[ts].sumo
                            _props, _lv = moe_experts.propose(
                                ts, _sumo, env.traffic_signals[ts].green_phase)
                            _m = _mg.gate_mask(_lv, moe_lex, presence=moe_presence)
                            _k = int(agent.take_action(initial_states[ts], mask=_m))  # type: ignore[call-arg]  # DQN here
                            moe_k[ts] = _k
                            actions[ts] = int(_props[_k])
                    elif enum_tables is not None:
                        # enum_frap: action IS the dense green-phase index (identity)
                        actions = {ts: agent.take_action(initial_states[ts], ts)
                                   for ts in env.ts_ids}
                    elif action_meta_file:
                        actions = {ts: int(std2green[ts][agent.take_action(
                            initial_states[ts], mask=ts_mask[ts])]) for ts in env.ts_ids}
                    else:
                        actions = {ts: agent.take_action(initial_states[ts]) for ts in env.ts_ids}
                    s, r, done, info = env.step(action=actions)

                    if agent.loss is not None:
                        ep_losses.append(agent.loss)
                        if agent.grad_norm is not None:
                            ep_gnorms.append(agent.grad_norm)
                        if agent.q_mean is not None:
                            ep_qmeans.append(agent.q_mean)
                            ep_qmaxs.append(agent.q_abs_max)
                    step_counter += 1

                    # ── Track phase selection ─────────────────────────────────────
                    for ts_id in env.ts_ids:
                        p = env.traffic_signals[ts_id].green_phase
                        phase_counts[ts_id][p] = phase_counts[ts_id].get(p, 0) + 1

                    # ── Store experience ──────────────────────────────────────────
                    for ts in env.ts_ids:
                        actual_action = env.traffic_signals[ts].last_executed_action
                        ts_reward = r[ts]  # type: ignore[index]
                        if mh_reward_cache is not None:
                            # multihead: store the per-level vector; the
                            # scalar r[ts] (identical composite) stays in
                            # the episode logs untouched
                            ts_reward = mh_reward_cache[ts]
                        ts_next_state = tuple(s[ts])  # type: ignore[index]
                        ts_done = done[ts]  # type: ignore[index]
                        if moe is not None:
                            # MoE: the gate's action is the CHOSEN expert k —
                            # min_green holds are env dynamics (the expert WAS
                            # consulted). next_mask = L4 mask of s' (all-ones
                            # in pure mode; presence-scan in lexicographic).
                            import moe_glue as _mg
                            _nm = (_mg.gate_mask(
                                       moe_experts.presence(
                                           ts, env.traffic_signals[ts].sumo),
                                       moe_lex, presence=moe_presence)
                                   if (moe_lex or moe_presence)
                                   else np.ones(6, dtype=bool))
                            agent.replay_buffer.add(
                                initial_states[ts], moe_k[ts],
                                ts_reward, ts_next_state, ts_done,
                                next_mask=_nm,
                            )
                        elif enum_tables is not None:
                            # enum_frap: executed dense green index == action;
                            # transition carries its junction id (tensor lookup)
                            agent.replay_buffer.add(
                                initial_states[ts], actual_action,
                                ts_reward, ts_next_state, ts_done, ts,
                            )
                        elif action_meta_file:
                            # buffer stores the executed CANONICAL STD action
                            actual_action = int(green2std[ts][actual_action])
                            agent.replay_buffer.add(
                                initial_states[ts], actual_action,
                                ts_reward, ts_next_state, ts_done,
                                next_mask=ts_mask[ts],
                            )
                        else:
                            agent.replay_buffer.add(
                                initial_states[ts], actual_action,
                                ts_reward, ts_next_state, ts_done,
                            )

                    initial_states = s

                    # ── Update ────────────────────────────────────────────────────
                    if agent.replay_buffer.size() > agent.mini_size:
                        agent.epsilon = (
                            agent.eps_end
                            + (agent.eps_start - agent.eps_end)
                            * math.exp(-1.0 * agent.count / agent.eps_decay)
                        )
                        # enum_frap: agent samples its own (tls-keyed) batch
                        if enum_tables is not None:
                            agent.learn_step()  # type: ignore[attr-defined]  # FRAPAgent
                        # PER: sample with current beta, get weights + indices
                        elif agent.use_per:
                            b_s, b_a, b_r, b_ns, b_d, b_w, b_idx = \
                                agent.replay_buffer.sample(agent.batch_size, beta=agent.current_beta)  # type: ignore[call-arg]
                            agent.update({
                                "states": b_s, "actions": b_a,
                                "next_states": b_ns, "rewards": b_r, "dones": b_d,
                                "weights": b_w, "indices": b_idx,
                            })
                        else:
                            _batch = agent.replay_buffer.sample(agent.batch_size)  # type: ignore[call-arg]
                            if len(_batch) == 6:
                                b_s, b_a, b_r, b_ns, b_d, b_m = _batch
                                agent.update({
                                    "states": b_s, "actions": b_a,
                                    "next_states": b_ns, "rewards": b_r, "dones": b_d,
                                    "next_masks": b_m,
                                })
                            else:
                                b_s, b_a, b_r, b_ns, b_d = _batch
                                agent.update({
                                    "states": b_s, "actions": b_a,
                                    "next_states": b_ns, "rewards": b_r, "dones": b_d,
                                })
            except Exception as e:
                import traceback
                print(f"\n[ERROR] SUMO error at episode {episode}, step {step_counter}:")
                traceback.print_exc()
                print(f"[ERROR] Resetting environment and continuing from episode {episode + 1}...\n")
                try:
                    initial_states = _masked_reset(env, training_seed)
                except Exception as reset_err:
                    print(f"[ERROR] Reset also failed: {reset_err}")
                    raise
                continue

            # ── End of episode ────────────────────────────────────────────────
            if agent.start_train and logging_mode != "none":
                print(f"[{exp_name}] ep={episode:5d}  epsilon={agent.epsilon:.4f}  phases={phase_counts}")
                if logging_mode == "simple":
                    ep_log: dict = {}
                    if ep_losses:
                        ep_log["train/loss"] = sum(ep_losses) / len(ep_losses)
                else:  # basic / full
                    phase_log = {}
                    for ts_id, counts in phase_counts.items():
                        total = sum(counts.values()) or 1
                        for p, c in counts.items():
                            phase_log[f"phase/{ts_id}/phase{p}_ratio"] = c / total
                    ep_log = {
                        "train/episode": episode,
                        "train/epsilon": agent.epsilon,
                        **phase_log,
                    }
                    if ep_losses:
                        ep_log["train/loss"] = sum(ep_losses) / len(ep_losses)
                # Optimizer-side diagnostics (present for all archs; grad_norm only when clipping on)
                if ep_gnorms:
                    ep_log["train/grad_norm"] = sum(ep_gnorms) / len(ep_gnorms)
                if ep_qmeans:
                    ep_log["train/q_mean"]    = sum(ep_qmeans) / len(ep_qmeans)
                    ep_log["train/q_abs_max"] = max(ep_qmaxs)
                # Log PER β annealing (only meaningful when PER on; else it's a constant 1.0)
                if agent.use_per:
                    ep_log["train/per_beta"] = agent.current_beta
                # D2 (CHANGES_AUDIT 2026-06-12): insertion backlog — vehicles
                # held OUTSIDE the network are invisible to both state and
                # reward; end-of-episode backlog makes boundary-hidden
                # congestion measurable. Logging only, no behaviour change.
                try:
                    ep_log["train/pending_veh"] = len(env.sumo.simulation.getPendingVehicles())
                except Exception:
                    pass
                if ep_log:
                    wandb.log(ep_log, step=step_counter)

                if episode % checkpoint_interval == 0:
                    ckpt_path = save_checkpoint(agent, episode, model_dir)
                    print(f"  → ckpt saved: {ckpt_path}")
                    wandb.save(ckpt_path, base_path=".")

            # ── Evaluation episode ────────────────────────────────────────────
            if eval_interval > 0 and episode % eval_interval == 0:
                eps_backup    = agent.epsilon
                agent.epsilon = 0.0                  # greedy policy

                eval_obs  = _masked_reset(env, eval_seed)
                eval_done: dict = {"__all__": False}
                eval_mc = EpisodeMetricsCollector(
                    ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green
                )
                eval_ts_reward: dict = {ts: 0.0 for ts in env.ts_ids}
                attn_ent: list = []; attn_mx: list = []; attn_amb: list = []
                while not eval_done["__all__"]:
                    eval_mc.collect_step(env.sumo)
                    if moe is not None:
                        import moe_glue as _mg
                        eval_actions = {}
                        for ts in env.ts_ids:
                            _props, _lv = moe_experts.propose(
                                ts, env.traffic_signals[ts].sumo,
                                env.traffic_signals[ts].green_phase)
                            _m = _mg.gate_mask(_lv, moe_lex, presence=moe_presence)
                            _k = int(agent.take_action(eval_obs[ts], mask=_m))  # type: ignore[call-arg]
                            eval_actions[ts] = int(_props[_k])
                    elif enum_tables is not None:
                        eval_actions = {ts: agent.take_action(eval_obs[ts], ts)
                                        for ts in env.ts_ids}
                    elif action_meta_file:
                        eval_actions = {ts: int(std2green[ts][agent.take_action(
                            eval_obs[ts], mask=ts_mask[ts])]) for ts in env.ts_ids}
                    else:
                        eval_actions = {ts: agent.take_action(eval_obs[ts]) for ts in env.ts_ids}
                    if attn_diag is not None:
                        xt = torch.tensor(np.asarray(eval_obs[last_ts_id]),
                                          dtype=torch.float32, device=agent.device)
                        attn = agent.q_net.cls_attention(xt)[0]          # (n_tokens,) raw CLS→token mass
                        p = attn.clamp_min(1e-12); p = p / p.sum()       # renormalize over tokens for entropy
                        attn_ent.append(float(-(p * p.log()).sum()))
                        attn_mx.append(float(attn.max()))
                        if attn_diag["amb_off"] is not None:
                            H = attn_diag["layout"]["header_dim"]
                            D = attn_diag["layout"]["token_dim"]
                            toks = xt[H:].reshape(-1, D)
                            amb_mask = toks[:, attn_diag["amb_off"]] > 0
                            if bool(amb_mask.any()):
                                attn_amb.append(float(attn[amb_mask].sum()))
                    eval_obs, eval_rew, eval_done, _ = env.step(action=eval_actions)  # type: ignore[misc]
                    for ts in env.ts_ids:
                        eval_ts_reward[ts] += eval_rew.get(ts, 0.0)

                eval_mc.collect_step(env.sumo)       # capture final step
                eval_mc.finalize(env.sumo)
                eval_mean = sum(eval_ts_reward.values()) / len(env.ts_ids)
                # B2 fix (ABT_1X3_RESULTS_2026-06-11 Part 4): the best-ckpt
                # criterion is the SLIDING MEAN of the last K evals, not a
                # single-eval argmax — one lucky eval episode must not freeze
                # the best checkpoint (K = cfg best_eval_window, default 3;
                # comparison starts once the window is full).
                _bw = int(cfg.get("best_eval_window", 3))
                best_eval_hist.append(eval_mean)
                if len(best_eval_hist) > _bw:
                    del best_eval_hist[: len(best_eval_hist) - _bw]
                eval_sliding = sum(best_eval_hist) / len(best_eval_hist)
                if len(best_eval_hist) == _bw and eval_sliding > best_eval_reward:
                    best_eval_reward = eval_sliding
                    save_checkpoint(agent, episode, model_dir, filename="best.pth")
                    # Snapshot the best ckpt's full eval metrics (all scopes x vTypes x all metric fields)
                    # for offline analysis / thesis tables. Overwrites each new-best event.
                    best_metrics = {
                        "_meta": {
                            "episode":          episode,
                            "eval_mean_reward": float(best_eval_reward),
                            "eval_last_reward": float(eval_mean),
                            "best_criterion":   f"sliding{_bw}",
                            "timestamp":        timestamp,
                            "ckpt_filename":    "best.pth",
                        },
                        "metrics": eval_mc.summary(),
                    }
                    with open(os.path.join(model_dir, "best_metrics.json"), "w") as _f:
                        json.dump(best_metrics, _f, indent=2, default=float)
                    print(f"  → best ckpt updated (sliding{_bw}={best_eval_reward:.4f}, last={eval_mean:.4f})")
                if logging_mode != "none":
                    _BASIC = {
                        "avg_stopped_time", "avg_stop_events", "avg_speed",
                        # Metric A: per-vehicle normalized average (each vehicle 1 vote, k_v skip-empty)
                        "avg_stopped_time_per_visit", "avg_stop_events_per_visit",
                        # Metric B: cross-ts mean (each intersection 1 vote, skip ts where n_vehicles==0)
                        "xts_avg_stopped_time", "xts_avg_stop_events", "xts_avg_speed",
                    }
                    _BASIC_SYS = _BASIC | {"completion_rate"}
                    all_mc = {k.replace("eval/", "eval_", 1): v
                              for k, v in eval_mc.to_flat_dict(prefix="eval").items()}
                    if logging_mode == "full":
                        mc_log = all_mc
                    elif logging_mode == "basic":
                        mc_log = {
                            k: v for k, v in all_mc.items()
                            if (k.startswith("eval_system/") and k.split("/")[-1] in _BASIC_SYS)
                            or (not k.startswith("eval_system/") and k.split("/")[-1] in _BASIC)
                        }
                    else:  # simple
                        mc_log = {}
                    eval_reward_log = {f"eval_{ts}/reward": eval_ts_reward[ts] for ts in env.ts_ids}
                    eval_reward_log["eval_system/mean_reward"] = eval_mean
                    eval_reward_log["eval_system/best_reward"] = best_eval_reward
                    try:    # D2: end-of-eval insertion backlog (see train log)
                        eval_reward_log["eval_system/pending_veh"] = len(
                            env.sumo.simulation.getPendingVehicles())
                    except Exception:
                        pass
                    if attn_ent:
                        eval_reward_log["eval_diag/cls_attn_entropy"] = sum(attn_ent) / len(attn_ent)
                        eval_reward_log["eval_diag/cls_attn_max"]     = sum(attn_mx) / len(attn_mx)
                        if attn_amb:   # only on evals where an ambulance was present
                            eval_reward_log["eval_diag/cls_attn_on_amb"] = sum(attn_amb) / len(attn_amb)
                    wandb.log({**mc_log, **eval_reward_log}, step=step_counter)
                print(f"  → eval ep={episode}")

                agent.epsilon = eps_backup           # restore training epsilon

        env.close()

    if logging_mode != "none":
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO-RL DQN unified trainer")
    parser.add_argument("--config", default="./configs/exp18_presslight.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0); -1 for CPU")
    parser.add_argument("--use_double", action="store_true", default=None,
                        help="Enable Double DQN (overrides YAML if set)")
    parser.add_argument("--use_per", action="store_true", default=None,
                        help="Enable Prioritized Experience Replay (overrides YAML if set)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI > YAML > default(False); write resolved value back into cfg so wandb logs it
    cfg["use_double"] = (
        args.use_double if args.use_double is not None
        else cfg.get("use_double", False)
    )
    cfg["use_per"] = (
        args.use_per if args.use_per is not None
        else cfg.get("use_per", False)
    )

    # Timestamp fixed once per launch so all paths are consistent
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Save a copy of the config next to the checkpoints for reproducibility
    config_save_dir = os.path.join("./models", cfg["name"], timestamp)
    os.makedirs(config_save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(config_save_dir, "config.yaml"))

    train(cfg, timestamp)
