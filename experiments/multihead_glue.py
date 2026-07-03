"""Glue between train.py and the multi-head B-line agent (agent_arch:
multihead). Import-only from the multihead branch — old configs never
reach this module.

make_priority_avg_waiting_reward_vec: the EXACT per-level decomposition
of sumo_rl.environment.rewards.make_priority_avg_waiting_reward —

    scalar  = -( Σ_l  l · avg_wait_l ) / 100
    vec[l-1] = -avg_wait_l / 100          (unweighted level channel)
    scalar == Σ_l  l · vec[l-1]           (identity, unit-tested)

The traversal is a line-for-line mirror of the original (single pass
over incoming lanes, env.vehicles cross-lane wait correction, vid→level
memo) so the SCALAR this returns is bit-equal to what the scalar B-line
trains on — verified empirically by a fixed-policy dual-run probe.
rewards.py itself is untouched (frozen-core discipline).
"""
from __future__ import annotations

import numpy as np

N_LEVELS = 5


def make_priority_avg_waiting_reward_vec(table: dict):
    """-> (reward_fn, cache). reward_fn returns the usual scalar (env/
    logging unchanged); cache[ts_id] holds this step's per-level vector
    for the trainer to store in the replay buffer."""
    from sumo_rl.environment.priority_map import DEFAULT_PRIORITY
    type_w = {t: float(p) for t, p in table.items()}
    default_w = float(DEFAULT_PRIORITY)
    vid_w: dict = {}            # vid → weight memo (vehicle type is immutable)
    cache: dict = {}            # ts_id → np.ndarray(N_LEVELS,) — this step

    def fn(ts) -> float:
        if len(vid_w) > 50000:  # safety valve; SUMO ids are unique, never reused
            vid_w.clear()
        sumo = ts.sumo
        env_vehicles = ts.env.vehicles
        totals: dict = {}
        counts: dict = {}
        for lane in ts.lanes:
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                veh_lane = sumo.vehicle.getLaneID(vid)
                acc = sumo.vehicle.getAccumulatedWaitingTime(vid)
                if vid not in env_vehicles:
                    env_vehicles[vid] = {veh_lane: acc}
                else:
                    env_vehicles[vid][veh_lane] = acc - sum(
                        env_vehicles[vid][l]
                        for l in env_vehicles[vid] if l != veh_lane
                    )
                lane_wait = env_vehicles[vid][veh_lane]
                w = vid_w.get(vid)
                if w is None:
                    w = type_w.get(sumo.vehicle.getTypeID(vid), default_w)
                    vid_w[vid] = w
                if w in totals:
                    totals[w] += lane_wait
                    counts[w] += 1
                else:
                    totals[w] = lane_wait
                    counts[w] = 1
        vec = np.zeros(N_LEVELS)
        r = 0.0
        for w, tot in totals.items():
            avg = tot / counts[w]
            r += w * avg                     # same accumulation as original
            vec[int(w) - 1] = -avg / 100.0   # unweighted level channel
        cache[ts.id] = vec
        return -r / 100.0

    return fn, cache


def build_multihead_agent(cfg, starting_state, state_space, action_space,
                          device):
    from sumo_rl.agents.dqn_multihead_agent import DQNMultiHead
    w = cfg.get("multihead_weights")     # optional override for weight sweeps
    return DQNMultiHead(
        starting_state=starting_state,
        state_space=state_space,
        hidden_dim=cfg.get("hidden_dim", 64),
        action_space=action_space,
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
        weights=(np.asarray(w, dtype=np.float64) if w is not None else None),
        use_double=cfg.get("use_double", False),
        loss_fn=cfg.get("loss_fn", "mse"),
        target_clip_max=cfg.get("target_clip_max", None),
        grad_clip=cfg.get("grad_clip", None),
    )
