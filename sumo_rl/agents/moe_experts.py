"""MoE-Gcμ rule experts — depth-1 MPC on weighted avg time-loss.

Design: RESEARCH_ROADMAP_2026-07-02.txt §A1 (final generic version).
Six zero-training experts; expert-k uses weights w_c = 5 (c==k) else 1,
expert-0 is all-ones (efficiency fallback). For every phase p in the
junction's menu the expert scores the PREDICTED time-loss increment over
the next control interval and picks argmin:

  Δ_k(p) = Σ_c w_c^(k)/max(n_c,1) · [
      ① queued on a movement NOT served by p ............ +Δt each
      ② queued on a served movement beyond the discharge
         capacity ⌊Δt_eff/headway⌋ × n_lanes (queue tail) . +Δt each
      ③ moving, ETA < Δt (levels in eta_levels only —
         the /n_c structure makes low-level terms negligible):
         red → (Δt−ETA)+ ; green-after-switch → (yellow−ETA)+
      ④ p ≠ current phase → Δt_eff = Δt − yellow  (transition cost) ]
  tie → keep current phase (second anti-thrash guard, derived not tuned)

Every quantity is a system constant (Δt, yellow), physics (ETA = distance/
speed), an engineering standard (saturation headway ≈ 2 s/veh/lane) or the
BNF weight convention. No per-network tuning knobs by design.

Data access: direct per-vehicle traci reads on the menu's incoming lanes —
the same access class the reward function uses; NO new obs features.
Vehicle-to-movement assignment is INTENT-EXACT (route lookup, memoized),
mirroring the B-family obs convention (slot_stats="intent", the exp209
default): each vehicle counts once, under its actual turning movement.
"""
from __future__ import annotations

import math

import numpy as np

N_EXPERTS = 6            # 0 = equal-weight efficiency, 1..5 = priority levels
LEVELS = (1, 2, 3, 4, 5)
SAT_HEADWAY = 2.0        # s/veh/lane discharge headway (≈1800 veh/h/lane)
ETA_LEVELS = (4, 5)      # anticipation only where /n_c leaves signal (design §A1)
STOP_SPEED = 0.1         # SUMO waiting convention


def weight_matrix():
    W = np.ones((N_EXPERTS, len(LEVELS)), dtype=np.float64)
    for k in range(1, N_EXPERTS):
        W[k, k - 1] = 5.0
    return W


class MoEExperts:
    """tables[ts] = {"phase_slots": [frozenset(slot_idx), ...] in green order,
                     "slot_lanes": {slot_idx: [lane_id, ...]}}
    prio_of_type: vehicle type -> level (BNF table), with default level."""

    def __init__(self, tables, delta_time, yellow_time, prio_of_type,
                 default_level=1):
        self.tables = tables
        self.dt = float(delta_time)
        self.yt = float(yellow_time)
        self.prio = dict(prio_of_type)
        self.default_level = int(default_level)
        self.W = weight_matrix()
        self._lane_len = {}          # lane_id -> length (cached once)
        self._vid_level = {}         # vid -> level memo (type is immutable)
        self._vid_slot = {}          # (vid, from_edge) -> slot memo (route is
        #                              fixed; from_edge pins the junction)

    def _level(self, sumo, vid):
        lv = self._vid_level.get(vid)
        if lv is None:
            lv = int(self.prio.get(sumo.vehicle.getTypeID(vid),
                                   self.default_level))
            self._vid_level[vid] = lv
            if len(self._vid_level) > 50000:
                self._vid_level.clear()
        return lv

    def _length(self, sumo, lane):
        L = self._lane_len.get(lane)
        if L is None:
            L = sumo.lane.getLength(lane)
            self._lane_len[lane] = L
        return L

    def _slot(self, sumo, vid, from_edge, by_edges):
        """Intent-exact movement of vid (obs _vehicle_slot convention):
        next route edge -> (from_edge, next) -> slot; -1 = uncontrolled."""
        key = (vid, from_edge)
        s = self._vid_slot.get(key)
        if s is None:
            route = sumo.vehicle.getRoute(vid)
            idx = sumo.vehicle.getRouteIndex(vid)
            nxt = route[idx + 1] if idx + 1 < len(route) else None
            s = by_edges.get((from_edge, nxt), -1)
            self._vid_slot[key] = s
            if len(self._vid_slot) > 50000:
                self._vid_slot.clear()
        return s

    def propose(self, ts_id, sumo, current_phase):
        """Returns (proposals (N_EXPERTS,) int array of green-phase indices,
        levels_present set)."""
        tab = self.tables[ts_id]
        phase_slots = tab["phase_slots"]
        by_edges = tab["movement_by_edges"]
        # ---- ONE pass over the unique incoming lanes; each vehicle counted
        # once, under its intent-exact movement (route-based, like the obs) --
        queued = {}      # slot -> list of (dist_to_stop, level), near first
        arriving = {}    # slot -> list of (eta, level), eta < dt
        n_c = np.zeros(len(LEVELS))
        for lane, from_edge in tab["lanes"].items():
            L = self._length(sumo, lane)
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                slot = self._slot(sumo, vid, from_edge, by_edges)
                if slot < 0:          # no controlled next movement here
                    continue
                lv = self._level(sumo, vid)
                n_c[lv - 1] += 1
                speed = sumo.vehicle.getSpeed(vid)
                dist = max(0.0, L - sumo.vehicle.getLanePosition(vid))
                if speed < STOP_SPEED:
                    queued.setdefault(slot, []).append((dist, lv))
                elif lv in ETA_LEVELS:
                    eta = dist / max(speed, STOP_SPEED)
                    if eta < self.dt:
                        arriving.setdefault(slot, []).append((eta, lv))
        for q in queued.values():
            q.sort()
        levels_present = {LEVELS[i] for i in range(len(LEVELS)) if n_c[i] > 0}
        inv_n = 1.0 / np.maximum(n_c, 1.0)

        # ---- score every phase for all 6 experts at once ----
        n_phases = len(phase_slots)
        scores = np.zeros((N_EXPERTS, n_phases))
        for p, served in enumerate(phase_slots):
            switch = (p != current_phase)
            dt_eff = self.dt - (self.yt if switch else 0.0)
            cap_lane = int(math.floor(max(0.0, dt_eff) / SAT_HEADWAY))
            mass = np.zeros(len(LEVELS))           # predicted loss per level
            for slot, q in queued.items():
                if slot in served:
                    cap = cap_lane * max(1, len(tab["slot_lanes"][slot]))
                    for _, lv in q[cap:]:          # ② queue tail
                        mass[lv - 1] += self.dt
                else:
                    for _, lv in q:                # ① unserved queue
                        mass[lv - 1] += self.dt
            for slot, arr in arriving.items():     # ③ anticipation
                for eta, lv in arr:
                    if slot in served:
                        loss = max(0.0, self.yt - eta) if switch else 0.0
                    else:
                        loss = max(0.0, self.dt - eta)
                    mass[lv - 1] += loss
            scores[:, p] = self.W @ (mass * inv_n)

        proposals = np.empty(N_EXPERTS, dtype=int)
        cur = int(current_phase) if 0 <= int(current_phase) < n_phases else 0
        for k in range(N_EXPERTS):
            best = int(np.argmin(scores[k]))
            # tie-break: keep current phase when it is (near-)optimal
            if scores[k, cur] <= scores[k, best] + 1e-9:
                best = cur
            proposals[k] = best
        return proposals, levels_present

    def presence(self, ts_id, sumo):
        """Light scan: which levels are present (for L4 next-state masks)."""
        levels = set()
        for lane in self.tables[ts_id]["lanes"]:
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                levels.add(self._level(sumo, vid))
        return levels
