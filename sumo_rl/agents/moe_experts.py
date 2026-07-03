"""MoE-Gcμ rule experts.

v2 (default, 2026-07-03) — clearance-horizon MPC with FOCAL experts.
Post-mortem of exp213/214/215 (EXP213-215_MOE_RESULTS_ANALYSIS_2026-07-03)
showed two design failures in v1:
  M-A  all six experts share one mass vector and differ only by a weight
       row -> identical argmins on 96% of steps (gate has no leverage);
  M-B  the depth-1 (one Δt) horizon cannot price abandoning a discharging
       queue -> phase thrash (96 switches / 199 steps).
v2 removes both with zero new tuning knobs:

  * FOCAL experts: expert-k scores phases by level-k vehicles ONLY
    (expert-0 = all vehicles, the efficiency expert). No level-k vehicle
    at the junction -> the expert abstains (proposes keep-current), so
    proposals genuinely diverge whenever more than one class is present.
  * CLEARANCE horizon: score(p) = predicted additional waiting of the
    focal vehicles under the plan "switch to p now (pay yellow if p≠cur),
    hold, discharge at saturation", accounted over a COMMON horizon H so
    plans with different clearance lengths are comparable (a per-plan
    horizon biases toward short/empty phases — freeze risk, caught by
    hand-computed tests). Hysteresis EMERGES: mid-discharge the current
    phase's residual work is small, so finishing it dominates — no
    commitment timer needed.

    delay0    = yellow_time if p != current else 0
    t_j       = delay0 + (j // n_lanes + 1) * SAT_HEADWAY    (queue pos j)
    T_clear(p)= delay0 + max over served slots of ceil(q/n_lanes)*headway
    H         = clip(max_p T_clear(p), delta_time, max_green) (env consts)
    served queued j ........ min(t_j, H)
    unserved queued ........ H
    served arriving  ....... (delay0 − ETA)+
    unserved arriving ...... (H − ETA)+
    tie -> keep current phase.

Every quantity is a system constant (Δt, yellow, max_green), physics
(ETA = distance/speed), or the engineering saturation headway (≈2 s/veh/
lane). No per-network knobs; menu (enum or 8std), junction size and class
mix are all free — genericity by construction.

v1 (design=1) is kept verbatim for forensic reruns of exp213/214/215
(depth-1 weighted time-loss; see the analysis doc for why it fails).

Data access: direct per-vehicle traci reads on the menu's incoming lanes —
the same access class the reward function uses; NO new obs features.
Vehicle-to-movement assignment is INTENT-EXACT (route lookup, memoized),
mirroring the B-family obs convention (slot_stats="intent").
"""
from __future__ import annotations

import math

import numpy as np

N_EXPERTS = 6            # 0 = equal-weight efficiency, 1..5 = priority levels
LEVELS = (1, 2, 3, 4, 5)
SAT_HEADWAY = 2.0        # s/veh/lane discharge headway (≈1800 veh/h/lane)
ETA_LEVELS = (4, 5)      # v1 only: anticipation where /n_c leaves signal
STOP_SPEED = 0.1         # SUMO waiting convention


def weight_matrix():
    """v1 weight rows (kept for design=1)."""
    W = np.ones((N_EXPERTS, len(LEVELS)), dtype=np.float64)
    for k in range(1, N_EXPERTS):
        W[k, k - 1] = 5.0
    return W


class MoEExperts:
    """tables[ts] = {"phase_slots": [frozenset(slot_idx), ...] in green order,
                     "slot_lanes": {slot_idx: [lane_id, ...]}}
    prio_of_type: vehicle type -> level (BNF table), with default level."""

    def __init__(self, tables, delta_time, yellow_time, prio_of_type,
                 default_level=1, design=2, max_green=60.0):
        self.tables = tables
        self.dt = float(delta_time)
        self.yt = float(yellow_time)
        self.mg = float(max_green)
        self.design = int(design)
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

    def _scan(self, tab, sumo):
        """ONE pass over the unique incoming lanes; each vehicle counted
        once, under its intent-exact movement (route-based, like the obs).
        arriving collects ALL levels with ETA < max_green horizon; the v1
        scorer applies its own (ETA_LEVELS, Δt) filter."""
        by_edges = tab["movement_by_edges"]
        queued = {}      # slot -> list of (dist_to_stop, level), near first
        arriving = {}    # slot -> list of (eta, level)
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
                else:
                    eta = dist / max(speed, STOP_SPEED)
                    if eta < self.mg:
                        arriving.setdefault(slot, []).append((eta, lv))
        for q in queued.values():
            q.sort()
        return queued, arriving, n_c

    # ---- v1 scorer (verbatim exp213/214/215 semantics; forensics only) ----
    def _score_v1(self, tab, queued, arriving, n_c, current_phase):
        phase_slots = tab["phase_slots"]
        inv_n = 1.0 / np.maximum(n_c, 1.0)
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
                    for _, lv in q[cap:]:          # queue tail
                        mass[lv - 1] += self.dt
                else:
                    for _, lv in q:                # unserved queue
                        mass[lv - 1] += self.dt
            for slot, arr in arriving.items():     # anticipation (v1 filter)
                for eta, lv in arr:
                    if lv not in ETA_LEVELS or eta >= self.dt:
                        continue
                    if slot in served:
                        loss = max(0.0, self.yt - eta) if switch else 0.0
                    else:
                        loss = max(0.0, self.dt - eta)
                    mass[lv - 1] += loss
            scores[:, p] = self.W @ (mass * inv_n)
        return scores

    # ---- v2 scorer: clearance-horizon plan cost, per level ----
    def _mass_v2(self, tab, queued, arriving, current_phase):
        """-> (n_levels, n_phases) predicted additional waiting per level."""
        phase_slots = tab["phase_slots"]
        n_phases = len(phase_slots)
        cur_served = (phase_slots[current_phase]
                      if 0 <= current_phase < n_phases else frozenset())
        mass = np.zeros((len(LEVELS), n_phases))
        # per-slot start delay matches the ACTUAL transition semantics of
        # traffic_signal._build_phases: yellow is per-link (G->r only), so a
        # slot green in BOTH cur and p keeps its green through the switch —
        # zero start delay; only newly-green slots wait out the yellow.
        def _delay(p, slot):
            return 0.0 if (p == current_phase or slot in cur_served) else self.yt
        # common accounting horizon H (see module docstring)
        h_max = 0.0
        for p, served in enumerate(phase_slots):
            for slot, q in queued.items():
                if slot in served:
                    n_lanes = max(1, len(tab["slot_lanes"][slot]))
                    h_max = max(h_max, _delay(p, slot)
                                + math.ceil(len(q) / n_lanes) * SAT_HEADWAY)
        H = min(max(h_max, self.dt), self.mg)
        for p, served in enumerate(phase_slots):
            for slot, q in queued.items():
                if slot in served:
                    d0 = _delay(p, slot)
                    n_lanes = max(1, len(tab["slot_lanes"][slot]))
                    for j, (_, lv) in enumerate(q):
                        t_j = d0 + (j // n_lanes + 1) * SAT_HEADWAY
                        mass[lv - 1, p] += min(t_j, H)
                else:
                    for _, lv in q:
                        mass[lv - 1, p] += H
            for slot, arr in arriving.items():
                for eta, lv in arr:
                    if slot in served:
                        mass[lv - 1, p] += max(0.0, _delay(p, slot) - eta)
                    else:
                        mass[lv - 1, p] += max(0.0, H - eta)
        return mass

    def propose(self, ts_id, sumo, current_phase):
        """Returns (proposals (N_EXPERTS,) int array of green-phase indices,
        levels_present set)."""
        tab = self.tables[ts_id]
        n_phases = len(tab["phase_slots"])
        queued, arriving, n_c = self._scan(tab, sumo)
        levels_present = {LEVELS[i] for i in range(len(LEVELS)) if n_c[i] > 0}
        cur = int(current_phase) if 0 <= int(current_phase) < n_phases else 0

        proposals = np.empty(N_EXPERTS, dtype=int)
        if self.design == 1:
            scores = self._score_v1(tab, queued, arriving, n_c, cur)
            for k in range(N_EXPERTS):
                best = int(np.argmin(scores[k]))
                if scores[k, cur] <= scores[k, best] + 1e-9:
                    best = cur              # tie-break: keep current phase
                proposals[k] = best
            return proposals, levels_present

        mass = self._mass_v2(tab, queued, arriving, cur)
        total = mass.sum(axis=0)
        for k in range(N_EXPERTS):
            if k == 0:
                row = total                          # efficiency: everyone
            elif n_c[k - 1] > 0:
                row = mass[k - 1]                    # focal: level-k only
            else:
                proposals[k] = cur                   # abstain: no focal veh
                continue
            best = int(np.argmin(row))
            if row[cur] <= row[best] + 1e-9:
                best = cur                  # tie-break: keep current phase
            proposals[k] = best
        return proposals, levels_present

    def presence(self, ts_id, sumo):
        """Light scan: which levels are present, INTENT-FILTERED (slot>=0)
        exactly like propose()'s levels_present — the next-state mask must
        use the same validity criterion as the act-time mask, or the TD
        target bootstraps from experts that are unselectable at the next
        decision (superset bias; matters on Dublin where many trips end on
        approach edges). Route lookups hit the same memo as propose()."""
        tab = self.tables[ts_id]
        by_edges = tab["movement_by_edges"]
        levels = set()
        for lane, from_edge in tab["lanes"].items():
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                if self._slot(sumo, vid, from_edge, by_edges) < 0:
                    continue
                levels.add(self._level(sumo, vid))
        return levels
