"""Observation functions for traffic signals."""
import math
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal
from .priority_map import load_priority_table, PRIORITY_LEVELS, DEFAULT_PRIORITY


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

class QueuePrioObservationFunction(ObservationFunction):
    """Observation: 分别统计每条进口车道的 car/bus density 和 queue。"""

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        print("This is QueuePrioObservationFunction with car/bus density and queue.")

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        car_density, bus_density = self.ts.get_lanes_density_by_type()
        car_queue, bus_queue = self.ts.get_lanes_queue_by_type()
        observation = np.array(
            phase_id + min_green + car_density + bus_density + car_queue + bus_queue,
            dtype=np.float32
        )
        return observation

    def observation_space(self) -> spaces.Box:
        num_phases = self.ts.num_green_phases
        num_lanes = len(self.ts.lanes)
        # phase + min_green + 4*num_lanes (car_density, bus_density, car_queue, bus_queue)
        dim = num_phases + 1 + 4 * num_lanes
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.ones(dim, dtype=np.float32),
        )
    
class PressLightObservationFunction(ObservationFunction):
    """PressLight observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize PressLight observation function."""
        super().__init__(ts)
        print("This is local obs.py")

    def __call__(self) -> np.ndarray:
        """Return the PressLight observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        list_inc_vehicle_counts, list_out_vehicle_counts = self._get_vehicle_count_by_lanes(self.ts.id)
        # d= self.ts.out_lanes
    
        observation = np.array(phase_id + min_green + list_inc_vehicle_counts + list_out_vehicle_counts, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        a= len(self.ts.lanes) #12
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

    def _get_vehicle_count_by_lanes(self, ts_id):
        """
        获取一个信号灯路口的每个进出车道的车辆数量
        :param ts_id: 信号灯的ID
        :return: 一个字典，键是车道ID，值是该车道上的车辆数量
        """
        inc_vehicle_counts = {}  # 存储结果的字典

        # 获取信号灯控制的所有车道
        inc_lanes = self.ts.sumo.trafficlight.getControlledLanes(ts_id)

        # 对于每个车道，计算并存储车辆数量
        for lane_id in inc_lanes:
            inc_vehicle_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane_id)
            inc_vehicle_counts[lane_id] = inc_vehicle_count

        # print(inc_vehicle_counts)

        out_vehicle_counts = {}
        # outgoing_lanes = self._get_outgoing_lanes(ts_id)
        outgoing_lanes = self.ts.out_lanes

        # 对于每个出口车道，计算并存储车辆数量
        for lane_id in outgoing_lanes:
            out_vehicle_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane_id)
            out_vehicle_counts[lane_id] = out_vehicle_count

        # print(out_vehicle_counts)

        #将字典转换为list
        list_inc_vehicle_counts = [inc_vehicle_counts[key] for key in inc_vehicle_counts]
        list_out_vehicle_counts = [out_vehicle_counts[key] for key in out_vehicle_counts]

        return list_inc_vehicle_counts, list_out_vehicle_counts


class PressLightNormObservationFunction(ObservationFunction):
    """PressLight observation with vehicle counts normalized to [0, 1] by lane capacity.

    Observation vector:
        [phase_one_hot (n), min_green (1), inc_density (n_in), out_density (n_out)]
    """

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        inc_density = self.ts.get_lanes_density()
        out_density = self.ts.get_out_lanes_density()
        return np.array(phase_id + min_green + inc_density + out_density, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        dim = self.ts.num_green_phases + 1 + len(self.ts.lanes) + len(self.ts.out_lanes)
        return spaces.Box(low=np.zeros(dim, dtype=np.float32), high=np.ones(dim, dtype=np.float32))
    
    
    
class PriorityObservationFunction(ObservationFunction):
    """Priority observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize Priority observation function."""
        super().__init__(ts)
        print("This is modified PriorityObservationFunction with car/truck state.")

    def __call__(self) -> np.ndarray:
        """Return the PressLight observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        # Get separate car/truck counts
        inc_car, inc_truck, out_car, out_truck = self._get_vehicle_count_by_type(self.ts.id)
        # d= self.ts.out_lanes
    
        # Build observation vector
        observation = np.array(
            phase_id
            + min_green
            + inc_car
            + out_car
            + inc_truck
            + out_truck,
            dtype=np.float32
        )
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        num_phases = self.ts.num_green_phases
        num_lanes = len(self.ts.lanes)
        # state length = num_phases + 1 + 4 * num_lanes
        dim = num_phases + 1 + 4 * num_lanes
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.ones(dim, dtype=np.float32),
        )


    def _get_vehicle_count_by_type(self, ts_id):
            """
            Return counts of cars and trucks on incoming and outgoing lanes.
            先按 type=="car" 计数，其余车辆一律归为 truck。
            :return: inc_car, inc_truck, out_car, out_truck as lists ordered by lane order
            """
            inc_lanes = self.ts.sumo.trafficlight.getControlledLanes(ts_id)
            out_lanes = self.ts.out_lanes  # assume exists

            inc_car, inc_truck = [], []
            for lane in inc_lanes:
                vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
                cars = sum(1 for v in vids if self.ts.sumo.vehicle.getTypeID(v) == 'car')
                inc_car.append(cars)
                inc_truck.append(len(vids) - cars)

            out_car, out_truck = [], []
            for lane in out_lanes:
                vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
                cars = sum(1 for v in vids if self.ts.sumo.vehicle.getTypeID(v) == 'car')
                out_car.append(cars)
                out_truck.append(len(vids) - cars)

            return inc_car, inc_truck, out_car, out_truck


class PriorityBCAObservationFunction(ObservationFunction):
    """Priority observation with car/bus/ambulance 3-class counts over all lanes.

    Observation vector:
        [phase_one_hot (n), min_green (1),
         inc_car (n_in), out_car (n_out),
         inc_bus (n_in), out_bus (n_out),
         inc_amb (n_in), out_amb (n_out)]
    """

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id  = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        inc_car, inc_bus, inc_amb = self._count_by_type(self.ts.lanes)
        out_car, out_bus, out_amb = self._count_by_type(self.ts.out_lanes)
        return np.array(
            phase_id + min_green
            + inc_car + out_car
            + inc_bus + out_bus
            + inc_amb + out_amb,
            dtype=np.float32,
        )

    def observation_space(self) -> spaces.Box:
        num_phases = self.ts.num_green_phases
        n_in       = len(self.ts.lanes)
        n_out      = len(self.ts.out_lanes)
        dim        = num_phases + 1 + 3 * (n_in + n_out)
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.full(dim, np.inf, dtype=np.float32),
        )

    def _count_by_type(self, lanes):
        car_l, bus_l, amb_l = [], [], []
        for lane in lanes:
            c = b = a = 0
            for v in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
                vtype = self.ts.sumo.vehicle.getTypeID(v)
                if vtype == "ambulance":
                    a += 1
                elif vtype == "bus":
                    b += 1
                else:
                    c += 1
            car_l.append(c)
            bus_l.append(b)
            amb_l.append(a)
        return car_l, bus_l, amb_l


class PriorityPhaseObservationFunction(ObservationFunction):
    """Scheme A — phase-level priority-bucket observation (action space unchanged).

    Builds on the per-type counting idea of ``PriorityBCAObservationFunction`` but
    (1) aggregates vehicles **by green phase** (over each phase's served G/g incoming
    lanes) instead of flat per-lane, and (2) replaces "count per car/bus/amb type"
    with a fixed **priority-bucket featurizer φ** over levels {1..5}, so the agent
    sees priority buckets (swap the policy table → zero-shot) rather than vehicle types.

    Two phase-state encodings (``phase_state``), so the timing/identity block can be
    swapped without touching φ:
      - "legacy": the previously-working encoding — a global front block
            [phase_one_hot(K), min_green_binary(1)]  then per phase k:  φ_k
        (matches DefaultObservationFunction / PriorityBCA's phase block exactly; no timing).
      - "perphase": header [min_green_ok, phase_elapsed]  then per phase k:  [is_cur] + φ_k
        (current-phase one-hot distributed into each row as is_cur, plus the continuous
         time-in-phase scalar — once, in the header, since it is global not per-phase).

    Layout lengths:
        legacy   : (K + 1) + K × φ_dim
        perphase : 2       + K × (1 + φ_dim)
        φ_k = for p in 1..5:  [<fields>]   (per priority bucket)

    Per-bucket fields are configurable via ``fields`` so the feature set can be ablated:
      - "mean_awt": REQUIRED in spirit — the reward is priority-weighted avg waiting
        time, so the value function must observe waiting time or it predicts an
        unobserved quantity (the measured |TD| floor of the count-only obs).
      - "queue": magnitude/discharge demand; orthogonal to mean_awt (leading vs lagging).
        Fresh arrivals show up in queue before they age into mean_awt — dropping it
        blinds the agent to new demand. Standard TSC state variable.
      - "count": total demand incl. moving vehicles; correlated with queue (queue ⊆ count),
        so the most droppable. Always computed internally (needed for mean) — emitting is free.
      - "max_awt": dilution-robust / leading indicator of future mean; redundant with mean
        for the n=1 ambulance bucket. Clean ablation knob.
    Lean core = ("queue", "mean_awt"); full = ("count","queue","mean_awt","max_awt").

    Perf: phase→served-lanes and per-phase capacity are cached once in __init__; each
    step iterates every incoming lane's vehicles exactly once (shared lanes reused across
    phases) and memoizes vid→priority (vehicle type is immutable). If per-vehicle TraCI
    (getAccumulatedWaitingTime/getSpeed) ever dominates runtime, switch to TraCI
    subscriptions — the layout here is unchanged by that.
    """

    _ALL_FIELDS = ("count", "queue", "mean_awt", "max_awt")
    _PHASE_STATES = ("legacy", "perphase")

    def __init__(self, ts: TrafficSignal,
                 fields: tuple = ("count", "queue", "mean_awt", "max_awt"),
                 phase_state: str = "perphase", priority_source=None,
                 normalize: bool = True, awt_scale: float = 100.0, veh_len: float = 5.0):
        super().__init__(ts)
        bad = [f for f in fields if f not in self._ALL_FIELDS]
        if bad:
            raise ValueError(f"unknown φ fields {bad}; allowed {self._ALL_FIELDS}")
        if phase_state not in self._PHASE_STATES:
            raise ValueError(f"phase_state must be one of {self._PHASE_STATES}")
        self.fields = tuple(fields)
        self.phase_state = phase_state
        self._is_legacy = (phase_state == "legacy")
        self.normalize = normalize
        self.awt_scale = awt_scale
        self._phi_dim = len(PRIORITY_LEVELS) * len(self.fields)

        # policy = single source of truth (priority_map.py / external file), compiled once.
        self._prio_table = load_priority_table(priority_source)

        # --- static caches (computed once) ---
        self._phase_lanes = self._build_phase_served_lanes()        # k -> [served incoming lanes]
        self._lane_cap = {                                          # lane -> jam capacity (#veh)
            lane: max(ts.lanes_length[lane] / (ts.MIN_GAP + veh_len), 1.0)
            for lane in ts.lanes
        }
        self._phase_cap = {                                         # k -> Σ served-lane capacity
            k: max(sum(self._lane_cap[l] for l in lanes), 1.0)
            for k, lanes in self._phase_lanes.items()
        }
        self._active = sorted(set(self._prio_table.values()))       # priority levels the table uses
        self._vid_prio: dict = {}                                   # vid -> priority memo

    def _build_phase_served_lanes(self) -> dict:
        """k -> incoming lanes that get G/g in green phase k (ts.lanes order, deduped)."""
        raw = self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)   # indices align phase.state
        served = {}
        for k, ph in enumerate(self.ts.green_phases):
            seen = {raw[i] for i, ch in enumerate(ph.state) if ch in ('G', 'g')}
            served[k] = [lane for lane in self.ts.lanes if lane in seen]
        return served

    def _vehicle_priority(self, vid) -> int:
        p = self._vid_prio.get(vid)
        if p is None:
            p = self._prio_table.get(self.ts.sumo.vehicle.getTypeID(vid), DEFAULT_PRIORITY)
            self._vid_prio[vid] = p
        return p

    def _lane_bucket_stats(self, lane) -> dict:
        """One pass over a lane's vehicles → {p: [count, queue, sum_awt, max_awt]}."""
        sumo = self.ts.sumo
        stats = {p: [0, 0, 0.0, 0.0] for p in PRIORITY_LEVELS}
        for vid in sumo.lane.getLastStepVehicleIDs(lane):
            s = stats[self._vehicle_priority(vid)]
            w = sumo.vehicle.getAccumulatedWaitingTime(vid)
            s[0] += 1
            if sumo.vehicle.getSpeed(vid) < 0.1:
                s[1] += 1
            s[2] += w
            if w > s[3]:
                s[3] = w
        return stats

    def __call__(self) -> np.ndarray:
        ts = self.ts
        if len(self._vid_prio) > 50000:          # safety valve against unbounded growth
            self._vid_prio.clear()

        min_green_ok = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0
        if self._is_legacy:
            # previously-working encoding: front phase one-hot + binary min_green
            obs = [1.0 if ts.green_phase == i else 0.0 for i in range(ts.num_green_phases)]
            obs.append(min_green_ok)
        else:
            # per-phase encoding: min_green_ok + continuous time-in-phase (global, once)
            obs = [min_green_ok, ts.time_since_last_phase_change / 100.0]

        # per-lane stats once; shared lanes are reused across phase groups
        lane_stats = {lane: self._lane_bucket_stats(lane) for lane in ts.lanes}

        for k in range(ts.num_green_phases):
            if not self._is_legacy:
                obs.append(1.0 if ts.green_phase == k else 0.0)   # is_cur per phase row
            agg = {p: [0, 0, 0.0, 0.0] for p in PRIORITY_LEVELS}
            for lane in self._phase_lanes[k]:
                ls = lane_stats[lane]
                for p in PRIORITY_LEVELS:
                    a, s = agg[p], ls[p]
                    a[0] += s[0]; a[1] += s[1]; a[2] += s[2]
                    if s[3] > a[3]:
                        a[3] = s[3]
            cap = self._phase_cap[k]
            for p in PRIORITY_LEVELS:
                c, q, sw, mw = agg[p]
                mean = (sw / c) if c else 0.0
                if self.normalize:
                    vals = {"count": c / cap, "queue": q / cap,
                            "mean_awt": mean / self.awt_scale, "max_awt": mw / self.awt_scale}
                else:
                    vals = {"count": c, "queue": q, "mean_awt": mean, "max_awt": mw}
                obs += [vals[f] for f in self.fields]
        return np.array(obs, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        K = self.ts.num_green_phases
        if self._is_legacy:
            dim = (K + 1) + K * self._phi_dim
        else:
            dim = 2 + K * (1 + self._phi_dim)
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.full(dim, np.inf, dtype=np.float32),
        )


# ── Scheme A — 6 ready-to-use variants (phase_state × fields) ───────────────────
# 2 phase-state encodings × 3 field-sets. All share φ, policy table, action space.
#   legacy   = the previously-working phase block (front one-hot + binary min_green)
#   perphase = is_cur per phase row + [min_green_ok, phase_elapsed]
_CQ   = ("count", "queue")
_CQM  = ("count", "queue", "mean_awt")
_CQMM = ("count", "queue", "mean_awt", "max_awt")


class PriorityPhaseLegacyCQ(PriorityPhaseObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQ, **kw)


class PriorityPhaseLegacyCQM(PriorityPhaseObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQM, **kw)


class PriorityPhaseLegacyCQMM(PriorityPhaseObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQMM, **kw)


class PriorityPhaseElapsedCQ(PriorityPhaseObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQ, **kw)


class PriorityPhaseElapsedCQM(PriorityPhaseObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQM, **kw)


class PriorityPhaseElapsedCQMM(PriorityPhaseObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQMM, **kw)


class PriorityMovementObservationFunction(ObservationFunction):
    """Scheme B — movement-level priority-bucket observation (action space unchanged).

    Same φ priority-bucket featurizer and same two phase-state encodings as Scheme A,
    but aggregates vehicles **by turning movement** instead of by phase. Movements are
    12 fixed slots = {N,E,S,W} approach × {L,T,R} turn. A vehicle's slot = (the compass
    side it approaches from, its turn) — both derived from geometry + its route, so the
    layout is generic (no edge-name or TraCI direction-index dependency).

    Layout lengths (N_SLOTS = 12):
        legacy   : (K + 1) + 12 × φ_dim                      # front phase one-hot + min_green
        perphase : 2       + 12 × (1 + φ_dim)                # [min_green_ok, phase_elapsed]; per-slot is_green
    Per slot the perphase ``is_green`` = 1 if that movement is served (G/g) by the current
    green phase (multi-hot: a phase serves several movements) — the Scheme-B analog of A's
    per-phase is_cur. Empty/impossible movements are simply all-zero φ (no separate mask
    channel needed for the MLP head).

    ``fields`` and ``phase_state`` behave exactly as in PriorityPhaseObservationFunction.
    Perf: movement slots, per-phase served-slot sets, and per-slot capacity are cached once;
    each step iterates incoming-lane vehicles once and memoizes vid→(priority, slot)
    (both immutable per vehicle).
    """

    _ALL_FIELDS = ("count", "queue", "mean_awt", "max_awt")
    _PHASE_STATES = ("legacy", "perphase")
    _APPROACHES = ("N", "E", "S", "W")
    _TURNS = ("L", "T", "R")
    N_SLOTS = 12

    def __init__(self, ts: TrafficSignal,
                 fields: tuple = ("count", "queue", "mean_awt", "max_awt"),
                 phase_state: str = "perphase", priority_source=None,
                 normalize: bool = True, awt_scale: float = 100.0, veh_len: float = 5.0):
        super().__init__(ts)
        bad = [f for f in fields if f not in self._ALL_FIELDS]
        if bad:
            raise ValueError(f"unknown φ fields {bad}; allowed {self._ALL_FIELDS}")
        if phase_state not in self._PHASE_STATES:
            raise ValueError(f"phase_state must be one of {self._PHASE_STATES}")
        self.fields = tuple(fields)
        self.phase_state = phase_state
        self._is_legacy = (phase_state == "legacy")
        self.normalize = normalize
        self.awt_scale = awt_scale
        self._phi_dim = len(PRIORITY_LEVELS) * len(self.fields)

        self._prio_table = load_priority_table(priority_source)   # policy source of truth

        # --- static movement geometry, cached once ---
        links = ts.sumo.trafficlight.getControlledLinks(ts.id)    # index = phase.state position
        self._link_slot = {}                  # link index -> slot id
        self._movement_by_edges = {}          # (from_edge, to_edge) -> slot id  (vehicle lookup)
        lane_feeds = {}                       # lane -> set(slot)
        for i, grp in enumerate(links):
            if not grp:
                continue
            fl, tl = grp[0][0], grp[0][1]
            slot = self._slot_idx(self._approach(fl), self._turn(fl, tl))
            self._link_slot[i] = slot
            self._movement_by_edges[(self._edge(fl), self._edge(tl))] = slot
            lane_feeds.setdefault(fl, set()).add(slot)

        # per-phase served-slot sets (slot served if ANY of its links is G/g this phase)
        self._phase_serves = {}
        for k, ph in enumerate(ts.green_phases):
            served = set()
            for i, slot in self._link_slot.items():
                if i < len(ph.state) and ph.state[i] in ('G', 'g'):
                    served.add(slot)
            self._phase_serves[k] = served

        # per-slot capacity = Σ jam-capacity of lanes feeding that movement
        cap = {s: 0.0 for s in range(self.N_SLOTS)}
        for lane, slots in lane_feeds.items():
            lcap = ts.lanes_length.get(lane, 0.0) / (ts.MIN_GAP + veh_len)
            for s in slots:
                cap[s] += lcap
        self._slot_cap = {s: max(c, 1.0) for s, c in cap.items()}

        self._vid_prio: dict = {}
        self._vid_slot: dict = {}

    # ── geometry helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _edge(lane: str) -> str:
        return lane.rsplit('_', 1)[0]

    def _heading(self, lane: str) -> float:
        s = self.ts.sumo.lane.getShape(lane)
        (x0, y0), (x1, y1) = s[-2], s[-1]
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    def _approach(self, lane: str) -> str:
        """Compass side the vehicle approaches FROM (opposite of its heading into junction)."""
        a = (self._heading(lane) + 180) % 360
        return ("E", "N", "W", "S")[round(a / 90) % 4]

    def _turn(self, fl: str, tl: str) -> str:
        d = (self._heading(tl) - self._heading(fl) + 180) % 360 - 180
        if -45 < d <= 45:
            return "T"
        if 45 < d <= 135:
            return "L"
        if -135 < d <= -45:
            return "R"
        return "L"          # U-turn folded into L (first version)

    def _slot_idx(self, approach: str, turn: str) -> int:
        return self._APPROACHES.index(approach) * 3 + self._TURNS.index(turn)

    # ── per-vehicle memoized lookups ──────────────────────────────────────────
    def _vehicle_priority(self, vid) -> int:
        p = self._vid_prio.get(vid)
        if p is None:
            p = self._prio_table.get(self.ts.sumo.vehicle.getTypeID(vid), DEFAULT_PRIORITY)
            self._vid_prio[vid] = p
        return p

    def _vehicle_slot(self, vid, from_edge: str) -> int:
        """Movement slot of vid (−1 if it has no controlled next movement). Memoized."""
        if vid in self._vid_slot:
            return self._vid_slot[vid]
        sumo = self.ts.sumo
        route = sumo.vehicle.getRoute(vid)
        idx = sumo.vehicle.getRouteIndex(vid)
        nxt = route[idx + 1] if idx + 1 < len(route) else None
        slot = self._movement_by_edges.get((from_edge, nxt), -1)
        self._vid_slot[vid] = slot
        return slot

    # ── observation ───────────────────────────────────────────────────────────
    def __call__(self) -> np.ndarray:
        ts = self.ts
        if len(self._vid_prio) > 50000:
            self._vid_prio.clear(); self._vid_slot.clear()

        # aggregate vehicles into (slot, priority) buckets — single pass over incoming lanes
        slot_stats = {s: {p: [0, 0, 0.0, 0.0] for p in PRIORITY_LEVELS} for s in range(self.N_SLOTS)}
        for lane in ts.lanes:
            from_edge = self._edge(lane)
            for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
                slot = self._vehicle_slot(vid, from_edge)
                if slot < 0:
                    continue
                s = slot_stats[slot][self._vehicle_priority(vid)]
                w = ts.sumo.vehicle.getAccumulatedWaitingTime(vid)
                s[0] += 1
                if ts.sumo.vehicle.getSpeed(vid) < 0.1:
                    s[1] += 1
                s[2] += w
                if w > s[3]:
                    s[3] = w

        min_green_ok = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0
        if self._is_legacy:
            obs = [1.0 if ts.green_phase == i else 0.0 for i in range(ts.num_green_phases)]
            obs.append(min_green_ok)
        else:
            obs = [min_green_ok, ts.time_since_last_phase_change / 100.0]

        served = self._phase_serves.get(ts.green_phase, set())
        for slot in range(self.N_SLOTS):
            if not self._is_legacy:
                obs.append(1.0 if slot in served else 0.0)      # is_green per movement slot
            cap = self._slot_cap[slot]
            for p in PRIORITY_LEVELS:
                c, q, sw, mw = slot_stats[slot][p]
                mean = (sw / c) if c else 0.0
                if self.normalize:
                    vals = {"count": c / cap, "queue": q / cap,
                            "mean_awt": mean / self.awt_scale, "max_awt": mw / self.awt_scale}
                else:
                    vals = {"count": c, "queue": q, "mean_awt": mean, "max_awt": mw}
                obs += [vals[f] for f in self.fields]
        return np.array(obs, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        K = self.ts.num_green_phases
        if self._is_legacy:
            dim = (K + 1) + self.N_SLOTS * self._phi_dim
        else:
            dim = 2 + self.N_SLOTS * (1 + self._phi_dim)
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.full(dim, np.inf, dtype=np.float32),
        )


# ── Scheme B — 6 ready-to-use variants (phase_state × fields) ───────────────────
class PriorityMovementLegacyCQ(PriorityMovementObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQ, **kw)


class PriorityMovementLegacyCQM(PriorityMovementObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQM, **kw)


class PriorityMovementLegacyCQMM(PriorityMovementObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQMM, **kw)


class PriorityMovementElapsedCQ(PriorityMovementObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQ, **kw)


class PriorityMovementElapsedCQM(PriorityMovementObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQM, **kw)


class PriorityMovementElapsedCQMM(PriorityMovementObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQMM, **kw)


class PriorityLaneTokenObservationFunction(PriorityPhaseObservationFunction):
    """Scheme T (simplified) — lane-level priority-bucket observation (action unchanged).

    The finest granularity: one token per incoming lane. Parallel to A (per phase) and
    B (per movement); same φ featurizer, same two phase_state encodings, same fields.
    Reuses the parent's φ helpers (_lane_bucket_stats / _vehicle_priority).

    ⚠ SIMPLIFIED on purpose: this is a FLAT per-lane vector for the current MLP, and it
    OMITS the static per-lane geometry block (length / approach-angle / capacity / turn
    one-hot / free-right) of the full design. The real Transformer head (per-lane token
    matrix + CLS + attention, lane-count-agnostic) and that static block are the planned
    next step ("后面再加"); this version exists to slot the lane-granularity obs into the
    existing pipeline now.

    Layout (N_lanes = len(ts.lanes); svc = K if include_phase_service else 0):
        legacy   : (K + 1) + N_lanes × (svc + φ_dim)            # front phase one-hot + min_green
        perphase : 2       + N_lanes × (1 + svc + φ_dim)        # [min_green_ok, phase_elapsed]; per-lane is_green
    per-lane is_green = 1 if that lane gets G/g in the current green phase.
    per-lane service (K multi-hot, default ON) = which green phases serve this lane —
    the lane-identity fix for permutation-invariant token nets (R1).
    """

    def __init__(self, ts: TrafficSignal, include_phase_service: bool = True, **kw):
        super().__init__(ts, **kw)
        self._lanes = list(ts.lanes)
        self._n_lanes = len(self._lanes)
        self._phase_served_set = {k: set(lanes) for k, lanes in self._phase_lanes.items()}
        # Lane identity for set-pooling Q-nets (R1 fix): a K-dim "served-by-phase"
        # multi-hot per token — which green phases give this lane G/g. Without it the
        # transformer's CLS pooling is permutation-invariant over tokens, so e.g.
        # "NS-left queued" vs "EW-left queued" produce identical Q-values (measured).
        # This is NOT static geometry — it is the action-aligned identity ("which
        # phase releases this queue"). Disable only as an identity-ablation arm.
        self.include_phase_service = include_phase_service
        K = ts.num_green_phases
        self._lane_service = {
            lane: [1.0 if lane in self._phase_served_set[k] else 0.0 for k in range(K)]
            for lane in self._lanes
        }

    def token_layout(self) -> dict:
        """Flat-vector layout for token-based Q-nets (e.g. LaneTokenTransformerQNet).

        Mirrors __call__/observation_space exactly:
          legacy  : header = phase_onehot(K)+min_green, token = [service(K)?]+φ
          perphase: header = [min_green_ok, phase_elapsed], token = [is_green]+[service(K)?]+φ
        """
        K = self.ts.num_green_phases
        svc = K if self.include_phase_service else 0
        if self._is_legacy:
            return {"header_dim": K + 1, "n_tokens": self._n_lanes, "token_dim": svc + self._phi_dim}
        return {"header_dim": 2, "n_tokens": self._n_lanes, "token_dim": 1 + svc + self._phi_dim}

    def __call__(self) -> np.ndarray:
        ts = self.ts
        if len(self._vid_prio) > 50000:
            self._vid_prio.clear()

        lane_stats = {lane: self._lane_bucket_stats(lane) for lane in self._lanes}

        min_green_ok = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0
        if self._is_legacy:
            obs = [1.0 if ts.green_phase == i else 0.0 for i in range(ts.num_green_phases)]
            obs.append(min_green_ok)
        else:
            obs = [min_green_ok, ts.time_since_last_phase_change / 100.0]

        served = self._phase_served_set.get(ts.green_phase, set())
        for lane in self._lanes:
            if not self._is_legacy:
                obs.append(1.0 if lane in served else 0.0)      # is_green per lane token
            if self.include_phase_service:
                obs += self._lane_service[lane]                 # K-dim lane identity (R1)
            cap = self._lane_cap[lane]
            ls = lane_stats[lane]
            for p in PRIORITY_LEVELS:
                c, q, sw, mw = ls[p]
                mean = (sw / c) if c else 0.0
                if self.normalize:
                    vals = {"count": c / cap, "queue": q / cap,
                            "mean_awt": mean / self.awt_scale, "max_awt": mw / self.awt_scale}
                else:
                    vals = {"count": c, "queue": q, "mean_awt": mean, "max_awt": mw}
                obs += [vals[f] for f in self.fields]
        return np.array(obs, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        K = self.ts.num_green_phases
        svc = K if self.include_phase_service else 0
        if self._is_legacy:
            dim = (K + 1) + self._n_lanes * (svc + self._phi_dim)
        else:
            dim = 2 + self._n_lanes * (1 + svc + self._phi_dim)
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.full(dim, np.inf, dtype=np.float32),
        )


# ── Scheme T (simplified) — 6 ready-to-use variants (phase_state × fields) ──────
class PriorityLaneLegacyCQ(PriorityLaneTokenObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQ, **kw)


class PriorityLaneLegacyCQM(PriorityLaneTokenObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQM, **kw)


class PriorityLaneLegacyCQMM(PriorityLaneTokenObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="legacy", fields=_CQMM, **kw)


class PriorityLaneElapsedCQ(PriorityLaneTokenObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQ, **kw)


class PriorityLaneElapsedCQM(PriorityLaneTokenObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQM, **kw)


class PriorityLaneElapsedCQMM(PriorityLaneTokenObservationFunction):
    def __init__(self, ts, **kw):
        super().__init__(ts, phase_state="perphase", fields=_CQMM, **kw)


class PriorityCtrlBCAObservationFunction(ObservationFunction):
    """Priority observation with car/bus/ambulance 3-class counts on signal_controlled_lanes only.

    Observation vector:
        [phase_one_hot (n), min_green (1),
         inc_car (n_ctrl), out_car (n_ctrl_out),
         inc_bus (n_ctrl), out_bus (n_ctrl_out),
         inc_amb (n_ctrl), out_amb (n_ctrl_out)]
    """

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        ctrl_set = set(ts.signal_controlled_lanes)
        ordered: dict = {}
        for link_group in ts.sumo.trafficlight.getControlledLinks(ts.id):
            if link_group:
                from_lane, to_lane, _ = link_group[0]
                if from_lane in ctrl_set:
                    ordered[to_lane] = None
        self._ctrl_out_lanes = list(ordered.keys())

    def __call__(self) -> np.ndarray:
        phase_id  = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        inc_car, inc_bus, inc_amb = self._count_by_type(self.ts.signal_controlled_lanes)
        out_car, out_bus, out_amb = self._count_by_type(self._ctrl_out_lanes)
        return np.array(
            phase_id + min_green
            + inc_car + out_car
            + inc_bus + out_bus
            + inc_amb + out_amb,
            dtype=np.float32,
        )

    def observation_space(self) -> spaces.Box:
        num_phases = self.ts.num_green_phases
        n_ctrl     = len(self.ts.signal_controlled_lanes)
        n_ctrl_out = len(self._ctrl_out_lanes)
        dim        = num_phases + 1 + 3 * (n_ctrl + n_ctrl_out)
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.full(dim, np.inf, dtype=np.float32),
        )

    def _count_by_type(self, lanes):
        car_l, bus_l, amb_l = [], [], []
        for lane in lanes:
            c = b = a = 0
            for v in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
                vtype = self.ts.sumo.vehicle.getTypeID(v)
                if vtype == "ambulance":
                    a += 1
                elif vtype == "bus":
                    b += 1
                else:
                    c += 1
            car_l.append(c)
            bus_l.append(b)
            amb_l.append(a)
        return car_l, bus_l, amb_l


class PriorityCtrlObservationFunction(ObservationFunction):
    """Priority observation using only signal_controlled_lanes (excludes always-green right-turn lanes).

    Both incoming and outgoing lanes are filtered: only lanes connected to/from
    signal_controlled_lanes are included.

    Observation vector:
        [phase_one_hot (n), min_green (1),
         inc_car (n_ctrl), out_car (n_ctrl_out),
         inc_truck (n_ctrl), out_truck (n_ctrl_out)]
    """

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        ctrl_set = set(ts.signal_controlled_lanes)
        ctrl_out_ordered = {}
        for link_group in ts.sumo.trafficlight.getControlledLinks(ts.id):
            if link_group:
                from_lane, to_lane, _ = link_group[0]
                if from_lane in ctrl_set:
                    ctrl_out_ordered[to_lane] = None
        self._ctrl_out_lanes = list(ctrl_out_ordered.keys())
        print(
            f"PriorityCtrlObservationFunction: "
            f"{len(ts.signal_controlled_lanes)} ctrl inc lanes, "
            f"{len(self._ctrl_out_lanes)} ctrl out lanes "
            f"(vs {len(ts.lanes)} / {len(ts.out_lanes)} total).\n"
            f"  ctrl inc:      {ts.signal_controlled_lanes}\n"
            f"  always-green:  {ts.always_green_lanes}\n"
            f"  ctrl out:      {self._ctrl_out_lanes}"
        )

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        inc_car, inc_truck, out_car, out_truck = self._get_vehicle_count_by_type_ctrl()
        return np.array(phase_id + min_green + inc_car + out_car + inc_truck + out_truck, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        num_phases = self.ts.num_green_phases
        num_ctrl = len(self.ts.signal_controlled_lanes)
        num_ctrl_out = len(self._ctrl_out_lanes)
        dim = num_phases + 1 + 2 * num_ctrl + 2 * num_ctrl_out
        return spaces.Box(low=np.zeros(dim, dtype=np.float32), high=np.ones(dim, dtype=np.float32))

    def _get_vehicle_count_by_type_ctrl(self):
        inc_car, inc_truck = [], []
        for lane in self.ts.signal_controlled_lanes:
            vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            cars = sum(1 for v in vids if self.ts.sumo.vehicle.getTypeID(v) == "car")
            inc_car.append(cars)
            inc_truck.append(len(vids) - cars)
        out_car, out_truck = [], []
        for lane in self._ctrl_out_lanes:
            vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            cars = sum(1 for v in vids if self.ts.sumo.vehicle.getTypeID(v) == "car")
            out_car.append(cars)
            out_truck.append(len(vids) - cars)
        return inc_car, inc_truck, out_car, out_truck


class PriorityNormObservationFunction(ObservationFunction):
    """Priority observation function with car/truck density normalised to [0, 1].

    Observation vector:
        [phase_one_hot (n), min_green (1),
         inc_car_density (n_in), out_car_density (n_out),
         inc_truck_density (n_in), out_truck_density (n_out)]

    Each vehicle-count feature is divided by the lane's maximum vehicle capacity
    (= lane_length / (MIN_GAP + avg_vehicle_length)), then clipped to [0, 1].
    This mirrors the normalisation used by get_lanes_density_by_type() for
    incoming lanes, extended here to outgoing lanes and the car/truck split.
    """

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        print("This is PriorityNormObservationFunction with normalised car/truck density.")

    def __call__(self) -> np.ndarray:
        phase_id  = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]

        inc_car_d, inc_truck_d = self._density_by_type(self.ts.lanes)
        out_car_d, out_truck_d = self._density_by_type(self.ts.out_lanes)

        return np.array(
            phase_id + min_green
            + inc_car_d + out_car_d
            + inc_truck_d + out_truck_d,
            dtype=np.float32,
        )

    def observation_space(self) -> spaces.Box:
        num_phases = self.ts.num_green_phases
        n_in       = len(self.ts.lanes)
        n_out      = len(self.ts.out_lanes)
        dim        = num_phases + 1 + 2 * n_in + 2 * n_out
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.ones(dim, dtype=np.float32),
        )

    def _density_by_type(self, lanes):
        """Return (car_density, truck_density) lists for the given lanes, each in [0, 1]."""
        car_d, truck_d = [], []
        for lane in lanes:
            vids     = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            capacity = self.ts.lanes_length[lane] / (
                self.ts.MIN_GAP + self.ts.sumo.lane.getLastStepLength(lane)
            )
            if capacity <= 0:
                car_d.append(0.0)
                truck_d.append(0.0)
                continue
            cars   = sum(1 for v in vids if self.ts.sumo.vehicle.getTypeID(v) == "car")
            trucks = len(vids) - cars
            car_d.append(min(1.0, cars   / capacity))
            truck_d.append(min(1.0, trucks / capacity))
        return car_d, truck_d


class DiffWaitingObservationFunction(ObservationFunction):
    """Observation aligned with diff-waiting-time reward.

    Observation vector:
        [phase_one_hot (n), min_green (1), wait_per_lane (n_in)]

    wait_per_lane: sum of getAccumulatedWaitingTime for all vehicles on each
    incoming lane, scaled by /100. Unbounded; use norm_obs=true (VecNormalize)
    when training with PPO, or rely on DQN's replay buffer for implicit scaling.
    """

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id  = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        wait      = [self._lane_wait(lane) / 100.0 for lane in self.ts.lanes]
        return np.array(phase_id + min_green + wait, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        n_phases = self.ts.num_green_phases
        n_in     = len(self.ts.lanes)
        low  = np.zeros(n_phases + 1 + n_in, dtype=np.float32)
        high = np.concatenate([
            np.ones(n_phases + 1, dtype=np.float32),          # phase + min_green bounded [0,1]
            np.full(n_in, np.inf, dtype=np.float32),           # wait: unbounded
        ])
        return spaces.Box(low=low, high=high)

    def _lane_wait(self, lane: str) -> float:
        return sum(
            self.ts.sumo.vehicle.getAccumulatedWaitingTime(v)
            for v in self.ts.sumo.lane.getLastStepVehicleIDs(lane)
        )


class PriorityDiffWaitingObservationFunction(ObservationFunction):
    """Observation aligned with 51-diff-waiting-time reward (car/bus split).

    Observation vector:
        [phase_one_hot (n), min_green (1), car_wait_per_lane (n_in), bus_wait_per_lane (n_in)]

    Each value is sum of getAccumulatedWaitingTime for that vehicle type on the lane,
    scaled by /100. Unbounded; use norm_obs=true (VecNormalize) for PPO.
    """

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id  = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        car_wait, bus_wait = [], []
        for lane in self.ts.lanes:
            car_w, bus_w = self._lane_wait_by_type(lane)
            car_wait.append(car_w / 100.0)
            bus_wait.append(bus_w / 100.0)
        return np.array(phase_id + min_green + car_wait + bus_wait, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        n_phases = self.ts.num_green_phases
        n_in     = len(self.ts.lanes)
        low  = np.zeros(n_phases + 1 + 2 * n_in, dtype=np.float32)
        high = np.concatenate([
            np.ones(n_phases + 1, dtype=np.float32),           # phase + min_green bounded [0,1]
            np.full(2 * n_in, np.inf, dtype=np.float32),       # car/bus wait: unbounded
        ])
        return spaces.Box(low=low, high=high)

    def _lane_wait_by_type(self, lane: str):
        car_w = bus_w = 0.0
        for v in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
            acc = self.ts.sumo.vehicle.getAccumulatedWaitingTime(v)
            if self.ts.sumo.vehicle.getTypeID(v) == "car":
                car_w += acc
            else:
                bus_w += acc
        return car_w, bus_w


class PriorityWaitingBCAObservationFunction(ObservationFunction):
    """Observation aligned with bus/car/ambulance avg-waiting-time reward.

    Uses all incoming lanes (ts.lanes), including always-green right-turn lanes.

    Observation vector:
        [phase_one_hot (n), min_green (1),
         car_wait_per_lane (n_in), bus_wait_per_lane (n_in), amb_wait_per_lane (n_in)]

    Each value is sum of getAccumulatedWaitingTime for that vehicle type on the lane,
    scaled by /100. Unbounded; use norm_obs=true (VecNormalize) for PPO.
    """

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id  = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        car_wait, bus_wait, amb_wait = [], [], []
        for lane in self.ts.lanes:
            c, b, a = self._lane_wait_by_type(lane)
            car_wait.append(c / 100.0)
            bus_wait.append(b / 100.0)
            amb_wait.append(a / 100.0)
        return np.array(phase_id + min_green + car_wait + bus_wait + amb_wait, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        n_phases = self.ts.num_green_phases
        n_in     = len(self.ts.lanes)
        low  = np.zeros(n_phases + 1 + 3 * n_in, dtype=np.float32)
        high = np.concatenate([
            np.ones(n_phases + 1, dtype=np.float32),
            np.full(3 * n_in, np.inf, dtype=np.float32),
        ])
        return spaces.Box(low=low, high=high)

    def _lane_wait_by_type(self, lane: str):
        car_w = bus_w = amb_w = 0.0
        for v in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
            acc = self.ts.sumo.vehicle.getAccumulatedWaitingTime(v)
            vtype = self.ts.sumo.vehicle.getTypeID(v)
            if vtype == "ambulance":
                amb_w += acc
            elif vtype == "bus":
                bus_w += acc
            else:
                car_w += acc
        return car_w, bus_w, amb_w


class CTBPriorityObservationFunction(ObservationFunction):
    """Priority observation with optimized vehicle-type counting for performance."""

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        
    def __call__(self) -> np.ndarray:
        ts = self.ts
        # one-hot encoding of current green phase
        phase_id = [1.0 if ts.green_phase == i else 0.0
                    for i in range(ts.num_green_phases)]
        # minimum green flag
        min_green = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0

        # fast count
        inc_car, inc_truck, inc_bus, out_car, out_truck, out_bus = self._get_vehicle_count_by_type(ts.id)

        # build and return observation
        obs = phase_id + [min_green] + inc_car + out_car + inc_truck + out_truck + inc_bus + out_bus
        return np.array(obs, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        num_phases = self.ts.num_green_phases
        num_lanes = len(self.ts.lanes)
        dim = num_phases + 1 + 6 * num_lanes
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.ones(dim, dtype=np.float32),
        )

    def _get_vehicle_count_by_type(self, ts_id):
        """Efficient single-pass counting of cars, trucks, and buses per lane."""
        sumo = self.ts.sumo
        get_ids = sumo.lane.getLastStepVehicleIDs
        get_type = sumo.vehicle.getTypeID

        inc_lanes = sumo.trafficlight.getControlledLanes(ts_id)
        out_lanes = self.ts.out_lanes

        def count_types(lanes):
            cars, trucks, buses = [], [], []
            for lane in lanes:
                vids = get_ids(lane)
                c = t = b = 0
                for vid in vids:
                    vtype = get_type(vid)
                    if vtype == 'truck':
                        t += 1
                    elif vtype == 'bus':
                        b += 1
                    else:
                        c += 1
                cars.append(c)
                trucks.append(t)
                buses.append(b)
            return cars, trucks, buses

        inc_car, inc_truck, inc_bus = count_types(inc_lanes)
        out_car, out_truck, out_bus = count_types(out_lanes)
        return inc_car, inc_truck, inc_bus, out_car, out_truck, out_bus

# cursor写的，还没验证还不好用
class PolicyAwareObservationFunction(ObservationFunction):
    """基于BNF Policy的道路结构感知观察函数"""

    def __init__(self, ts: TrafficSignal, bnf_parser=None):
        super().__init__(ts)
        self.bnf_parser = bnf_parser
        self.lane_structure = self._analyze_lane_structure()
        self.policy_count = len([k for k in bnf_parser.bnf_rules.keys() if k.startswith('policy')]) if bnf_parser else 0
        
    def _analyze_lane_structure(self):
        """分析道路结构：每个方向的车道数量和类型"""
        structure = {
            'north_lanes': [],
            'south_lanes': [],
            'west_lanes': [],
            'east_lanes': [],
            'total_incoming_lanes': len(self.ts.lanes),
            'total_outgoing_lanes': len(self.ts.out_lanes)
        }
        
        # 根据车道ID分析方向
        for lane in self.ts.lanes:
            if 'n_' in lane:
                structure['north_lanes'].append(lane)
            elif 's_' in lane:
                structure['south_lanes'].append(lane)
            elif 'w_' in lane:
                structure['west_lanes'].append(lane)
            elif 'e_' in lane:
                structure['east_lanes'].append(lane)
        
        # 添加车道数量信息
        structure['north_count'] = len(structure['north_lanes'])
        structure['south_count'] = len(structure['south_lanes'])
        structure['west_count'] = len(structure['west_lanes'])
        structure['east_count'] = len(structure['east_lanes'])
        
        return structure

    def __call__(self) -> np.ndarray:
        ts = self.ts
        observation = []
        
        # 1. 基础信息：相位和时间
        phase_id = [1.0 if ts.green_phase == i else 0.0 
                   for i in range(ts.num_green_phases)]
        min_green = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0
        observation.extend(phase_id)
        observation.append(min_green)
        
        # 2. 道路结构信息（归一化）
        max_lanes = 4  # 假设最大车道数
        structure_info = [
            self.lane_structure['north_count'] / max_lanes,
            self.lane_structure['south_count'] / max_lanes,
            self.lane_structure['west_count'] / max_lanes,
            self.lane_structure['east_count'] / max_lanes,
            self.lane_structure['total_incoming_lanes'] / (max_lanes * 4),
            self.lane_structure['total_outgoing_lanes'] / (max_lanes * 4)
        ]
        observation.extend(structure_info)
        
        # 3. 每条车道的详细信息
        lane_observations = self._get_lane_based_observations()
        observation.extend(lane_observations)
        
        # 4. 方向级别的汇总信息
        direction_observations = self._get_direction_based_observations()
        observation.extend(direction_observations)
        
        return np.array(observation, dtype=np.float32)

    def _get_lane_based_observations(self):
        """获取每条车道的观察信息"""
        lane_obs = []
        
        # 处理所有进入车道
        for lane in self.ts.lanes:
            # 获取车道上的所有车辆
            vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            total_vehicles = len(vids)
            
            # 按policy分组统计
            policy_counts = self._count_vehicles_by_policy(vids)
            
            # 车道容量归一化
            lane_capacity = self.ts.sumo.lane.getLength(lane) / 7.5  # 假设每车7.5米
            
            # 添加观察：总车辆数 + 各policy车辆数
            lane_obs.append(total_vehicles / max(lane_capacity, 1.0))
            
            # 添加每个policy的车辆数（归一化）
            for policy_num in range(1, self.policy_count + 1):
                policy_key = f'policy{policy_num}'
                count = policy_counts.get(policy_key, 0)
                lane_obs.append(count / max(lane_capacity, 1.0))
            
            # 添加等待车辆数
            waiting_vehicles = self.ts.sumo.lane.getLastStepHaltingNumber(lane)
            lane_obs.append(waiting_vehicles / max(lane_capacity, 1.0))
            
            # 添加平均速度
            if vids:
                avg_speed = np.mean([self.ts.sumo.vehicle.getSpeed(vid) for vid in vids])
                lane_obs.append(avg_speed / 13.89)  # 归一化到50km/h
            else:
                lane_obs.append(0.0)
        
        return lane_obs

    def _get_direction_based_observations(self):
        """获取按方向汇总的观察信息"""
        direction_obs = []
        
        directions = ['north_lanes', 'south_lanes', 'west_lanes', 'east_lanes']
        
        for direction in directions:
            lanes = self.lane_structure[direction]
            
            if not lanes:
                # 如果该方向没有车道，填充零
                direction_obs.extend([0.0] * (2 + self.policy_count))
                continue
            
            total_vehicles = 0
            total_waiting = 0
            policy_totals = {f'policy{i}': 0 for i in range(1, self.policy_count + 1)}
            
            # 汇总该方向所有车道的信息
            for lane in lanes:
                vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
                total_vehicles += len(vids)
                total_waiting += self.ts.sumo.lane.getLastStepHaltingNumber(lane)
                
                # 按policy统计
                policy_counts = self._count_vehicles_by_policy(vids)
                for policy, count in policy_counts.items():
                    if policy in policy_totals:
                        policy_totals[policy] += count
            
            # 计算该方向的总容量
            direction_capacity = sum(self.ts.sumo.lane.getLength(lane) / 7.5 for lane in lanes)
            
            # 归一化并添加到观察
            direction_obs.append(total_vehicles / max(direction_capacity, 1.0))
            direction_obs.append(total_waiting / max(direction_capacity, 1.0))
            
            # 添加各policy的车辆数
            for policy_num in range(1, self.policy_count + 1):
                policy_key = f'policy{policy_num}'
                count = policy_totals.get(policy_key, 0)
                direction_obs.append(count / max(direction_capacity, 1.0))
        
        return direction_obs

    def _count_vehicles_by_policy(self, vehicle_ids):
        """统计车辆按policy分组的数量"""
        policy_counts = {}
        
        if not self.bnf_parser:
            return policy_counts
        
        for vid in vehicle_ids:
            try:
                vehicle_type = self.ts.sumo.vehicle.getTypeID(vid)
                context = self._get_vehicle_context(vid)
                
                # 找到匹配的policy
                matching_policy = self._find_matching_policy(vehicle_type, context)
                
                if matching_policy:
                    if matching_policy not in policy_counts:
                        policy_counts[matching_policy] = 0
                    policy_counts[matching_policy] += 1
            except:
                # 如果获取车辆信息失败，跳过
                continue
        
        return policy_counts

    def _find_matching_policy(self, vehicle_type: str, context: dict) -> str:
        """找到车辆匹配的policy"""
        if not self.bnf_parser:
            return None
        
        # 遍历所有policy规则，找到匹配的
        for policy_name, policy_def in self.bnf_parser.bnf_rules.items():
            if policy_name.startswith('policy') and self._matches_policy_rule(vehicle_type, context, policy_def):
                return policy_name
        
        return None

    def _matches_policy_rule(self, vehicle_type: str, context: dict, policy_def: str) -> bool:
        """检查车辆是否匹配特定的policy规则"""
        # 简化的匹配逻辑，您可以根据BNF语法完善
        vehicle_type_map = {
            'car': 'Car',
            'truck': 'Truck', 
            'bus': 'Bus',
            'ambulance': 'Ambulance'
        }
        
        bnf_vehicle_type = vehicle_type_map.get(vehicle_type.lower(), vehicle_type)
        
        # 检查车辆类型是否匹配
        if f'"{bnf_vehicle_type}"' not in policy_def:
            return False
        
        # 这里可以添加更复杂的上下文匹配逻辑
        # 比如匹配task, location, time等
        
        return True

    def _get_vehicle_context(self, vehicle_id: str) -> dict:
        """获取车辆上下文信息"""
        ts = self.ts
        
        # 获取时间信息
        current_time = ts.sumo.simulation.getTime()
        current_hour = (current_time // 3600) % 24
        is_peak = 7 <= current_hour <= 9 or 17 <= current_hour <= 19
        
        # 获取位置信息
        try:
            lane_id = ts.sumo.vehicle.getLaneID(vehicle_id)
            route_id = ts.sumo.vehicle.getRouteID(vehicle_id)
        except:
            lane_id = ""
            route_id = ""
        
        return {
            'task': 'on',  # 简化假设
            'np': 'M',     # 简化假设
            'location': self._infer_location(lane_id, route_id),
            'time': 'Peak' if is_peak else 'non',
            'state': 'Normal'  # 简化假设
        }

    def _infer_location(self, lane_id: str, route_id: str) -> str:
        """根据车道ID和路线ID推断位置"""
        combined = (lane_id + route_id).lower()
        
        if 'hospital' in combined:
            return 'Hospital'
        elif 'center' in combined or 'city' in combined:
            return 'City Centre'
        elif 'industrial' in combined:
            return 'Industrial'
        elif 'residential' in combined:
            return 'Residential'
        elif 'parking' in combined:
            return 'Parking'
        else:
            return '-'

    def observation_space(self) -> spaces.Box:
        """定义观察空间"""
        # 计算观察向量的总维度
        num_phases = self.ts.num_green_phases
        num_lanes = len(self.ts.lanes)
        
        # 基础信息：相位(4) + 最小绿灯(1) = 5
        basic_dim = num_phases + 1
        
        # 道路结构信息：6个特征
        structure_dim = 6
        
        # 每条车道信息：总车辆数(1) + 各policy车辆数(policy_count) + 等待车辆数(1) + 平均速度(1)
        lane_features_per_lane = 3 + self.policy_count
        lane_dim = num_lanes * lane_features_per_lane
        
        # 方向汇总信息：4个方向 × (总车辆数(1) + 等待车辆数(1) + 各policy车辆数(policy_count))
        direction_features_per_direction = 2 + self.policy_count
        direction_dim = 4 * direction_features_per_direction
        
        total_dim = basic_dim + structure_dim + lane_dim + direction_dim
        
        return spaces.Box(
            low=np.zeros(total_dim, dtype=np.float32),
            high=np.ones(total_dim, dtype=np.float32),
        )
