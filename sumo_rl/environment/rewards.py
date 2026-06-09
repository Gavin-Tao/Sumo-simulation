"""Reward functions for traffic signals.

All functions take a TrafficSignal instance (ts) as the first argument,
matching the calling convention in TrafficSignal.compute_reward():
    self.last_reward = self.reward_fn(self)
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .traffic_signal import TrafficSignal


# ---------------------------------------------------------------------------
# Helper: vehicle list
# ---------------------------------------------------------------------------

def _get_veh_list(ts: "TrafficSignal") -> list:
    veh_list = []
    for lane in ts.lanes:
        veh_list += ts.sumo.lane.getLastStepVehicleIDs(lane)
    return veh_list


# ---------------------------------------------------------------------------
# Metric getters (used by reward functions below)
# ---------------------------------------------------------------------------

def get_accumulated_waiting_time_per_lane(ts: "TrafficSignal") -> List[float]:
    wait_time_per_lane = []
    for lane in ts.lanes:
        veh_list = ts.sumo.lane.getLastStepVehicleIDs(lane)
        wait_time = 0.0
        for veh in veh_list:
            veh_lane = ts.sumo.vehicle.getLaneID(veh)
            acc = ts.sumo.vehicle.getAccumulatedWaitingTime(veh)
            if veh not in ts.env.vehicles:
                ts.env.vehicles[veh] = {veh_lane: acc}
            else:
                ts.env.vehicles[veh][veh_lane] = acc - sum(
                    [ts.env.vehicles[veh][l] for l in ts.env.vehicles[veh].keys() if l != veh_lane]
                )
            wait_time += ts.env.vehicles[veh][veh_lane]
        wait_time_per_lane.append(wait_time)
    return wait_time_per_lane


def get_average_speed(ts: "TrafficSignal") -> float:
    avg_speed = 0.0
    vehs = _get_veh_list(ts)
    if len(vehs) == 0:
        return 1.0
    for v in vehs:
        avg_speed += ts.sumo.vehicle.getSpeed(v) / ts.sumo.vehicle.getAllowedSpeed(v)
    return avg_speed / len(vehs)


def get_total_queued(ts: "TrafficSignal") -> int:
    return sum(ts.sumo.lane.getLastStepHaltingNumber(lane) for lane in ts.lanes)


def get_pressure(ts: "TrafficSignal") -> float:
    return (
        sum(ts.sumo.lane.getLastStepVehicleNumber(lane) for lane in ts.out_lanes)
        - sum(ts.sumo.lane.getLastStepVehicleNumber(lane) for lane in ts.lanes)
    )


def get_priority_pressure(ts: "TrafficSignal", alpha: float = 1.0, beta: float = 1.25) -> float:
    """α*(#out_car - #in_car) + β*(#out_truck - #in_truck)"""
    in_car = in_truck = out_car = out_truck = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getTypeID(vid) == "truck":
                in_truck += 1
            else:
                in_car += 1
    for lane in ts.out_lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getTypeID(vid) == "truck":
                out_truck += 1
            else:
                out_car += 1
    return alpha * (out_car - in_car) + beta * (out_truck - in_truck)


def _make_priority_pressure_bc(alpha: float, beta: float):
    """Factory: bus/car priority pressure with given weights."""
    def fn(ts: "TrafficSignal") -> float:
        in_car = in_bus = out_car = out_bus = 0
        for lane in ts.lanes:
            for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
                if ts.sumo.vehicle.getTypeID(vid) == "car":
                    in_car += 1
                else:
                    in_bus += 1
        for lane in ts.out_lanes:
            for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
                if ts.sumo.vehicle.getTypeID(vid) == "car":
                    out_car += 1
                else:
                    out_bus += 1
        return alpha * (out_car - in_car) + beta * (out_bus - in_bus)
    return fn


get_priority_pressure_52  = lambda ts: get_priority_pressure(ts, alpha=1.0, beta=2.50)
get_priority_pressure_21  = _make_priority_pressure_bc(1.0, 2.0)
get_priority_pressure_31  = _make_priority_pressure_bc(1.0, 3.0)
get_priority_pressure_41  = _make_priority_pressure_bc(1.0, 4.0)
get_priority_pressure_51  = _make_priority_pressure_bc(1.0, 5.0)
get_priority_pressure_61  = _make_priority_pressure_bc(1.0, 6.0)
get_priority_pressure_71  = _make_priority_pressure_bc(1.0, 7.0)
get_priority_pressure_81  = _make_priority_pressure_bc(1.0, 8.0)
get_priority_pressure_91  = _make_priority_pressure_bc(1.0, 9.0)
get_priority_pressure_10_1 = _make_priority_pressure_bc(1.0, 10.0)
get_priority_pressure_20_1 = _make_priority_pressure_bc(1.0, 20.0)
get_priority_pressure_50_1 = _make_priority_pressure_bc(1.0, 50.0)


def _get_signal_controlled_out_lanes(ts: "TrafficSignal") -> list:
    """Lazily compute & cache out lanes connected from signal_controlled_lanes."""
    if not hasattr(ts, "_signal_controlled_out_lanes"):
        ctrl_set = set(ts.signal_controlled_lanes)
        ordered: dict = {}
        for link_group in ts.sumo.trafficlight.getControlledLinks(ts.id):
            if link_group:
                from_lane, to_lane, _ = link_group[0]
                if from_lane in ctrl_set:
                    ordered[to_lane] = None
        ts._signal_controlled_out_lanes = list(ordered.keys())  # type: ignore[attr-defined]
    return ts._signal_controlled_out_lanes  # type: ignore[attr-defined]


def _make_priority_pressure_bcA(alpha: float, beta: float, gamma: float):
    """Factory: car/bus/ambulance priority pressure (3-type) over all lanes/out_lanes."""
    def fn(ts: "TrafficSignal") -> float:
        in_car = in_bus = in_amb = out_car = out_bus = out_amb = 0
        for lane in ts.lanes:
            for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
                vtype = ts.sumo.vehicle.getTypeID(vid)
                if vtype == "ambulance":
                    in_amb += 1
                elif vtype == "bus":
                    in_bus += 1
                else:
                    in_car += 1
        for lane in ts.out_lanes:
            for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
                vtype = ts.sumo.vehicle.getTypeID(vid)
                if vtype == "ambulance":
                    out_amb += 1
                elif vtype == "bus":
                    out_bus += 1
                else:
                    out_car += 1
        return (alpha * (out_car - in_car)
                + beta  * (out_bus - in_bus)
                + gamma * (out_amb - in_amb))
    return fn


def _make_priority_pressure_bcA_ctrl(alpha: float, beta: float, gamma: float):
    """Factory: car/bus/ambulance priority pressure (3-type) on signal_controlled_lanes only."""
    def fn(ts: "TrafficSignal") -> float:
        in_car = in_bus = in_amb = out_car = out_bus = out_amb = 0
        for lane in ts.signal_controlled_lanes:
            for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
                vtype = ts.sumo.vehicle.getTypeID(vid)
                if vtype == "ambulance":
                    in_amb += 1
                elif vtype == "bus":
                    in_bus += 1
                else:
                    in_car += 1
        for lane in _get_signal_controlled_out_lanes(ts):
            for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
                vtype = ts.sumo.vehicle.getTypeID(vid)
                if vtype == "ambulance":
                    out_amb += 1
                elif vtype == "bus":
                    out_bus += 1
                else:
                    out_car += 1
        return (alpha * (out_car - in_car)
                + beta  * (out_bus - in_bus)
                + gamma * (out_amb - in_amb))
    return fn


_521_priority_pressure_fn      = _make_priority_pressure_bcA(1.0, 2.0, 5.0)
_521_priority_pressure_ctrl_fn = _make_priority_pressure_bcA_ctrl(1.0, 2.0, 5.0)


def _521_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return _521_priority_pressure_fn(ts)


def _521_priority_pressure_ctrl_reward(ts: "TrafficSignal") -> float:
    return _521_priority_pressure_ctrl_fn(ts)


def get_priority_pressure_45(ts: "TrafficSignal") -> float:
    """α*(#out_car-#in_car) + β*(#out_truck-#in_truck), car weight=1.25"""
    in_car = in_truck = out_car = out_truck = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getTypeID(vid) == "truck":
                in_truck += 1
            else:
                in_car += 1
    for lane in ts.out_lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getTypeID(vid) == "truck":
                out_truck += 1
            else:
                out_car += 1
    return 1.25 * (out_car - in_car) + 1.0 * (out_truck - in_truck)


def _make_priority_pressure_bc_norm(alpha: float, beta: float):
    """Factory: bus/car priority-pressure with per-lane density normalisation.

    Same weights as the raw-count variant but each lane contributes a value
    in [0, 1] (vehicles / capacity) instead of a raw integer count.
    This keeps the reward magnitude in a stable range regardless of lane length
    or the number of lanes, making it easier for PPO's value function to learn.
    """
    def fn(ts: "TrafficSignal") -> float:
        MIN_GAP  = ts.MIN_GAP
        get_ids  = ts.sumo.lane.getLastStepVehicleIDs
        get_len  = ts.sumo.lane.getLastStepLength
        get_type = ts.sumo.vehicle.getTypeID

        in_car_d = in_bus_d = out_car_d = out_bus_d = 0.0

        for lane in ts.lanes:
            vids = get_ids(lane)
            cap  = ts.lanes_length[lane] / (MIN_GAP + get_len(lane))
            if cap <= 0:
                continue
            cars = sum(1 for v in vids if get_type(v) == "car")
            buses = len(vids) - cars
            in_car_d += min(1.0, cars  / cap)
            in_bus_d += min(1.0, buses / cap)

        for lane in ts.out_lanes:
            vids = get_ids(lane)
            cap  = ts.lanes_length[lane] / (MIN_GAP + get_len(lane))
            if cap <= 0:
                continue
            cars = sum(1 for v in vids if get_type(v) == "car")
            buses = len(vids) - cars
            out_car_d += min(1.0, cars  / cap)
            out_bus_d += min(1.0, buses / cap)

        return alpha * (out_car_d - in_car_d) + beta * (out_bus_d - in_bus_d)
    return fn


get_priority_pressure_41_norm = _make_priority_pressure_bc_norm(1.0, 4.0)
get_priority_pressure_51_norm = _make_priority_pressure_bc_norm(1.0, 5.0)


def get_CTB_priority_pressure(ts: "TrafficSignal",
                               alpha: float = 1.0,
                               beta: float = 1.25,
                               gamma: float = 2.5) -> float:
    """Car/Truck/Bus weighted pressure."""
    pressure = 0.0
    get_ids = ts.sumo.lane.getLastStepVehicleIDs
    get_type = ts.sumo.vehicle.getTypeID
    for sign, lanes in ((1, ts.out_lanes), (-1, ts.lanes)):
        for lane in lanes:
            for vid in get_ids(lane):
                vtype = get_type(vid)
                if vtype == "truck":
                    pressure += beta * sign
                elif vtype == "bus":
                    pressure += gamma * sign
                else:
                    pressure += alpha * sign
    return pressure


def get_priority_queue_51(ts: "TrafficSignal") -> float:
    """-(1*car_queue + 5*bus_queue)"""
    car_q = bus_q = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getSpeed(vid) < 0.1:
                if ts.sumo.vehicle.getTypeID(vid) == "car":
                    car_q += 1
                else:
                    bus_q += 1
    return -(1.0 * car_q + 5.0 * bus_q)


def get_priority_queue_21(ts: "TrafficSignal") -> float:
    """-(1*car_queue + 2*bus_queue)"""
    car_q = bus_q = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getSpeed(vid) < 0.1:
                if ts.sumo.vehicle.getTypeID(vid) == "car":
                    car_q += 1
                else:
                    bus_q += 1
    return -(1.0 * car_q + 2.0 * bus_q)


def get_priority_queue_31(ts: "TrafficSignal") -> float:
    """-(1*car_queue + 3*bus_queue)"""
    car_q = bus_q = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getSpeed(vid) < 0.1:
                if ts.sumo.vehicle.getTypeID(vid) == "car":
                    car_q += 1
                else:
                    bus_q += 1
    return -(1.0 * car_q + 3.0 * bus_q)


def get_priority_queue_41(ts: "TrafficSignal") -> float:
    """-(1*car_queue + 4*bus_queue)"""
    car_q = bus_q = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getSpeed(vid) < 0.1:
                if ts.sumo.vehicle.getTypeID(vid) == "car":
                    car_q += 1
                else:
                    bus_q += 1
    return -(1.0 * car_q + 4.0 * bus_q)


# ---------------------------------------------------------------------------
# Reward functions (called as reward_fn(ts))
# ---------------------------------------------------------------------------

def diff_waiting_time_reward(ts: "TrafficSignal") -> float:
    ts_wait = sum(get_accumulated_waiting_time_per_lane(ts)) / 100.0
    reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait
    return reward


def _get_weighted_waiting_time(ts: "TrafficSignal", alpha: float, beta: float) -> float:
    """Weighted accumulated waiting time: alpha*car_wait + beta*bus_wait across all incoming lanes.

    Uses ts.env.vehicles for cross-lane correction (same as get_accumulated_waiting_time_per_lane),
    so a vehicle that switched lanes only contributes its wait on the current lane.
    """
    total = 0.0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            veh_lane = ts.sumo.vehicle.getLaneID(vid)
            acc      = ts.sumo.vehicle.getAccumulatedWaitingTime(vid)
            if vid not in ts.env.vehicles:
                ts.env.vehicles[vid] = {veh_lane: acc}
            else:
                ts.env.vehicles[vid][veh_lane] = acc - sum(
                    ts.env.vehicles[vid][l]
                    for l in ts.env.vehicles[vid] if l != veh_lane
                )
            lane_wait = ts.env.vehicles[vid][veh_lane]
            if ts.sumo.vehicle.getTypeID(vid) == "car":
                total += alpha * lane_wait
            else:
                total += beta * lane_wait
    return total / 100.0


def _get_weighted_avg_waiting_time(ts: "TrafficSignal", alpha: float, beta: float) -> float:
    """Per-vehicle weighted average waiting time: alpha*car_avg + beta*bus_avg.

    Averages accumulated waiting time across vehicles of each type separately,
    then combines with weights. Returns 0 for a type with no vehicles present.
    Uses cross-lane correction identical to _get_weighted_waiting_time.
    """
    car_total = bus_total = 0.0
    car_count = bus_count = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            veh_lane = ts.sumo.vehicle.getLaneID(vid)
            acc = ts.sumo.vehicle.getAccumulatedWaitingTime(vid)
            if vid not in ts.env.vehicles:
                ts.env.vehicles[vid] = {veh_lane: acc}
            else:
                ts.env.vehicles[vid][veh_lane] = acc - sum(
                    ts.env.vehicles[vid][l]
                    for l in ts.env.vehicles[vid] if l != veh_lane
                )
            lane_wait = ts.env.vehicles[vid][veh_lane]
            if ts.sumo.vehicle.getTypeID(vid) == "car":
                car_total += lane_wait
                car_count += 1
            else:
                bus_total += lane_wait
                bus_count += 1
    car_avg = (car_total / car_count) if car_count > 0 else 0.0
    bus_avg = (bus_total / bus_count) if bus_count > 0 else 0.0
    return (alpha * car_avg + beta * bus_avg) / 100.0


def _make_avg_waiting_bc(alpha: float, beta: float):
    """Factory: negative weighted average waiting time reward."""
    def fn(ts: "TrafficSignal") -> float:
        return -_get_weighted_avg_waiting_time(ts, alpha, beta)
    return fn


def _make_avg_diff_waiting_bc(alpha: float, beta: float):
    """Factory: differential weighted average waiting time reward."""
    def fn(ts: "TrafficSignal") -> float:
        curr = _get_weighted_avg_waiting_time(ts, alpha, beta)
        reward = getattr(ts, "_last_weighted_avg_wait", 0.0) - curr
        ts._last_weighted_avg_wait = curr  # type: ignore[attr-defined]
        return reward
    return fn


_51_avg_waiting_fn      = _make_avg_waiting_bc(1.0, 5.0)
_51_avg_diff_waiting_fn = _make_avg_diff_waiting_bc(1.0, 5.0)


def _51_avg_waiting_reward(ts: "TrafficSignal") -> float:
    return _51_avg_waiting_fn(ts)


def _51_avg_diff_waiting_reward(ts: "TrafficSignal") -> float:
    return _51_avg_diff_waiting_fn(ts)


def _get_weighted_avg_waiting_time_bcA(
    ts: "TrafficSignal",
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Per-vehicle weighted average waiting time over all incoming lanes (ts.lanes),
    split by 3 types: car / bus / ambulance.

    Returns alpha*avg_car + beta*avg_bus + gamma*avg_amb. Cross-lane wait correction
    identical to _get_weighted_avg_waiting_time.
    """
    car_total = bus_total = amb_total = 0.0
    car_count = bus_count = amb_count = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            veh_lane = ts.sumo.vehicle.getLaneID(vid)
            acc = ts.sumo.vehicle.getAccumulatedWaitingTime(vid)
            if vid not in ts.env.vehicles:
                ts.env.vehicles[vid] = {veh_lane: acc}
            else:
                ts.env.vehicles[vid][veh_lane] = acc - sum(
                    ts.env.vehicles[vid][l]
                    for l in ts.env.vehicles[vid] if l != veh_lane
                )
            lane_wait = ts.env.vehicles[vid][veh_lane]
            vtype = ts.sumo.vehicle.getTypeID(vid)
            if vtype == "ambulance":
                amb_total += lane_wait
                amb_count += 1
            elif vtype == "bus":
                bus_total += lane_wait
                bus_count += 1
            else:
                car_total += lane_wait
                car_count += 1
    car_avg = (car_total / car_count) if car_count > 0 else 0.0
    bus_avg = (bus_total / bus_count) if bus_count > 0 else 0.0
    amb_avg = (amb_total / amb_count) if amb_count > 0 else 0.0
    return (alpha * car_avg + beta * bus_avg + gamma * amb_avg) / 100.0


def _make_avg_waiting_bcA(alpha: float, beta: float, gamma: float):
    def fn(ts: "TrafficSignal") -> float:
        return -_get_weighted_avg_waiting_time_bcA(ts, alpha, beta, gamma)
    return fn


_521_avg_waiting_fn = _make_avg_waiting_bcA(1.0, 2.0, 5.0)
_531_avg_waiting_fn = _make_avg_waiting_bcA(1.0, 3.0, 5.0)  # stronger bus weight variant
_541_avg_waiting_fn = _make_avg_waiting_bcA(1.0, 4.0, 5.0)  # even stronger bus weight variant
_641_avg_waiting_fn = _make_avg_waiting_bcA(1.0, 4.0, 6.0)  # higher amb weight to fix priority inversion at high bus weight


def _521_avg_waiting_reward(ts: "TrafficSignal") -> float:
    return _521_avg_waiting_fn(ts)


def _531_avg_waiting_reward(ts: "TrafficSignal") -> float:
    return _531_avg_waiting_fn(ts)


def _541_avg_waiting_reward(ts: "TrafficSignal") -> float:
    return _541_avg_waiting_fn(ts)


def _641_avg_waiting_reward(ts: "TrafficSignal") -> float:
    return _641_avg_waiting_fn(ts)


# ---------------------------------------------------------------------------
# Mixed: car/bus use AVG waiting time, ambulance uses MAX waiting time.
# Rationale: ambulance is safety-critical (only 1-2 per episode);
#  what matters is WORST-CASE latency, not average.
# Same per-lane wait correction as _get_weighted_avg_waiting_time_bcA.
# ---------------------------------------------------------------------------

def _get_weighted_mixed_waiting_time_bcA(
    ts: "TrafficSignal",
    alpha: float,  # car weight
    beta:  float,  # bus weight
    gamma: float,  # ambulance weight
) -> float:
    """car/bus → avg waiting; ambulance → max waiting. Returns alpha*avg_car + beta*avg_bus + gamma*max_amb (/100)."""
    car_total = bus_total = 0.0
    car_count = bus_count = 0
    amb_max   = 0.0  # max wait across all amb on lanes (0 if none)
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            veh_lane = ts.sumo.vehicle.getLaneID(vid)
            acc = ts.sumo.vehicle.getAccumulatedWaitingTime(vid)
            if vid not in ts.env.vehicles:
                ts.env.vehicles[vid] = {veh_lane: acc}
            else:
                ts.env.vehicles[vid][veh_lane] = acc - sum(
                    ts.env.vehicles[vid][l]
                    for l in ts.env.vehicles[vid] if l != veh_lane
                )
            lane_wait = ts.env.vehicles[vid][veh_lane]
            vtype = ts.sumo.vehicle.getTypeID(vid)
            if vtype == "ambulance":
                if lane_wait > amb_max:
                    amb_max = lane_wait      # ← MAX, not avg
            elif vtype == "bus":
                bus_total += lane_wait
                bus_count += 1
            else:
                car_total += lane_wait
                car_count += 1
    car_avg = (car_total / car_count) if car_count > 0 else 0.0
    bus_avg = (bus_total / bus_count) if bus_count > 0 else 0.0
    return (alpha * car_avg + beta * bus_avg + gamma * amb_max) / 100.0


def _make_mixed_waiting_bcA(alpha: float, beta: float, gamma: float):
    def fn(ts: "TrafficSignal") -> float:
        return -_get_weighted_mixed_waiting_time_bcA(ts, alpha, beta, gamma)
    return fn


_541_mixed_waiting_fn = _make_mixed_waiting_bcA(1.0, 4.0, 5.0)  # car/bus avg + amb max, weights 5-4-1
_531_mixed_waiting_fn = _make_mixed_waiting_bcA(1.0, 3.0, 5.0)  # car/bus avg + amb max, weights 5-3-1
_521_mixed_waiting_fn = _make_mixed_waiting_bcA(1.0, 2.0, 5.0)  # car/bus avg + amb max, weights 5-2-1


def _541_mixed_waiting_reward(ts: "TrafficSignal") -> float:
    return _541_mixed_waiting_fn(ts)


def _531_mixed_waiting_reward(ts: "TrafficSignal") -> float:
    return _531_mixed_waiting_fn(ts)


def _521_mixed_waiting_reward(ts: "TrafficSignal") -> float:
    return _521_mixed_waiting_fn(ts)


# ---------------------------------------------------------------------------
# Plain (vehicle-type-agnostic) avg waiting reward — "no priority" baseline.
# Each vehicle counted as 1 unit regardless of type. Cross-lane wait correction
# identical to _get_weighted_avg_waiting_time. Used as ablation against the
# 5-2-1 weighted variant to prove that priority behavior comes from weights,
# not from RL training or observation encoding alone.
# ---------------------------------------------------------------------------

def _get_plain_avg_waiting_time(ts: "TrafficSignal") -> float:
    """Plain per-vehicle average waiting time across all incoming lanes.

    No per-type splitting / weighting: every vehicle contributes its
    accumulated wait once, divided by total vehicle count on ts.lanes.
    """
    total = 0.0
    count = 0
    for lane in ts.lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            veh_lane = ts.sumo.vehicle.getLaneID(vid)
            acc = ts.sumo.vehicle.getAccumulatedWaitingTime(vid)
            if vid not in ts.env.vehicles:
                ts.env.vehicles[vid] = {veh_lane: acc}
            else:
                ts.env.vehicles[vid][veh_lane] = acc - sum(
                    ts.env.vehicles[vid][l]
                    for l in ts.env.vehicles[vid] if l != veh_lane
                )
            total += ts.env.vehicles[vid][veh_lane]
            count += 1
    return (total / count) / 100.0 if count > 0 else 0.0


def _avg_waiting_reward(ts: "TrafficSignal") -> float:
    """Vehicle-type-agnostic avg waiting time reward (no priority bias)."""
    return -_get_plain_avg_waiting_time(ts)


def _get_weighted_avg_waiting_time_ctrl(ts: "TrafficSignal", alpha: float, beta: float) -> float:
    """Same as _get_weighted_avg_waiting_time but only over signal_controlled_lanes."""
    car_total = bus_total = 0.0
    car_count = bus_count = 0
    for lane in ts.signal_controlled_lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            veh_lane = ts.sumo.vehicle.getLaneID(vid)
            acc = ts.sumo.vehicle.getAccumulatedWaitingTime(vid)
            if vid not in ts.env.vehicles:
                ts.env.vehicles[vid] = {veh_lane: acc}
            else:
                ts.env.vehicles[vid][veh_lane] = acc - sum(
                    ts.env.vehicles[vid][l]
                    for l in ts.env.vehicles[vid] if l != veh_lane
                )
            lane_wait = ts.env.vehicles[vid][veh_lane]
            if ts.sumo.vehicle.getTypeID(vid) == "car":
                car_total += lane_wait
                car_count += 1
            else:
                bus_total += lane_wait
                bus_count += 1
    car_avg = (car_total / car_count) if car_count > 0 else 0.0
    bus_avg = (bus_total / bus_count) if bus_count > 0 else 0.0
    return (alpha * car_avg + beta * bus_avg) / 100.0


def _make_avg_waiting_bc_ctrl(alpha: float, beta: float):
    def fn(ts: "TrafficSignal") -> float:
        return -_get_weighted_avg_waiting_time_ctrl(ts, alpha, beta)
    return fn


def _make_avg_diff_waiting_bc_ctrl(alpha: float, beta: float):
    def fn(ts: "TrafficSignal") -> float:
        curr = _get_weighted_avg_waiting_time_ctrl(ts, alpha, beta)
        reward = getattr(ts, "_last_weighted_avg_wait_ctrl", 0.0) - curr
        ts._last_weighted_avg_wait_ctrl = curr  # type: ignore[attr-defined]
        return reward
    return fn


_51_avg_waiting_ctrl_fn      = _make_avg_waiting_bc_ctrl(1.0, 5.0)
_51_avg_diff_waiting_ctrl_fn = _make_avg_diff_waiting_bc_ctrl(1.0, 5.0)


def _51_avg_waiting_ctrl_reward(ts: "TrafficSignal") -> float:
    return _51_avg_waiting_ctrl_fn(ts)


def _51_avg_diff_waiting_ctrl_reward(ts: "TrafficSignal") -> float:
    return _51_avg_diff_waiting_ctrl_fn(ts)


def _get_weighted_avg_waiting_time_ctrl_bcA(
    ts: "TrafficSignal",
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Per-vehicle weighted average waiting time over signal_controlled_lanes,
    split by 3 types: car / bus / ambulance.

    Returns alpha*avg_car + beta*avg_bus + gamma*avg_amb.
    Per-type avg uses only vehicles of that type currently in signal_controlled_lanes;
    types with zero count contribute 0. Cross-lane wait correction identical to
    _get_weighted_avg_waiting_time_ctrl.
    """
    car_total = bus_total = amb_total = 0.0
    car_count = bus_count = amb_count = 0
    for lane in ts.signal_controlled_lanes:
        for vid in ts.sumo.lane.getLastStepVehicleIDs(lane):
            veh_lane = ts.sumo.vehicle.getLaneID(vid)
            acc = ts.sumo.vehicle.getAccumulatedWaitingTime(vid)
            if vid not in ts.env.vehicles:
                ts.env.vehicles[vid] = {veh_lane: acc}
            else:
                ts.env.vehicles[vid][veh_lane] = acc - sum(
                    ts.env.vehicles[vid][l]
                    for l in ts.env.vehicles[vid] if l != veh_lane
                )
            lane_wait = ts.env.vehicles[vid][veh_lane]
            vtype = ts.sumo.vehicle.getTypeID(vid)
            if vtype == "ambulance":
                amb_total += lane_wait
                amb_count += 1
            elif vtype == "bus":
                bus_total += lane_wait
                bus_count += 1
            else:
                car_total += lane_wait
                car_count += 1
    car_avg = (car_total / car_count) if car_count > 0 else 0.0
    bus_avg = (bus_total / bus_count) if bus_count > 0 else 0.0
    amb_avg = (amb_total / amb_count) if amb_count > 0 else 0.0
    return (alpha * car_avg + beta * bus_avg + gamma * amb_avg) / 100.0


def _make_avg_waiting_bcA_ctrl(alpha: float, beta: float, gamma: float):
    def fn(ts: "TrafficSignal") -> float:
        return -_get_weighted_avg_waiting_time_ctrl_bcA(ts, alpha, beta, gamma)
    return fn


_521_avg_waiting_ctrl_fn = _make_avg_waiting_bcA_ctrl(1.0, 2.0, 5.0)


def _521_avg_waiting_ctrl_reward(ts: "TrafficSignal") -> float:
    return _521_avg_waiting_ctrl_fn(ts)


def _make_diff_waiting_bc(alpha: float, beta: float):
    """Factory: weighted diff-waiting-time reward with car/bus weights."""
    def fn(ts: "TrafficSignal") -> float:
        ts_wait = _get_weighted_waiting_time(ts, alpha, beta)
        reward  = getattr(ts, "_last_weighted_wait", 0.0) - ts_wait
        ts._last_weighted_wait = ts_wait  # type: ignore[attr-defined]
        return reward
    return fn


_51_diff_waiting_fn = _make_diff_waiting_bc(1.0, 5.0)
_41_diff_waiting_fn = _make_diff_waiting_bc(1.0, 4.0)


def _51_diff_waiting_reward(ts: "TrafficSignal") -> float:
    return _51_diff_waiting_fn(ts)


def _41_diff_waiting_reward(ts: "TrafficSignal") -> float:
    return _41_diff_waiting_fn(ts)


def average_speed_reward(ts: "TrafficSignal") -> float:
    return get_average_speed(ts)


def queue_reward(ts: "TrafficSignal") -> float:
    return -get_total_queued(ts)


def pressure_reward(ts: "TrafficSignal") -> float:
    return get_pressure(ts)


def pressure_norm_reward(ts: "TrafficSignal") -> float:
    return sum(ts.get_out_lanes_density()) - sum(ts.get_lanes_density())


def priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure(ts)


def _52_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_52(ts)


def _21_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_21(ts)


def _31_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_31(ts)


def _41_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_41(ts)


def _51_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_51(ts)


def _61_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_61(ts)


def _71_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_71(ts)


def _81_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_81(ts)


def _91_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_91(ts)


def _10_1_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_10_1(ts)


def _20_1_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_20_1(ts)


def _50_1_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_50_1(ts)


def _45_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_45(ts)


def _41_priority_pressure_norm_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_41_norm(ts)


def _51_priority_pressure_norm_reward(ts: "TrafficSignal") -> float:
    return get_priority_pressure_51_norm(ts)


def CTB_priority_pressure_reward(ts: "TrafficSignal") -> float:
    return get_CTB_priority_pressure(ts)


def _51_priority_queue_reward(ts: "TrafficSignal") -> float:
    return get_priority_queue_51(ts)


def _41_priority_queue_reward(ts: "TrafficSignal") -> float:
    return get_priority_queue_41(ts)


def _31_priority_queue_reward(ts: "TrafficSignal") -> float:
    return get_priority_queue_31(ts)


def _21_priority_queue_reward(ts: "TrafficSignal") -> float:
    return get_priority_queue_21(ts)


# ---------------------------------------------------------------------------
# Registry: maps string name → function
# ---------------------------------------------------------------------------

REWARD_REGISTRY = {
    "diff-waiting-time":            diff_waiting_time_reward,
    "51-diff-waiting-time":         _51_diff_waiting_reward,
    "41-diff-waiting-time":         _41_diff_waiting_reward,
    "51-avg-waiting-time":          _51_avg_waiting_reward,
    "51-avg-diff-waiting-time":     _51_avg_diff_waiting_reward,
    "51-avg-waiting-time-ctrl":     _51_avg_waiting_ctrl_reward,
    "51-avg-diff-waiting-time-ctrl": _51_avg_diff_waiting_ctrl_reward,
    "5-2-1-avg-waiting-time":       _521_avg_waiting_reward,
    "5-3-1-avg-waiting-time":       _531_avg_waiting_reward,
    "5-4-1-avg-waiting-time":       _541_avg_waiting_reward,
    "6-4-1-avg-waiting-time":       _641_avg_waiting_reward,
    "5-4-1-avg-waiting-time-ambmax": _541_mixed_waiting_reward,   # car/bus avg + amb MAX
    "5-3-1-avg-waiting-time-ambmax": _531_mixed_waiting_reward,
    "5-2-1-avg-waiting-time-ambmax": _521_mixed_waiting_reward,
    "5-2-1-avg-waiting-time-ctrl":  _521_avg_waiting_ctrl_reward,
    "avg-waiting-time":             _avg_waiting_reward,

    "5-2-1-priority-pressure":      _521_priority_pressure_reward,
    "5-2-1-priority-pressure-ctrl": _521_priority_pressure_ctrl_reward,
    "average-speed":           average_speed_reward,
    "queue":                   queue_reward,
    "pressure":                pressure_reward,
    "pressure-norm":           pressure_norm_reward,
    "priority-pressure":       priority_pressure_reward,
    "52-priority-pressure":    _52_priority_pressure_reward,
    "21-priority-pressure":    _21_priority_pressure_reward,
    "31-priority-pressure":    _31_priority_pressure_reward,
    "41-priority-pressure":    _41_priority_pressure_reward,
    "51-priority-pressure":    _51_priority_pressure_reward,
    "61-priority-pressure":    _61_priority_pressure_reward,
    "71-priority-pressure":    _71_priority_pressure_reward,
    "81-priority-pressure":    _81_priority_pressure_reward,
    "91-priority-pressure":    _91_priority_pressure_reward,
    "10_1-priority-pressure":  _10_1_priority_pressure_reward,
    "20_1-priority-pressure":  _20_1_priority_pressure_reward,
    "50_1-priority-pressure":  _50_1_priority_pressure_reward,
    "45-priority-pressure":    _45_priority_pressure_reward,
    "41-priority-pressure-norm": _41_priority_pressure_norm_reward,
    "51-priority-pressure-norm": _51_priority_pressure_norm_reward,
    "CTB_priority-pressure":   CTB_priority_pressure_reward,
    "51-priority-queue":       _51_priority_queue_reward,
    "41-priority-queue":       _41_priority_queue_reward,
    "31-priority-queue":       _31_priority_queue_reward,
    "21-priority-queue":       _21_priority_queue_reward,
}
