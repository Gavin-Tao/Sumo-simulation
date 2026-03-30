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
    "diff-waiting-time":       diff_waiting_time_reward,
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
    "CTB_priority-pressure":   CTB_priority_pressure_reward,
    "51-priority-queue":       _51_priority_queue_reward,
    "41-priority-queue":       _41_priority_queue_reward,
    "31-priority-queue":       _31_priority_queue_reward,
    "21-priority-queue":       _21_priority_queue_reward,
}
