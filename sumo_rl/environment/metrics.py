"""Evaluation metrics collectors for SUMO-RL environments.

Usage in env.py:
    from .metrics import CTBMetricsCollector, DirectionalMetricsCollector

    # in __init__:
    self.metrics_collector = CTBMetricsCollector()

    # in _sumo_step:
    self.metrics_collector.collect_step(self.sumo)

Results are accessible via:
    env.metrics_collector.CTB_Metrics      – latest step snapshot
    env.metrics_collector.step_history     – full per-step history
"""

from __future__ import annotations


class DirectionalMetricsCollector:
    """Collects per-direction (N_S / W_E) metrics (legacy version).

    Hardcoded for the 1x1 syc intersection layout.
    Call collect_step(sumo) every simulation step during evaluation.
    """

    N_S_LANES = [
        "n_t_0", "n_t_1", "n_t_2",
        "t_s_0", "t_s_1", "t_s_2",
        "s_t_0", "s_t_1", "s_t_2",
        "t_n_0", "t_n_1", "t_n_2",
    ]
    W_E_LANES = [
        "w_t_0", "w_t_1", "w_t_2",
        "t_e_0", "t_e_1", "t_e_2",
        "e_t_0", "e_t_1", "e_t_2",
        "t_w_0", "t_w_1", "t_w_2",
    ]
    N_S_INCOMING = ["n_t_0", "n_t_1", "n_t_2", "s_t_0", "s_t_1", "s_t_2"]
    W_E_INCOMING = ["w_t_0", "w_t_1", "w_t_2", "e_t_0", "e_t_1", "e_t_2"]

    def __init__(self):
        self.throughput_per_direction = {"N_S": 0, "W_E": 0}
        self.throughput_N_S: dict = {}
        self.throughput_W_E: dict = {}
        self.total_throughput: int = 0
        self.number_produced_vehicles = {"N_S": 0, "W_E": 0}

        self.generated_vehicle_ids_N_S: set = set()
        self.generated_vehicle_ids_W_E: set = set()

        # Per-step history lists
        self.list_average_speed_per_direction = {"N_S": [], "W_E": []}
        self.list_overall_average_speed: list = []
        self.list_average_waiting_time_per_direction = {"N_S": [], "W_E": []}
        self.list_overall_average_waiting_time: list = []
        self.list_sum_waiting_time_per_direction = {"N_S": [], "W_E": []}
        self.list_sum_overall_waiting_time: list = []
        self.list_stopped_times_per_direction = {"N_S": [], "W_E": []}
        self.list_total_stopped_times: list = []

    def collect_step(self, sumo) -> None:
        """Collect metrics for the current simulation step."""
        sum_wait = {"N_S": 0.0, "W_E": 0.0}
        sum_speed = {"N_S": 0.0, "W_E": 0.0}
        veh_count = {"N_S": 0, "W_E": 0}
        stopped = {"N_S": 0.0, "W_E": 0.0}
        lanes_vehicles: list = []

        # N-S lanes
        for lane in self.N_S_LANES:
            vehs = sumo.lane.getLastStepVehicleIDs(lane)
            veh_count["N_S"] += len(vehs)
            for v in vehs:
                lanes_vehicles.append(v)
                sum_wait["N_S"] += sumo.vehicle.getWaitingTime(v)
                sum_speed["N_S"] += sumo.vehicle.getSpeed(v)
            stopped["N_S"] += sumo.lane.getLastStepHaltingNumber(lane)

        # W-E lanes
        for lane in self.W_E_LANES:
            vehs = sumo.lane.getLastStepVehicleIDs(lane)
            veh_count["W_E"] += len(vehs)
            for v in vehs:
                lanes_vehicles.append(v)
                sum_wait["W_E"] += sumo.vehicle.getWaitingTime(v)
                sum_speed["W_E"] += sumo.vehicle.getSpeed(v)
            stopped["W_E"] += sumo.lane.getLastStepHaltingNumber(lane)

        # Intersection vehicles (all - lane vehicles)
        all_ids = sumo.vehicle.getIDList()
        for v in set(all_ids) - set(lanes_vehicles):
            route = sumo.vehicle.getRoute(v)
            if route and len(route) > 0:
                direction = route[0]
                if direction in ("n_t", "s_t"):
                    sum_speed["N_S"] += sumo.vehicle.getSpeed(v)
                    veh_count["N_S"] += 1
                    self.throughput_N_S[v] = 1
                elif direction in ("w_t", "e_t"):
                    sum_speed["W_E"] += sumo.vehicle.getSpeed(v)
                    veh_count["W_E"] += 1
                    self.throughput_W_E[v] = 1

        # Track generated vehicles
        for lane in self.N_S_INCOMING:
            self.generated_vehicle_ids_N_S.update(sumo.lane.getLastStepVehicleIDs(lane))
        for lane in self.W_E_INCOMING:
            self.generated_vehicle_ids_W_E.update(sumo.lane.getLastStepVehicleIDs(lane))

        self.number_produced_vehicles["N_S"] = len(self.generated_vehicle_ids_N_S)
        self.number_produced_vehicles["W_E"] = len(self.generated_vehicle_ids_W_E)
        self.throughput_per_direction["N_S"] = len(self.throughput_N_S)
        self.throughput_per_direction["W_E"] = len(self.throughput_W_E)
        self.total_throughput = (
            self.throughput_per_direction["N_S"] + self.throughput_per_direction["W_E"]
        )

        # Aggregate
        avg_wait = {
            d: (sum_wait[d] / veh_count[d]) if veh_count[d] > 0 else 0.0
            for d in ("N_S", "W_E")
        }
        avg_speed = {
            d: (sum_speed[d] / veh_count[d]) if veh_count[d] > 0 else 0.0
            for d in ("N_S", "W_E")
        }
        total_count = veh_count["N_S"] + veh_count["W_E"]
        total_stopped = stopped["N_S"] + stopped["W_E"]
        overall_avg_speed = (
            (sum_speed["N_S"] + sum_speed["W_E"]) / total_count if total_count > 0 else 0.0
        )
        sum_overall_wait = sum_wait["N_S"] + sum_wait["W_E"]
        overall_avg_wait = sum_overall_wait / total_count if total_count > 0 else 0.0

        # Append to history
        self.list_average_speed_per_direction["N_S"].append(avg_speed["N_S"])
        self.list_average_speed_per_direction["W_E"].append(avg_speed["W_E"])
        self.list_overall_average_speed.append(overall_avg_speed)
        self.list_average_waiting_time_per_direction["N_S"].append(avg_wait["N_S"])
        self.list_average_waiting_time_per_direction["W_E"].append(avg_wait["W_E"])
        self.list_overall_average_waiting_time.append(overall_avg_wait)
        self.list_sum_waiting_time_per_direction["N_S"].append(sum_wait["N_S"])
        self.list_sum_waiting_time_per_direction["W_E"].append(sum_wait["W_E"])
        self.list_sum_overall_waiting_time.append(sum_overall_wait)
        self.list_stopped_times_per_direction["N_S"].append(stopped["N_S"])
        self.list_stopped_times_per_direction["W_E"].append(stopped["W_E"])
        self.list_total_stopped_times.append(total_stopped)


class CTBMetricsCollector:
    """Collects per-vehicle-type (Car / Truck / Bus) metrics.

    Hardcoded for the 1x1 syc intersection layout.
    Call collect_step(sumo) every simulation step during evaluation.

    Attributes:
        CTB_Metrics:   latest step snapshot  {per_type, overall}
        step_history:  per-step lists for all metrics
    """

    LANES = [
        "n_t_0", "n_t_1", "n_t_2", "t_s_0", "t_s_1", "t_s_2",
        "s_t_0", "s_t_1", "s_t_2", "t_n_0", "t_n_1", "t_n_2",
        "w_t_0", "w_t_1", "w_t_2", "t_e_0", "t_e_1", "t_e_2",
        "e_t_0", "e_t_1", "e_t_2", "t_w_0", "t_w_1", "t_w_2",
    ]
    TYPES = ("car", "truck", "bus", "ambulance")

    def __init__(self):
        self.total_generated_vehicle_ids: dict[str, set] = {t: set() for t in self.TYPES}
        self.total_throughput_per_type: dict[str, set] = {t: set() for t in self.TYPES}
        self.total_number_produced_vehicles: dict[str, int] = {t: 0 for t in self.TYPES}

        # Latest step snapshot
        self.CTB_Metrics: dict = {}

        # Full per-step history
        self.step_history: dict = {
            "per_type": {
                t: {
                    "count": [], "sum_wait": [], "sum_speed": [],
                    "avg_wait": [], "avg_speed": [],
                    "stopped": [], "produced": [], "throughput": [],
                }
                for t in self.TYPES
            },
            "overall": {
                "total_count": [], "avg_wait": [], "avg_speed": [],
                "total_stopped": [], "total_produced": [], "total_throughput": [],
            },
        }

    def collect_step(self, sumo) -> None:
        """Collect metrics for the current simulation step."""
        sum_wait: dict[str, float] = {t: 0.0 for t in self.TYPES}
        sum_speed: dict[str, float] = {t: 0.0 for t in self.TYPES}
        count_veh: dict[str, int] = {t: 0 for t in self.TYPES}
        sum_stopped: dict[str, int] = {t: 0 for t in self.TYPES}
        lane_ids: set = set()

        # Lane vehicles
        for lane in self.LANES:
            vehs = sumo.lane.getLastStepVehicleIDs(lane)
            lane_ids.update(vehs)
            for vid in vehs:
                t = sumo.vehicle.getTypeID(vid)
                if t not in self.TYPES:
                    continue
                sum_wait[t] += sumo.vehicle.getAccumulatedWaitingTime(vid)
                sum_speed[t] += sumo.vehicle.getSpeed(vid)
                count_veh[t] += 1
                if sumo.vehicle.getSpeed(vid) < 0.1:
                    sum_stopped[t] += 1
                self.total_generated_vehicle_ids[t].add(vid)

        # Intersection vehicles
        for vid in set(sumo.vehicle.getIDList()) - lane_ids:
            t = sumo.vehicle.getTypeID(vid)
            if t not in self.TYPES:
                continue
            sum_speed[t] += sumo.vehicle.getSpeed(vid)
            sum_wait[t] += sumo.vehicle.getAccumulatedWaitingTime(vid)
            count_veh[t] += 1
            if sumo.vehicle.getSpeed(vid) < 0.1:
                sum_stopped[t] += 1
            self.total_throughput_per_type[t].add(vid)
            self.total_generated_vehicle_ids[t].add(vid)

        # Update total produced counts
        for t in self.TYPES:
            self.total_number_produced_vehicles[t] = len(self.total_generated_vehicle_ids[t])

        # Build per_type and overall snapshots
        throughput = {t: len(self.total_throughput_per_type[t]) for t in self.TYPES}
        per_type: dict = {}
        for t in self.TYPES:
            c = count_veh[t]
            per_type[t] = {
                "count":      c,
                "sum_wait":   sum_wait[t],
                "sum_speed":  sum_speed[t],
                "avg_wait":   (sum_wait[t]  / c) if c else 0.0,
                "avg_speed":  (sum_speed[t] / c) if c else 0.0,
                "stopped":    sum_stopped[t],
                "produced":   self.total_number_produced_vehicles[t],
                "throughput": throughput[t],
            }

        total_count = sum(count_veh.values())
        total_wait  = sum(sum_wait.values())
        total_speed = sum(sum_speed.values())
        overall: dict = {
            "total_count":      total_count,
            "avg_wait":         (total_wait  / total_count) if total_count else 0.0,
            "avg_speed":        (total_speed / total_count) if total_count else 0.0,
            "total_stopped":    sum(sum_stopped.values()),
            "total_produced":   sum(self.total_number_produced_vehicles.values()),
            "total_throughput": sum(throughput.values()),
        }

        # Append to history
        for t, stats in per_type.items():
            buf = self.step_history["per_type"][t]
            buf["count"].append(     stats["count"])
            buf["sum_wait"].append(  stats["sum_wait"])
            buf["sum_speed"].append( stats["sum_speed"])
            buf["avg_wait"].append(  stats["avg_wait"])
            buf["avg_speed"].append( stats["avg_speed"])
            buf["stopped"].append(   stats["stopped"])
            buf["produced"].append(  stats["produced"])
            buf["throughput"].append(stats["throughput"])

        ob = self.step_history["overall"]
        ob["total_count"].append(     overall["total_count"])
        ob["avg_wait"].append(        overall["avg_wait"])
        ob["avg_speed"].append(       overall["avg_speed"])
        ob["total_stopped"].append(   overall["total_stopped"])
        ob["total_produced"].append(  overall["total_produced"])
        ob["total_throughput"].append(overall["total_throughput"])

        self.CTB_Metrics = {"per_type": per_type, "overall": overall}


# ---------------------------------------------------------------------------
# Accurate episode-level metrics collector
# ---------------------------------------------------------------------------

class EpisodeMetricsCollector:
    """Accurate per-episode evaluation metrics.

    Implements the unbiased methods described in the paper:

    - **avg_speed**   : time-weighted  sum(v_it) / sum(N_t)
    - **avg_wait**    : per-vehicle accumulated waiting time at departure/end,
                        averaged over all vehicles
    - **n_stop_events**: moving->stopped transitions (not per-step halting count)
    - **throughput**  : vehicles that completed their route (arrived at dest.)

    Scopes available after summary():
        "system"        -- all intersections, all vehicle types
        "<vtype>"       -- system-level but filtered to one vehicle type
        "<ts_id>"       -- vehicles observed on that intersection's approach lanes
        "<ts_id>_<vtype>" -- intersection x vehicle-type

    Usage
    -----
        ts_lane_map = {ts_id: env.traffic_signals[ts_id].lanes
                       for ts_id in env.ts_ids}
        mc = EpisodeMetricsCollector(ts_lane_map, delta_time=env.delta_time)

        while not done["__all__"]:
            mc.collect_step(env.sumo)          # BEFORE env.step()
            obs, rew, done, info = env.step(actions)

        mc.finalize(env.sumo)                  # capture remaining vehicles
        results   = mc.summary()               # nested dict
        flat      = mc.to_flat_dict()          # for wandb.log / csv
    """

    STOP_THRESHOLD = 0.1   # m/s — below this speed counts as "stopped"

    def __init__(
        self,
        ts_lane_map: dict,
        vtypes: tuple = ("car", "truck", "bus", "ambulance"),
        delta_time: int = 5,
        excluded_lanes: set = None,
    ):
        """
        Args:
            ts_lane_map    : {ts_id: [incoming_lane_id, ...]}
                             Should contain only signal-controlled lanes (lanes that
                             ever get red). Build as:
                                 {ts: env.traffic_signals[ts].signal_controlled_lanes
                                  for ts in env.ts_ids}
            vtypes         : vehicle types to report separately
            delta_time     : seconds per simulation step (for stopped-time calc)
            excluded_lanes : set of lane IDs that are always green (free right turns).
                             Vehicles seen exclusively on these lanes are excluded from
                             system-level wait/stop metrics so they don't bias averages.
                             If None, no exclusion is applied.
        """
        self.ts_lane_map    = ts_lane_map
        self.vtypes         = vtypes
        self.delta_time     = delta_time
        self._excluded_lanes: set = set(excluded_lanes) if excluded_lanes else set()
        # M2 observability: per-vehicle TraCI queries skipped because the vehicle
        # vanished (teleport/removal). Skips keep prior semantics (record kept,
        # that step's contribution dropped) but are now counted, not silent.
        self.query_skips: int = 0

        # lane_id -> ts_id  (fast lookup)
        self._lane_to_ts: dict = {
            lane: ts for ts, lanes in ts_lane_map.items() for lane in lanes
        }

        # vid -> live record (cleared when vehicle leaves)
        self._active: dict = {}

        # vid -> final record (populated when vehicle leaves or finalize() called)
        self._finalized: dict = {}

        # ── time-weighted speed accumulators ──────────────────────────────────
        # keyed by scope string: "system", vtype, ts_id, f"{ts_id}_{vtype}"
        self._spd: dict = self._zero_speed_accum()

        # Parallel accumulator used only when excluded_lanes is active:
        # counts speed/steps only for vehicles observed on signal-controlled lanes,
        # so system avg_speed stays consistent with the filtered n_vehicles.
        self._spd_ctrl: dict = self._zero_speed_accum() if self._excluded_lanes else {}

        # ── per-ts stopped-time (seconds) accumulator ─────────────────────────
        # Only counts time vehicles spent stopped on that ts's approach lanes
        self._ts_stopped_sec: dict = {ts: 0.0 for ts in ts_lane_map}
        for ts in ts_lane_map:
            for t in vtypes:
                self._ts_stopped_sec[f"{ts}_{t}"] = 0.0

        # ── per-ts local accumulated-waiting-time (seconds) accumulator ───────
        # Increments of sumo.vehicle.getAccumulatedWaitingTime are attributed to
        # whichever ts approach lane the vehicle is on at that step. Unlike
        # `last_acc_wait` (whole-trip wait), this lets per-ts.avg_wait reflect
        # *only* wait that happened on that ts (no downstream contamination).
        self._ts_local_wait_sec: dict = {ts: 0.0 for ts in ts_lane_map}
        for ts in ts_lane_map:
            for t in vtypes:
                self._ts_local_wait_sec[f"{ts}_{t}"] = 0.0

        # ── per-ts stop-event counter ─────────────────────────────────────────
        self._ts_stop_evt: dict = {ts: 0 for ts in ts_lane_map}
        for ts in ts_lane_map:
            for t in vtypes:
                self._ts_stop_evt[f"{ts}_{t}"] = 0

        # ── per-ts throughput (set of vids that visited approach lanes) ────────
        self._ts_seen: dict = {ts: set() for ts in ts_lane_map}
        for ts in ts_lane_map:
            for t in vtypes:
                self._ts_seen[f"{ts}_{t}"] = set()

        # ── per-ts cleared (vehicles that actually passed through the intersection)
        # A vehicle is "cleared" when it was previously on a ts's approach lanes
        # but is no longer there — meaning it moved through the intersection.
        # This is the correct throughput denominator (vs _ts_seen which includes
        # vehicles still queuing at episode end).
        self._ts_cleared: dict = {ts: set() for ts in ts_lane_map}
        for ts in ts_lane_map:
            for t in vtypes:
                self._ts_cleared[f"{ts}_{t}"] = set()

    # ── internal helpers ─────────────────────────────────────────────────────

    def _zero_speed_accum(self) -> dict:
        """Return a fresh speed-accumulator dict (sum, count) per scope."""
        scopes = ["system"] + list(self.vtypes)
        for ts in self.ts_lane_map:
            scopes.append(ts)
            for t in self.vtypes:
                scopes.append(f"{ts}_{t}")
        return {s: [0.0, 0] for s in scopes}

    def _new_record(self, vtype: str) -> dict:
        return {
            "type":              vtype if vtype in self.vtypes else "other",
            "prev_speed":        None,   # for system-level stop detection
            "stop_count":        0,      # system-level moving->stopped transitions
            "last_acc_wait":     0.0,    # accumulated waiting time (last observed)
            "completed":         False,  # True if vehicle arrived at destination
            "ever_on_controlled": False, # True once seen on a signal-controlled lane
            # per-ts (updated only when on a ts's approach lanes)
            "ts_prev_speed":      {},  # ts_id -> prev speed (for ts-level stop detection)
            "ts_visited":         set(),  # set of ts_ids this vehicle has been on (k_v for Metric A)
            "per_ts_stopped_sec": {},  # ts_id -> seconds this vehicle was stopped on ts's approach lanes (Metric A)
            "per_ts_stop_events": {},  # ts_id -> stop-event count this vehicle had on ts's approach lanes (Metric A)
            "per_ts_local_wait_sec": {},  # ts_id -> seconds of local accumulated wait increment on ts's approach lanes
        }

    # ── main API ─────────────────────────────────────────────────────────────

    def collect_step(self, sumo) -> None:
        """Collect metrics for the current step.

        Call this BEFORE calling sumo.simulationStep() / env.step().
        """
        current_ids: set = set(sumo.vehicle.getIDList())

        # Build per-ts vehicle sets (vehicles on each ts's approach lanes)
        ts_vehs: dict = {}
        for ts_id, lanes in self.ts_lane_map.items():
            vehs: set = set()
            for lane in lanes:
                vehs.update(sumo.lane.getLastStepVehicleIDs(lane))
            ts_vehs[ts_id] = vehs

        # Build vehicle -> current_ts lookup (a vehicle can only be on one ts's
        # approach lanes at a time; last write wins if somehow on two)
        veh_ts: dict = {}
        for ts_id, vehs in ts_vehs.items():
            for vid in vehs:
                veh_ts[vid] = ts_id

        # Register newly appeared vehicles
        for vid in current_ids - set(self._active):
            vtype = sumo.vehicle.getTypeID(vid)
            self._active[vid] = self._new_record(vtype)

        # Update all active vehicles
        for vid in list(current_ids):
            if vid not in self._active:
                continue
            rec = self._active[vid]
            try:
                speed            = sumo.vehicle.getSpeed(vid)
                current_acc_wait = sumo.vehicle.getAccumulatedWaitingTime(vid)
            except Exception:
                self.query_skips += 1
                continue
            vtype = rec["type"]
            cur_ts = veh_ts.get(vid)     # None if not on any ts's approach lanes

            # Δ accumulated-waiting in the last delta_time, attributed to cur_ts.
            # Clamped to [0, delta_time]:
            #   - negative means SUMO's 100s memory window dropped old wait (rare here)
            #   - >delta_time shouldn't happen, but cap as a safety net.
            delta_wait = current_acc_wait - rec["last_acc_wait"]
            if delta_wait < 0.0:
                delta_wait = 0.0
            elif delta_wait > self.delta_time:
                delta_wait = float(self.delta_time)

            # ── time-weighted speed (system + per-type, all vehicles) ────────────
            # Used by _build_system_summary when excluded_lanes is not set.
            # When excluded_lanes IS set, _spd_ctrl (populated below) is used instead.
            self._spd["system"][0] += speed
            self._spd["system"][1] += 1
            if vtype in self.vtypes:
                self._spd[vtype][0] += speed
                self._spd[vtype][1] += 1

            # ── per-ts metrics (only when on approach lanes) ──────────────────
            if cur_ts is not None:
                # Vehicle reached a signal-controlled lane — count it in system metrics
                rec["ever_on_controlled"] = True
                # Record that this vehicle visited this ts (k_v for Metric A,
                # independent of whether it stopped — stopped vehicles AND moving
                # vehicles both count as "visited 1 intersection here").
                rec["ts_visited"].add(cur_ts)

                # system-level stop transition — counted only on signal-controlled
                # approach lanes so that system n_stop_events is consistent with
                # per-ts n_stop_events and excludes spillback stops on road segments.
                # prev_speed is still updated every step (below) so the transition
                # detector stays continuous across in/out of approach lanes.
                if rec["prev_speed"] is not None:
                    if rec["prev_speed"] >= self.STOP_THRESHOLD > speed:
                        rec["stop_count"] += 1
                # Mark as seen at this ts (for throughput counting)
                self._ts_seen[cur_ts].add(vid)
                if vtype in self.vtypes:
                    self._ts_seen[f"{cur_ts}_{vtype}"].add(vid)

                # Time-weighted speed for this ts
                self._spd[cur_ts][0] += speed
                self._spd[cur_ts][1] += 1
                if vtype in self.vtypes:
                    self._spd[f"{cur_ts}_{vtype}"][0] += speed
                    self._spd[f"{cur_ts}_{vtype}"][1] += 1

                # Controlled-only speed accumulator (for consistent avg_speed when
                # right-turn vehicles are excluded from system summary).
                if self._excluded_lanes:
                    self._spd_ctrl["system"][0] += speed
                    self._spd_ctrl["system"][1] += 1
                    if vtype in self.vtypes:
                        self._spd_ctrl[vtype][0] += speed
                        self._spd_ctrl[vtype][1] += 1

                # Stopped time on this ts's approach lanes
                if speed < self.STOP_THRESHOLD:
                    self._ts_stopped_sec[cur_ts] += self.delta_time
                    if vtype in self.vtypes:
                        self._ts_stopped_sec[f"{cur_ts}_{vtype}"] += self.delta_time
                    # per-vehicle aggregation (for Metric A: per-vehicle normalized avg)
                    rec["per_ts_stopped_sec"][cur_ts] = (
                        rec["per_ts_stopped_sec"].get(cur_ts, 0.0) + self.delta_time
                    )

                # Local accumulated-waiting-time increment on this ts's approach lanes.
                # Δ of SUMO's getAccumulatedWaitingTime since last observed step is
                # attributed entirely to cur_ts (the ts the vehicle is on right now).
                # This gives a per-ts wait metric that EXCLUDES wait incurred at
                # other intersections / downstream lanes — solves the "avg_wait gets
                # contaminated by downstream queues" problem of `last_acc_wait`.
                if delta_wait > 0.0:
                    self._ts_local_wait_sec[cur_ts] += delta_wait
                    if vtype in self.vtypes:
                        self._ts_local_wait_sec[f"{cur_ts}_{vtype}"] += delta_wait
                    rec["per_ts_local_wait_sec"][cur_ts] = (
                        rec["per_ts_local_wait_sec"].get(cur_ts, 0.0) + delta_wait
                    )

                # Stop transition on this ts's approach lanes.
                # Two cases:
                #   1. Normal: vehicle was already on this ts's lane last step
                #      → use ts_prev_speed for the transition check.
                #   2. First entry: vehicle just arrived on this lane this step
                #      → ts_prev_speed is None, but global prev_speed carries
                #        the speed from the previous step (possibly off-lane).
                #        Use it so "entered lane already stopped" is counted.
                prev_ts_spd = rec["ts_prev_speed"].get(cur_ts)
                is_first_entry = prev_ts_spd is None
                if is_first_entry:
                    # Compensate for the "arrived already stopped" case using
                    # global prev_speed, keeping per-ts consistent with system.
                    if rec["prev_speed"] is not None:
                        if rec["prev_speed"] >= self.STOP_THRESHOLD > speed:
                            self._ts_stop_evt[cur_ts] += 1
                            if vtype in self.vtypes:
                                self._ts_stop_evt[f"{cur_ts}_{vtype}"] += 1
                            # per-vehicle aggregation (for Metric A)
                            rec["per_ts_stop_events"][cur_ts] = (
                                rec["per_ts_stop_events"].get(cur_ts, 0) + 1
                            )
                else:
                    if prev_ts_spd >= self.STOP_THRESHOLD > speed:
                        self._ts_stop_evt[cur_ts] += 1
                        if vtype in self.vtypes:
                            self._ts_stop_evt[f"{cur_ts}_{vtype}"] += 1
                        # per-vehicle aggregation (for Metric A)
                        rec["per_ts_stop_events"][cur_ts] = (
                            rec["per_ts_stop_events"].get(cur_ts, 0) + 1
                        )
                rec["ts_prev_speed"][cur_ts] = speed

            # ── always update prev_speed and last_acc_wait (every vehicle, every step)
            # prev_speed must be updated unconditionally so the stop-transition detector
            # in the cur_ts block above has a valid previous speed even after the vehicle
            # has been off a controlled lane for several steps.
            rec["prev_speed"]    = speed
            rec["last_acc_wait"] = current_acc_wait

        # ── Track intersection clearance ──────────────────────────────────────
        # A vehicle "clears" a ts when it was previously on its approach lanes
        # but is no longer there this step (moved through or departed the network).
        # Vehicles departing this step are still in _active (finalized below).
        for ts_id in self.ts_lane_map:
            new_cleared = self._ts_seen[ts_id] - ts_vehs[ts_id] - self._ts_cleared[ts_id]
            for vid in new_cleared:
                rec_lookup = self._active.get(vid) or self._finalized.get(vid)
                if rec_lookup is None:
                    continue
                self._ts_cleared[ts_id].add(vid)
                vtype = rec_lookup["type"]
                if vtype in self.vtypes:
                    self._ts_cleared[f"{ts_id}_{vtype}"].add(vid)

        # Finalize vehicles that left the network this step
        for vid in set(self._active) - current_ids:
            rec = self._active.pop(vid)
            rec["completed"] = True
            self._finalized[vid] = rec

    def finalize(self, sumo=None) -> None:
        """Finalize vehicles still in the network at episode end.

        Args:
            sumo: if provided, captures the latest accumulated waiting time
                  for still-active vehicles before finalizing them.
        """
        if sumo is not None:
            # Guard with the live ID set: vehicles that left the network after the
            # last collect_step would make libsumo PRINT a C++-side error before the
            # exception even reaches us ("Vehicle 'x' is not known" spam at episode
            # end). Skipping them is semantics-identical to the old except-pass
            # (last_acc_wait keeps its previous value).
            current_ids = set(sumo.vehicle.getIDList())
            for vid in list(self._active):
                if vid not in current_ids:
                    continue
                try:
                    self._active[vid]["last_acc_wait"] = (
                        sumo.vehicle.getAccumulatedWaitingTime(vid)
                    )
                except Exception:
                    self.query_skips += 1
        if self.query_skips:
            print(f"[metrics] {self.query_skips} per-vehicle TraCI queries skipped "
                  f"this episode (vanished vehicles — see PROJECT_AUDIT M2)")
        for vid, rec in self._active.items():
            rec["completed"] = False
            self._finalized[vid] = rec
        self._active.clear()

    def summary(self) -> dict:
        """Return a nested summary dict.

        Structure::

            {
                "system":   {"all": {metrics}, "car": {metrics}, ...},
                "<ts_id>":  {"all": {metrics}, "car": {metrics}, ...},
                ...
            }

        Each leaf ``{metrics}`` dict contains:
            avg_speed, avg_wait, n_stop_events, throughput, n_vehicles
        """
        out: dict = {}
        out["system"] = self._build_system_summary()
        for ts_id in self.ts_lane_map:
            out[ts_id] = self._build_ts_summary(ts_id)
        return out

    def to_flat_dict(self, prefix: str = "eval") -> dict:
        """Return a flat dict suitable for wandb.log() or CSV.

        Keys follow the pattern:
            ``{prefix}/{scope}/{vtype}/{metric}``
        e.g. ``eval/system/all/avg_speed``,  ``eval/J1/bus/n_stop_events``
        """
        flat: dict = {}
        for scope, type_stats in self.summary().items():
            for vtype, metrics in type_stats.items():
                for metric, value in metrics.items():
                    flat[f"{prefix}/{scope}/{vtype}/{metric}"] = value
        return flat

    # ── internal summary builders ─────────────────────────────────────────────

    def _leaf_metrics_spd(self, spd_table: dict, spd_key: str, records: list) -> dict:
        """Build the leaf metrics dict for a given scope, using an explicit speed table.

        Note: throughput is NOT included here — system-level throughput (mean of per-ts
        cleared counts) and completed_trips are added by _build_system_summary directly.
        """
        import numpy as _np
        s = spd_table[spd_key]
        n = len(records)
        stop_total = sum(r["stop_count"] for r in records)
        waits = [r["last_acc_wait"] for r in records]
        return {
            "n_vehicles":      n,
            "avg_speed":       s[0] / s[1] if s[1] else 0.0,
            "avg_wait":        sum(waits) / n if n else 0.0,
            # Robust wait distribution (avg_wait is sensitive to single stuck vehicles)
            "median_wait":     float(_np.median(waits)) if waits else 0.0,
            "p95_wait":        float(_np.percentile(waits, 95)) if waits else 0.0,
            "max_wait":        float(max(waits)) if waits else 0.0,
            "n_stop_events":   stop_total,
            "avg_stop_events": stop_total / n if n else 0.0,
        }

    def _per_visit_mean(self, records: list, field: str) -> float:
        """Metric A: per-vehicle normalized average across visited intersections.

        For each vehicle: (sum of metric across visited ts) / (# ts visited).
        Then averaged across vehicles. Each vehicle gets equal weight regardless
        of route length — complements the existing avg_stop_events_per_all_visit
        which is a global ratio weighted by route length.

        IMPORTANT: k_v is the number of intersections the vehicle was **observed
        on** (via `ts_visited`), not the number where it stopped. A vehicle that
        drove through 1 intersection without stopping has k_v=1, s_v=0, so it
        contributes 0 to the average — which keeps the denominator honest and
        avoids the bias of "averaging only over vehicles that stopped".

        Vehicles that never reached any approach lane (k_v=0) are skipped,
        consistent with the existing system-level filtering.

        Args:
            records: finalized vehicle records of the target vType.
            field:   record key holding {ts_id: value}, one of
                     "per_ts_stopped_sec", "per_ts_stop_events", or
                     "per_ts_local_wait_sec".
        """
        vals = []
        for r in records:
            k_v = len(r.get("ts_visited", set()))
            if k_v == 0:
                continue
            s_v = sum(r.get(field, {}).values())
            vals.append(s_v / k_v)
        return sum(vals) / len(vals) if vals else float('nan')

    def _xts_mean(self, vtype: str, metric_name: str) -> float:
        """Metric B: cross-intersection mean, skipping ts where vtype is absent.

        Averages the per-ts metric values across all intersections where
        n_vehicles_of_vtype > 0 (skip-empty to avoid 0-bias).

        Each intersection gets equal weight — complements Metric A's
        per-vehicle view.

        Args:
            vtype:       "all" or one of self.vtypes.
            metric_name: one of "avg_stopped_time", "avg_stop_events",
                         "avg_speed", "avg_waiting_inc".
        """
        vals = []
        for ts_id in self.ts_lane_map:
            key = ts_id if vtype == "all" else f"{ts_id}_{vtype}"
            n = len(self._ts_seen[key])
            if n == 0:
                continue
            if metric_name == "avg_stopped_time":
                v = self._ts_stopped_sec[key] / n
            elif metric_name == "avg_stop_events":
                v = self._ts_stop_evt[key] / n
            elif metric_name == "avg_speed":
                s = self._spd[key]
                v = s[0] / s[1] if s[1] else 0.0
            elif metric_name == "avg_waiting_inc":
                v = self._ts_local_wait_sec[key] / n
            else:
                raise ValueError(f"Unknown metric_name for xts_mean: {metric_name}")
            vals.append(v)
        return sum(vals) / len(vals) if vals else float('nan')

    def _build_system_summary(self) -> dict:
        # When excluded_lanes is active, filter to vehicles that reached at least
        # one signal-controlled lane (right-turn-only vehicles are excluded so they
        # don't bias avg_wait / stop metrics for cars vs buses).
        # avg_speed uses _spd_ctrl so its denominator is consistent with n_vehicles.
        if self._excluded_lanes:
            all_recs  = [r for r in self._finalized.values() if r["ever_on_controlled"]]
            spd_table = self._spd_ctrl
        else:
            all_recs  = list(self._finalized.values())
            spd_table = self._spd
        out: dict = {}

        n_ts = len(self.ts_lane_map)

        out["all"] = self._leaf_metrics_spd(spd_table, "system", all_recs)
        total_all = sum(self._ts_stopped_sec.get(ts, 0.0) for ts in self.ts_lane_map)
        total_local_wait_all = sum(self._ts_local_wait_sec.get(ts, 0.0) for ts in self.ts_lane_map)
        n_all = out["all"]["n_vehicles"]
        out["all"]["stopped_time"]     = total_all
        out["all"]["avg_stopped_time"] = total_all / n_all if n_all else 0.0
        # Local accumulated-waiting-time aggregates (approach-lane only, no downstream contamination)
        out["all"]["waiting_inc"]      = total_local_wait_all
        out["all"]["avg_waiting_inc"]  = total_local_wait_all / n_all if n_all else 0.0
        # System throughput: sum and mean of per-ts cleared counts
        ts_cleared_counts = [len(self._ts_cleared[ts]) for ts in self.ts_lane_map]
        out["all"]["throughput"]      = sum(ts_cleared_counts)
        out["all"]["mean_throughput"] = sum(ts_cleared_counts) / n_ts if n_ts else 0.0
        # Completed trips = vehicles that finished their entire route and left the network
        out["all"]["completed_trips"]  = sum(1 for r in all_recs if r["completed"])
        out["all"]["completion_rate"]  = out["all"]["completed_trips"] / n_all if n_all else 0.0
        # Stop events normalised by total intersection visits (removes route-length bias)
        total_visits_all = sum(len(self._ts_seen[ts]) for ts in self.ts_lane_map)
        out["all"]["avg_stop_events_per_all_visit"] = (
            out["all"]["n_stop_events"] / total_visits_all if total_visits_all else 0.0
        )
        # Metric A — per-vehicle normalized averages (each vehicle 1 vote, skip k_v=0)
        out["all"]["avg_stopped_time_per_visit"] = self._per_visit_mean(all_recs, "per_ts_stopped_sec")
        out["all"]["avg_stop_events_per_visit"]  = self._per_visit_mean(all_recs, "per_ts_stop_events")
        out["all"]["avg_waiting_inc_per_visit"]  = self._per_visit_mean(all_recs, "per_ts_local_wait_sec")
        # Metric B — cross-ts mean (each intersection 1 vote, skip ts where n_vehicles==0)
        out["all"]["xts_avg_stopped_time"] = self._xts_mean("all", "avg_stopped_time")
        out["all"]["xts_avg_stop_events"]  = self._xts_mean("all", "avg_stop_events")
        out["all"]["xts_avg_speed"]        = self._xts_mean("all", "avg_speed")
        out["all"]["xts_avg_waiting_inc"]  = self._xts_mean("all", "avg_waiting_inc")

        for t in self.vtypes:
            recs = [r for r in all_recs if r["type"] == t]
            leaf = self._leaf_metrics_spd(spd_table, t, recs)
            total_t            = sum(self._ts_stopped_sec.get(f"{ts}_{t}", 0.0) for ts in self.ts_lane_map)
            total_local_wait_t = sum(self._ts_local_wait_sec.get(f"{ts}_{t}", 0.0) for ts in self.ts_lane_map)
            n_t = leaf["n_vehicles"]
            leaf["stopped_time"]     = total_t
            leaf["avg_stopped_time"] = total_t / n_t if n_t else 0.0
            leaf["waiting_inc"]      = total_local_wait_t
            leaf["avg_waiting_inc"]  = total_local_wait_t / n_t if n_t else 0.0
            ts_cleared_t = [len(self._ts_cleared[f"{ts}_{t}"]) for ts in self.ts_lane_map]
            leaf["throughput"]      = sum(ts_cleared_t)
            leaf["mean_throughput"] = sum(ts_cleared_t) / n_ts if n_ts else 0.0
            leaf["completed_trips"] = sum(1 for r in recs if r["completed"])
            leaf["completion_rate"] = leaf["completed_trips"] / n_t if n_t else 0.0
            total_visits_t = sum(len(self._ts_seen[f"{ts}_{t}"]) for ts in self.ts_lane_map)
            leaf["avg_stop_events_per_all_visit"] = (
                leaf["n_stop_events"] / total_visits_t if total_visits_t else 0.0
            )
            # Metric A — per-vehicle normalized averages (each vehicle 1 vote, skip k_v=0)
            leaf["avg_stopped_time_per_visit"] = self._per_visit_mean(recs, "per_ts_stopped_sec")
            leaf["avg_stop_events_per_visit"]  = self._per_visit_mean(recs, "per_ts_stop_events")
            leaf["avg_waiting_inc_per_visit"]  = self._per_visit_mean(recs, "per_ts_local_wait_sec")
            # Metric B — cross-ts mean (each intersection 1 vote, skip ts where n_vehicles==0)
            leaf["xts_avg_stopped_time"] = self._xts_mean(t, "avg_stopped_time")
            leaf["xts_avg_stop_events"]  = self._xts_mean(t, "avg_stop_events")
            leaf["xts_avg_speed"]        = self._xts_mean(t, "avg_speed")
            leaf["xts_avg_waiting_inc"]  = self._xts_mean(t, "avg_waiting_inc")
            out[t] = leaf

        return out

    def _build_ts_summary(self, ts_id: str) -> dict:
        """Per-ts summary.

        - avg_speed     : time-weighted speed of vehicles on ts's approach lanes
        - avg_wait      : mean per-vehicle total accumulated wait of vehicles
                          that visited this ts (proxy; includes wait at other ts)
        - avg_waiting_inc: mean per-vehicle local wait (Δ getAccumulatedWaitingTime
                          attributed only to this ts's approach lanes; no downstream
                          contamination — preferred over avg_wait for fair per-ts comparison)
        - n_stop_events    : stop transitions occurring on ts's approach lanes
        - avg_stop_events  : n_stop_events / n_vehicles
        - stopped_time     : total vehicle-seconds spent stopped on approach lanes
        - avg_stopped_time : stopped_time / n_vehicles
        - throughput    : vehicles that cleared the intersection (moved past approach lanes)
        """
        all_seen_vids = self._ts_seen[ts_id]
        all_recs = [self._finalized[v] for v in all_seen_vids if v in self._finalized]

        def ts_leaf(spd_key: str, records: list, type_key: str) -> dict:
            import numpy as _np
            s = self._spd[spd_key]
            n = len(records)
            waits = [r["last_acc_wait"] for r in records]
            return {
                "n_vehicles":    n,
                "throughput":    len(self._ts_cleared[type_key]),
                "avg_speed":     s[0] / s[1] if s[1] else 0.0,
                "avg_wait":        sum(waits) / n if n else 0.0,
                # Robust wait distribution
                "median_wait":     float(_np.median(waits)) if waits else 0.0,
                "p95_wait":        float(_np.percentile(waits, 95)) if waits else 0.0,
                "max_wait":        float(max(waits)) if waits else 0.0,
                "n_stop_events":   self._ts_stop_evt[type_key],
                "avg_stop_events": self._ts_stop_evt[type_key] / n if n else 0.0,
                "stopped_time":    self._ts_stopped_sec[type_key],
                "avg_stopped_time": self._ts_stopped_sec[type_key] / n if n else 0.0,
                # Local-only wait (approach lanes of this ts ONLY; no downstream contamination)
                "waiting_inc":     self._ts_local_wait_sec[type_key],
                "avg_waiting_inc": self._ts_local_wait_sec[type_key] / n if n else 0.0,
            }

        out: dict = {}
        out["all"] = ts_leaf(ts_id, all_recs, ts_id)
        for t in self.vtypes:
            seen_vids_t = self._ts_seen[f"{ts_id}_{t}"]
            recs_t = [self._finalized[v] for v in seen_vids_t if v in self._finalized]
            out[t] = ts_leaf(f"{ts_id}_{t}", recs_t, f"{ts_id}_{t}")
        return out
