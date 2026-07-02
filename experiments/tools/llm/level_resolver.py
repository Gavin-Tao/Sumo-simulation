"""IR -> runtime rule engine: entry-time classification of one vehicle.

classify(vtype, entry_time, entry_edge, share_draw) -> level (1..5)
First matching assignment wins (validator guarantees a catch-all exists).
Predicate whitelist v1: type / time_window / region / share_lt.
sched_delay_min_gt is phase-2: constructing a resolver over an IR that uses
it raises, matching the validator warning.
"""
from __future__ import annotations


class LevelResolver:
    def __init__(self, ir: dict, regions: dict | None = None):
        self.rules = ir["assignments"]
        self.regions = dict(ir.get("regions") or {})
        if regions:
            self.regions.update(regions)
        for i, r in enumerate(self.rules):
            if "sched_delay_min_gt" in r["match"]:
                raise NotImplementedError(
                    f"rule {i}: sched_delay_min_gt is phase-2 (GTFS sidecar)")
            reg = r["match"].get("region")
            if reg and reg not in self.regions:
                raise KeyError(f"rule {i}: region '{reg}' undefined "
                               f"(supply via ir.regions or regions arg)")
        self._region_sets = {k: set(v) for k, v in self.regions.items()}

    def classify(self, vtype: str, entry_time: float = 0.0,
                 entry_edge: str = "", share_draw: float = 1.0) -> int:
        base = vtype.split("@", 1)[0]          # derived types classify by base
        for r in self.rules:
            m = r["match"]
            if m["type"] != "*" and m["type"] != base:
                continue
            tw = m.get("time_window")
            if tw and not (tw[0] <= entry_time < tw[1]):
                continue
            reg = m.get("region")
            if reg and entry_edge not in self._region_sets[reg]:
                continue
            sh = m.get("share_lt")
            if sh is not None and not (share_draw < sh):
                continue
            return int(r["level"])
        raise RuntimeError("no rule matched — validator should have "
                           "guaranteed a catch-all")

    def base_table(self) -> dict:
        """type -> level from UNCONDITIONAL rules only (the static part;
        conditional deviations are applied by the runtime via retyping)."""
        out = {}
        for r in self.rules:
            m = r["match"]
            if set(m) == {"type"} and m["type"] != "*" and m["type"] not in out:
                out[m["type"]] = int(r["level"])
        return out

    def default_level(self) -> int:
        return int(self.rules[-1]["level"])
