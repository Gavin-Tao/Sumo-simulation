"""Runtime enforcement of a compiled policy — the derived-vType retyping
daemon (the architectural pivot: dynamic levels pass through every existing
consumer's STATIC type->level table, so obs/reward/MoE-experts need zero
modification).

Per step: newly departed vehicles are classified (entry-time semantics);
when the assigned level differs from the base type's table level, the
vehicle is retyped to `<base>@l<k>` (derived vType created once via
vehicletype.copy). The extended priority table maps every derived name to
its level, so all downstream lookups resolve naturally.
"""
from __future__ import annotations

import random


class PolicyRuntime:
    def __init__(self, resolver, base_priority_table: dict, seed: int = 0):
        self.res = resolver
        self.base_table = dict(base_priority_table)
        self.rng = random.Random(seed)
        self._made_types = set()
        self.stats = {"seen": 0, "retyped": 0, "by_derived": {}}

    def extended_table(self) -> dict:
        """base table + every possible derived name (type@l1..5)."""
        t = dict(self.base_table)
        for base in list(self.base_table):
            for k in range(1, 6):
                t[f"{base}@l{k}"] = k
        return t

    def reset(self):
        self._made_types.clear()
        self.stats = {"seen": 0, "retyped": 0, "by_derived": {}}

    def step(self, sumo, sim_time: float):
        """Call once per control step, right after env.step()."""
        for vid in sumo.simulation.getDepartedIDList():
            self.stats["seen"] += 1
            vtype = sumo.vehicle.getTypeID(vid)
            base = vtype.split("@", 1)[0]
            try:
                edge = sumo.vehicle.getRoadID(vid)
            except Exception:
                edge = ""
            level = self.res.classify(base, entry_time=sim_time,
                                      entry_edge=edge,
                                      share_draw=self.rng.random())
            base_level = self.base_table.get(base)
            if base_level is not None and level == base_level:
                continue                       # nominal level, no retype
            derived = f"{base}@l{level}"
            if derived not in self._made_types:
                sumo.vehicletype.copy(base, derived)
                self._made_types.add(derived)
            sumo.vehicle.setType(vid, derived)
            self.stats["retyped"] += 1
            self.stats["by_derived"][derived] = \
                self.stats["by_derived"].get(derived, 0) + 1
