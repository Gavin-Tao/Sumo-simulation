"""Vehicle priority policy — the single source of truth, compiled once to a flat table.

The agent / φ featurizer never sees vehicle *types*, only priority *buckets* {1..5}.
That indirection is the point: swapping the policy (this table now, a context-aware BNF
policy later) is zero-shot — the observation layout and the network are unchanged.

Design (agreed): the policy lives HERE (or in an external file referenced by config),
NOT hardcoded inside an observation/reward class. It is loaded ONCE at startup into a
plain dict; the per-step hot loop only does an O(1) dict lookup (callers additionally
memoize vid→priority since a vehicle's type is immutable). Never parse a policy file in
the per-vehicle loop.

Usage:
    from .priority_map import load_priority_table, PRIORITY_LEVELS, DEFAULT_PRIORITY
    table = load_priority_table(source)          # source=None → built-in default
    prio  = table.get(vtype, DEFAULT_PRIORITY)
"""
from __future__ import annotations

import json

PRIORITY_LEVELS = (1, 2, 3, 4, 5)   # fixed bucket space (zero-shot invariant)
DEFAULT_PRIORITY = 1                 # fallback for unknown vehicle types

# Built-in default static table. Each type → its own bucket so the agent sees them
# separately; the reward weights (e.g. 5-4-1) are a SEPARATE concern from this mapping.
# (truck is absent from the current scenarios → its bucket simply stays empty.)
DEFAULT_TABLE = {"ambulance": 5, "bus": 3, "truck": 2, "car": 1}


def load_priority_table(source: str | dict | None = None) -> dict:
    """Compile the policy into a flat {vehicle_type: priority} dict (called once).

    source:
        None  → built-in DEFAULT_TABLE (a copy).
        dict  → used directly (validated/copied).
        str   → path to a ``.json`` file holding a {type: priority} object.
                (Phase-2: a ``.bnf`` path will be parsed into a table here — same
                 return type, so callers never change.)
    """
    if source is None:
        table = dict(DEFAULT_TABLE)
    elif isinstance(source, dict):
        table = dict(source)
    elif isinstance(source, str) and source.endswith(".json"):
        with open(source) as f:
            table = json.load(f)
    else:
        raise ValueError(f"unsupported priority source: {source!r} "
                         "(expected None, a dict, or a .json path)")

    out = {}
    for vtype, p in table.items():
        p = int(p)
        if p not in PRIORITY_LEVELS:
            raise ValueError(f"priority {p} for {vtype!r} outside {PRIORITY_LEVELS}")
        out[str(vtype)] = p
    return out
