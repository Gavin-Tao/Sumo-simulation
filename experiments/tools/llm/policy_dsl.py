"""Priority-policy DSL (the IR): schema + deterministic validator.

The IR is the safety gate between the LLM compiler and the runtime: small
enough to be checked field-by-field, so any hallucination dies here with an
actionable error message (fed back into the compile retry loop).

IR v1 (YAML/JSON):
  version: 1
  assignments:                # ORDERED rules, first match wins
    - match: {type: "bus", time_window: [0, 1800], region: "hospital_zone",
              share_lt: 0.5, sched_delay_min_gt: 5}   # all keys optional
      level: 4                # 1..5
    - match: {type: "*"}      # catch-all REQUIRED as last rule
      level: 1
  guarantees: {lexicographic_min_level: 5}     # optional (MoE L4 switch)
  kpis:                       # optional, verified by verify_policy.py
    - {metric: "bus_wait_p90", op: "<", value: 40}
    - {metric: "car_wait_mean_degradation", op: "<", value: 0.10}
  regions: {hospital_zone: ["edge1", "edge2"]}  # optional inline defs

Semantics (v1): classification happens ONCE, when a vehicle enters the
network ("entry-time classification") — consistent with the runtime's
derived-vType retyping trick that leaves obs/reward/expert memos untouched.
`sched_delay_min_gt` is declared but phase-2 (validator warns).
"""
from __future__ import annotations

import json

import jsonschema
import yaml

MATCH_KEYS = {"type", "time_window", "region", "share_lt", "sched_delay_min_gt"}
KPI_METRIC_RE = r"^[a-zA-Z_][a-zA-Z0-9_]*_(wait_mean|wait_p90|count|wait_mean_degradation|wait_p90_degradation)$"

SCHEMA = {
    "type": "object",
    "required": ["version", "assignments"],
    "additionalProperties": False,
    "properties": {
        "version": {"const": 1},
        "assignments": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["match", "level"],
                "additionalProperties": False,
                "properties": {
                    "match": {
                        "type": "object",
                        "required": ["type"],
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "minLength": 1},
                            "time_window": {
                                "type": "array", "minItems": 2, "maxItems": 2,
                                "items": {"type": "number", "minimum": 0}},
                            "region": {"type": "string"},
                            "share_lt": {"type": "number",
                                         "exclusiveMinimum": 0, "maximum": 1},
                            "sched_delay_min_gt": {"type": "number", "minimum": 0},
                        },
                    },
                    "level": {"type": "integer", "minimum": 1, "maximum": 5},
                },
            },
        },
        "guarantees": {
            "type": "object", "additionalProperties": False,
            "properties": {"lexicographic_min_level":
                           {"type": "integer", "minimum": 1, "maximum": 5}},
        },
        "kpis": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["metric", "op", "value"],
                "additionalProperties": False,
                "properties": {
                    "metric": {"type": "string", "pattern": KPI_METRIC_RE},
                    "op": {"enum": ["<", "<=", ">", ">="]},
                    "value": {"type": "number"},
                },
            },
        },
        "regions": {
            "type": "object",
            "additionalProperties": {"type": "array",
                                     "items": {"type": "string"}},
        },
    },
}


def validate(ir: dict):
    """-> (ok, errors, warnings). Errors block; warnings inform."""
    errors, warnings = [], []
    try:
        jsonschema.validate(ir, SCHEMA)
    except jsonschema.ValidationError as e:
        return False, [f"schema: {e.message} (at {'/'.join(str(p) for p in e.absolute_path)})"], []

    rules = ir["assignments"]
    # catch-all completeness
    if rules[-1]["match"].get("type") != "*" or \
            set(rules[-1]["match"]) != {"type"}:
        errors.append("last assignment must be the unconditional catch-all "
                      "{match: {type: '*'}, level: k}")
    # unconditional-rule shadowing: any rule after an unconditional match of
    # the same type (or '*') can never fire
    seen_uncond = set()
    for i, r in enumerate(rules):
        m = r["match"]
        t = m["type"]
        if t in seen_uncond or "*" in seen_uncond:
            warnings.append(f"rule {i} is shadowed by an earlier "
                            f"unconditional rule and can never fire")
        if set(m) == {"type"}:
            seen_uncond.add(t)
        if m.get("time_window") and m["time_window"][0] >= m["time_window"][1]:
            errors.append(f"rule {i}: empty time_window {m['time_window']}")
        if m.get("region") and m["region"] not in (ir.get("regions") or {}):
            warnings.append(f"rule {i}: region '{m['region']}' not defined "
                            f"inline — must be supplied at runtime")
        if "sched_delay_min_gt" in m:
            warnings.append(f"rule {i}: sched_delay_min_gt is a PHASE-2 "
                            f"predicate (GTFS sidecar required); the v1 "
                            f"resolver rejects it")
    return (not errors), errors, warnings


def load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) if path.endswith((".yaml", ".yml")) \
            else json.load(f)


def dump(ir: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(ir, f, sort_keys=False, allow_unicode=True)


DSL_SPEC_FOR_PROMPT = __doc__  # the compiler embeds this docstring verbatim


if __name__ == "__main__":
    import sys
    ir = load(sys.argv[1])
    ok, errs, warns = validate(ir)
    for w in warns:
        print("WARN:", w)
    for e in errs:
        print("ERROR:", e)
    print("VALID" if ok else "INVALID")
    sys.exit(0 if ok else 1)
