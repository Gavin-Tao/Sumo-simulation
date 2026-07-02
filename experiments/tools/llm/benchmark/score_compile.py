"""Compilation benchmark scorer — SEMANTIC equivalence, not string match.

Two IRs are equivalent iff they classify identically over a synthetic
context grid (types × entry times × regions × share draws) AND agree on
guarantees. Ambiguity cases score on behavior: predicted output must be a
clarification object (its wording is not graded).

Usage:
  python score_compile.py --pred-dir <dir with <case>.yaml predictions>
  python score_compile.py --self-test          # gold vs gold -> 100%
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import policy_dsl  # noqa: E402
from level_resolver import LevelResolver  # noqa: E402

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold")
TYPES = ["ambulance", "bus", "car", "tram", "truck", "unknown_type"]
TIMES = [0.0, 900.0, 1799.0, 1800.0, 2700.0]
DRAWS = [0.05, 0.25, 0.49, 0.51, 0.95]


def contexts(ir):
    regions = list((ir.get("regions") or {}).keys())
    edges = ["__none__"] + [f"__in_{r}__" for r in regions]
    for t in TYPES:
        for tm in TIMES:
            for e in edges:
                for d in DRAWS:
                    yield t, tm, e, d


def behavior(ir):
    regions = {r: [f"__in_{r}__"] for r in (ir.get("regions") or {})}
    res = LevelResolver(ir, regions=regions)
    return [res.classify(t, tm, e, d) for t, tm, e, d in contexts(ir)], res


def equivalent(pred, gold):
    if "clarification_needed" in gold:
        return "clarification_needed" in pred
    if "clarification_needed" in pred:
        return False
    ok, errs, _ = policy_dsl.validate(pred)
    if not ok:
        return False
    try:
        bg, _ = behavior(gold)
        # evaluate pred over the GOLD's context grid (same regions universe)
        regions = {r: [f"__in_{r}__"] for r in (gold.get("regions") or {})}
        rp = LevelResolver(pred, regions={**regions,
                           **{r: [f"__in_{r}__"] for r in (pred.get("regions") or {})}})
        bp = [rp.classify(t, tm, e, d) for t, tm, e, d in contexts(gold)]
    except Exception as e:
        print("  resolver error:", e)
        return False
    if bg != bp:
        return False
    return (gold.get("guarantees") or {}) == (pred.get("guarantees") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    n = hit = 0
    for gpath in sorted(glob.glob(os.path.join(GOLD, "*.yaml"))):
        case = os.path.basename(gpath)[:-5]
        gold = yaml.safe_load(open(gpath))["ir"]
        if args.self_test:
            pred = gold
        else:
            ppath = os.path.join(args.pred_dir, case + ".yaml")
            if not os.path.exists(ppath):
                print(f"{case}: MISSING prediction")
                n += 1
                continue
            pred = yaml.safe_load(open(ppath))
            pred = pred.get("ir", pred)
        ok = equivalent(pred, gold)
        n += 1
        hit += ok
        print(f"{case}: {'OK' if ok else 'MISMATCH'}")
    print(f"\nsemantic accuracy: {hit}/{n} = {hit / n:.1%}")


if __name__ == "__main__":
    main()
