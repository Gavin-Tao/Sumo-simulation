"""Enumerate ALL maximal protected (zero-conflict incl. merging) movement
phases per TLS -> dublin_enum.net.xml + dublin_enum_meta.json.

Spec: experiments/analysis/FRAP_ENUM_DESIGN_2026-07-02.txt §1/§2.5.
Conflict(slot a, slot b) = (not same-arm AND any link-pair areFoes)
                           OR any link-pair shares (to_edge, to_lane).
Relation codes (meta movement_rel): -1 nonexistent slot pair, 0 same-arm,
1 compatible, 2 merge (all conflict evidence shares a to_edge), 3 crossing.
Protected-only hard rule (user 2026-07-02): phases are all-'G'; no 'g' ever.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from reindex_8std import are_foes, same_arm, yellow_between  # noqa: E402

SLOTS = [(a, t) for a in ("N", "E", "S", "W") for t in ("L", "T", "R")]
SLOT_IDX = {s: k for k, s in enumerate(SLOTS)}
GREEN_DUR, YELLOW_DUR = "30", "3"
NET_ENUM = os.path.join(common.OUT_DIR, "dublin_enum.net.xml")
META_ENUM = os.path.join(common.OUT_DIR, "dublin_enum_meta.json")


def slot_tables(mov):
    """slot (approach, turn) -> list of tl link indices. Uses conns[0] per
    index — same convention as train.py's ts_turnmap and the obs rebind."""
    st = {}
    for i, conns in mov["links"].items():
        st.setdefault((conns[0]["approach"], conns[0]["turn"]), []).append(i)
    return st


def _lane_share(mov, i, j):
    ti = {(c["to_edge"], c["to_lane"]) for c in mov["links"][i]}
    tj = {(c["to_edge"], c["to_lane"]) for c in mov["links"][j]}
    return bool(ti & tj)


def _edge_share(mov, i, j):
    ti = {c["to_edge"] for c in mov["links"][i]}
    tj = {c["to_edge"] for c in mov["links"][j]}
    return bool(ti & tj)


def intra_slot_conflicts(mov, nodes, st):
    bad = []
    for slot, idxs in st.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if (not same_arm(mov, i, j) and are_foes(mov, nodes, i, j)) \
                        or _lane_share(mov, i, j):
                    bad.append((slot, i, j))
    return bad


def movement_rel(mov, nodes, st):
    rel = [[-1] * 12 for _ in range(12)]
    for sa, ia in st.items():
        ka = SLOT_IDX[sa]
        for sb, ib in st.items():
            kb = SLOT_IDX[sb]
            if ka == kb:
                rel[ka][kb] = 0
                continue
            if sa[0] == sb[0]:                        # same approach arm
                rel[ka][kb] = 0
                continue
            evid = [(i, j) for i in ia for j in ib
                    if are_foes(mov, nodes, i, j) or _lane_share(mov, i, j)]
            if not evid:
                rel[ka][kb] = 1
            else:
                all_merge = all(_edge_share(mov, i, j) for i, j in evid)
                rel[ka][kb] = 2 if all_merge else 3
    return rel


def enumerate_menu(rel, st):
    """All maximal conflict-free slot sets (conflict = rel>=2), sorted by
    12-bit multi-hot tuple for reproducibility. No ordering prior — complete."""
    exist = sorted(SLOT_IDX[s] for s in st)
    res = []

    def grow(cur, cand):
        ext = [c for c in cand if all(rel[c][x] < 2 for x in cur)]
        if not ext:
            if cur:
                res.append(frozenset(cur))
            return
        for k, c in enumerate(ext):
            grow(cur | {c}, ext[k + 1:])
    grow(set(), exist)
    maximal = [s for s in set(res) if not any(s < r for r in res if r != s)]
    key = lambda p: tuple(1 if m in p else 0 for m in range(12))  # noqa: E731
    return sorted(maximal, key=key)


def phase_state(mov, st, members, n_state):
    state = ["r"] * n_state
    inv = {SLOT_IDX[s]: idxs for s, idxs in st.items()}
    for m in members:
        for i in inv[m]:
            state[i] = "G"
    return "".join(state)


def verify_phase(mov, nodes, state):
    """Hard verifier (spec §6 V1): protected-only, zero foe, zero lane-merge."""
    if "g" in state:
        raise RuntimeError("protected-only violated: 'g' present")
    gset = [i for i in sorted(mov["links"]) if state[i] == "G"]
    for a in range(len(gset)):
        for b in range(a + 1, len(gset)):
            i, j = gset[a], gset[b]
            if not same_arm(mov, i, j) and are_foes(mov, nodes, i, j):
                raise RuntimeError(f"foe conflict {i},{j} in {state}")
            if _lane_share(mov, i, j):
                raise RuntimeError(f"merge conflict {i},{j} in {state}")
    for i in gset:                                    # shared-index self merge
        here = set()
        for c in mov["links"][i]:
            key = (c["to_edge"], c["to_lane"])
            if key in here:
                raise RuntimeError(f"shared-index merge {i}->{key}")
            here.add(key)
    return True
