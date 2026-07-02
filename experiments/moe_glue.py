"""Glue between train.py and the MoE-Gcμ scheme (action_scheme: moe).

Import-only from the moe branch — no side effects; old configs never reach
this module. Builds the expert tables from EITHER menu meta schema:
  * enum meta  (enum_phases / enum_phases_net): tls[t]["phase_movements"]
  * 8std meta  (reindex_8std / reindex_8std_net): tls[t]["actions"] states
so the L1 menu (enum vs 8std) is a pure config choice, as decided
2026-07-02 (RESEARCH_ROADMAP §A1: menu-agnostic experts).
"""
import json

import numpy as np

SLOTS = [(a, t) for a in ("N", "E", "S", "W") for t in ("L", "T", "R")]
SLOT_IDX = {s: k for k, s in enumerate(SLOTS)}


def load_moe_tables(meta_path):
    """-> {"tables": {ts: {"phase_slots": [frozenset], "slot_lanes": {slot: [lane]}}},
          "turnmap": {ts: {link_idx: (approach, turn)}}}"""
    meta = json.load(open(meta_path))["tls"]
    tables, turnmap = {}, {}
    for tid, t in meta.items():
        links = {int(i): c for i, c in t["links"].items()}
        turnmap[tid] = {i: (c[0]["approach"], c[0]["turn"])
                        for i, c in links.items()}
        link_slot = {i: SLOT_IDX[(c[0]["approach"], c[0]["turn"])]
                     for i, c in links.items()}
        slot_lanes = {}          # slot -> lanes (for discharge-capacity count)
        lanes = {}               # unique lane_id -> from_edge (single scan set)
        by_edges = {}            # (from_edge, to_edge) -> slot (intent lookup,
        #                          same convention as obs _movement_by_edges)
        for i, conns in links.items():
            s = link_slot[i]
            for c in conns:
                lane = f"{c['from_edge']}_{c['from_lane']}"
                slot_lanes.setdefault(s, [])
                if lane not in slot_lanes[s]:
                    slot_lanes[s].append(lane)
                lanes[lane] = c["from_edge"]
                by_edges[(c["from_edge"], c["to_edge"])] = s
        if "phase_movements" in t:                      # enum meta
            phase_slots = [frozenset(m for m in range(12) if row[m])
                           for row in t["phase_movements"]]
        else:                                            # 8std meta
            # greens in dense order = std actions sorted by their green index
            s2g = {int(a): gi for a, gi in t["std_to_green_index"].items()}
            order = sorted(s2g, key=lambda a: s2g[a])
            phase_slots = []
            for a in order:
                state = t["actions"][f"a{a}"]["state"]
                served = frozenset(link_slot[i] for i in links
                                   if i < len(state) and state[i] == "G")
                phase_slots.append(served)
        assert phase_slots, tid
        tables[tid] = {"phase_slots": phase_slots, "slot_lanes": slot_lanes,
                       "lanes": lanes, "movement_by_edges": by_edges}
    return {"tables": tables, "turnmap": turnmap}


def build_experts(cfg, moe, env):
    from sumo_rl.agents.moe_experts import MoEExperts
    from sumo_rl.environment.priority_map import load_priority_table, DEFAULT_PRIORITY
    prio = load_priority_table(cfg.get("priority_source"))
    return MoEExperts(moe["tables"], delta_time=cfg.get("delta_time", 5),
                      yellow_time=cfg.get("yellow_time", 2),
                      prio_of_type=prio, default_level=int(DEFAULT_PRIORITY))


def gate_mask(levels_present, lexicographic, lex_min=5, n_experts=6):
    """L4: presence-triggered lexicographic mask over the 6 experts.
    lexicographic off -> all-valid (pure learning mode)."""
    m = np.ones(n_experts, dtype=bool)
    if lexicographic:
        top = max((l for l in levels_present if l >= lex_min), default=0)
        if top:
            m[:] = False
            m[top] = True     # expert index k == level k (index 0 = efficiency)
    return m
