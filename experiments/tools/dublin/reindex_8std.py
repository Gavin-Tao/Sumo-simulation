"""M0 — reindex every TLS of dublin_8action.net.xml to 8 standard phases.

Standard actions (left-hand traffic, FRAP-style; design Part 2.2):
  a0 NS-straight  a1 EW-straight  a2 NS-protected-right  a3 EW-protected-right
  a4 N-all        a5 S-all        a6 E-all               a7 W-all
Single-arm cores (a4-a7) include ALL movements of the arm (L+T+R) so every
link is guaranteed green in at least one phase (fixes the 5 broken junctions
with never-green links in the source net).

Per phase the state is the maximal compatible closure (design Part 2.3):
  1. core links -> 'G' (a core link that hard-conflicts with an already
     placed core link of another arm is downgraded to 'g' = yield)
  2. closure: any remaining link with no foe among current 'G' links -> 'G'
  3. permissive: remaining crossing right turns whose own arm's straight is
     running -> 'g' (D3)
Conflicts come from the junction <request> foes matrix via sumolib
Node.areFoes (D4) — exact, no geometry heuristics. Links from the same
incoming edge never block each other (diverging/merging same-arm flows).

The source net keeps GAP indices in tl states (stripped pedestrian links);
states are emitted at max(linkIndex)+1 — internal gap chars stay 'r',
TRAILING padding beyond the last real link is dropped (SUMO warns on it).

Outputs: nets/dublin/dublin_8std.net.xml + dublin_8std_meta.json
Runs V1 checks (design Part 7) and exits non-zero on failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GREEN_DUR = "30"
YELLOW_DUR = "3"
# user decision 2026-06-11: PROTECTED-ONLY phases — no 'g' (yield-green)
# anywhere in the programs. All three 'g' sources (core-clash downgrade,
# same-lane merge downgrade, permissive crossing right) are conflicting by
# construction; granting them as uncontrolled yields caused peak yield-
# starvation and adds behaviour the agent cannot control. Coverage is safe:
# the permissive-only audit shows every movement has a protected 'G' phase.
# Cost: some green utilisation — accepted for clean action semantics.
ALLOW_PERMISSIVE = False

CORES = {  # std action -> (sectors, turns); single-arm phases take all turns
    0: (("N", "S"), ("T",)),
    1: (("E", "W"), ("T",)),
    2: (("N", "S"), ("R",)),
    3: (("E", "W"), ("R",)),
    4: (("N",), ("L", "T", "R")),
    5: (("S",), ("L", "T", "R")),
    6: (("E",), ("L", "T", "R")),
    7: (("W",), ("L", "T", "R")),
}
TURN_RANK = {"L": 0, "T": 1, "R": 2}


def are_foes(mov, nodes, i, j):
    for a in mov["links"][i]:
        for b in mov["links"][j]:
            if a["junction"] != b["junction"]:
                continue
            if nodes[a["junction"]].areFoes(a["junction_index"], b["junction_index"]):
                return True
    return False


def same_arm(mov, i, j):
    fi = {c["from_edge"] for c in mov["links"][i]}
    fj = {c["from_edge"] for c in mov["links"][j]}
    return bool(fi & fj)


def build_phase(mov, nodes, action, n_state):
    """Return state string for one std action, or None if core is empty."""
    links = mov["links"]
    idxs = sorted(links)
    sectors, turns = CORES[action]
    core = [i for i in idxs
            if any(c["approach"] in sectors and c["turn"] in turns for c in links[i])]
    if not core:
        return None
    state = ["r"] * n_state
    # 1. core (links of the same incoming edge never block each other)
    for i in core:
        clash = any(state[j] == "G" and not same_arm(mov, i, j)
                    and are_foes(mov, nodes, i, j) for j in idxs)
        state[i] = "g" if clash else "G"
    # 2. maximal compatible closure (near-side lefts first, then straights, rights)
    rest = sorted((i for i in idxs if state[i] == "r"),
                  key=lambda i: (min(TURN_RANK[c["turn"]] for c in links[i]), i))
    for i in rest:
        if not any(state[j] == "G" and not same_arm(mov, i, j)
                   and are_foes(mov, nodes, i, j) for j in idxs):
            state[i] = "G"
    # 2b. merge safety: two 'G' links of different arms targeting the same
    # lane are NOT in the foes matrix but are unsafe (SUMO warning) — keep
    # the better-ranked one 'G', downgrade the rest to 'g' (yield-merge).
    by_target = {}
    for i in idxs:
        if state[i] != "G":
            continue
        for c in links[i]:
            by_target.setdefault((c["to_edge"], c["to_lane"]), []).append(
                (TURN_RANK[c["turn"]], i, c["from_edge"]))
    for tgt, members in by_target.items():
        distinct = sorted({(r, i) for r, i, _ in members})
        if len(distinct) < 2:
            continue  # single tl index (even with 2 conns) — cannot split chars
        keep = distinct[0][1]
        for _, i in distinct:
            if i != keep:
                state[i] = "g"
    # 3. permissive right where own arm's straight is running (D3)
    running = {c["approach"] for i in idxs if state[i] in "Gg"
               for c in links[i] if c["turn"] == "T"}
    for i in idxs:
        if state[i] == "r" and any(c["turn"] == "R" and c["approach"] in running
                                   for c in links[i]):
            state[i] = "g"
    if not ALLOW_PERMISSIVE:
        # 'g' grants never influenced closure decisions (those test 'G'
        # only), so stripping them to 'r' here is identical to never
        # granting — dedupe/mask/coverage evaluate the final strings.
        state = ["r" if ch == "g" else ch for ch in state]
    return "".join(state)


def yellow_between(cur, nxt):
    out = []
    for c, x in zip(cur, nxt):
        if c in "Gg" and x == "r":
            out.append("y")
        elif c in "Gg":
            out.append(c)
        else:
            out.append("r")
    return "".join(out)


def reindex_tls(net, tls_id, n_state):
    mov = common.tls_movements(net, tls_id)
    nodes = mov["nodes"]
    idxs = sorted(mov["links"])
    # state length = highest real link index + 1. Do NOT inherit the source
    # program's longer state strings (trailing padding from stripped
    # pedestrian links) — SUMO warns "Unused states ... after tl-index N".
    n_state = mov["n_links"]

    actions = {}
    canonical = {}
    state_to_std = {}
    for a in range(8):
        st = build_phase(mov, nodes, a, n_state)
        if st is None or ("G" not in st and "g" not in st):
            actions[a] = {"valid": False, "reason": "empty-core"}
            continue
        if st in state_to_std:
            actions[a] = {"valid": False, "reason": f"dup-of-a{state_to_std[st]}", "state": st}
            canonical[a] = state_to_std[st]
        else:
            state_to_std[st] = a
            actions[a] = {"valid": True, "state": st}
            canonical[a] = a

    canon_order = [a for a in range(8) if actions[a].get("valid")]
    greens = [actions[a]["state"] for a in canon_order]
    if len(greens) < 2:
        raise RuntimeError(f"{tls_id}: only {len(greens)} distinct phases")

    # V1: hard-conflict freedom among 'G' links of every phase
    for st in greens:
        gset = [i for i in idxs if st[i] == "G"]
        for ii, i in enumerate(gset):
            for j in gset[ii + 1:]:
                if not same_arm(mov, i, j) and are_foes(mov, nodes, i, j):
                    raise RuntimeError(f"{tls_id}: hard conflict {i},{j} in phase {st}")
    # V1: every controlled link served by some canonical phase
    orphan = [i for i in idxs if all(st[i] == "r" for st in greens)]
    if orphan:
        raise RuntimeError(f"{tls_id}: never-green links {orphan}")
    # V1 (user contract 2026-06-11): ZERO merge conflicts — no two 'G' links
    # of a phase may feed the same target lane (any arms, any indices), and
    # no single shared index may carry two connections into one lane.
    for st in greens:
        tgt = {}
        for i in idxs:
            if st[i] != "G":
                continue
            seen_here = set()
            for c in mov["links"][i]:
                key = (c["to_edge"], c["to_lane"])
                if key in seen_here:
                    raise RuntimeError(f"{tls_id}: shared-index merge {i}->{key}")
                seen_here.add(key)
                if key in tgt and tgt[key] != i:
                    raise RuntimeError(f"{tls_id}: merge conflict {tgt[key]},{i}->{key} in {st}")
                tgt[key] = i

    phases = []  # green then yellow-to-next, cyclically
    for k, st in enumerate(greens):
        phases.append((GREEN_DUR, st))
        ystate = yellow_between(st, greens[(k + 1) % len(greens)])
        # a transition where NOTHING turns red has no 'y' — emitting it would
        # make sumo_rl's _build_phases count it as an EXTRA green phase and
        # shift every std->green index (caught by V1 re-verification)
        if "y" in ystate:
            phases.append((YELLOW_DUR, ystate))

    meta = {
        "n_links": mov["n_links"],
        "n_state": n_state,
        "controlled_indices": idxs,
        "mask": [bool(actions[a].get("valid")) for a in range(8)],
        "canonical": {str(a): canonical[a] for a in canonical},
        "std_to_green_index": {str(a): canon_order.index(a) for a in canon_order},
        "actions": {f"a{a}": actions[a] for a in range(8)},
        "arms": mov["arms"],
        "links": {str(i): mov["links"][i] for i in idxs},
    }
    return phases, meta


def main():
    os.makedirs(common.OUT_DIR, exist_ok=True)
    # source: the demoted net if the demotion step has run (see
    # demote_uncontested.py), else the raw source net
    demoted = os.path.join(common.OUT_DIR, "dublin_8action_demoted.net.xml")
    src = sys.argv[1] if len(sys.argv) > 1 else (
        demoted if os.path.exists(demoted) else common.SRC_NET)
    print("source net:", src)
    net = common.load_net(src)
    tree = ET.parse(src)
    root = tree.getroot()

    orig_len = {tl.get("id"): len(tl.find("phase").get("state"))
                for tl in root.findall("tlLogic")}

    all_meta = {}
    new_programs = {}
    for tls_id, n_state in orig_len.items():
        phases, meta = reindex_tls(net, tls_id, n_state)
        all_meta[tls_id] = meta
        new_programs[tls_id] = phases

    for tl in root.findall("tlLogic"):
        tls_id = tl.get("id")
        tl.set("programID", "8std")
        tl.set("offset", "0")
        tl.set("type", "static")
        for p in list(tl):
            tl.remove(p)
        for dur, st in new_programs[tls_id]:
            ET.SubElement(tl, "phase", duration=dur, state=st)

    tree.write(common.NET_8STD, encoding="UTF-8", xml_declaration=True)

    entries, exits = common.boundary_edges(net)
    sidecar = {
        "source_net": os.path.relpath(src, common.REPO),
        "boundary": {"entries": sorted(e.getID() for e in entries),
                     "exits": sorted(e.getID() for e in exits)},
        "tls": all_meta,
    }
    with open(common.META_8STD, "w") as f:
        json.dump(sidecar, f, indent=1)

    # ---- V1 report
    n_valid = Counter(sum(m["mask"]) for m in all_meta.values())
    print(f"TLS reindexed: {len(all_meta)}  | valid-action count distribution: "
          f"{dict(sorted(n_valid.items()))}")
    print(f"boundary: {len(entries)} entries / {len(exits)} exits")
    res = subprocess.run(
        ["sumo", "-n", common.NET_8STD, "--no-step-log", "-e", "0"],
        capture_output=True, text=True)
    bad = [l for l in (res.stderr or "").splitlines() if "Error" in l]
    if res.returncode != 0 or bad:
        print(res.stderr)
        sys.exit("V1 FAIL: sumo cannot load the new net")
    warn = [l for l in (res.stderr or "").splitlines() if "Warning" in l]
    print(f"sumo load test: OK ({len(warn)} warnings)")
    for w in warn[:8]:
        print("  ", w)
    print(f"wrote {common.NET_8STD}\nwrote {common.META_8STD}")


if __name__ == "__main__":
    main()
