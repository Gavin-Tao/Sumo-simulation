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


def split_merged_slots(mov, nodes):
    """槽粒度修正 (2026-09-30 用户批准; 仅 --split-slots 时调用, 默认路径不变)。

    问题: 几何转向分类可能把同一进口的两个不同去向并进同一个槽 (approach, turn)。
    槽是整体进出的, 其中任一条 link 与别的进口冲突, 整个槽就被判冲突, 于是
    "只放其中不冲突的那条 link" 的组合在槽粒度上不存在, 枚举器无从生成
    (cluster_21620852_...: Harcourt 左转 link 5 与 Cuffe 相容, 却因同槽的 link 6 被判冲突)。
    规则 (只在确实藏住了相容组合时触发, 其它路口逐位不变):
      槽内 link 去向 >= 2 个, 且存在另一进口的槽 B 使得 "整槽与 B 冲突, 但槽内至少一条
      link 与 B 完全不冲突" -> 把槽内 netconvert 标注转向 (dir_turn) 与几何转向不同的 link
      改归到该进口的 (approach, dir_turn) 槽, 前提是那个槽原本为空。
    原几何转向保留在 conns[*]["turn_geometric"] 供审计。返回 [(link, 旧槽, 新槽), ...]。
    """
    st = slot_tables(mov)

    def lconf(i, j):
        return (not same_arm(mov, i, j) and are_foes(mov, nodes, i, j)) \
            or _lane_share(mov, i, j)

    moved = []
    for slot, idxs in sorted(st.items()):
        if len({c["to_edge"] for i in idxs for c in mov["links"][i]}) < 2:
            continue
        hidden = False
        for sb, ib in st.items():
            if sb[0] == slot[0]:
                continue
            if any(lconf(i, j) for i in idxs for j in ib):
                free = [i for i in idxs if not any(lconf(i, j) for j in ib)]
                if free and len(free) < len(idxs):
                    hidden = True
        if not hidden:
            continue
        cand = []
        for i in idxs:
            lab = {c["dir_turn"] for c in mov["links"][i]}
            if len(lab) == 1:
                lab = next(iter(lab))
                if lab in ("L", "T", "R") and lab != slot[1] and (slot[0], lab) not in st:
                    cand.append((i, lab))
        if not cand or len(cand) == len(idxs):      # 不把原槽搬空
            continue
        for i, lab in cand:
            for c in mov["links"][i]:
                c["turn_geometric"] = c["turn"]
                c["turn"] = lab
            moved.append((i, list(slot), [slot[0], lab]))
    return moved


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


def main():
    demoted = os.path.join(common.OUT_DIR, "dublin_8action_demoted.net.xml")
    # 2026-09-30: 可选开关 (默认全关, 无参数时行为与输出逐位不变):
    #   --split-slots   启用 split_merged_slots 槽粒度修正
    #   --tag X         输出 dublin_enum_X.net.xml / dublin_enum_X_meta.json (不覆盖原文件, 不重写 *_enum.sumocfg)
    #   --out-dir D     输出目录 (默认 nets/dublin; 给出时同样不重写 sumocfg)
    argv = sys.argv[1:]
    split = "--split-slots" in argv
    tag = argv[argv.index("--tag") + 1] if "--tag" in argv else None
    out_dir = argv[argv.index("--out-dir") + 1] if "--out-dir" in argv else None
    skip = set()
    for flag in ("--tag", "--out-dir"):
        if flag in argv:
            skip.update({argv.index(flag), argv.index(flag) + 1})
    pos = [a for k, a in enumerate(argv) if k not in skip and a != "--split-slots"]
    src = pos[0] if pos else demoted
    stem = f"dublin_enum_{tag}" if tag else "dublin_enum"
    NET_ENUM = os.path.join(out_dir or common.OUT_DIR, stem + ".net.xml")
    META_ENUM = os.path.join(out_dir or common.OUT_DIR, stem + "_meta.json")
    slot_split = {}
    print("source net:", src)
    net = common.load_net(src)
    tree = ET.parse(src)
    root = tree.getroot()
    tls_ids = [tl.get("id") for tl in root.findall("tlLogic")]
    all_meta, programs, kmax = {}, {}, 0
    for tid in tls_ids:
        mov = common.tls_movements(net, tid)
        nodes = mov["nodes"]
        if split:
            moved = split_merged_slots(mov, nodes)
            if moved:
                slot_split[tid] = moved
                print(f"split-slots {tid}: {moved}")
        st = slot_tables(mov)
        bad = intra_slot_conflicts(mov, nodes, st)
        if bad:
            sys.exit(f"V1 FAIL {tid}: intra-slot conflicts {bad}")
        rel = movement_rel(mov, nodes, st)
        menu = enumerate_menu(rel, st)
        greens = []
        for p in menu:
            s = phase_state(mov, st, p, mov["n_links"])
            verify_phase(mov, nodes, s)
            greens.append(s)
        kmax = max(kmax, len(menu))
        phases = []
        for k, s in enumerate(greens):
            phases.append((GREEN_DUR, s))
            y = yellow_between(s, greens[(k + 1) % len(greens)])
            if "y" in y:
                phases.append((YELLOW_DUR, y))
        programs[tid] = phases
        all_meta[tid] = {
            "n_phases": len(menu),
            "phase_movements": [[1 if m in p else 0 for m in range(12)] for p in menu],
            "movement_rel": rel,
            "links": {str(i): mov["links"][i] for i in sorted(mov["links"])},
        }
    for m in all_meta.values():                       # pad masks to global K_max
        m["mask"] = [k < m["n_phases"] for k in range(kmax)]
    for tl in root.findall("tlLogic"):
        tid = tl.get("id")
        tl.set("programID", "enum")
        tl.set("offset", "0")
        tl.set("type", "static")
        for p in list(tl):
            tl.remove(p)
        for dur, s in programs[tid]:
            ET.SubElement(tl, "phase", duration=dur, state=s)
    tree.write(NET_ENUM, encoding="UTF-8", xml_declaration=True)
    entries, exits = common.boundary_edges(net)
    doc = {"action_scheme": "enum_frap", "n_actions": kmax,
           "source_net": os.path.relpath(src, common.REPO),
           "boundary": {"entries": sorted(e.getID() for e in entries),
                        "exits": sorted(e.getID() for e in exits)},
           "tls": all_meta}
    if split:                                         # 默认路径不加此键 -> 旧 meta 逐位不变
        doc["slot_split"] = slot_split
    json.dump(doc, open(META_ENUM, "w"), indent=1)
    sizes = sorted(m["n_phases"] for m in all_meta.values())
    print(f"TLS: {len(all_meta)}  K_max={kmax}  menu sizes={sizes}  "
          f"mean={sum(sizes) / len(sizes):.1f}")
    res = subprocess.run(["sumo", "-n", NET_ENUM, "--no-step-log", "-e", "0"],
                         capture_output=True, text=True)
    bad_lines = [l for l in (res.stderr or "").splitlines() if "Error" in l]
    if res.returncode != 0 or bad_lines:
        print(res.stderr)
        sys.exit("V1 FAIL: sumo cannot load the enum net")
    warn = [l for l in (res.stderr or "").splitlines() if "Warning" in l]
    print(f"sumo load: OK ({len(warn)} warnings)")
    for w in warn[:8]:
        print("  ", w)
    print(f"wrote {NET_ENUM}\nwrote {META_ENUM}")
    if tag or out_dir:                                # 变体输出: 不改动既有 *_enum.sumocfg
        return
    for h in ("02", "11", "18"):
        src_cfg = os.path.join(common.OUT_DIR, f"weekday_{h}h", f"dublin_weekday_{h}h.sumocfg")
        dst_cfg = src_cfg.replace(".sumocfg", "_enum.sumocfg")
        txt = open(src_cfg).read().replace("dublin_8std.net.xml", "dublin_enum.net.xml")
        assert "dublin_enum.net.xml" in txt, src_cfg
        open(dst_cfg, "w").write(txt)
        print(f"wrote {dst_cfg}")


if __name__ == "__main__":
    main()
