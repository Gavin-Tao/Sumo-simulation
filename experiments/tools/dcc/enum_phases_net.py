"""Generic enum-phases wrapper for ANY net (1x1, 1x6, dcc, ...) — reuses
experiments/tools/dublin/enum_phases.py UNCHANGED by overriding common's
path constants (same pattern as reindex_8std_net.py). Per-TLS fault
tolerance: un-enumerable TLS keep their original program and are reported.

Usage (repo root):
  python experiments/tools/dcc/enum_phases_net.py <net.xml> <prefix>
  -> <dir>/<prefix>_enum.net.xml + <prefix>_enum_meta.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "dublin"))
import common  # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    net_path = os.path.abspath(sys.argv[1])
    prefix = sys.argv[2]
    out_dir = os.path.dirname(net_path)
    common.OUT_DIR = out_dir
    import enum_phases as EP  # noqa: E402  (dublin module, constants unused below)

    net = common.load_net(net_path)
    tree = ET.parse(net_path)
    root = tree.getroot()
    tls_ids = [tl.get("id") for tl in root.findall("tlLogic")]
    all_meta, programs, skipped, kmax = {}, {}, {}, 0
    for tid in tls_ids:
        try:
            mov = common.tls_movements(net, tid)
            nodes = mov["nodes"]
            st = EP.slot_tables(mov)
            bad = EP.intra_slot_conflicts(mov, nodes, st)
            if bad:
                raise RuntimeError(f"intra-slot conflicts {bad}")
            rel = EP.movement_rel(mov, nodes, st)
            menu = EP.enumerate_menu(rel, st)
            greens = []
            for p in menu:
                s = EP.phase_state(mov, st, p, mov["n_links"])
                EP.verify_phase(mov, nodes, s)
                greens.append(s)
            kmax = max(kmax, len(menu))
            phases = []
            for k, s in enumerate(greens):
                phases.append((EP.GREEN_DUR, s))
                y = EP.yellow_between(s, greens[(k + 1) % len(greens)])
                if "y" in y:
                    phases.append((EP.YELLOW_DUR, y))
            programs[tid] = phases
            all_meta[tid] = {
                "n_phases": len(menu),
                "phase_movements": [[1 if m in p else 0 for m in range(12)]
                                    for p in menu],
                "movement_rel": rel,
                "links": {str(i): mov["links"][i] for i in sorted(mov["links"])},
            }
        except Exception as e:
            skipped[tid] = str(e)[:120]
    for m in all_meta.values():
        m["mask"] = [k < m["n_phases"] for k in range(kmax)]
    for tl in root.findall("tlLogic"):
        tid = tl.get("id")
        if tid not in programs:
            continue
        tl.set("programID", "enum")
        tl.set("offset", "0")
        tl.set("type", "static")
        for p in list(tl):
            tl.remove(p)
        for dur, s in programs[tid]:
            ET.SubElement(tl, "phase", duration=dur, state=s)
    out_net = os.path.join(out_dir, f"{prefix}_enum.net.xml")
    out_meta = os.path.join(out_dir, f"{prefix}_enum_meta.json")
    tree.write(out_net, encoding="UTF-8", xml_declaration=True)
    entries, exits = common.boundary_edges(net)
    json.dump({"action_scheme": "enum_frap", "n_actions": kmax,
               "source_net": net_path,
               "boundary": {"entries": sorted(e.getID() for e in entries),
                            "exits": sorted(e.getID() for e in exits)},
               "skipped_tls": skipped, "tls": all_meta},
              open(out_meta, "w"), indent=1)
    sizes = sorted(m["n_phases"] for m in all_meta.values())
    print(f"TLS total={len(tls_ids)} enumerated={len(all_meta)} "
          f"skipped={len(skipped)} K_max={kmax} menus={sizes[:20]}")
    res = subprocess.run(["sumo", "-n", out_net, "--no-step-log", "-e", "0"],
                         capture_output=True, text=True)
    if res.returncode != 0 or any("Error" in l for l in (res.stderr or "").splitlines()):
        print(res.stderr[-800:])
        sys.exit("V1 FAIL: sumo cannot load")
    print(f"sumo load: OK\nwrote {out_net}\nwrote {out_meta}")


if __name__ == "__main__":
    main()
