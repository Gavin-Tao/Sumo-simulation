"""Generic 8-std reindex for non-dublin nets (1x6, dcc) — REUSES the dublin
pipeline modules UNCHANGED (experiments/tools/dublin/{common,reindex_8std}.py)
by overriding common's path constants at runtime. New file by design: the
dublin scripts stay byte-identical and keep working for dublin.

Differences vs reindex_8std.main():
  * per-TLS fault tolerance — a TLS that cannot be 8-std-ified (e.g. <2
    distinct phases, never-green links) is SKIPPED and reported, instead of
    aborting the whole net (mandatory for dcc's 1044 signals);
  * skipped TLS keep their ORIGINAL program in the output net (control
    decision for them — demote/retype — is a separate later step);
  * output paths derived from the input net: <dir>/<prefix>_8std.net.xml
    + <prefix>_8std_meta.json.

Usage (from repo root):
  python experiments/tools/dcc/reindex_8std_net.py nets/dcc/1x6.net.xml 1x6
  python experiments/tools/dcc/reindex_8std_net.py nets/dcc/dcc.net.xml dcc
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "dublin"))
import common  # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    net_path = os.path.abspath(sys.argv[1])
    prefix = sys.argv[2]
    out_dir = os.path.dirname(net_path)

    # sanity: same source export as dublin -> same netOffset -> SCATS/GTFS
    # coordinate constants in common stay valid. Hard-fail if it differs.
    loc = ET.parse(net_path).getroot().find("location")
    off = tuple(float(x) for x in loc.get("netOffset").split(","))
    assert off == common.NET_OFFSET, \
        f"netOffset {off} != common.NET_OFFSET {common.NET_OFFSET} — " \
        f"different source export; extend wrapper before proceeding"

    # override the dublin path constants for this run (originals untouched)
    common.OUT_DIR = out_dir
    common.NET_8STD = os.path.join(out_dir, f"{prefix}_8std.net.xml")
    common.META_8STD = os.path.join(out_dir, f"{prefix}_8std_meta.json")

    import reindex_8std as R8  # noqa: E402  (import after patching)

    net = common.load_net(net_path)
    tree = ET.parse(net_path)
    root = tree.getroot()
    orig_len = {tl.get("id"): len(tl.find("phase").get("state"))
                for tl in root.findall("tlLogic")}

    all_meta, new_programs, skipped = {}, {}, {}
    for tls_id, n_state in orig_len.items():
        try:
            phases, meta = R8.reindex_tls(net, tls_id, n_state)
            all_meta[tls_id] = meta
            new_programs[tls_id] = phases
        except Exception as e:                       # per-TLS fault tolerance
            skipped[tls_id] = str(e)[:120]

    for tl in root.findall("tlLogic"):
        tls_id = tl.get("id")
        if tls_id not in new_programs:               # skipped: keep original
            continue
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
        "source_net": os.path.relpath(net_path, common.REPO),
        "boundary": {"entries": sorted(e.getID() for e in entries),
                     "exits": sorted(e.getID() for e in exits)},
        "skipped_tls": skipped,
        "tls": all_meta,
    }
    with open(common.META_8STD, "w") as f:
        json.dump(sidecar, f, indent=1)

    n_valid = Counter(sum(m["mask"]) for m in all_meta.values())
    print(f"TLS total={len(orig_len)}  reindexed={len(all_meta)}  "
          f"skipped={len(skipped)}")
    print(f"valid-action distribution (reindexed): {dict(sorted(n_valid.items()))}")
    if skipped:
        reasons = Counter(v.split(":")[-1].strip()[:40] for v in skipped.values())
        print("skip reasons:", dict(reasons.most_common(5)))
    print(f"boundary: {len(entries)} entries / {len(exits)} exits")
    res = subprocess.run(["sumo", "-n", common.NET_8STD, "--no-step-log", "-e", "0"],
                         capture_output=True, text=True)
    bad = [l for l in (res.stderr or "").splitlines() if "Error" in l]
    if res.returncode != 0 or bad:
        print("\n".join(bad[:5]))
        sys.exit("V1 FAIL: sumo cannot load the new net")
    print(f"sumo load: OK\nwrote {common.NET_8STD}\nwrote {common.META_8STD}")


if __name__ == "__main__":
    main()
