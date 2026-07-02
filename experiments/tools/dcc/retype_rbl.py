"""Retype all right_before_left junctions of a net to priority — Ireland has
no right-before-left rule, and its symmetric waits deadlock in SUMO (the
dublin Grafton mini-gyratory lesson; demote_uncontested.py did this for the
dublin subnet). New generic script: applies to any net via netconvert node
patch, verifies connections unchanged, writes IN PLACE (net is a generated
artifact; the uploaded source net stays untouched).

Usage: python experiments/tools/dcc/retype_rbl.py nets/dcc/dcc_8std.net.xml
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET


def conn_set(path):
    return sorted((c.get("from"), c.get("to"), c.get("fromLane"), c.get("toLane"))
                  for c in ET.parse(path).getroot().findall("connection")
                  if not (c.get("from") or "").startswith(":"))


def main():
    net_path = os.path.abspath(sys.argv[1])
    root = ET.parse(net_path).getroot()
    rbl = [j.get("id") for j in root.findall("junction")
           if j.get("type") == "right_before_left"]
    if not rbl:
        print("no right_before_left junctions — nothing to do")
        return
    patch = net_path + ".rbl.nod.xml"
    with open(patch, "w") as f:
        f.write("<nodes>\n")
        for jid in rbl:
            f.write(f'  <node id="{jid}" type="priority"/>\n')
        f.write("</nodes>\n")
    before = conn_set(net_path)
    tmp = net_path + ".retyped.net.xml"
    res = subprocess.run(
        ["netconvert", "-s", net_path, "-n", patch, "-o", tmp,
         "--offset.disable-normalization"],
        capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-1500:])
        sys.exit("netconvert failed")
    after = conn_set(tmp)
    assert before == after, f"connection drift: {len(before)} vs {len(after)}"
    # netOffset must survive (SCATS/GTFS coordinate logic depends on it)
    off_a = ET.parse(net_path).getroot().find("location").get("netOffset")
    off_b = ET.parse(tmp).getroot().find("location").get("netOffset")
    assert off_a == off_b, f"netOffset drift: {off_a} -> {off_b}"
    os.replace(tmp, net_path)
    os.remove(patch)
    left = sum(1 for j in ET.parse(net_path).getroot().findall("junction")
               if j.get("type") == "right_before_left")
    print(f"retyped {len(rbl)} right_before_left -> priority "
          f"(remaining rbl: {left}); connections identical ({len(before)})")


if __name__ == "__main__":
    main()
