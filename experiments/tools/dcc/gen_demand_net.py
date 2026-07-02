"""Dublin-style 3-period demand generation for non-dublin nets (1x6, dcc) —
REUSES the dublin pipeline modules UNCHANGED (scats_pipeline, gtfs_bus,
calibrate_nnls, gen_routes) by overriding common's path constants at runtime.
New file by design: dublin scripts stay byte-identical and keep working.

Pipeline per net (same order as the dublin campaign):
  1. scats_pipeline   (once)      -> calibration/scats_targets.json
  2. gtfs_bus <hour>              -> bus_weekday_{H}h.rou.xml + bus_crossings
  3. calibrate_nnls <hour>        -> turn_ratios / boundary_rates
  4. gen_routes <hour> <seed>     -> car/amb rou.xml
  5. (wrapper) write <prefix>_weekday_{H}h.sumocfg

Routing net = <dir>/<prefix>_8std.net.xml (reindex_8std_net.py output;
connections identical to the source net). netOffset asserted equal to
dublin's so SCATS/GTFS/DFB coordinate logic is valid unchanged.

Usage (repo root):
  python experiments/tools/dcc/gen_demand_net.py nets/dcc/1x6_8std.net.xml 1x6 11 [seed]
  hours: run once per hour in (2, 11, 18).
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "dublin"))
import common  # noqa: E402

SUMOCFG = """<configuration>
  <input>
    <net-file value="../{net_base}"/>
    <route-files value="car_weekday_{h:02d}h.rou.xml,bus_weekday_{h:02d}h.rou.xml,amb_weekday_{h:02d}h.rou.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="3600"/>
  </time>
  <processing>
    <ignore-junction-blocker value="60"/>
  </processing>
  <report>
    <no-step-log value="true"/>
    <duration-log.statistics value="true"/>
  </report>
</configuration>
"""


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    net_path = os.path.abspath(sys.argv[1])
    prefix = sys.argv[2]
    hour = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    out_dir = os.path.dirname(net_path)

    loc = ET.parse(net_path).getroot().find("location")
    off = tuple(float(x) for x in loc.get("netOffset").split(","))
    assert off == common.NET_OFFSET, f"netOffset mismatch: {off}"

    # override dublin constants for this run (originals untouched). SRC_NET
    # points at the reindexed net; load_net_demand() falls back to SRC_NET
    # because <out_dir>/dublin_8action_demoted.net.xml does not exist here.
    common.SRC_NET = net_path
    common.OUT_DIR = out_dir
    common.CALIB_DIR = os.path.join(out_dir, f"calibration_{prefix}")
    common.NET_8STD = net_path
    common.META_8STD = os.path.join(out_dir, f"{prefix}_8std_meta.json")
    os.makedirs(common.CALIB_DIR, exist_ok=True)
    # CRITICAL (bug found 2026-07-02): load_net's DEFAULT ARG bound the
    # dublin path at def-time — patching common.SRC_NET alone does NOT
    # redirect bare common.load_net() calls (scats_pipeline uses one).
    # Rebind the module attribute with a closure over the target net.
    _orig_load_net = common.load_net
    common.load_net = lambda path=None: _orig_load_net(path or net_path)

    import scats_pipeline  # noqa: E402
    import gtfs_bus  # noqa: E402
    import calibrate_nnls  # noqa: E402
    import gen_routes  # noqa: E402

    targets = os.path.join(common.CALIB_DIR, "scats_targets.json")
    if not os.path.exists(targets):
        print(f"== [1/4] scats_pipeline ({prefix})")
        scats_pipeline.main()
    else:
        print(f"== [1/4] scats_targets.json exists — skip")

    print(f"== [2/4] gtfs_bus hour={hour}")
    sys.argv = ["gtfs_bus.py", str(hour)]
    gtfs_bus.main()

    print(f"== [3/4] calibrate_nnls hour={hour}")
    sys.argv = ["calibrate_nnls.py", str(hour)]
    calibrate_nnls.main()

    print(f"== [4/4] gen_routes hour={hour} seed={seed}")
    sys.argv = ["gen_routes.py", str(hour), str(seed)]
    gen_routes.main()

    cfg_dir = os.path.join(out_dir, f"weekday_{hour:02d}h")
    cfg_path = os.path.join(cfg_dir, f"{prefix}_weekday_{hour:02d}h.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(SUMOCFG.format(net_base=os.path.basename(net_path), h=hour))
    print(f"wrote {cfg_path}")


if __name__ == "__main__":
    main()
