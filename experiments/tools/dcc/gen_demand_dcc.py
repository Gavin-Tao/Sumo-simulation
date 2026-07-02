"""DCC citywide 3-period demand — hybrid calibration (spec:
DCC_DEMAND_DESIGN_2026-07-02.txt). L1 = randomTrips candidate pool +
routeSampler against SCATS edge counts; L2 = background randomTrips at
BG_SHARE of L1 volume; bus = existing gtfs_bus (unchanged, via constant
override); amb = Poisson(city rate) station->hospital internal routes.

Deviation from the dublin subnet convention, recorded: DCC vehicles are NOT
lane-locked (citywide lane chains impractical for sampled routes); buses
keep gtfs_bus's own locking.

Usage (repo root): python experiments/tools/dcc/gen_demand_dcc.py <hour> [seed]
Prereqs: dcc_8std.net.xml (reindexed+retyped+pruned), calibration_dcc/
         scats_targets.json (scats_pipeline via gen_demand_net.py run).
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "dublin"))
import common  # noqa: E402

REPO = common.REPO
NET = os.path.join(REPO, "nets", "dcc", "dcc_8std.net.xml")
META = os.path.join(REPO, "nets", "dcc", "dcc_8std_meta.json")
CALIB = os.path.join(REPO, "nets", "dcc", "calibration_dcc")
import sumolib  # noqa: E402
TOOLS = os.path.join(os.path.dirname(os.path.dirname(sumolib.__file__)),
                     "sumo", "tools")
BG_SHARE = float(os.environ.get("DCC_BG_SHARE", 0.1))    # L2 share of L1
DEMAND_SCALE = float(os.environ.get("DCC_DEMAND_SCALE", 0.5))  # capacity-feasible
# scaling of SCATS counts (unscaled 11h gridlocks the fixed-time baseline:
# 34% backlog / 15.5k teleports — spec V-b record). Both knobs recorded below.
POOL_TRIPS = 120_000      # candidate pool size for routeSampler
AMB_RATE = {2: 5.9, 11: 11.1, 18: 12.2}   # DFB citywide incidents/h (measured)

SUMOCFG = """<configuration>
  <input>
    <net-file value="../dcc_8std.net.xml"/>
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


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd[:6]), "...")
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if res.returncode != 0:
        print(res.stderr[-2000:])
        sys.exit(f"FAILED: {cmd[0]}")
    return res


def count_vehicles(rou):
    return sum(1 for _, el in ET.iterparse(rou) if el.tag == "vehicle")


def gen_ambulances(net, hour, seed, out_path):
    import gen_routes as GR  # dublin module: only STATIONS/HOSPITALS consts used
    rng = random.Random(seed)
    n = max(1, int(rng.gauss(AMB_RATE[hour], math.sqrt(AMB_RATE[hour]))))

    def nearest_edge(lat, lon):
        x, y = common.latlon_to_net(lat, lon)
        cands = net.getNeighboringEdges(x, y, r=400)
        cands = [(d, e) for e, d in cands
                 if e.getFunction() != "internal" and e.allows("passenger")]
        return min(cands, key=lambda t: t[0])[1] if cands else None

    st_edges = [e for e in (nearest_edge(*p) for p in GR.STATIONS.values()) if e]
    ho_edges = [e for e in (nearest_edge(*p) for p in GR.HOSPITALS.values()) if e]
    assert st_edges and ho_edges, "no station/hospital edges found in net"
    vehs, tries = [], 0
    while len(vehs) < n and tries < n * 20:
        tries += 1
        o = rng.choice(st_edges)
        d = rng.choice(ho_edges)
        if o.getID() == d.getID():
            continue
        path, _ = net.getShortestPath(o, d, vClass="passenger")
        if not path or len(path) < 3:
            continue
        vehs.append((rng.uniform(0, 3600), [e.getID() for e in path]))
    vehs.sort()
    with open(out_path, "w") as f:
        f.write('<routes>\n  <vType id="ambulance" vClass="emergency" '
                'guiShape="emergency" color="0,1,0"/>\n')
        for k, (dep, edges) in enumerate(vehs):
            f.write(f'  <vehicle id="amb_{k}" type="ambulance" depart="{dep:.1f}" '
                    f'departLane="best" departSpeed="max">\n'
                    f'    <route edges="{" ".join(edges)}"/>\n  </vehicle>\n')
        f.write("</routes>\n")
    print(f"ambulances: target n={n}, emitted {len(vehs)} (tries {tries})")


def main():
    hour = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    out_dir = os.path.join(REPO, "nets", "dcc", f"weekday_{hour:02d}h")
    os.makedirs(out_dir, exist_ok=True)
    tools = os.path.abspath(TOOLS)
    scratch = os.path.join(CALIB, "pool")
    os.makedirs(scratch, exist_ok=True)

    # [0] bus via existing gtfs_bus (constant-override, dublin module unchanged)
    bus_out = os.path.join(out_dir, f"bus_weekday_{hour:02d}h.rou.xml")
    if not os.path.exists(bus_out):
        common.SRC_NET = NET
        common.OUT_DIR = os.path.join(REPO, "nets", "dcc")
        common.CALIB_DIR = CALIB
        _orig = common.load_net
        common.load_net = lambda path=None: _orig(path or NET)
        import gtfs_bus
        sys.argv = ["gtfs_bus.py", str(hour)]
        gtfs_bus.main()

    # [1] SCATS -> edge counts (with capacity-feasible scale)
    print(f"knobs: DEMAND_SCALE={DEMAND_SCALE} BG_SHARE={BG_SHARE}")
    run([sys.executable, os.path.join(_HERE, "scats_to_edgecounts.py"),
         NET, META, CALIB, str(hour), str(DEMAND_SCALE)])
    edgecounts = os.path.join(CALIB, f"edgecounts_{hour:02d}h.xml")

    # [2] candidate pool (built once, reused across hours)
    pool_routes = os.path.join(scratch, "candidates.rou.xml")
    if not os.path.exists(pool_routes):
        trips = os.path.join(scratch, "pool.trips.xml")
        run([sys.executable, os.path.join(tools, "randomTrips.py"), "-n", NET,
             "-o", trips, "-e", str(POOL_TRIPS), "-p", "1", "--seed", "1",
             "--fringe-factor", "5", "--min-distance", "300",
             "--vclass", "passenger", "--validate"])
        run(["duarouter", "-n", NET, "--route-files", trips, "-o", pool_routes,
             "--ignore-errors", "--no-warnings", "--no-step-log",
             "--repair", "--remove-loops"])
        print(f"candidate pool: {count_vehicles(pool_routes)} routes")

    # [3] L1 routeSampler against SCATS counts
    l1 = os.path.join(scratch, f"l1_{hour:02d}h.rou.xml")
    res = run([sys.executable, os.path.join(tools, "routeSampler.py"),
         "-r", pool_routes, "--edgedata-files", edgecounts, "-o", l1,
         "--edgedata-attribute", "count",   # default is 'entered' — silent 0 otherwise
         "--attributes", 'type="car" departLane="best" departSpeed="max"',
         "--seed", str(seed), "--mismatch-output",
         os.path.join(CALIB, f"mismatch_{hour:02d}h.xml")])
    for line in (res.stdout or "").splitlines():
        if "achieving" in line or "GEH" in line:
            print("  routeSampler:", line.strip())
    n_l1 = count_vehicles(l1)

    # [4] L2 background at BG_SHARE of L1 (uncovered-area prior; explicit knob)
    n_bg = int(n_l1 * BG_SHARE)
    bg_trips = os.path.join(scratch, f"bg_{hour:02d}h.trips.xml")
    bg = os.path.join(scratch, f"bg_{hour:02d}h.rou.xml")
    run([sys.executable, os.path.join(tools, "randomTrips.py"), "-n", NET,
         "-o", bg_trips, "-b", "0", "-e", "3600",
         "--insertion-rate", str(max(1, n_bg)), "--seed", str(seed + 7),
         "--fringe-factor", "2", "--min-distance", "300", "--vclass", "passenger"])
    run(["duarouter", "-n", NET, "--route-files", bg_trips, "-o", bg,
         "--ignore-errors", "--no-warnings", "--no-step-log",
         "--repair", "--remove-loops"])

    # [5] merge L1+L2 -> car rou (with vType, departs sorted)
    car_out = os.path.join(out_dir, f"car_weekday_{hour:02d}h.rou.xml")
    vehs = []
    for src, tag in ((l1, "c"), (bg, "b")):
        for _, el in ET.iterparse(src):
            if el.tag == "vehicle":
                dep = float(el.get("depart"))
                edges = el.find("route").get("edges")
                vehs.append((dep, tag, edges))
    vehs.sort()
    with open(car_out, "w") as f:
        f.write('<routes>\n  <vType id="car" vClass="passenger"/>\n')
        for k, (dep, tag, edges) in enumerate(vehs):
            f.write(f'  <vehicle id="car_{tag}{k}" type="car" depart="{dep:.1f}" '
                    f'departLane="best" departSpeed="max">\n'
                    f'    <route edges="{edges}"/>\n  </vehicle>\n')
        f.write("</routes>\n")
    print(f"cars: L1(calibrated)={n_l1} + L2(background)={count_vehicles(bg)} "
          f"-> {len(vehs)} total; wrote {car_out}")

    # [5.5] depart smoothing + origin dispersal (spawn-flood cap at source)
    run([sys.executable, os.path.join(_HERE, "smooth_departs.py"), NET, car_out])

    # [6] ambulances
    net = common.load_net(NET)
    gen_ambulances(net, hour, seed, os.path.join(out_dir, f"amb_weekday_{hour:02d}h.rou.xml"))

    # [7] sumocfg
    cfg = os.path.join(out_dir, f"dcc_weekday_{hour:02d}h.sumocfg")
    with open(cfg, "w") as f:
        f.write(SUMOCFG.format(h=hour))
    print(f"wrote {cfg}")


if __name__ == "__main__":
    main()
