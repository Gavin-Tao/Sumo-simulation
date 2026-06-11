"""Route generation — lane-locked cars + DFB ambulances + sumocfg.

Cars (design Part 3.1/3.5):
  * Poisson arrivals per boundary entry at the NNLS-calibrated rate
  * lane-consistent Markov walk: every hop is sampled ONLY among edges
    reachable through a real lane->lane connection from the CURRENT lane,
    weighted by the calibrated turn ratios; departLane = chain start.
    A vehicle therefore NEVER needs a lane change (D2): vType disables
    discretionary AND strategic changing outright — the chain guarantees
    the current lane always connects to the next route edge.
  * no junction revisit (G2 guard); rejection-resample on dead ends

Ambulances (design Part 4.2, assumption A6):
  * lambda_city(h) from the DFB 2023 log (weekday incidents per hour)
  * subnet rate = lambda_city(h) * sum_j exposure(j) / mean TLS crossings
    per car route, exposure(j) = station + hospital distance decay
  * route: entry weighted by station proximity, exit weighted by hospital
    direction, shortest path between them, lane chain via DP; no bluelight
    privilege — priority is the RL agent's job, as in the 1x3 scenarios

Outputs (nets/dublin/weekday_<H>h/):
  car_weekday_<H>h.rou.xml, amb_weekday_<H>h.rou.xml, dublin_weekday_<H>h.sumocfg
Usage: python3 gen_routes.py [hour] [seed]   (default 11, 42)
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MAX_ROUTE_TRIES = 80

STATIONS = {  # DFB ambulance stations (lat, lon)
    "Tallaght Fire Station": (53.2884, -6.3640),
    "Tara Street Fire Station": (53.3458, -6.2547),
    "Kilbarrack Fire Station": (53.3878, -6.1500),
    "Phibsborough Fire Station": (53.3603, -6.2730),
    "Dolphins Barn Fire Station": (53.3322, -6.2920),
    "Finglas Fire Station": (53.3865, -6.2956),
    "Rathfarnham Fire Station": (53.2960, -6.2810),
    "Blanchardstown Fire Station": (53.3850, -6.3760),
    "North Strand Fire Station": (53.3573, -6.2456),
    "Swords Fire Station": (53.4585, -6.2225),
    "Donnybrook Fire Station": (53.3215, -6.2316),
}
HOSPITALS = {  # acute hospitals around the subnet (lat, lon)
    "St James Hospital": (53.3403, -6.2949),
    "Mater Hospital": (53.3603, -6.2674),
    "St Vincent's": (53.3175, -6.2155),
}


def lane_edge_conns(edge):
    """{from_lane_index: {to_edge_id: [to_lane_index, ...]}}"""
    table = defaultdict(lambda: defaultdict(list))
    for to_edge, conns in edge.getOutgoing().items():
        for c in conns:
            table[c.getFromLane().getIndex()][to_edge.getID()].append(
                c.getToLane().getIndex())
    return table


class CarWalker:
    def __init__(self, net, ratios, rng):
        self.net = net
        self.ratios = ratios
        self.rng = rng
        self.conn_table = {e.getID(): lane_edge_conns(e) for e in net.getEdges()}
        self.tls_nodes = {t.getID() for t in net.getTrafficLights()}

    def walk(self, entry):
        """One lane-consistent route.

        Returns (edge_ids, depart_lane, crossed_tls_ids) or None.
        """
        for _ in range(MAX_ROUTE_TRIES):
            r = self._try(entry)
            if r:
                return r
        return None

    def _try(self, entry):
        # 1) pure EDGE-level Markov walk following the calibrated ratios —
        #    this is exactly the distribution the NNLS propagation assumed.
        #    (Sampling restricted by the current lane instead provably skews
        #    turn shares toward lane counts and broke V3 — see design doc.)
        e = entry
        path = [e.getID()]
        visited = set()
        crossed = []
        while e.getOutgoing():
            node = e.getToNode().getID()
            cand = {t: w for t, w in self.ratios.get(e.getID(), {}).items()
                    if w > 0 and self.net.getEdge(t).getToNode().getID() not in visited}
            if not cand:
                return None
            if node in self.tls_nodes:
                crossed.append(node)
            visited.add(node)
            nxt = self._sample(cand)
            path.append(nxt)
            e = self.net.getEdge(nxt)
        # 2) lane-locked feasibility: a lane chain must exist for the path
        depart_lane = lane_chain_dp(self.net, path, self.rng)
        if depart_lane is None:
            return None  # no lane-locked chain — resample
        return path, depart_lane, crossed

    def _sample(self, weights):
        tot = sum(weights.values())
        r = self.rng.random() * tot
        acc = 0.0
        for k, w in sorted(weights.items()):
            acc += w
            if r <= acc:
                return k
        return sorted(weights)[-1]


def lane_chain_dp(net, path_edges, rng=None):
    """Lane-consistent chain along a FIXED edge path.

    Returns the depart lane index, or None if no lane-locked chain exists.
    With `rng`, samples uniformly per step among feasible lanes/parents —
    WITHOUT it every vehicle on the same path takes the identical chain
    (lowest lane index), leaving parallel feasible lanes permanently empty
    (e.g. 4472590#0: 782/249/20 split, user review 2026-06-11).
    """
    edges = [net.getEdge(eid) for eid in path_edges]
    layers = [{l.getIndex(): [None] for l in edges[0].getLanes()}]  # lane -> parents
    for a, b in zip(edges, edges[1:]):
        table = lane_edge_conns(a)
        nxt = {}
        for la in layers[-1]:
            for lb in table.get(la, {}).get(b.getID(), []):
                nxt.setdefault(lb, []).append(la)
        if not nxt:
            return None
        layers.append(nxt)
    lanes = sorted(layers[-1])
    lane = rng.choice(lanes) if rng else lanes[0]
    for layer in reversed(layers[1:]):
        parents = sorted(set(layer[lane]))
        lane = rng.choice(parents) if rng else parents[0]
    return lane


def dfb_hourly_rate(hour):
    """Citywide weekday incidents/hour at the given hour, from the DFB log."""
    counts = defaultdict(int)
    days = set()
    with open(common.DFB_CSV, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                d = datetime.strptime(r["Date"], "%d/%m/%Y")
                h = int(r["TOC"].split(":")[0])
            except (ValueError, AttributeError, IndexError):
                continue
            if d.weekday() >= 5:
                continue
            days.add(r["Date"])
            counts[h] += 1
    return counts[hour] / max(len(days), 1)


def ambulance_exposure(net, mean_crossings):
    """sum_j exposure(j) / mean crossings  (design Part 4.2, MD decay form)."""
    total = 0.0
    for t in net.getTrafficLights():
        x, y = net.getNode(t.getID()).getCoord()
        lat, lon = common.utm_to_latlon(x, y)
        ds = min(common.latlon_dist_m(lat, lon, *p) for p in STATIONS.values())
        dh = min(common.latlon_dist_m(lat, lon, *p) for p in HOSPITALS.values())
        total += max(0.005, 0.05 * math.exp(-ds / 800)) + max(0.003, 0.03 * math.exp(-dh / 800))
    return total / max(mean_crossings, 1.0)


def edge_latlon(edge, end=True):
    x, y = edge.getShape()[-1 if end else 0]
    return common.utm_to_latlon(x, y)


def main():
    hour = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rng = random.Random(seed)
    out_dir = os.path.join(common.OUT_DIR, f"weekday_{hour:02d}h")
    os.makedirs(out_dir, exist_ok=True)

    net = common.load_net_demand()
    ratios = json.load(open(os.path.join(common.CALIB_DIR,
                                       f"turn_ratios_{hour:02d}h.json")))["ratios"]
    rates = json.load(open(os.path.join(common.CALIB_DIR,
                                        f"boundary_rates_{hour:02d}h.json")))["rates"]
    walker = CarWalker(net, ratios, rng)
    entries, exits = common.boundary_edges(net)

    # ---------------- cars
    cars = []
    n_fail = 0
    tls_cross_sum = 0
    for eid, rate in sorted(rates.items()):
        n = _poisson(rate, rng)
        for _ in range(n):
            r = walker.walk(net.getEdge(eid))
            if r is None:
                n_fail += 1
                continue
            path, depart_lane, crossed = r
            tls_cross_sum += len(crossed)
            cars.append((rng.uniform(0, 3600), path, depart_lane))
    cars.sort()
    mean_cross = tls_cross_sum / max(len(cars), 1)
    print(f"cars: {len(cars)} generated, {n_fail} walk failures, "
          f"mean TLS crossings/route: {mean_cross:.2f}")

    car_path = os.path.join(out_dir, f"car_weekday_{hour:02d}h.rou.xml")
    with open(car_path, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="car" color="1,0,0" lcStrategic="0" lcCooperative="0" '
                'lcSpeedGain="0" lcKeepRight="0"/>\n')
        for k, (t, path, lane) in enumerate(cars):
            f.write(f'  <vehicle id="car_{k}" type="car" depart="{t:.1f}" '
                    f'departLane="{lane}" departSpeed="max">\n'
                    f'    <route edges="{" ".join(path)}"/>\n  </vehicle>\n')
        f.write("</routes>\n")

    # ---------------- ambulances
    lam_city = dfb_hourly_rate(hour)
    lam_sub = lam_city * ambulance_exposure(net, mean_cross)
    # zero-truncated Poisson: a scenario without any priority vehicle cannot
    # exercise the priority mechanism, so we condition on N >= 1 (documented)
    n_amb = max(1, _poisson(lam_sub, rng)) if lam_sub > 0 else 0
    print(f"ambulances: city {lam_city:.1f}/h -> subnet {lam_sub:.2f}/h -> n={n_amb}")

    entry_w = {}
    for e in entries:
        lat, lon = edge_latlon(e, end=False)
        ds = min(common.latlon_dist_m(lat, lon, *p) for p in STATIONS.values())
        entry_w[e.getID()] = math.exp(-ds / 1500)
    exit_w = {}
    for e in exits:
        lat, lon = edge_latlon(e, end=True)
        exit_w[e.getID()] = sum(math.exp(-common.latlon_dist_m(lat, lon, *p) / 2000)
                                for p in HOSPITALS.values())

    ambs = []
    tries = 0
    while len(ambs) < n_amb and tries < 200:
        tries += 1
        ein = walker._sample({k: v for k, v in entry_w.items()})
        eout = walker._sample({k: v for k, v in exit_w.items()})
        sp, _cost = net.getShortestPath(net.getEdge(ein), net.getEdge(eout))
        if not sp or len(sp) < 3:
            continue
        path = [e.getID() for e in sp]
        n_tls = sum(1 for e in sp[:-1] if e.getToNode().getID() in walker.tls_nodes)
        if n_tls < 2:
            continue  # must actually transit the controlled area
        lane = lane_chain_dp(net, path, rng)
        if lane is None:
            continue
        ambs.append((rng.uniform(0, 3600), path, lane))
    ambs.sort()
    print(f"ambulances: emitted {len(ambs)} (tries {tries})")

    amb_path = os.path.join(out_dir, f"amb_weekday_{hour:02d}h.rou.xml")
    with open(amb_path, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="ambulance" vClass="emergency" guiShape="emergency" '
                'color="0,1,0" lcStrategic="0" lcCooperative="0" lcSpeedGain="0" '
                'lcKeepRight="0"/>\n')
        for k, (t, path, lane) in enumerate(ambs):
            f.write(f'  <vehicle id="amb_{k}" type="ambulance" depart="{t:.1f}" '
                    f'departLane="{lane}" departSpeed="max">\n'
                    f'    <route edges="{" ".join(path)}"/>\n  </vehicle>\n')
        f.write("</routes>\n")

    # ---------------- sumocfg
    cfg_path = os.path.join(out_dir, f"dublin_weekday_{hour:02d}h.sumocfg")
    with open(cfg_path, "w") as f:
        f.write(f"""<configuration>
  <input>
    <net-file value="../dublin_8std.net.xml"/>
    <route-files value="car_weekday_{hour:02d}h.rou.xml,bus_weekday_{hour:02d}h.rou.xml,amb_weekday_{hour:02d}h.rou.xml"/>
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
""")
    print("wrote", car_path)
    print("wrote", amb_path)
    print("wrote", cfg_path)


def _poisson(lam, rng):
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


if __name__ == "__main__":
    main()
