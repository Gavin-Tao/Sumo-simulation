"""Shared helpers for the Dublin validation pipeline.

Single source of truth for: paths, bbox, UTM conversion, boundary entry/exit
edges (connection-graph truth), and per-TLS movement extraction (compass
sector + tie-break). Both the action reindexer (M0) and the route generators
(M2) import from here so state/action/demand share one movement semantics.

Design doc: experiments/analysis/DUBLIN_VALIDATION_DESIGN_2026-06-11.txt
"""
from __future__ import annotations

import math
import os
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_DIR = os.path.join(REPO, "dublin_sumo_files")
SRC_NET = os.path.join(SRC_DIR, "dublin_8action.net.xml")
OUT_DIR = os.path.join(REPO, "nets", "dublin")
NET_8STD = os.path.join(OUT_DIR, "dublin_8std.net.xml")
META_8STD = os.path.join(OUT_DIR, "dublin_8std_meta.json")
CALIB_DIR = os.path.join(OUT_DIR, "calibration")

SCATS_CSV = os.path.join(SRC_DIR, "SCATSJanuary2023.csv")
SCATS_SITES_CSV = os.path.join(SRC_DIR, "dcc_traffic_signals_20221130.csv")
DFB_CSV = os.path.join(SRC_DIR, "2023-open-data-dfb-ambulance.csv")
GTFS_DIR = os.path.join(SRC_DIR, "bus")

NET_OFFSET = (-677749.85, -5911444.78)  # from <location> of the source net
SECTORS = ("N", "E", "S", "W")

# ---------------------------------------------------------------- UTM zone 29N


def utm_to_latlon(x_net: float, y_net: float) -> tuple[float, float]:
    """Net coordinates -> (lat, lon). Inverse UTM zone 29N, WGS84 (~<1m)."""
    E = x_net - NET_OFFSET[0]
    N = y_net - NET_OFFSET[1]
    k0 = 0.9996
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    M = N / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 * e1 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    C1 = ep2 * math.cos(phi1) ** 2
    T1 = math.tan(phi1) ** 2
    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    R1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    D = (E - 500000) / (N1 * k0)
    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D * D / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ep2) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ep2 - 3 * C1 * C1) * D ** 6 / 720)
    lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6
           + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ep2 + 24 * T1 * T1) * D ** 5 / 120) / math.cos(phi1)
    lon0 = math.radians(29 * 6 - 183)
    return math.degrees(lat), math.degrees(lon0 + lon)


def latlon_to_net(lat: float, lon: float) -> tuple[float, float]:
    """(lat, lon) -> net coordinates. Forward UTM zone 29N, WGS84 (~<1m)."""
    k0 = 0.9996
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi = math.radians(lat)
    lam = math.radians(lon)
    lam0 = math.radians(29 * 6 - 183)
    Nn = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = ep2 * math.cos(phi) ** 2
    A = (lam - lam0) * math.cos(phi)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    E = k0 * Nn * (A + (1 - T + C) * A ** 3 / 6
                   + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    No = k0 * (M + Nn * math.tan(phi) * (A * A / 2
               + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
               + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return E + NET_OFFSET[0], No + NET_OFFSET[1]


def latlon_dist_m(lat1, lon1, lat2, lon2):
    return math.hypot((lat1 - lat2) * 111320.0,
                      (lon1 - lon2) * 111320.0 * math.cos(math.radians(53.34)))


# --------------------------------------------------------------- net utilities


def load_net(path: str = SRC_NET):
    import sumolib
    return sumolib.net.readNet(path, withInternal=False)


def load_net_demand():
    """Net for demand/lane-chain work: the demoted + lane-reach-patched net
    when the demotion step has run (lane tables MUST match what vehicles will
    drive on), else the raw source net."""
    p = os.path.join(OUT_DIR, "dublin_8action_demoted.net.xml")
    return load_net(p if os.path.exists(p) else SRC_NET)


def boundary_edges(net):
    """(entries, exits) by connection-graph truth (design D7).

    Entry edge: no connection leads into it -> traffic can only originate there.
    Exit edge: no connection leaves it -> traffic can only terminate there.
    NOTE: geometric dead-end heuristics misclassify two-way mid-street nodes;
    this criterion is the ground truth (see design doc Part 3.1).
    """
    entries, exits = [], []
    for e in net.getEdges():
        if not any(True for inc in e.getIncoming()):
            entries.append(e)
        if not any(True for out in e.getOutgoing()):
            exits.append(e)
    return entries, exits


def edge_bearing_in(edge) -> float:
    """Compass bearing (0=N, 90=E) of travel direction at the edge's END."""
    shape = edge.getShape()
    (x1, y1), (x2, y2) = shape[-2], shape[-1]
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360


def edge_bearing_out(edge) -> float:
    """Compass bearing of travel direction at the edge's START."""
    shape = edge.getShape()
    (x1, y1), (x2, y2) = shape[0], shape[1]
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360


DIR_TO_TURN = {  # SUMO connection dir -> movement turn (left-hand traffic)
    "s": "T",            # straight
    "l": "L", "L": "L",  # near-side left (non-crossing in left-hand traffic)
    "r": "R", "R": "R",  # crossing right
    "t": "R", "T": "R",  # u-turn crosses oncoming -> grouped with crossing right
}


def lane_end_bearing(lane) -> float:
    """Compass bearing (0=N, 90=E) of the lane's LAST shape segment."""
    shape = lane.getShape()
    (x1, y1), (x2, y2) = shape[-2], shape[-1]
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360


def geometric_turn(conn) -> str:
    """Turn classification by geometry — IDENTICAL rule to the B-scheme
    observation classifier (observations.py `_turn`): compare the END-segment
    bearing of the outgoing lane with the END-segment bearing of the incoming
    lane. The end-of-outgoing-lane captures where the movement actually heads
    once a curved edge unfolds (e.g. the St Stephen's Green gyratory corner,
    where netconvert labels the main continuation dir='s' although the
    movement bends left — caught by user review 2026-06-11).

    Left-hand traffic: counterclockwise = L (near-side), clockwise = R
    (crossing). U-turns (|delta|>135) fold into L to stay aligned with the
    observation classifier.
    """
    d = (lane_end_bearing(conn.getToLane()) - lane_end_bearing(conn.getFromLane())
         + 180.0) % 360.0 - 180.0  # compass: positive = clockwise = right
    if -45.0 < d <= 45.0:
        return "T"
    if 45.0 < d <= 135.0:
        return "R"
    if -135.0 < d <= -45.0:
        return "L"
    return "L"  # U-turn folded into L (matches observations.py)


def tls_movements(net, tls_id: str):
    """Movement table for one TLS.

    Returns dict with:
      links:   {tl_index: [ {junction, junction_index, from_edge, from_lane,
                to_edge, to_lane, dir, approach (N/E/S/W), turn (L/T/R)}, ...]}
               (sparse: the source net has gap indices from stripped
               pedestrian links; one index can carry several connections)
      n_links: max tl index + 1
      arms:    {approach_sector: {edges: [...], bearing: deg}}
    Approach sector = compass sector of the approach ORIGIN bearing
    (= into-junction bearing + 180), with two refinements (design 1.1/D5):
      * incoming edges whose bearings differ < 30 deg merge into one arm
      * sector collisions resolved by relocating the arm nearest an
        adjacent EMPTY sector into that sector
    """
    tls = net.getTLS(tls_id)
    conns = tls.getConnections()  # (fromLane, toLane, tlIndex)
    # recover rich Connection objects via the nodes; NOTE the source net has
    # GAPS in TL link indices (pedestrian links were stripped but indices kept)
    # and one index may carry several connections — keep a list per index.
    rich = defaultdict(list)
    nodes = set()
    for fl, _tl_lane, _idx in conns:
        nodes.add(fl.getEdge().getToNode())
    for node in nodes:
        for c in node.getConnections():
            if c.getTLSID() == tls_id:
                rich[c.getTLLinkIndex()].append((c, node))
    n_links = max(rich) + 1

    # --- arms: group incoming edges by bearing (<30 deg apart = same arm)
    in_edges = {}
    for idx, pairs in rich.items():
        for c, node in pairs:
            e = c.getFrom()
            in_edges[e.getID()] = (e, (edge_bearing_in(e) + 180.0) % 360.0)
    arms = []  # [ [ (edge_id, origin_bearing), ...], ... ]
    for eid, (e, b) in sorted(in_edges.items(), key=lambda kv: kv[1][1]):
        placed = False
        for arm in arms:
            db = abs((b - arm[0][1] + 180) % 360 - 180)
            if db < 30:
                arm.append((eid, b))
                placed = True
                break
        if not placed:
            arms.append([(eid, b)])

    def mean_bearing(arm):
        sx = sum(math.sin(math.radians(b)) for _, b in arm)
        sy = sum(math.cos(math.radians(b)) for _, b in arm)
        return math.degrees(math.atan2(sx, sy)) % 360

    def sector_of(bearing):
        return int(((bearing + 45.0) % 360.0) // 90.0)

    arm_sector = {}
    occupied = defaultdict(list)
    for ai, arm in enumerate(arms):
        s = sector_of(mean_bearing(arm))
        arm_sector[ai] = s
        occupied[s].append(ai)

    # --- collision tie-break (D5)
    for s, members in sorted(occupied.items()):
        while len(members) > 1:
            best = None  # (angular cost, arm, target sector)
            for ai in members:
                b = mean_bearing(arms[ai])
                for ds in (-1, +1):
                    tgt = (s + ds) % 4
                    if occupied[tgt]:
                        continue
                    center = tgt * 90.0
                    cost = abs((b - center + 180) % 360 - 180)
                    if best is None or cost < best[0]:
                        best = (cost, ai, tgt)
            if best is None:
                raise RuntimeError(f"{tls_id}: unresolvable sector collision in {SECTORS[s]}")
            _, ai, tgt = best
            members.remove(ai)
            occupied[tgt].append(ai)
            arm_sector[ai] = tgt

    edge_to_sector = {}
    arm_info = {}
    for ai, arm in enumerate(arms):
        sec = SECTORS[arm_sector[ai]]
        arm_info[sec] = {"edges": [eid for eid, _ in arm], "bearing": round(mean_bearing(arm), 1)}
        for eid, _ in arm:
            edge_to_sector[eid] = sec

    links = {}
    for idx in sorted(rich):
        links[idx] = [{
            "tl_index": idx,
            "junction": node.getID(),
            "junction_index": c.getJunctionIndex(),
            "from_edge": c.getFrom().getID(),
            "from_lane": c.getFromLane().getIndex(),
            "to_edge": c.getTo().getID(),
            "to_lane": c.getToLane().getIndex(),
            "dir": c.getDirection(),
            "dir_turn": DIR_TO_TURN[c.getDirection()],  # netconvert label (audit)
            "approach": edge_to_sector[c.getFrom().getID()],
            "turn": geometric_turn(c),  # geometry — same rule as B-scheme obs
        } for c, node in rich[idx]]
    return {"links": links, "n_links": n_links, "arms": arm_info,
            "nodes": {n.getID(): n for n in nodes}}
