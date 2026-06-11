"""Bus pipeline — TFI GTFS -> real bus routes/stops/schedule inside the subnet.

Steps (design Part 4.1):
  1. pick ONE concrete Tuesday inside the feed window (calendar + exceptions;
     avoids double-counting overlapping seasonal timetables)
  2. bus trips (route_type=3) serving stops within the padded subnet bbox
     whose first in-bbox arrival falls in the target hour
  3. map-match each distinct shape to a connected edge path (grid-indexed
     nearest-edge with heading gate; gaps repaired via vClass-aware shortest
     path; trips whose shape cannot be matched are dropped and logged)
  4. emit bus_weekday_<H>h.rou.xml + per-TLS crossing counts
     (calibration/bus_crossings_<H>h.json, used as car = SCATS - bus)
  NOTE (user decisions):
  * no bus stops / dwell — ALL vehicles enter the net and drive through
    without stopping; GTFS stop_times only time each bus's entry.
  * buses are LANE-LOCKED like cars (no mid-route lane change = no bias):
    each bus runs the longest contiguous sub-path of its matched route that
    admits a full lane->lane chain (same salvage philosophy as the shape
    matching); departLane pins the chain, vType disables all lane changing.

Usage: python3 gtfs_bus.py [hour]   (default 11)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from gen_routes import lane_chain_dp, lane_edge_conns  # noqa: E402

import sumolib  # noqa: E402
from sumolib import geomhelper  # noqa: E402

PAD = 150.0
CAND_DIST = 20.0
CAND_HEAD = 60.0
REPAIR_MAX_EDGES = 8
import random as _random
_BUS_RNG = _random.Random(4242)  # lane-chain spreading for buses
BUS_SPEED = 6.0      # m/s, rough in-city average for depart-time back-projection


def gtfs_time(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def pick_tuesday(gtfs_dir):
    """The concrete Tuesday with the most active services, honouring exceptions."""
    cal = list(csv.DictReader(open(os.path.join(gtfs_dir, "calendar.txt"))))
    lo = min(r["start_date"] for r in cal)
    hi = max(r["end_date"] for r in cal)
    d = date(int(lo[:4]), int(lo[4:6]), int(lo[6:]))
    d += timedelta(days=(1 - d.weekday()) % 7)
    end = date(int(hi[:4]), int(hi[4:6]), int(hi[6:]))
    exceptions = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(gtfs_dir, "calendar_dates.txt"))):
        exceptions[r["date"]].append((r["service_id"], r["exception_type"]))
    best_key, best_active = None, set()
    while d <= end:
        key = d.strftime("%Y%m%d")
        active = {r["service_id"] for r in cal
                  if r["tuesday"] == "1" and r["start_date"] <= key <= r["end_date"]}
        for sid, etype in exceptions.get(key, ()):
            if etype == "1":
                active.add(sid)
            else:
                active.discard(sid)
        if len(active) > len(best_active):
            best_key, best_active = key, active
        d += timedelta(days=7)
    return best_key, best_active


class EdgeIndex:
    """50 m grid over edge shapes for nearest-edge queries with heading gate."""

    def __init__(self, net, cell=50.0):
        self.cell = cell
        self.grid = defaultdict(list)
        self.shapes = {}
        for e in net.getEdges():
            shape = e.getShape()
            self.shapes[e] = shape
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            for cx in range(int(min(xs) // cell), int(max(xs) // cell) + 1):
                for cy in range(int(min(ys) // cell), int(max(ys) // cell) + 1):
                    self.grid[(cx, cy)].append(e)

    def candidates(self, x, y):
        cx, cy = int(x // self.cell), int(y // self.cell)
        seen = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for e in self.grid.get((cx + dx, cy + dy), ()):
                    seen.add(e)
        return seen

    def best(self, x, y, bearing):
        best, score = None, 1e18
        for e in self.candidates(x, y):
            shape = self.shapes[e]
            d, segb = 1e18, 0.0
            for (x1, y1), (x2, y2) in zip(shape, shape[1:]):
                dd = geomhelper.distancePointToLine((x, y), (x1, y1), (x2, y2))
                if dd < d:
                    d = dd
                    segb = math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360
            if d > CAND_DIST:
                continue
            db = abs((bearing - segb + 180) % 360 - 180)
            if db > CAND_HEAD:
                continue
            sc = d + 0.2 * db
            if sc < score:
                best, score = e, sc
        return best


def resample(pts, step=8.0):
    """Polyline -> ~step-spaced samples with forward bearings."""
    out = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg < 1e-3:
            continue
        b = math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360
        k = max(1, int(seg // step))
        for t in range(k):
            f = t / k
            out.append((x1 + f * (x2 - x1), y1 + f * (y2 - y1), b))
    if pts:
        out.append((*pts[-1], out[-1][2] if out else 0.0))
    return out


def lane_locked_subpath(net, path_edges):
    """Longest contiguous sub-path admitting a full lane chain.

    Returns (sub_path, depart_lane) or None. Forward feasible-lane-set sweep
    yields maximal lane-lockable segments; the longest (by metres) is kept.
    """
    segments = []
    i, n = 0, len(path_edges)
    while i < n:
        edges = [net.getEdge(e) for e in path_edges[i:]]
        feas = {l.getIndex() for l in edges[0].getLanes()}
        j = i
        prev = edges[0]
        for k in range(1, len(edges)):
            table = lane_edge_conns(prev)
            nxt = set()
            for la in feas:
                nxt.update(table.get(la, {}).get(edges[k].getID(), []))
            if not nxt:
                break
            feas, prev, j = nxt, edges[k], i + k
        segments.append((i, j))
        i = j + 1 if j > i else i + 1
    best = max(segments,
               key=lambda s: sum(net.getEdge(e).getLength()
                                 for e in path_edges[s[0]:s[1] + 1]))
    sub = path_edges[best[0]:best[1] + 1]
    if len(sub) < 2 or sum(net.getEdge(e).getLength() for e in sub) < 150.0:
        return None
    lane = lane_chain_dp(net, sub, _BUS_RNG)
    if lane is None:
        return None
    return sub, lane


def match_shape(net, index, pts_xy):
    """Longest connected in-net sub-path of the shape (segment salvage).

    City-centre shapes often touch streets that are missing from the subnet
    (bus gates, contraflow bus lanes, holes at the boundary). Instead of
    failing the whole trip, the chain is broken there and the longest
    connected segment is kept — that segment IS the bus's in-net traversal.
    """
    samples = resample(pts_xy)
    segments = []
    path, cur = [], None
    for x, y, b in samples:
        e = index.best(x, y, b)
        if e is None or e is cur:
            continue
        if cur is None:
            path, cur = [e], e
            continue
        if e in (o for o in cur.getOutgoing()):
            path.append(e)
            cur = e
            continue
        sp, _cost = net.getShortestPath(cur, e, vClass="bus")
        if sp and 1 < len(sp) <= REPAIR_MAX_EDGES:
            path.extend(sp[1:])
            cur = e
        else:
            segments.append(path)
            path, cur = [e], e
    segments.append(path)
    best = max(segments, key=lambda s: sum(e.getLength() for e in s), default=None)
    if not best or len(best) < 2 or sum(e.getLength() for e in best) < 150.0:
        return None
    return best


def main():
    hour = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    h0, h1 = hour * 3600, (hour + 1) * 3600
    out_dir = os.path.join(common.OUT_DIR, f"weekday_{hour:02d}h")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(common.CALIB_DIR, exist_ok=True)

    net = common.load_net_demand()
    xmin, ymin, xmax, ymax = net.getBoundary()
    bx0, by0, bx1, by1 = xmin - PAD, ymin - PAD, xmax + PAD, ymax + PAD
    index = EdgeIndex(net)
    tls_ids = {t.getID() for t in net.getTrafficLights()}

    day, services = pick_tuesday(common.GTFS_DIR)
    print(f"service day {day}: {len(services)} active weekday services")

    bus_routes = {r["route_id"]: r["route_short_name"]
                  for r in csv.DictReader(open(os.path.join(common.GTFS_DIR, "routes.txt")))
                  if r["route_type"] == "3"}

    stops_xy = {}
    for r in csv.DictReader(open(os.path.join(common.GTFS_DIR, "stops.txt"))):
        try:
            x, y = common.latlon_to_net(float(r["stop_lat"]), float(r["stop_lon"]))
        except ValueError:
            continue
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            stops_xy[r["stop_id"]] = (x, y, r["stop_name"])
    print(f"stops in padded bbox: {len(stops_xy)}")

    trips = {}
    for r in csv.DictReader(open(os.path.join(common.GTFS_DIR, "trips.txt"))):
        if r["route_id"] in bus_routes and r["service_id"] in services:
            trips[r["trip_id"]] = (r["route_id"], r["shape_id"])
    print(f"candidate bus trips on {day}: {len(trips)}")

    trip_hits = defaultdict(list)  # trip -> [(arr, stop_id, seq)]
    with open(os.path.join(common.GTFS_DIR, "stop_times.txt")) as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["trip_id"] in trips and r["stop_id"] in stops_xy:
                trip_hits[r["trip_id"]].append(
                    (gtfs_time(r["arrival_time"]), r["stop_id"], int(r["stop_sequence"])))

    window = {}
    for t, hits in trip_hits.items():
        hits.sort(key=lambda h: h[2])
        arr = hits[0][0]
        # GTFS encodes post-midnight trips as >24:00:00 of the PREVIOUS service
        # day — a 02:xx bus appears as 26:xx. Accept both encodings (weekday
        # service sets of adjacent days are near-identical).
        if h0 <= arr < h1 or h0 + 86400 <= arr < h1 + 86400:
            window[t] = [(a - 86400 if a >= 86400 else a, sid, sq) for a, sid, sq in hits]
    print(f"trips entering subnet in [{hour}:00,{hour + 1}:00): {len(window)}")

    need_shapes = {trips[t][1] for t in window}
    shape_pts = defaultdict(list)
    with open(os.path.join(common.GTFS_DIR, "shapes.txt")) as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r["shape_id"] in need_shapes:
                shape_pts[r["shape_id"]].append(
                    (int(r["shape_pt_sequence"]), float(r["shape_pt_lat"]), float(r["shape_pt_lon"])))

    shape_path = {}
    fail = []
    for sid, pts in shape_pts.items():
        pts.sort()
        xy = [common.latlon_to_net(la, lo) for _, la, lo in pts]
        xy = [(x, y) for x, y in xy if bx0 <= x <= bx1 and by0 <= y <= by1]
        if len(xy) < 2:
            fail.append(sid)
            continue
        p = match_shape(net, index, xy)
        if p is None:
            fail.append(sid)
        else:
            shape_path[sid] = p
    print(f"shapes matched: {len(shape_path)}/{len(need_shapes)}  (failed: {len(fail)})")

    # --- vehicles (NO stop dwell — user decision: buses are through traffic
    # like cars; stop_times are used ONLY to time the entry into the subnet)
    def nearest_path_edge_rank(path, edge_rank, stop_id):
        x, y, _name = stops_xy[stop_id]
        best = None
        for e in path:
            d = geomhelper.distancePointToPolygon((x, y), e.getShape())
            if best is None or d < best[0]:
                best = (d, e)
        if best is None or best[0] > 30.0 or best[1].getID() not in edge_rank:
            return None
        return edge_rank[best[1].getID()]

    vehicles = []
    n_unlockable = 0
    crossings = defaultdict(int)
    for t, hits in sorted(window.items(), key=lambda kv: kv[1][0][0]):
        rid, sid = trips[t]
        path = shape_path.get(sid)
        if path is None:
            continue
        locked = lane_locked_subpath(net, [e.getID() for e in path])
        if locked is None:
            n_unlockable += 1
            continue
        path_edges, depart_lane = locked
        path = [net.getEdge(eid) for eid in path_edges]
        edge_rank = {eid: k for k, eid in enumerate(path_edges)}
        # depart: back-project from first in-bbox stop arrival to path start
        first_arr = hits[0][0]
        rank = nearest_path_edge_rank(path, edge_rank, hits[0][1])
        lead = sum(net.getEdge(eid).getLength() for eid in path_edges[:rank]) if rank else 0.0
        depart = min(max(0.0, first_arr - h0 - lead / BUS_SPEED), 3599.0)
        vehicles.append((depart, f"bus_{bus_routes[rid]}_{t[-14:]}", path_edges, depart_lane))
        for e in path[:-1]:
            n = e.getToNode().getID()
            if n in tls_ids:
                crossings[n] += 1

    rou_path = os.path.join(out_dir, f"bus_weekday_{hour:02d}h.rou.xml")
    with open(rou_path, "w") as f:
        f.write("<routes>\n")
        f.write('  <vType id="bus" vClass="bus" length="12.0" color="0,0,1" '
                'lcStrategic="0" lcCooperative="0" lcSpeedGain="0" '
                'lcKeepRight="0"/>\n')
        for depart, vid, path_edges, depart_lane in sorted(vehicles):
            f.write(f'  <vehicle id="{escape(vid)}" type="bus" depart="{depart:.1f}" '
                    f'departLane="{depart_lane}" departSpeed="max">\n')
            f.write(f'    <route edges="{" ".join(path_edges)}"/>\n')
            f.write("  </vehicle>\n")
        f.write("</routes>\n")

    cross_path = os.path.join(common.CALIB_DIR, f"bus_crossings_{hour:02d}h.json")
    with open(cross_path, "w") as f:
        json.dump({"hour": hour, "service_day": day, "n_buses": len(vehicles),
                   "crossings": dict(sorted(crossings.items()))}, f, indent=1)

    print(f"buses emitted: {len(vehicles)} lane-locked "
          f"(no stops; {n_unlockable} dropped — no lockable sub-path >=150m)")
    print(f"TLS crossing counts (top): "
          f"{sorted(crossings.items(), key=lambda kv: -kv[1])[:6]}")
    print("wrote", rou_path)
    print("wrote", cross_path)


if __name__ == "__main__":
    main()
