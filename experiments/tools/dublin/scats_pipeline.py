"""M1 — SCATS extraction: TLS<->site matching + hourly junction-total targets.

For every TLS junction: nearest SCATS site (UTM-inverse lat/lon matching),
hourly site totals (sum over detectors) for the clean weekday 2023-01-03.
A site is TRUSTED for calibration iff match distance <= 130 m AND its daily
total passes a liveness threshold AND its detectors are sufficiently live
(live >= 4 or live/total >= 0.5). Partial-coverage sites (e.g. 668: 1/24
detectors, 662: 2/24) measure a subset of approaches, so their "junction
total" poisons calibration targets — data-driven rule, no hand blacklist.

Output: nets/dublin/calibration/scats_targets.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

WEEKDAY_DATE = "20230103"  # the single clean weekday in the local extract
MATCH_MAX_M = 130.0
DAILY_ALIVE_MIN = 2000.0   # veh/day; real junctions here do >5k, dead/partial ~<1k


def main():
    os.makedirs(common.CALIB_DIR, exist_ok=True)
    net = common.load_net()

    sites = {}
    with open(common.SCATS_SITES_CSV) as f:
        for row in csv.DictReader(f):
            try:
                sites[row["SiteID"]] = (float(row["Lat"]), float(row["Long"]),
                                        row["Site_Description_Cap"])
            except (ValueError, KeyError):
                continue

    # hourly totals + detector liveness/flakiness per site for the chosen date
    hourly = defaultdict(lambda: [0.0] * 24)
    det_all = defaultdict(set)
    det_live = defaultdict(set)
    det_hours = defaultdict(lambda: defaultdict(dict))  # site -> det -> hour -> v
    with open(common.SCATS_CSV) as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            ts = row[0]
            if ts[:8] != WEEKDAY_DATE:
                continue
            v = float(row[4]) if row[4] else 0.0
            hourly[row[2]][int(ts[8:10])] += v
            det_all[row[2]].add(row[3])
            if v > 0:
                det_live[row[2]].add(row[3])
            det_hours[row[2]][row[3]][int(ts[8:10])] = v

    # flaky site: a load-bearing detector (>=100 veh/day) with >=2 zero hours
    # inside 07-20h reports intermittently (e.g. site 911 det 2 dies at 18:00)
    # -> its hourly "junction total" is not comparable across hours
    flaky_sites = set()
    for site, dets in det_hours.items():
        for det, hh in dets.items():
            if sum(hh.values()) < 100.0:
                continue
            zeros = sum(1 for h in range(7, 21) if hh.get(h, 0.0) == 0.0)
            if zeros >= 2:
                flaky_sites.add(site)
                break

    out = {"date": WEEKDAY_DATE, "tls": {}}
    print(f"{'TLS':>44s} {'site':>5s} {'dist':>5s} {'daily':>7s} {'11h':>5s} {'18h':>5s} trusted")
    for tls_id in sorted(t.getID() for t in net.getTrafficLights()):
        node = net.getNode(tls_id)
        x, y = node.getCoord()
        lat, lon = common.utm_to_latlon(x, y)
        best, bestd = None, 1e18
        for sid, (sla, slo, _name) in sites.items():
            d = common.latlon_dist_m(lat, lon, sla, slo)
            if d < bestd:
                best, bestd = sid, d
        hh = hourly.get(best, [0.0] * 24)
        daily = sum(hh)
        live, total = len(det_live.get(best, ())), len(det_all.get(best, ()))
        live_ok = live >= 4 or (total > 0 and live / total >= 0.5)
        # mid-block pedestrian signals ("... PED") measure one street only,
        # never a junction total — they are not calibration targets
        is_ped = "PED" in sites[best][2].upper()
        trusted = (bestd <= MATCH_MAX_M and daily >= DAILY_ALIVE_MIN and live_ok
                   and not is_ped and best not in flaky_sites)
        out["tls"][tls_id] = {
            "site": best, "site_name": sites[best][2], "dist_m": round(bestd, 1),
            "daily_total": round(daily), "hourly": [round(v) for v in hh],
            "detectors_live": live, "detectors_total": total,
            "trusted": trusted,
        }
        print(f"{tls_id[:44]:>44s} {best:>5s} {bestd:5.0f} {daily:7.0f} "
              f"{hh[11]:5.0f} {hh[18]:5.0f} {live:2d}/{total:<2d} {trusted}")

    # ---- street-level sites: mid-block "PED" signals are invalid as junction
    # totals but are VALID single-street cross-section counts — reuse them as
    # street-flow calibration rows (fixes e.g. Dawson St starving at zero
    # while site 919 measures ~257 veh/h).
    xmin, ymin, xmax, ymax = net.getBoundary()
    street_sites = {}
    for sid, (sla, slo, name) in sites.items():
        if "PED" not in name.upper() or sid not in hourly:
            continue
        x, y = common.latlon_to_net(sla, slo)
        if not (xmin - 30 <= x <= xmax + 30 and ymin - 30 <= y <= ymax + 30):
            continue
        if sum(hourly[sid]) < 500:
            continue
        street = name.split()[0].rstrip("/").capitalize()  # "DAWSON ST.." -> "Dawson"
        edges = []
        for e in net.getEdges():
            if street.lower() not in (e.getName() or "").lower():
                continue
            from sumolib import geomhelper
            if geomhelper.distancePointToPolygon((x, y), e.getShape()) <= 25.0:
                edges.append(e.getID())
        if edges:
            street_sites[sid] = {"name": name, "street": street, "edges": edges,
                                 "hourly": [round(v) for v in hourly[sid]]}
            print(f"street site {sid} ({name[:40]}): edges {edges}")
    out["street_sites"] = street_sites

    n_tr = sum(1 for v in out["tls"].values() if v["trusted"])
    print(f"\ntrusted calibration targets: {n_tr}/{len(out['tls'])} "
          f"+ {len(street_sites)} street-level rows")
    path = os.path.join(common.CALIB_DIR, "scats_targets.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
