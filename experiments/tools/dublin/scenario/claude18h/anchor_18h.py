"""任务 1: SCATS 18 点路口总量 (在 NET18B 的 18 个 RL 路口上重新匹配站点) + 旧 18h 小汽车路由的逐转向先验。
匹配与可信规则沿用 tools/dublin/scats_pipeline.py (≤130 m, 日总量 ≥2000, 检测器在线率, 非 PED, 非间歇站)。"""
import collections, csv, json, xml.etree.ElementTree as ET
import sumolib
import common18 as C
import common as DC   # tools/dublin/common.py

HOUR = 18; DATE = "20230103"; MATCH_MAX_M = 130.0; DAILY_ALIVE_MIN = 2000.0
OLD18 = C.NETS / "weekday_18h_15_southclosed"

def scats():
    sites = {}
    for row in csv.DictReader(open(DC.SCATS_SITES_CSV)):
        try: sites[row["SiteID"]] = (float(row["Lat"]), float(row["Long"]), row["Site_Description_Cap"])
        except (ValueError, KeyError): continue
    hourly = collections.defaultdict(lambda: [0.0] * 24); det_all = collections.defaultdict(set); det_live = collections.defaultdict(set)
    det_hours = collections.defaultdict(lambda: collections.defaultdict(dict))
    rd = csv.reader(open(DC.SCATS_CSV)); next(rd)
    for row in rd:
        ts = row[0]
        if ts[:8] != DATE: continue
        v = float(row[4]) if row[4] else 0.0; h = int(ts[8:10])
        hourly[row[2]][h] += v; det_all[row[2]].add(row[3])
        if v > 0: det_live[row[2]].add(row[3])
        det_hours[row[2]][row[3]][h] = v
    flaky = set()
    for site, dets in det_hours.items():
        for det, hh in dets.items():
            if sum(hh.values()) >= 100.0 and sum(1 for h in range(7, 21) if hh.get(h, 0.0) == 0.0) >= 2:
                flaky.add(site); break
    return sites, hourly, det_all, det_live, flaky

def movement_counts(meta, files):
    mv = {}
    for tid, t in meta.items():
        for g in t["links"].values():
            for c in g: mv[(c["from_edge"], c["to_edge"])] = (tid, c["approach"] + "-" + c["turn"])
    cnt = collections.Counter(); n = 0
    for f in files:
        for v in ET.parse(f).getroot().iter("vehicle"):
            r = v.find("route"); es = r.get("edges").split(); n += 1
            for a, b in zip(es, es[1:]):
                if (a, b) in mv: cnt[mv[(a, b)]] += 1
    return cnt, n

def main():
    C.check_hashes(); meta = C.load_meta(); net = sumolib.net.readNet(str(C.NET))
    sites, hourly, det_all, det_live, flaky = scats()
    anchors = {}
    for tid in sorted(meta):
        x, y = net.getNode(tid).getCoord(); lat, lon = DC.utm_to_latlon(x, y)
        best, bestd = min(((sid, DC.latlon_dist_m(lat, lon, s[0], s[1])) for sid, s in sites.items()), key=lambda z: z[1])
        hh = hourly.get(best, [0.0] * 24); daily = sum(hh); live, total = len(det_live.get(best, ())), len(det_all.get(best, ()))
        live_ok = live >= 4 or (total > 0 and live / total >= 0.5)
        trusted = bestd <= MATCH_MAX_M and daily >= DAILY_ALIVE_MIN and live_ok and "PED" not in sites[best][2].upper() and best not in flaky
        anchors[tid] = dict(site=best, site_name=sites[best][2].strip(), match_m=round(bestd, 1), daily=round(daily), total_18h=round(hh[HOUR]),
                            detectors=f"{live}/{total}", trusted=trusted)
    cars, ncar = movement_counts(meta, [OLD18 / "car_weekday_18h.rou.xml"])
    buses, nbus = movement_counts(meta, [OLD18 / "bus_weekday_18h.rou.xml"])
    prior = {f"{t}|{m}": dict(car=cars[(t, m)], bus=buses[(t, m)]) for (t, m) in sorted(set(cars) | set(buses))}
    for tid in anchors:
        anchors[tid]["old18h_car"] = sum(v for (t, m), v in cars.items() if t == tid)
        anchors[tid]["old18h_bus"] = sum(v for (t, m), v in buses.items() if t == tid)
    C.dump(dict(hour=HOUR, date=DATE, anchors=anchors, prior_old18h=prior, n_old_cars=ncar, n_old_buses=nbus,
                note="prior = 旧 18h 真实小汽车路由 car_weekday_18h (不含人工增量 added_flows) 与 GTFS 公交在 NET18B 各转向上的过境数"), "anchors_18h.json")
    print(f"{'路口':16s} {'站点':5s} {'距离':>6s} {'18点':>5s} 可信  旧18h小汽车/公交")
    for tid, a in anchors.items():
        print(f"{tid.replace('cluster_','c')[:16]:16s} {a['site']:5s} {a['match_m']:6.1f} {a['total_18h']:5d} {str(a['trusted']):5s} {a['old18h_car']}/{a['old18h_bus']}  {a['site_name'][:30]}")

if __name__ == "__main__":
    main()
