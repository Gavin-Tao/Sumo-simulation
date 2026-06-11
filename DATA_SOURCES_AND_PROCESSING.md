# Dublin RL Traffic Signal Control - Data Sources & Processing Logic

This document describes all data sources, processing logic, and output files for the Dublin traffic simulation project.

## Project Overview

- **Goal**: Reinforcement Learning traffic signal control with multi-class priority (car / bus / ambulance) on Dublin City SUMO network
- **Affiliation**: PhD research at Trinity College Dublin (supervised by Prof. Ivana Dusparic)
- **Simulation target**: Full Dublin City network (~770 RL-controlled signalised intersections) or representative sub-network
- **Time scenarios**: 3 time-of-day windows per supervisor guidance:
  - Free-flow (00:00-04:00)
  - Normal/low traffic (10:00-12:00)
  - Evening peak (17:00-19:00)
- **Day types**: Weekday + Weekend

---

## 1. Dublin Road Network

### Source
- **OpenStreetMap (OSM)** raw data
- Converted using SUMO `netconvert`

### Output Files (already generated)
| File | Location | Description |
|---|---|---|
| `dublin_safe_tls.net.xml` | `/home/claude/` | 938 junctions with traffic lights |
| `dublin_8action.net.xml` | `/home/claude/` | 777 RL-controllable junctions (simplified) |
| `dcc_8action_final.net.xml` | `/home/claude/` | DCC region version |
| `dcc_lefthand.net.xml` | `/home/claude/` | Left-hand traffic corrected version |

---

## 2. Bus Data (★★★★★ Real)

### Source
- **Transport for Ireland (TFI) GTFS** schedule data
- Official: https://www.transportforireland.ie/transitData/PT_Data.html
- Files in `/mnt/user-data/uploads/`:
  - `agency.txt`
  - `calendar.txt` (service calendar)
  - `calendar_dates.txt` (holiday exceptions)
  - `routes.txt` (391 bus + 19 DART + 2 Luas)
  - `shapes.txt` (route geometry)
  - `stop_times.txt` (6M rows, every stop arrival time)
  - `stops.txt` (all stops with coordinates)
  - `trips.txt` (153,824 trips)
  - `feed_info.txt`

### Processing Logic
```python
# Pseudocode
1. Filter route_type:
   - route_type=3: Bus (391 routes) ← KEEP
   - route_type=0: Luas (EXCLUDE - tram)
   - route_type=2: DART (EXCLUDE - rail)

2. Junction-stop association:
   - For each SUMO junction, find all stops within 200m radius
   - One stop can belong to multiple junctions' coverage

3. Time filtering by service:
   - calendar.txt defines which services run on weekday/saturday/sunday
   - A single service can run on multiple day_types

4. Trip deduplication (CRITICAL):
   - Count each trip_id AT MOST ONCE per (junction, hour)
   - Same bus passing multiple stops within 200m is counted once

5. Output: bus count per junction per hour per day_type
```

### Output Files
| File | Description |
|---|---|
| `/mnt/user-data/outputs/d_junctions_bus_by_time_daytype.json` | 6 D-junctions × 24h × (weekday/saturday/sunday) |
| `/mnt/user-data/outputs/d_junctions_bus_only.json` | Pure bus (excluding Luas/DART) |
| `/home/claude/gtfs_dcc_stops.json` | Stops metadata |
| `/home/claude/gtfs_route_to_stops.json` | Route → stops mapping |
| `/home/claude/gtfs_trip_to_route.json` | Trip → route mapping |

### Reliability
- ★★★★★ Real GTFS schedule data
- ~±10% error (schedule vs actual operations)

---

## 3. Car Data (★★★ Estimated)

### Source
- **DCC SCATS Detector Volumes (January 2023)**
- File: `/mnt/user-data/uploads/SCATSJanuary2023.csv` (357MB, 10.4M rows)
- Official: https://data.smartdublin.ie/dataset/scats-detector-volume

- **DCC Traffic Signals metadata**:
  - `/mnt/user-data/uploads/dcc_traffic_signals_20221130.csv` (825 SCATS sites with lat/lon)
  - `/mnt/user-data/uploads/dcc-traffic-scats-signals-google-maps-1.csv` (backup, 1205 sites)

### Processing Logic
```python
# Step 1: SCATS site → SUMO junction spatial mapping
- For each SUMO junction, find nearest SCATS site
- Tier 1 (≤100m): Direct coverage, 146 junctions
- Tier 2 (100-500m, ≥3 sites): IDW interpolation, 414 junctions
- Tier 3 (>500m): Nearest site with distance decay, 217 junctions

# Step 2: SCATS data aggregation
- Sum all detectors per site per hour
- Average across weekdays separately from weekends

# Step 3: Car estimation via bus_ratio method (UNIFIED METHOD)
junction_class = {
    'residential':       {'bus_ratio': 0.08},
    'minor_arterial':    {'bus_ratio': 0.10},
    'arterial':          {'bus_ratio': 0.13},
    'bus_corridor':      {'bus_ratio': 0.25},
    'central_bus_hub':   {'bus_ratio': 0.45-0.50},
}

# For each junction:
total_flow = bus_GTFS / bus_ratio  # All approaches combined
car = total_flow - bus

# Step 4: Junction type auto-classification
- Based on bus / total ratio from SCATS-covered junctions
- 5 categories cover 100% of 777 RL junctions
```

### Output Files
| File | Description |
|---|---|
| `/mnt/user-data/outputs/dublin_full_traffic_config.json` | All 777 RL junctions, 8am + 17pm full config |
| `/mnt/user-data/outputs/d_junctions_scats_by_time_daytype.json` | 6 D-junctions × 24h × weekday/weekend SCATS totals |
| `/home/claude/d_to_scats_v2.json` | 6 D-junctions → SCATS site mapping |

### Reliability
- ★★★ bus_ratio is empirical (Dublin Canal Cordon Count 2022 calibrated)
- ±30-50% error
- bus_ratio assumption needs sensitivity analysis (±30%)

---

## 4. Ambulance Data (★★★ Real total, estimated path)

### Source
- **Dublin Fire Brigade 2023 Annual Ambulance Activity Log**
- File: `/mnt/user-data/uploads/2023-open-data-dfb-ambulance.csv`
- 80,916 real incidents, average 222/day
- Official: https://data.smartdublin.ie/dataset/fire-brigade-and-ambulance-call-outs

- **Data fields**: ID, Date, Station Name, criticality (E/D/C/B/A/O), TOC, ORD, MOB, IA, LS, AH, MAV, CD

### Processing Logic
```python
# Step 1: Extract real distributions from DFB 2023
- 24h hourly distribution (weekday vs weekend)
- Criticality distribution:
  Echo: 2.68%, Delta: 47.14%, Charlie: 15.47%
  Bravo: 14.45%, Alpha: 17.81%, Omega: 2.45%
- Per-station incident counts

# Step 2: Per-junction exposure coefficient
def exposure(d_station, d_hospital):
    # d in meters
    station_factor = max(0.005, 0.05 * exp(-d_station / 800))
    hospital_factor = max(0.003, 0.03 * exp(-d_hospital / 800))
    return station_factor + hospital_factor

# Step 3: Per-junction ambulance/hr
ambulance_per_hr = city_wide_incidents_per_hr × exposure_coef
# city_wide_incidents_per_hr from DFB real data (4.3-12.2/hr depending on hour)
```

### Hardcoded Reference Points

**8 Hospitals (lat, lon)**:
```python
hospitals = {
    'Coombe Maternity': (53.3349, -6.2840),
    'St James Hospital': (53.3403, -6.2949),
    'Mater Hospital': (53.3603, -6.2674),
    'Rotunda Maternity': (53.3517, -6.2618),
    'Beaumont Hospital': (53.3858, -6.2284),
    'Connolly Hospital': (53.3823, -6.3825),
    "St Vincent's": (53.3175, -6.2155),
    'Tallaght Hospital': (53.2871, -6.3712),
}
```

**12 DFB Stations (lat, lon)**:
```python
station_locs = {
    'Tallaght Fire Station': (53.2884, -6.3640),
    'Tara Street Fire Station': (53.3458, -6.2547),
    'Kilbarrack Fire Station': (53.3878, -6.1500),
    'Phibsborough Fire Station': (53.3603, -6.2730),
    'Dolphins Barn Fire Station': (53.3322, -6.2920),
    'Finglas Fire Station': (53.3865, -6.2956),
    'Rathfarnharm Fire Station': (53.2960, -6.2810),
    'Blanchardstown Fire Station': (53.3850, -6.3760),
    'North Strand Fire Station': (53.3573, -6.2456),
    'Swords Fire Station': (53.4585, -6.2225),
    'Donnybrook Fire Station': (53.3215, -6.2316),
}
```

### Output Files
| File | Description |
|---|---|
| `/mnt/user-data/outputs/dfb_2023_real_distribution.json` | Real 24h × weekday/weekend distribution, criticality, station stats |
| `/mnt/user-data/outputs/d_junctions_ambulance_real.json` | 6 D-junctions × 24h × weekday/weekend ambulance/hr |

### Limitations
- ★★★ DFB total is real (80,916 actual incidents)
- ✗ No incident addresses in data (cannot determine actual paths)
- ✗ Path through specific junction = estimation
- ±50% error per-junction

---

## 5. Final Integrated Output

### File: `/mnt/user-data/outputs/d_junctions_full_traffic.json` + `.csv`

**Structure**: 6 D-junctions × 24h × (weekday/weekend) × (car/bus/ambulance/total) = 288 data points

**Schema**:
```json
{
  "D1": {
    "name": "Sandymount",
    "type": "residential",
    "bus_ratio_assumed": 0.10,
    "nearest_station": "Donnybrook Fire Station",
    "station_distance_m": 1576,
    "nearest_hospital": "St Vincent's",
    "hospital_distance_m": 1619,
    "unit": "vehicles per hour (all approaches combined)",
    "method": "total = bus / 0.10, car = total - bus",
    "weekday": {
      "0": {"car": ..., "bus": ..., "ambulance": ..., "total": ...},
      ...
      "23": {...}
    },
    "weekend": {...}
  },
  "D2": {...}, "D3": {...}, "D4": {...}, "D5": {...}, "D6": {...}
}
```

**CSV columns**:
```
Junction, Name, Type, BusRatio, DayType, Hour,
Car, Bus, Ambulance, Total, Unit
```

**Unit**: `vehicles per hour (all approaches combined)` — total flow across all incoming approaches

---

## 6. The 6 D-Junctions Mapping

| Label | Name | SUMO jid | SCATS site | Type | bus_ratio |
|---|---|---|---|---|---:|
| D1 | Sandymount | 1420144368 | 959 | residential | 0.10 |
| D2 | Portobello | 389678 | 268 | bus_corridor | 0.25 |
| D3 | O'Connell | 389281 | 1 | central_bus_hub | 0.45 |
| D4 | Baggot St | 29400040 | 882 | arterial | 0.13 |
| D5 | Coombe | 32336040 | 962 | bus_corridor | 0.25 |
| D6 | Burgh Quay | 9101555 | 908 | central_bus_hub | 0.50 |

---

## 7. Data Quality Summary

| Data | Source | Reliability | Error | Notes |
|---|---|---|---|---|
| Road network | OSM | ★★★★★ | - | Truth |
| Bus volumes | GTFS | ★★★★★ | ±10% | Real schedule |
| Car volumes | bus_ratio reverse | ★★★ | ±30-50% | Calibrated to Canal Cordon |
| Ambulance total | DFB 2023 | ★★★★ | ±10% | Real |
| Ambulance per-junction | Distance decay | ★★★ | ±50% | No path data |
| Lane allocation | OSM + SUMO default | ★★★ | - | Auto by SUMO |
| Turning ratios | Literature default 50/25/25 | ★★ | - | No data for 775 junctions |
| OD patterns | randomTrips | ★★ | - | No real OD matrix |

---

## 8. Pending Data (NTA correspondence in progress)

**Requested from NTA**:
- ERM (Eastern Regional Model) SATURN assignments → would provide junction-level turning movements + OD matrices
- IDASO mytrafficcounts.com historical surveys → real lane-level counts
- DCC 2025 junction surveys

**Status**: Awaiting NTA response (David Conlon, Senior Transport Modeller)

If received, these would replace:
- bus_ratio assumptions → real turning data
- randomTrips OD → real OD matrix
- ±30% car error → ±10%

---

## 9. SUMO Simulation Generation Strategy

### Bus routes
- Use SUMO `gtfs2pt.py` to convert GTFS → SUMO bus routes
- Input: GTFS files + SUMO network
- Output: `bus.rou.xml` + `busstops.xml`

### Car routes
- Use SUMO `randomTrips.py` with calibration
- Period calibrated to total flow from `dublin_full_traffic_config.json`
- Edge weights from OSM road class:
  - motorway: weight 10
  - primary: weight 5
  - residential: weight 1
- Fringe-factor 5 (95% trips from network boundary, 5% internal)

### Ambulance routes
- Custom script:
  - For each ambulance: pick station by DFB count probability
  - Pick incident location randomly within station service area
  - Route: station → incident → nearest appropriate hospital
  - vClass="emergency" for signal priority
- Generation rate per `dfb_2023_real_distribution.json` hourly pattern

### Combine
- Merge three .rou.xml files
- Sort by depart time
- Output: `dublin_combined.rou.xml`

---

## 10. Key Decisions & Rationale

1. **Whole Dublin map, not sub-map**: Supervisor (Prof. Dusparic) explicitly required full Dublin network as evidence of algorithm generality, not individual junction tests.

2. **Three time windows**: Supervisor specified midnight free-flow, around 11am normal, and 17:00-19:00 peak.

3. **bus_ratio method (not SCATS-bus)**: SCATS detector coverage varies (1-16 active detectors per site), making direct SCATS-bus subtraction unreliable. bus_ratio method gives consistent "all approaches" total flow.

4. **Exclude Luas/DART from "bus"**: GTFS route_type filtering ensures only buses (route_type=3) are counted, not trams (route_type=0) or rail (route_type=2).

5. **Trip-deduplication for bus counting**: Same bus passing multiple stops within 200m of a junction is counted ONCE per hour.
