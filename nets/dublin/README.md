# Dublin city-centre validation scenarios (St Stephen's Green / Dame St)

Generated 2026-06-11. Design: `experiments/analysis/DUBLIN_VALIDATION_DESIGN_2026-06-11.txt`
(Part 9 = as-built). Methods narrative: `experiments/analysis/DUBLIN_PIPELINE_METHODS_2026-06-11.txt`.
Pipeline scripts: `experiments/tools/dublin/`.

## Scenarios

| dir | hour | car | bus | amb | V2 (3 h cohort) | V3 junction-flow MAPE |
|---|---|---|---|---|---|---|
| `weekday_02h/` | 02:00 free-flow | 1380 | 18 | 2 | 0 teleports | 29.0 % (vol-weighted 26.5 %) |
| `weekday_11h/` | 11:00 normal | 3808 | 233 | 7 | 14 teleports | 20.9 % (22.9 %) |
| `weekday_18h/` | 18:00 peak | 5041 | 262 | 1 | 22 teleports (0.4 %) | 24.6 % (26.3 %) |

Each dir: `car/bus/amb_weekday_<H>h.rou.xml` + `dublin_weekday_<H>h.sumocfg`.
**No vehicle ever stops/dwells** (user decision): cars, buses and ambulances
enter the net and drive through. GTFS stop_times only time bus entries.

## Network files

| file | what |
|---|---|
| `dublin_8action_demoted.net.xml` | source net with 9 no-competition TLS demoted to priority junctions (`demote_uncontested.py`: kept iff two cross-arm foe links both carry ≥10 veh/h in some scenario; pedestrian-exclusion rationale) |
| `dublin_8std.net.xml` | demoted net with every remaining tlLogic rebuilt as ≤8 standard phases (left-hand FRAP: NS-str / EW-str / NS-right / EW-right / N / S / E / W), maximal-compatible green closure; fixes 5 junctions whose original programs had never-green links |
| `dublin_8std_meta.json` | per-TLS sidecar: 8-dim `mask`, `canonical` dup→canonical map, `std_to_green_index` (std action → green-phase position as sumo-rl `_build_phases` sees it), link↔movement table, arm bearings; plus `boundary.entries/exits` (connection-graph truth, 15/16) |
| `calibration/scats_targets.json` | per-TLS nearest SCATS site + hourly totals (2023-01-03) + trust flags (≤130 m, alive, detector-liveness, not "PED", not flaky) — 16/27 trusted |
| `calibration/turn_ratios_<H>h.json` | A1 turn ratios: P(e→e') ∝ lanes(e')·site_hour_volume^1.5 (hour-specific!) |
| `calibration/boundary_rates_<H>h.json` | bounded-LSQ-calibrated Poisson rate per entry edge |
| `calibration/bus_crossings_<H>h.json` | GTFS buses crossing each TLS (car target = SCATS − bus) |

## Key properties (verified)

* **18 RL-controlled signals** (9 of 27 demoted — no competing flows once
  pedestrians are excluded); action space: fixed `Discrete(8)` + static
  per-junction mask → one shared agent; valid-action counts:
  {2:×5, 3:×7, 4:×2, 6:×3, 8:×1}
* **ALL vehicles are lane-locked** (`lc* = 0`, `departLane` pinned): cars &
  ambulances prove a lane chain by DP before emission; buses run the longest
  contiguous sub-path of their real GTFS route that admits a full chain
  (0 buses dropped). Zero lane-change deadlocks by construction; remaining
  teleports are priority-side-road yields in the Grafton quarter.
* turn semantics are GEOMETRIC (same classifier as the B-scheme obs):
  netconvert `dir` labels mislabel gyratory corners (6/111 relabeled).
* boundary rates: floors max(10, 0.15·prior) — every entry road alive (A9);
  road-class caps 800/lane arterial, 300 minor lane (no Duke-St artifacts).
* calibration is **generator-consistent**: the A-matrix is Monte-Carlo
  estimated with the same walker that emits vehicles. Do NOT use closed-form
  (I−Pᵀ)⁻¹ propagation — it counts gyratory loops real vehicles never make.

## Regenerate (any hour H ∈ {02,11,18,...}, seed S)

```bash
cd /home/xiaowen/sumo-rl
python3 experiments/tools/dublin/scats_pipeline.py      # targets (hour-independent)
python3 experiments/tools/dublin/gtfs_bus.py H          # per-hour
python3 experiments/tools/dublin/calibrate_nnls.py H    # per-hour (after bus)
python3 experiments/tools/dublin/gen_routes.py H S      # per-hour
# after ALL scenario hours exist:
python3 experiments/tools/dublin/demote_uncontested.py  # 9 no-competition TLS -> priority
python3 experiments/tools/dublin/reindex_8std.py        # net + meta (reads demoted net)
sumo -c nets/dublin/weekday_HHh/dublin_weekday_HHh.sumocfg
```

V3 re-check: add an `edgeData` additional (period 10800, run `-e 10800` for
cohort-complete counts) and compare per-TLS sums of outgoing-edge `entered`
against `calibration/scats_targets.json`.
