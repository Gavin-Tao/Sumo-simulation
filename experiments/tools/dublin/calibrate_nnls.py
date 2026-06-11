"""Calibration — turn ratios + linear flow propagation + regularized NNLS.

Design Part 3.2/3.3:
  * turn ratios (assumption A1): P(e->e') ∝ lanes(e') × site_daily(toNode(e'))^alpha,
    u-turn excluded unless it is the only option; one set for the whole day
  * A[j,k] = expected crossings of junction j per vehicle entering at entry k,
    estimated by MONTE CARLO with the ACTUAL route generator (CarWalker:
    no-junction-revisit + lane-chain feasibility). A closed-form (I-P^T)^{-1}
    propagation is NOT used: it counts gyratory loop traversals that real
    (loop-free) vehicles never make, inflating crossings ~1.7x (measured) —
    calibration must share the generator's distribution by construction.
  * junction model throughput (A x)_j vs target s_j = SCATS_hour - bus crossings
    rows weighted 1/n when n TLS share one SCATS site; solve
      min || W(Ax - s) ||^2 + lam || x - x_prior ||^2   s.t. x >= 0
    x_prior ∝ entry lane count, scaled to match total demand implied by s.

Outputs: calibration/turn_ratios_<H>h.json, boundary_rates_<H>h.json,
         calibration_report_<H>h.json (V3 offline residuals)
Usage: python3 calibrate_nnls.py [hour]   (default 11)
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ALPHA = 1.5
LAMBDA = 0.05          # prior regularization weight (relative, see scaling below)
MINOR_CAP = 800.0      # veh/h per arterial lane (saturation-order bound)
N_MC = 3000            # Monte-Carlo walks per entry for the A matrix
RAT_RUN_W = 0.3        # attraction multiplier for service-lane (prio<=4) targets
# streets with DEADLOCK evidence under lane-locked flows (teleport hotspots,
# user-directed): through-routing weight = 0 unless it is the only option.
# Reversible alternative to deleting them from the net.
BANNED_STREETS = ("William Street South", "Montague Street",
                  "Montague Lane", "Montague Court", "Digges Street Upper")
# pedestrianised-zone entries (real Dublin: Grafton St is car-free): cap to a
# delivery trickle. Kills the artificial flood through the demoted Nassau
# corridor junctions 135109491/32235616 (user, 2026-06-11).
PED_ZONE_ENTRIES = {"Grafton Street": 20.0, "Harry Street": 20.0,
                    # Harcourt St: PED site 668 measures the street at only
                    # 153-258 veh/h (the Luas runs on it) — the solver was
                    # flooding it to the 800 arterial cap, jamming the
                    # Harcourt/Clonmel corner (node 130159360, user report)
                    "Harcourt Street": 300.0}
N_IPF = 1              # attraction-correction iterations (1 = off; IPF gave no gain)


def main():
    hour = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    net = common.load_net_demand()
    targets = json.load(open(os.path.join(common.CALIB_DIR, "scats_targets.json")))
    bus = json.load(open(os.path.join(common.CALIB_DIR, f"bus_crossings_{hour:02d}h.json")))

    edges = sorted(net.getEdges(), key=lambda e: e.getID())
    eidx = {e.getID(): i for i, e in enumerate(edges)}
    n = len(edges)
    entries, exits = common.boundary_edges(net)
    entries = sorted(entries, key=lambda e: e.getID())

    # node attraction = the TARGET HOUR's volume of the nearest matched site
    # (TLS only), else median. Hour-specific matters: at night traffic
    # concentrates on the Camden/Georges nightlife corridor while office
    # streets (Kildare, Merrion Row) go quiet — daily totals miss that.
    node_daily = {}
    for tls_id, t in targets["tls"].items():
        node_daily[tls_id] = max(t["hourly"][hour], 1.0)
    med = float(np.median(list(node_daily.values()))) if node_daily else 1.0

    # ---- turn ratios (re-derived each IPF iteration from node_attr)
    node_attr = dict(node_daily)

    def build_ratios():
        out = {}
        for e in edges:
            outs = sorted(e.getOutgoing().keys(), key=lambda o: o.getID())
            if not outs:
                out[e.getID()] = {}
                continue
            rev = [o for o in outs if o.getToNode() is e.getFromNode()
                   and o.getFromNode() is e.getToNode()]
            cand = [o for o in outs if o not in rev] or outs  # u-turn only if forced
            w = {}
            for o in cand:
                attract = node_attr.get(o.getToNode().getID(), med)
                w[o.getID()] = o.getLaneNumber() * (attract ** ALPHA)
                # rat-run suppression: service lanes (prio<=4) stay routable
                # (hard exclusion shreds the graph) but unattractive
                if o.getPriority() <= 4 and any(True for _ in o.getOutgoing()):
                    w[o.getID()] *= RAT_RUN_W
                # deadlock-evidence ban list: zero through-routing
                if (o.getName() or "") in BANNED_STREETS and len(cand) > 1:
                    w[o.getID()] = 0.0
            if sum(w.values()) <= 0:   # all candidates banned — restore
                for o in cand:
                    w[o.getID()] = 1.0
            tot = sum(w.values())
            out[e.getID()] = {k: v / tot for k, v in w.items()}
        return out

    ratios = build_ratios()

    # ---- every edge must reach an exit through P>0 moves (V2 static check)
    reach = {e.getID() for e in exits}
    changed = True
    while changed:
        changed = False
        for e in edges:
            if e.getID() in reach:
                continue
            if any(t in reach for t in ratios[e.getID()]):
                reach.add(e.getID())
                changed = True
    dead = [e.getID() for e in edges if e.getID() not in reach]
    if dead:
        sys.exit(f"FATAL: edges cannot reach any exit: {dead}")

    import random
    from gen_routes import CarWalker
    tls_list = sorted(targets["tls"].keys())
    tls_pos = {tid: i for i, tid in enumerate(tls_list)}
    street = targets.get("street_sites", {})
    street_ids = sorted(street)
    street_edges = {sid: set(street[sid]["edges"]) for sid in street_ids}

    # ---- IPF outer loop (gravity-model attraction correction): after each
    # NNLS solve, scale each trusted junction's attraction by
    # (target/model)^0.7 (clipped) and re-derive turn ratios — fixes the
    # structural under-served rows (Merrion Row / SSG gyratory) that a fixed
    # attraction heuristic cannot reach, instead of flooding capped entries.
    for ipf_it in range(N_IPF):
      ratios = build_ratios() if ipf_it else ratios
      walker = CarWalker(net, ratios, random.Random(12345))
      # crossings counted at ALL measured junctions (SCATS targets) — 3
      # trusted targets are demoted priority junctions, still constraints
      walker.tls_nodes = set(targets["tls"].keys())
      A_full = np.zeros((len(tls_list), len(entries)))
      S_full = np.zeros((len(street_ids), len(entries)))  # street-level rows
      for k, e in enumerate(entries):
          cnt = np.zeros(len(tls_list))
          scnt = np.zeros(len(street_ids))
          for _ in range(N_MC):
              r = walker.walk(e)
              if r is None:
                  continue  # dropped vehicle contributes nothing — same as generator
              path, _lane, crossed = r
              for tid in crossed:
                  cnt[tls_pos[tid]] += 1
              for si, sid in enumerate(street_ids):
                  if street_edges[sid] & set(path):
                      scnt[si] += 1
          A_full[:, k] = cnt / N_MC
          S_full[:, k] = scnt / N_MC

      # rows over trusted TLS targets
      site_count = defaultdict(int)
      for tls_id, t in targets["tls"].items():
          if t["trusted"]:
              site_count[t["site"]] += 1
      rows, s, w_rows, row_ids = [], [], [], []
      for tls_id, t in sorted(targets["tls"].items()):
          if not t["trusted"]:
              continue
          target = max(t["hourly"][hour] - bus["crossings"].get(tls_id, 0), 0.0)
          rows.append(A_full[tls_pos[tls_id], :])
          s.append(target)
          w_rows.append(1.0 / site_count[t["site"]])
          row_ids.append(tls_id)
      A = np.array(rows)
      s = np.array(s, dtype=float)
      # row-feasibility weighting (bounded influence): a target the model
      # cannot serve even with ALL entries at cap must not be chased — the
      # solver otherwise floods capped entries to nudge an unreachable row,
      # overshooting everywhere else (18h went to 6.9k veh/h, 83 teleports).
      caps_pre = np.array([MINOR_CAP * e.getLaneNumber() if e.getPriority() >= 10
                           else 300.0 for e in entries])
      reach_max = A @ caps_pre
      for i in range(len(s)):
          if s[i] > 1.0:
              w_rows[i] *= min(1.0, reach_max[i] / s[i]) ** 2
      W = np.diag(np.array(w_rows))

      # prior: lane share, scaled so total junction-crossings match targets
      lanes = np.array([e.getLaneNumber() for e in entries], dtype=float)
      c = A.sum(axis=0)  # junction-crossings per unit inflow at entry k
      T_est = s.sum() / max((c * lanes / lanes.sum()).sum(), 1e-9)
      x_prior = lanes / lanes.sum() * T_est

    # ---- regularized bounded least squares with ROAD-CLASS caps:
    # arterials (OSM prio>=10): MINOR_CAP veh/h per lane; service lanes
    # (prio<=4, e.g. Duke St beside the pedestrianised Grafton quarter):
    # 300 veh/h (the net lacks Nassau contraflow / opens Grafton, so minor
    # lanes are the only feeders of the Dawson quarter — 100 starves it) — without this the solver parks collinear-corridor demand
    # on whichever minor entry is cheapest (Duke St solved to 700/h).
      caps = np.array([MINOR_CAP * e.getLaneNumber() if e.getPriority() >= 10
                       else 300.0 for e in entries])
      for _k, _e in enumerate(entries):
          if (_e.getName() or "") in PED_ZONE_ENTRIES:
              caps[_k] = PED_ZONE_ENTRIES[_e.getName()]
      scale = np.linalg.norm(W @ A) / max(np.linalg.norm(np.eye(len(entries))), 1e-9)
      lam = LAMBDA * scale
      A_aug = np.vstack([W @ A, lam * np.eye(len(entries))])
      b_aug = np.concatenate([W @ s, lam * x_prior])
      # floors (assumption A9): an open boundary road is never empty —
      # x_k >= max(10, 0.15*prior), clipped under the cap (Dame-St artifact).
      floors = np.minimum(np.maximum(0.15 * x_prior, 10.0), 0.8 * caps)
      x = lsq_linear(A_aug, b_aug, bounds=(floors, caps)).x

      model = A @ x
      mape_it = float(np.mean(np.abs(model - s) / np.maximum(s, 1.0)))
      print(f"  IPF iter {ipf_it}: MAPE {mape_it:.1%}  total inflow {x.sum():.0f}")
      if ipf_it < N_IPF - 1:
          for i, tid in enumerate(row_ids):
              ratio = (s[i] / max(model[i], 1.0)) ** 0.7
              node_attr[tid] = node_attr.get(tid, med) * min(max(ratio, 1/3), 3.0)

    with open(os.path.join(common.CALIB_DIR, f"turn_ratios_{hour:02d}h.json"), "w") as f:
        json.dump({"alpha": ALPHA, "ipf_iters": N_IPF, "ratios": ratios}, f, indent=1)

    err = model - s
    mape = float(np.mean(np.abs(err) / np.maximum(s, 1.0)))

    print(f"hour {hour}: targets {len(s)} trusted rows, entries {len(entries)}")
    print(f"NNLS MAPE (offline linear model): {mape:.1%}")
    print(f"{'TLS':>44s} {'target':>7s} {'model':>7s} {'err%':>6s}")
    for i, tid in enumerate(row_ids):
        print(f"{tid[:44]:>44s} {s[i]:7.0f} {model[i]:7.0f} "
              f"{100 * err[i] / max(s[i], 1):6.1f}")
    if street_ids:
        import xml.etree.ElementTree as ET
        bus_edge = defaultdict(int)
        bus_rou = os.path.join(common.OUT_DIR, f"weekday_{hour:02d}h",
                               f"bus_weekday_{hour:02d}h.rou.xml")
        if os.path.exists(bus_rou):
            for v in ET.parse(bus_rou).getroot().iter("vehicle"):
                for eid in v.find("route").get("edges").split():
                    bus_edge[eid] += 1
        print("street-level diagnostics (NOT fitted; PED single-street counts):")
        for si, sid in enumerate(street_ids):
            buses = sum(bus_edge[e] for e in street_edges[sid])
            tgt = max(street[sid]["hourly"][hour] - buses, 0.0)
            mdl = float(S_full[si, :] @ x)
            print(f"  street:{street[sid]['street']}({sid}) target={tgt:.0f} model={mdl:.0f}")
    print(f"\n{'entry edge':>20s} {'lanes':>5s} {'veh/h':>7s} {'per-lane':>8s}")
    flagged = []
    for k, e in enumerate(entries):
        per_lane = x[k] / max(lanes[k], 1)
        mark = " <<" if per_lane > MINOR_CAP else ""
        if mark:
            flagged.append(e.getID())
        print(f"{e.getID():>20s} {int(lanes[k]):5d} {x[k]:7.0f} {per_lane:8.0f}{mark}")
    if flagged:
        print(f"WARNING: per-lane rate above {MINOR_CAP}: {flagged}")

    with open(os.path.join(common.CALIB_DIR, f"boundary_rates_{hour:02d}h.json"), "w") as f:
        json.dump({"hour": hour, "lambda": LAMBDA, "mape_offline": mape,
                   "rates": {e.getID(): round(float(x[k]), 2)
                             for k, e in enumerate(entries)}}, f, indent=1)
    report = {
        "hour": hour, "mape_offline": mape,
        "rows": [{"tls": tid, "target": float(s[i]), "model": float(model[i])}
                 for i, tid in enumerate(row_ids)],
    }
    with open(os.path.join(common.CALIB_DIR, f"calibration_report_{hour:02d}h.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("wrote turn_ratios.json, boundary_rates, calibration_report")


if __name__ == "__main__":
    main()
