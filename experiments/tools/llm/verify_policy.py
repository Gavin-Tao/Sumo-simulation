"""Policy verification — the "compile-time test suite" for traffic policy.

Runs the scenario under a compiled IR (retyping daemon active), optionally
against a baseline IR, computes per-class outcome metrics, and evaluates the
IR's own KPI assertions -> pass/fail report.

Controllers:
  --controller fixed   fixed-time programs (net's tlLogic)
  --controller frap    a trained FRAP-enum checkpoint (config's exp211-style
                       yaml) — the controller SEES the policy through its
                       obs table (extended with derived types), so level
                       changes produce real behavioral responses.

Metrics grammar (per base class, split at '@'):
  <class>_wait_mean | <class>_wait_p90 | <class>_count
  <class>_wait_{mean,p90}_degradation   (requires --baseline-ir)

Usage (repo root):
  python experiments/tools/llm/verify_policy.py \
      --config experiments/configs/exp211_dublin11h_531_enumfrap.yaml \
      --ir policy.yaml [--baseline-ir default.yaml] --controller frap \
      [--num-seconds 3600] [--seed 123]
Read-only analysis: trains nothing, modifies no config.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, os.path.join(_REPO, "experiments", "tools", "kan"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml

import policy_dsl
from level_resolver import LevelResolver
from policy_runtime import PolicyRuntime


def run_once(cfg, ir, controller, seed, regions=None):
    from extract_frap_targets import build            # kan env constructor
    resolver = LevelResolver(ir, regions=regions)
    runtime = PolicyRuntime(resolver, cfg["priority_source"], seed=seed)
    cfg = dict(cfg)
    cfg["priority_source"] = runtime.extended_table()  # controller/obs see
    env = build(cfg)                                   # derived levels

    agent = None
    if controller == "frap":
        from frap_glue import load_enum_tables, build_frap_agent
        tables = load_enum_tables(cfg["enum_meta_file"])
        env.reset(int(seed))
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.observation_fn.rebind_movements(tables["turnmap"][tid])
            ts.observation_space = ts.observation_fn.observation_space()
        agent = build_frap_agent(cfg, tables, env, "cpu")
        ckpt = sorted(glob.glob(os.path.join(
            "models", cfg["name"], "*", "best.pth")))[-1]
        agent.q_net.load_state_dict(torch.load(
            ckpt, map_location="cpu", weights_only=False)["policy_state_dict"])
        agent.q_net.eval()
        states = {tid: env.traffic_signals[tid].observation_fn()
                  for tid in env.ts_ids}
    else:
        env.reset(int(seed))
        states = None

    sumo = env.traffic_signals[env.ts_ids[0]].sumo
    runtime.reset()
    wait, cls = {}, {}
    done = {"__all__": False}
    while not done["__all__"]:
        if agent is not None:
            acts = {}
            for i, ts in enumerate(agent._ids):
                x = torch.tensor(np.asarray(states[ts], np.float32)).unsqueeze(0)
                pm, rel, exist, mask = agent._tensors(torch.tensor([i]))
                with torch.no_grad():
                    q = agent.q_net(x, pm, rel, exist).masked_fill(~mask, -1e9)
                acts[ts] = int(q.argmax().item())
            states, _, done, _ = env.step(action=acts)
        else:
            _, _, done, _ = env.step(action={})
        runtime.step(sumo, sumo.simulation.getTime())
        for vid in sumo.vehicle.getIDList():
            wait[vid] = sumo.vehicle.getAccumulatedWaitingTime(vid)
            if vid not in cls:
                cls[vid] = sumo.vehicle.getTypeID(vid).split("@", 1)[0]
    env.close()

    per_cls = {}
    for vid, w in wait.items():
        per_cls.setdefault(cls[vid], []).append(w)
    metrics = {}
    for c, ws in per_cls.items():
        ws = np.asarray(ws)
        metrics[f"{c}_wait_mean"] = float(ws.mean())
        metrics[f"{c}_wait_p90"] = float(np.percentile(ws, 90))
        metrics[f"{c}_count"] = int(len(ws))
    return metrics, dict(runtime.stats)


def evaluate_kpis(kpis, metrics, base_metrics):
    ops = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}
    out = []
    for k in kpis or []:
        m = k["metric"]
        if m.endswith("_degradation"):
            root = m[: -len("_degradation")]
            if base_metrics is None or root not in base_metrics:
                out.append({**k, "status": "SKIP (no baseline)"})
                continue
            base = base_metrics[root]
            val = (metrics.get(root, base) - base) / base if base else 0.0
        else:
            if m not in metrics:
                out.append({**k, "status": "SKIP (metric absent)"})
                continue
            val = metrics[m]
        ok = ops[k["op"]](val, k["value"])
        out.append({**k, "observed": round(float(val), 4),
                    "status": "PASS" if ok else "FAIL"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--baseline-ir", default=None)
    ap.add_argument("--controller", choices=["fixed", "frap"], default="fixed")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num-seconds", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg_path, ir_path = os.path.abspath(args.config), os.path.abspath(args.ir)
    base_path = os.path.abspath(args.baseline_ir) if args.baseline_ir else None
    os.chdir(os.path.join(_REPO, "experiments"))
    cfg = yaml.safe_load(open(cfg_path))
    if args.num_seconds:
        cfg["num_seconds"] = args.num_seconds

    ir = policy_dsl.load(ir_path)
    ok, errs, warns = policy_dsl.validate(ir)
    assert ok, errs
    for w in warns:
        print("WARN:", w)

    base_metrics = base_stats = None
    if base_path:
        base_ir = policy_dsl.load(base_path)
        assert policy_dsl.validate(base_ir)[0]
        print("== baseline run ==")
        base_metrics, base_stats = run_once(cfg, base_ir, args.controller, args.seed)
    print("== policy run ==")
    metrics, stats = run_once(cfg, ir, args.controller, args.seed)

    report = {"config": cfg_path, "ir": ir_path, "controller": args.controller,
              "seed": args.seed, "retype_stats": stats,
              "metrics": metrics, "baseline_metrics": base_metrics,
              "baseline_retype_stats": base_stats,
              "kpis": evaluate_kpis(ir.get("kpis"), metrics, base_metrics)}
    report["verdict"] = ("PASS" if all(k["status"] == "PASS"
                                       for k in report["kpis"]
                                       if k["status"] in ("PASS", "FAIL"))
                         else "FAIL") if report["kpis"] else "NO-KPIS"
    out = args.out or (os.path.splitext(ir_path)[0] + "_verify.json")
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps({k: report[k] for k in
                      ("retype_stats", "kpis", "verdict")}, indent=1,
                     ensure_ascii=False))
    print("report ->", out)


if __name__ == "__main__":
    main()
