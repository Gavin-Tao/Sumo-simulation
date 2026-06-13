"""A1+B1: multi-seed eval of a ckpt. Aggregates ALL eval_system metrics by
class, reports mean±std over N seeds, and judges priority inversion using
ONLY fair (per-visit) metrics. Usage:
  python3 tools/multiseed_eval.py <ckpt> <config> <N> <seed_base>
Run from experiments/ cwd. Pure analysis (no training, no wandb)."""
import sys, os, json
import numpy as np
os.environ.setdefault('SUMO_RL_LIBSUMO', '1')
sys.path.insert(0, '/home/xiaowen/sumo-rl')
sys.path.insert(0, '/home/xiaowen/sumo-rl/experiments')
import yaml
from eval_unified import run_eval

ckpt, cfg_path = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
seed_base = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
cfg = yaml.safe_load(open(cfg_path))

eps = run_eval("dqn", cfg, ckpt, N, False, "cpu", None,
               wandb_log=False, warmup_steps=20, seed_base=seed_base)
# run_eval may double-append per episode; dedup by keeping unique dict ids
seen, rows = set(), []
for d in eps:
    if id(d) in seen: continue
    seen.add(id(d)); rows.append(d)
print(f"\n[multiseed] {len(rows)} episodes collected (seed {seed_base}..{seed_base+N-1})")

CLASSES = ["all", "car", "bus", "ambulance"]
# fair = per-visit (divides by #intersections crossed); unfair for amb = whole-trip
METRICS = [
    ("avg_speed", "speed↑", "m/s", False),
    ("avg_stopped_time", "stop↓", "s", "UNFAIR_amb"),
    ("avg_stopped_time_per_visit", "stop/visit↓", "s", "FAIR"),
    ("avg_stop_events_per_visit", "nstop/visit↓", "", "FAIR"),
    ("avg_waiting_inc_per_visit", "wait/visit↓", "s", "FAIR"),
    ("completion_rate", "complete↑", "", False),
]
def col(rows, key):
    a = np.array([r.get(key, np.nan) for r in rows], float)
    return a[~np.isnan(a)]

agg = {}  # (cls, metric) -> (mean, std, n)
for cls in CLASSES:
    for m, _, _, _ in METRICS:
        a = col(rows, f"eval_system/{cls}/{m}")
        agg[(cls, m)] = (a.mean() if len(a) else np.nan,
                         a.std() if len(a) else np.nan, len(a))

print("\n" + "="*92)
print(f"eval_system 全指标  mean±std over {len(rows)} seeds  (★=FAIR口径, 用于判反转)")
print("="*92)
hdr = f"{'metric':22}{'fair?':12}" + "".join(f"{c:>14}" for c in CLASSES)
print(hdr)
for m, lab, unit, fair in METRICS:
    tag = "★FAIR" if fair == "FAIR" else ("⚠不公平(amb)" if fair == "UNFAIR_amb" else "—")
    line = f"{lab+'('+unit+')':22}{tag:12}"
    for cls in CLASSES:
        mu, sd, n = agg[(cls, m)]
        line += f"{mu:>7.2f}±{sd:<5.1f}" if not np.isnan(mu) else f"{'—':>14}"
    print(line)

print("\n" + "="*92)
print("优先级反转判定  (ambulance < bus < car ?  仅用 FAIR 口径)")
print("="*92)
for m, lab, unit, fair in METRICS:
    if fair != "FAIR":
        continue
    amb = agg[("ambulance", m)][0]; bus = agg[("bus", m)][0]; car = agg[("car", m)][0]
    # lower is better for these → expect amb < bus < car
    ladder_ok = amb <= bus <= car
    amb_vs_car = "amb<car ✓" if amb < car else "amb≥car ✗反转!"
    amb_vs_bus = "amb<bus ✓" if amb < bus else "amb≥bus ✗"
    print(f"  {lab:16}: amb={amb:.2f} bus={bus:.2f} car={car:.2f}  "
          f"| {amb_vs_car} | {amb_vs_bus} | 阶梯{'成立✓' if ladder_ok else '破缺'}")
# explicit contrast: the UNFAIR metric and what it would wrongly say
m = "avg_stopped_time"
amb = agg[("ambulance", m)][0]; car = agg[("car", m)][0]
print(f"\n  对照(⚠不公平): 全程 stop  amb={amb:.2f} car={car:.2f} → "
      f"{'会误判反转!' if amb >= car else '碰巧不反转'} (amb穿越多路口, 此口径冤枉amb)")

# amb sample size note
amb_n = col(rows, "eval_system/ambulance/avg_stopped_time_per_visit")
print(f"\n  amb 样本: {len(amb_n)} episodes, per_visit 跨seed std="
      f"{amb_n.std():.2f} (CI95≈±{1.96*amb_n.std()/np.sqrt(len(amb_n)):.2f})")
json.dump({f"{c}/{m}": agg[(c,m)][:2] for c in CLASSES for m,_,_,_ in METRICS},
          open(f"/tmp/multiseed_{os.path.basename(ckpt)}.json","w"), default=float)
print("\nsaved /tmp/multiseed_*.json")
