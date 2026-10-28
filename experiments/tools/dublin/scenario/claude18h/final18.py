"""定稿标定: 1.0 档三种控制器 × 5 种子; 救护车/公交机会 (C11) 用时刻表 1–6 × 5 种子 (固定轮转, 第 7 份最终测试不跑);
0.6/0.8 档固定轮转与排队驱动 × 3 种子。最多 4 个进程并行。输出 build/final_calibration.json。"""
import collections, json, os, subprocess, sys
import numpy as np
import common18 as C
PY = sys.executable; OUT = C.WORK / "calib_final"; OUT.mkdir(parents=True, exist_ok=True)
jobs = []
R = lambda bg, k: f"background_cars_{bg}.rou.xml,bus_tt{k}.rou.xml,amb_tt{k}.rou.xml"
for pol in ("served", "fixed", "random"):
    for s in range(123, 128): jobs.append((f"L100_{pol}_tt1_{s}", pol, s, R("100", 1)))
for k in range(2, 7):
    for s in range(123, 128): jobs.append((f"L100_fixed_tt{k}_{s}", "fixed", s, R("100", k)))
for bg in ("060", "080"):
    for pol in ("served", "fixed"):
        for s in range(123, 126): jobs.append((f"L{bg}_{pol}_tt1_{s}", pol, s, R(bg, 1)))
running = []; done = []
env = dict(os.environ, OMP_NUM_THREADS="1")
while jobs or running:
    while jobs and len(running) < 6:
        name, pol, s, routes = jobs.pop(0)
        if (OUT / f"{name}.json").exists(): done.append(name); continue
        running.append((name, subprocess.Popen([PY, "calib18.py", "--policy", pol, "--seed", str(s), "--routes", routes, "--out", str(OUT / name)], cwd=str(C.TOOLS), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)))
    for x in list(running):
        if x[1].poll() is not None: running.remove(x); done.append(x[0])
    if running: running[0][1].wait(timeout=None) if len(running) >= 6 or not jobs else None
res = {p.stem: json.load(open(p)) for p in OUT.glob("*.json")}
def agg(prefix):
    ds = [d for k, d in res.items() if k.startswith(prefix)]
    if not ds: return None
    f = lambda path: [eval("d" + path) for d in ds]
    c4 = [d["C4"]["old_queue_share"] for d in ds]
    c14 = [max([v["grow"] - max(0.2 * v["first"], 2.0) for v in d["C14"].values()] or [0]) for d in ds]
    return dict(n=len(ds), C1_growth=f('["C1"]["growth"]'), C2_not_inserted=f('["C2"]["not_inserted"]'), C2_mean_delay=[round(x, 2) for x in f('["C2"]["mean_delay"]')],
                C2_p99=[round(x, 1) for x in f('["C2"]["p99_delay"]')], C3_blocked=f('["C3"]["blocked_events"]'), C3p_long=f('["C3p"]["long_wait_visits"]'),
                C4_missed=f('["C4"]["missed"]'), C4_old=[round(x, 4) for x in c4], C6_spill=f('["C6"]["spill_events"]'),
                C7_long_over_5pct=[sum(1 for v in d["C7"]["long_lanes"].values() if v > 0.05) for d in ds], C8=[round(d["C8"]["share"], 2) for d in ds],
                C9=[(d["C9"]["lane_changes"], d["C9"]["collisions"], d["C9"]["teleports"], d["C9"]["tracker_missing"], f'{d["C9"]["amb_arrived"]}/{d["C9"]["amb_total"]}') for d in ds],
                C13=f('["C13"]["early_wait_over_rotation"]'), C13_90s=f('["C13"]["early_long_wait_90"]'), C14_worst_excess=[round(x, 2) for x in c14],
                C17_cutins=[d["C17"]["n_cutins"] for d in ds], C17_prio_nonrl_stop_s=[{k: round(v) for k, v in d["C17"]["prio_nonrl_stop_s"].items()} for d in ds])
summary = {k: agg(k) for k in ["L100_served_tt1", "L100_fixed_tt1", "L100_random_tt1", "L080_served", "L080_fixed", "L060_served", "L060_fixed"]}
opp = collections.defaultdict(lambda: [0, 0, 0, 0])
for k, d in res.items():
    if not k.startswith("L100_fixed_tt"): continue
    for ty in ("ambulance_competing", "bus_competing"):
        if ty in d["C11"]: opp[ty][0] += d["C11"][ty]["visits"]; opp[ty][1] += d["C11"][ty]["with_opp"]
    for ty in ("ambulance", "bus"):
        if ty in d["C11"]: opp[ty + "_kinds"][2] += d["C11"][ty]["red_release"]; opp[ty + "_kinds"][3] += d["C11"][ty]["green_hold"]
n_ep = sum(1 for k in res if k.startswith("L100_fixed_tt"))
summary["C11_fixed_tt1to6"] = dict(episodes=n_ep, ambulance_competing_visits=opp["ambulance_competing"][0], ambulance_with_opp=opp["ambulance_competing"][1],
                                   ambulance_share=opp["ambulance_competing"][1] / max(1, opp["ambulance_competing"][0]), bus_with_opp_per_ep=opp["bus_competing"][1] / max(1, n_ep),
                                   amb_red_release=opp["ambulance_kinds"][2], amb_green_hold=opp["ambulance_kinds"][3], bus_red_release=opp["bus_kinds"][2], bus_green_hold=opp["bus_kinds"][3])
C.dump(summary, "final_calibration.json")
for k, v in summary.items(): print(k, json.dumps(v, ensure_ascii=False))
