# 仓库副本 (2026-10-14): 中间文件在 analysis/data/synthesis_1x1_2026-10-14/ (会话中原在临时目录); 逐集行为记录合并为 behav_out.jsonl。
"""离线读本地 .wandb (不调 API): 1x1 各臂训练中评估 (种子 123, 每 5 集) 曲线 → 学习速度与稳定性。只读。
输出 JSON: exp -> {E, stop, ret(若有), q_mean(末段)}。"""
import sys, json, glob, re, os, numpy as np
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
W = "logs/wandb/wandb"
def load(run_dir):
    fs = glob.glob(f"{run_dir}/*.wandb")
    if not fs: return []
    ds = datastore.DataStore(); ds.open_for_scan(fs[0]); H = []
    while True:
        try: rec = ds.scan_data()
        except AssertionError: break
        if rec is None: break
        r = pb.Record(); r.ParseFromString(rec)
        if r.WhichOneof("record_type") == "history":
            H.append({it.key or "/".join(it.nested_key): json.loads(it.value_json) for it in r.history.item})
    return H
runs = {}
for d in sorted(glob.glob(f"{W}/run-*")):
    c = f"{d}/files/config.yaml"
    if not os.path.exists(c): continue
    txt = open(c).read(); m = re.search(r"^name:\s*\n\s*value:\s*(\S+)", txt, re.M)
    if not m: continue
    mm = re.match(r"exp(\d+)_", m.group(1))
    if mm and "1x1" in m.group(1): runs.setdefault(int(mm.group(1)), []).append(d)
want = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else sorted(runs)
out = {}
for e in want:
    if e not in runs: continue
    best = None
    for d in runs[e]:          # 同一臂多次运行: 取评估点最多的那次 (完整训练)
        H = load(d); keys = set().union(*[h.keys() for h in H]) if H else set()
        k = [x for x in keys if x.startswith("eval") and x.endswith("system/all/avg_stopped_time_per_visit")]
        if not k: continue
        pts = [(h.get("eval/episode", h.get("episode")), h[k[0]], h) for h in H if k[0] in h and h[k[0]] is not None]
        if best is None or len(pts) > len(best[1]): best = (d, pts, keys)
    if best is None: continue
    d, pts, keys = best
    E = [p[0] for p in pts]
    if any(x is None for x in E): E = [5 * (i + 1) for i in range(len(pts))]
    rk = [x for x in keys if x.startswith("eval") and ("reward" in x or "return" in x)]
    rec = {"run": os.path.basename(d), "E": E, "stop": [p[1] for p in pts], "rkeys": sorted(rk)[:6]}
    for kk in rk[:3]: rec[kk] = [p[2].get(kk) for p in pts]
    out[e] = rec
json.dump(out, open("/home/xiaowen/sumo-rl/experiments/analysis/data/synthesis_1x1_2026-10-14/curves_1x1.json", "w"))
print(len(out), "臂;", "示例奖励键:", next(iter(out.values()))["rkeys"] if out else None)
