"""汇总 scan_locks.py 的输出: 每臂 (检查点 × 种子) 的锁死率、落点路口、最大等待; 用法: python lockscan_summary.py <dir> [--md out.md]"""
import argparse, collections, glob, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--md"); ap.add_argument("--thr", type=float, default=150); a = ap.parse_args()
NAME = {"458": "旧111", "456": "旧521", "449": "旧531", "457": "旧541", "460": "新111", "462": "新521", "461": "新531", "459": "新541"}
rows = []
for f in sorted(glob.glob(os.path.join(a.dir, "exp*_ep*_s*.json"))):
    m = re.match(r"exp(\d+)_ep(\d+)_s(\d+)\.json", os.path.basename(f)); d = json.load(open(f))
    locks = [(t, s["max_w"], s["t_max"]) for t, s in d["summary"].items() if s["max_w"] >= a.thr]
    rows.append(dict(exp=m.group(1), ep=int(m.group(2)), seed=int(m.group(3)), lock=locks, maxw=max(s["max_w"] for s in d["summary"].values()), reward=sum(s["reward_sum"] for s in d["summary"].values()), remaining=d["n_remaining"]))
out = []
out.append(f"| 臂 | 已扫 (检查点×种子) | 锁死集 (任一路口最大本次访问等待 ≥ {a.thr:.0f} s) | 锁死率 | 落点路口 (次) | 最大等待 (s) | 锁死集的检查点 |")
out.append("|---|---|---|---|---|---|---|")
for e in ["458", "456", "449", "457", "460", "462", "461", "459"]:
    R = [r for r in rows if r["exp"] == e]
    if not R: continue
    L = [r for r in R if r["lock"]]
    junc = collections.Counter(t.split("_")[0] + ("_" + t.split("_")[1] if t.startswith("cluster") else "") for r in L for t, _, _ in r["lock"])
    out.append(f"| {NAME.get(e, e)} (exp{e}) | {len(R)} | {len(L)} | {len(L)/len(R):.1%} | {dict(junc)} | {max([w for r in L for _, w, _ in r['lock']] or [0]):.0f} | {sorted(set((r['ep'], r['seed']) for r in L))} |")
out.append("")
out.append("| 组 | 已扫 | 锁死集 | 锁死率 |"); out.append("|---|---|---|---|")
for g, es in (("旧四臂", ["458", "456", "449", "457"]), ("新四臂", ["460", "462", "461", "459"])):
    R = [r for r in rows if r["exp"] in es]; L = [r for r in R if r["lock"]]
    if R: out.append(f"| {g} | {len(R)} | {len(L)} | {len(L)/len(R):.1%} |")
txt = "\n".join(out); print(txt)
if a.md: open(a.md, "w").write(txt + "\n")
json.dump(rows, open(os.path.join(a.dir, "lockscan_rows.json"), "w"))
