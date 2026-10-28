"""任务 9 的标定迭代: 生成 → 排队驱动控制器 3 种子 → 按硬判据定位不达标路口 → 下调 → 再生成。
规则 (计划 §5): 只有 "需求过高" 类 (C1/C4/C14, 以及归到路口的 C2/C3) 才下调; 可信路口 s_j 每次 −0.1, 下限 0.8 (到下限仍不过则报告);
无可信站点的路口加总量上限 = 本轮实现量 × 0.9。"""
import collections, json, subprocess, sys
import common18 as C
PY = sys.executable; TOOLS = str(C.TOOLS)
SEEDS = [123, 124, 125]

def run(cmd):
    r = subprocess.run(cmd, cwd=TOOLS, capture_output=True, text=True)
    if r.returncode != 0: print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit("失败: " + " ".join(cmd))
    return r.stdout

def generate(scale_file):
    run([PY, "demand18.py", "--scale-file", scale_file]); run([PY, "ambulances18.py"]); run([PY, "demand18.py", "--scale-file", scale_file]); run([PY, "write18.py"])

def calibrate_both(tag):
    return calibrate(tag, "served") + calibrate(tag, "fixed")

def calibrate(tag, policy="served", seeds=SEEDS, routes="background_cars_100.rou.xml,bus_tt1.rou.xml,amb_tt1.rou.xml"):
    out = C.WORK / "calib"; out.mkdir(parents=True, exist_ok=True); procs = []
    for s in seeds:
        procs.append(subprocess.Popen([PY, "calib18.py", "--policy", policy, "--seed", str(s), "--routes", routes, "--out", str(out / f"{tag}_{policy}_{s}")],
                                      cwd=TOOLS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(__import__("os").environ, OMP_NUM_THREADS="1")))
    res = []
    for p, s in zip(procs, seeds):
        o, _ = p.communicate()
        line = [l for l in o.splitlines() if l.startswith(policy)]
        print("   ", line[-1] if line else o[-800:])
        res.append(json.load(open(out / f"{tag}_{policy}_{s}.json")))
    return res

def entry_first_rl():
    r = C.load_router(); meta = C.load_meta(); out = {}
    for e in r.net.getEdges():
        cur = e; seen = 0
        while seen < 12:
            if cur.getToNode().getID() in meta: out[e.getID()] = cur.getToNode().getID(); break
            outs = list(cur.getOutgoing())
            if len(outs) != 1: break
            cur = outs[0]; seen += 1
    return out

_R = None
def downstream_rl_lanes(lane, depth=3):
    """出口车道往下游走 (最多 3 段) 遇到的第一批 RL 进口车道。"""
    global _R
    _R = _R or C.load_router(); meta = C.load_meta()
    rl_in = {c["from_edge"] + "_" + str(c["from_lane"]) for t in meta.values() for g in t["links"].values() for c in g}
    out = set(); front = {lane}
    for _ in range(depth):
        nxt = set()
        for l in front:
            for q in _R.g.successors(l):
                if q.startswith("__"): continue
                if q in rl_in: out.add(q)
                else: nxt.add(q)
        if out: return out
        front = nxt
    return out

def feeding_lanes(lane):
    meta = C.load_meta()
    return {c["from_edge"] + "_" + str(c["from_lane"]) for t in meta.values() for g in t["links"].values() for c in g if c["to_edge"] + "_" + str(c["to_lane"]) == lane}

def failures(res, efr):
    fail = collections.defaultdict(set)
    for d in res:
        for e, v in d["C14"].items():
            if v["grow"] > max(0.2 * v["first"], 2.0): fail[v["tid"]].add(f"C14 {e} +{v['grow']}"); fail["__lane__"].add(e)   # e 为车道 id
        agg = collections.defaultdict(lambda: [0, 0])
        for k, v in d["C4"]["by_slot"].items():
            t = k.split("|")[0]; agg[t][0] += v["events"]; agg[t][1] += v["green_ends"]
        for t, (ev, ge) in agg.items():
            if ge and ev / ge > 0.05: fail[t].add(f"C4 旧队列 {ev}/{ge}")
        for k, v in d["C5"]["by_slot"].items():       # 用户关注: 清空后很快又排满 (60 s 内回到 max(4 辆, 60% 容量)) 占清空次数 ≥30%
            if v["clears"] >= 10 and v["refill_lt60"] / v["clears"] >= 0.3:
                tid_, mv_ = k.split("|")
                for g in C.load_meta()[tid_]["links"].values():
                    for c in g:
                        if c["approach"] + "-" + c["turn"] == mv_: fail["__lane__"].add(c["from_edge"] + "_" + str(c["from_lane"]))
        for l, n in d["C6"]["by_lane"].items():      # 出口回溢: 下游没排空 → 下调下游的 RL 进口车道; 下游不是 RL 进口时下调喂这条出口的进口车道
            if n <= 0: continue
            down = downstream_rl_lanes(l)
            if down: fail["__lane__"] |= down
            else: fail["__lane__"] |= feeding_lanes(l)
            fail.setdefault("__spill__", set()).add(l)
        for l, sh in d["C7"]["long_lanes"].items():
            if sh > 0.05: fail["__lane__"].add(l)
        for k, v in d["C3"]["blocked"].items():
            fail[k.split("|")[0]].add(f"C3 堵死 {k.split('|')[1]}")
        for k, v in d["C2"]["by_entry_type"].items():
            e, ty = k.split("|"); req = v.get("requested", 0)
            if req and ((v.get("not_inserted", 0) > 0.01 * req) or (v["mean_delay"] or 0) > 2.0):
                t = efr.get(e)
                if t: fail[t].add(f"C2 入口 {e} 未插入 {v.get('not_inserted',0)} 延迟 {v['mean_delay']}")
    return fail

def main():
    anchors = C.load("anchors_18h.json")["anchors"]; sf = C.WORK / "scales.json"
    sc = json.load(open(sf)) if sf.exists() else {}
    caps = sc.pop("caps", {}); lf = sc.pop("lane_factor", {}); efr = entry_first_rl(); history = []
    for it in range(1, 9):
        json.dump(dict(sc, caps=caps, lane_factor=lf), open(sf, "w"), indent=1)
        generate(str(sf)); rep = C.load("demand_report.json")
        print(f"第 {it} 轮: s_j = {sc}; caps = {caps}; lane_factor = {lf}")
        res = calibrate_both(f"it{it}")     # 排队驱动 + 固定轮转都作硬判据 (C1 原口径、C14 按单条车道)
        F = failures(res, efr)
        g = [d["C1"]["growth"] for d in res]
        history.append(dict(it=it, scales=dict(sc), caps=dict(caps), lane_factor=dict(lf), fail={k: sorted(v) for k, v in F.items()}, C1=g))
        C.dump(dict(history=history), "iterate_history.json")
        if not {k: v for k, v in F.items() if k not in ("__lane__", "__spill__")} and not F.get("__lane__") and all(x is not None and x <= 0.10 for x in g):
            print("全部硬判据通过 (排队驱动 + 固定轮转)")
            break
        if any(x is not None and x > 0.10 for x in g) and not F.get("__lane__"):
            print("   C1 超 10% 但没有单条车道判为积压: 把各次运行中积队增长最大的车道各降一档")
            for d in res:
                if d["C1"]["growth"] is not None and d["C1"]["growth"] > 0.10 and d["C14"]:
                    e = max(d["C14"], key=lambda k: d["C14"][k]["grow"]); F.setdefault("__lane__", set()).add(e)
        changed = False
        # 进口层面: 持续积压 (C14) 或长进口队尾越界 (C7) 的进口, 车道容量上限 ×0.85 (不压整个路口)
        lane_fail = F.pop("__lane__", set()); F.pop("__spill__", None)
        for e in lane_fail:
            nf = round(lf.get(e, 1.0) * 0.85, 3)
            if nf >= 0.5: lf[e] = nf; changed = True
            print(f"   车道 {e}: 容量系数 → {lf.get(e)}")
        for t, why in F.items():
            if all(w.startswith("C14") for w in why): continue      # 已在进口层面处理
            jun = rep["junctions"][t]
            if anchors[t]["trusted"]:
                cur = sc.get(t, 1.0)
                if cur > 0.8 + 1e-9: sc[t] = round(cur - 0.1, 2); changed = True
            else:
                newcap = round(0.9 * jun["realized"], 1)
                if t not in caps or newcap < caps[t] - 1: caps[t] = newcap; changed = True
            print(f"   {t[:22]}: {sorted(why)[:4]}")
        if not changed:
            print("不达标路口都已到下限, 停止并报告"); break
    json.dump(dict(sc, caps=caps, lane_factor=lf), open(sf, "w"), indent=1)

if __name__ == "__main__":
    main()
