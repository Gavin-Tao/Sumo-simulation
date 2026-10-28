"""救护车选线 (实测, 第二版): 先要求放行取舍机会占比 ≥70% (固定轮转), 再选不分车型时 (排队驱动, 与 wandb 同口径按车道剔除恒绿) 救护车基准等待最接近公交、且不高于公交 +0.3 s 的路线 —
目的: 公交不天然占优、救护车也不天然占优, 531 下的顺序可归因于优先级规格。候选规则放宽到 ≥5 个 RL 路口、≥4 次竞争转向。
(第一版说明): 候选 = 满足规则的路线 (边界入口、≥6 RL 路口、≥5 次非恒绿竞争转向、含两相位支路与多相位路口、无让行、入口负荷 ≤450);
按冲突方向的平台期需求粗排, 前 K 条各放 20 辆救护车 (固定轮转, 种子 123, 背景 1.0 + 公交 tt1) 实测 "有取舍机会的访问" 占比, 取最高。
结果写 build/amb_route_choice.json, ambulances18.py 读取。"""
import collections, json, os, subprocess, sys, xml.etree.ElementTree as ET
import common18 as C
from ambulances18 import classify, score_route, first_rl_dist
from demand18 import conflict_pairs
PY = sys.executable; K = 14

def main():
    meta = C.load_meta(); anchors = C.load("anchors_18h.json"); E = C.load("entries.json"); r = C.load_router(); D = C.load("demand_report.json")
    zero = {tuple(m) for m in E["zero"]} | {tuple(m) for m in E["zero_lane_loop"]}
    ag, nact, minor2 = classify(meta, anchors); CP = conflict_pairs(meta)
    flow = {tuple(k.split("|")): v["realized_total"] for k, v in D["movements"].items()}
    cands = []
    for d in C.load("route_pool.json")["routes"]:
        trav = [tuple(x) for x in d["traversals"]]; comp = [m for m in trav if m not in ag and m not in zero]
        load_ok = D["entry_plateau_vph"].get(d["lanes"][0], 0.0) <= 450
        ok = (load_ok and d["entry_type"] == "boundary" and len(trav) >= 5 and len(comp) >= 4 and not d["yield_links"]
              and first_rl_dist(r, meta, d["lanes"]) >= 100 and any(minor2(t, m) for t, m in comp) and any(nact[t] >= 5 for t, _ in comp))
        if not ok: continue
        proxy = 0.0
        for t, m in comp:
            conf = sum(flow.get((t, b if a == m else a), 0.0) for a, b in CP[t] if m in (a, b))
            proxy += min(1.0, conf / 300.0)
        cands.append((proxy, d, comp))
    cands.sort(key=lambda x: -x[0]); seen = set(); top = []
    for p, d, comp in cands:
        k = tuple(d["edges"])
        if k in seen: continue
        seen.add(k); top.append((p, d, comp))
        if len(top) == K: break
    out = C.WORK / "amb_select"; out.mkdir(parents=True, exist_ok=True); res = []
    bg = {c["entry"]: [] for c in C.load("background.json")["cars"]}
    for i, (p, d, comp) in enumerate(top):
        root = ET.Element("routes"); t = 900.0; k = 0
        while t <= 2800:
            v = ET.SubElement(root, "vehicle", id=f"amb_{k + 1:02d}", type="ambulance", depart=f"{t:.1f}", departLane=str(d["lane"]), departSpeed="max")
            ET.SubElement(v, "route", edges=" ".join(d["edges"])); t += 100.0; k += 1
        ET.ElementTree(root).write(out / f"cand{i}.rou.xml", encoding="UTF-8", xml_declaration=True)
    procs = []; jobs = [(i, "fixed", 123) for i in range(len(top))] + [(i, "served", s_) for i in range(len(top)) for s_ in (123, 124, 125)]
    for i, pol, s_ in jobs:
        tag = f"cand{i}" if pol == "fixed" else f"cand{i}_served_{s_}"
        procs.append(subprocess.Popen([PY, "calib18.py", "--policy", pol, "--seed", str(s_), "--routes", f"background_cars_100.rou.xml,bus_tt1.rou.xml,{out}/cand{i}.rou.xml",
                                       "--out", str(out / tag)], cwd=str(C.TOOLS), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS="1")))
        if len(procs) >= 6:
            procs.pop(0).wait()
    for p_ in procs: p_.wait()
    for i, (p, d, comp) in enumerate(top):
        j = json.load(open(out / f"cand{i}.json")); a = j["C11"].get("ambulance_competing", {"visits": 0, "with_opp": 0})
        sv = [json.load(open(out / f"cand{i}_served_{s_}.json"))["C16"]["by_type"] for s_ in (123, 124, 125)]
        amb_w = sum(x["ambulance"][1] for x in sv) / max(1, sum(x["ambulance"][0] for x in sv)); bus_w = sum(x["bus"][1] for x in sv) / max(1, sum(x["bus"][0] for x in sv))
        res.append(dict(i=i, proxy=round(p, 2), share=a["with_opp"] / max(1, a["visits"]), visits=a["visits"], with_opp=a["with_opp"], arrived=j["C9"]["amb_arrived"],
                        amb_base=round(amb_w, 2), bus_base=round(bus_w, 2), gap=round(amb_w - bus_w, 2),
                        n_rl=len(d["traversals"]), comp=[f"{t[:12]} {m}" for t, m in comp], edges=d["edges"], entry=d["lanes"][0]))
    ok = [x for x in res if x["share"] >= 0.7 and x["gap"] <= 0.3]
    pool_ = ok if ok else [x for x in res if x["share"] >= 0.7] or res
    pool_.sort(key=lambda x: (abs(x["gap"]), -x["share"]))
    for x in sorted(res, key=lambda x: (abs(x["gap"]), -x["share"])):
        print(f"  候选{x['i']}: 机会 {x['share']:.0%} ({x['with_opp']}/{x['visits']}), 基准等待 救护车 {x['amb_base']} / 公交 {x['bus_base']} (差 {x['gap']:+.2f}), RL {x['n_rl']}, 到达 {x['arrived']}, {x['comp']}")
    best = pool_[0]
    json.dump(dict(edges=best["edges"], share_probe=best["share"], candidates=res), open(C.WORK / "amb_route_choice.json", "w"), indent=1, ensure_ascii=False)
    print("选中候选", best["i"])

if __name__ == "__main__":
    main()
