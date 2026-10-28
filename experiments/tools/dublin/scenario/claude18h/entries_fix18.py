"""中段插车合规循环: 生成 → 按全部车型、所有可能车道路径统计每个非边界发车车道的 "平台期上游穿行 / 本地发车" →
公交原出发边不合规的撤掉 (公交延伸到合规入口) → 再生成, 直到所有公交入口合规; 小汽车支路入口由线性规划约束, 结果一并报告。"""
import collections, json, subprocess, sys, xml.etree.ElementTree as ET
import common18 as C
PY = sys.executable

def run(*a):
    r = subprocess.run([PY, *a], cwd=str(C.TOOLS), capture_output=True, text=True)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-2000:]); raise SystemExit(a)

def compliance():
    r = C.load_router(); E = C.load("entries.json"); B = C.load("buses.json")
    lanes = sorted(set(E["side_lanes"]) | set(B["bus_side_entries"]))
    st = collections.defaultdict(collections.Counter); th = collections.defaultdict(collections.Counter); k = 3600 / (C.PLATEAU[1] - C.PLATEAU[0])
    for f in ["background_cars_100.rou.xml", "bus_tt1.rou.xml", "amb_tt1.rou.xml"]:
        for v in ET.parse(C.OUT / f).getroot().iter("vehicle"):
            t = float(v.get("depart"))
            if not (C.PLATEAU[0] <= t < C.PLATEAU[1]): continue
            es = v.find("route").get("edges").split(); ln = int(v.get("departLane")); allp = {x for p in r.lane_paths(es, ln) for x in p}; s0 = f"{es[0]}_{ln}"
            for l in lanes:
                if s0 == l: st[l][v.get("type")] += k
                elif l in allp: th[l][v.get("type")] += k
    rep = {}
    for l in lanes:
        S = sum(st[l].values()); T = sum(th[l].values())
        rep[l] = dict(kind="car_side" if l in E["side_lanes"] else "bus_entry", start=round(S, 1), through=round(T, 1), start_by=dict(st[l]), through_by=dict(th[l]), ok=T <= 0.1 * S + 1e-6)
    return rep

def main():
    sf = str(C.WORK / "scales.json")
    for it in range(1, 5):
        run("buses18.py"); run("routes18.py"); run("ambulances18.py"); run("demand18.py", "--scale-file", sf); run("ambulances18.py"); run("demand18.py", "--scale-file", sf); run("write18.py")
        rep = compliance()
        for l, v in rep.items(): print(f"  第{it}轮 {l:18s} {v['kind']:9s} 发车 {v['start']:6.1f} 穿行 {v['through']:6.1f} {'合规' if v['ok'] else '不合规'} {v['through_by']}")
        bad_bus = [l for l, v in rep.items() if v["kind"] == "bus_entry" and not v["ok"]]
        if not bad_bus: break
        ex = set(json.load(open(C.WORK / "bus_entry_excluded.json"))) | set(bad_bus)
        json.dump(sorted(ex), open(C.WORK / "bus_entry_excluded.json", "w")); print("  撤掉公交入口:", bad_bus)
    C.dump(rep, "entry_compliance.json")

if __name__ == "__main__":
    main()
