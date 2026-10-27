"""救护车密集 / 换车型探针路由 (2026-10-27, 用户令 "1-3 做一下"): 从 18h 小汽车路由里挑 36 条经过 ≥2 个 RL 路口的路线, 出发时刻 50 s 起每 100 s 一辆,
同一批车分别以 ambulance / bus / car 三种车型各写一个文件 (车辆 id 相同 probe_01..36), 评估时与原 car/bus 路由一起加载 (不含原 amb_1)。只用于评估, 不训练。"""
import json, random, re, xml.etree.ElementTree as ET, collections, os
NETDIR = "../nets/dublin/weekday_18h_reroute_v2"; META = "../nets/dublin/dublin_8std_splitslot_meta.json"
meta = json.load(open(META))["tls"]
edge2tls = {}
for tid, t in meta.items():
    for i, grp in t["links"].items():
        for c in grp: edge2tls[c["from_edge"]] = tid
cars = ET.parse(f"{NETDIR}/car_weekday_18h.rou.xml").getroot()
cands = []
for v in cars.iter("vehicle"):
    edges = v.find("route").get("edges").split(); tls = []
    for e in edges:
        if e in edge2tls and (not tls or tls[-1] != edge2tls[e]): tls.append(edge2tls[e])
    if len(set(tls)) >= 2 and len(edges) >= 5: cands.append((v.get("id"), edges, tls, v.get("departLane", "0")))
rng = random.Random(20261027); rng.shuffle(cands)
chosen = []; used_routes = set(); first_cnt = collections.Counter()
for k in range(36):
    for c in cands:
        key = tuple(c[1])
        if key in used_routes or first_cnt[c[2][0]] >= 6: continue
        chosen.append(c); used_routes.add(key); first_cnt[c[2][0]] += 1; break
assert len(chosen) == 36, len(chosen)
cover = collections.Counter(t for c in chosen for t in set(c[2]))
short = lambda t: t.split("_")[0] + ("_" + t.split("_")[1] if t.startswith("cluster") else "")
for var, vtype, header in (("amb", "ambulance", '  <vType id="ambulance" vClass="emergency" guiShape="emergency" color="0,1,0" lcStrategic="0" lcCooperative="0" lcSpeedGain="0" lcKeepRight="0"/>\n'), ("bus", "bus", ""), ("car", "car", "")):
    lines = [f'<!-- 2026-10-27 换车型探针 ({var}): 36 辆, 50 s 起每 100 s 一辆, 路线取自 car_weekday_18h (经 >=2 个 RL 路口, 两两不同), 三个文件车辆 id/路线/出发相同只换车型; 仅评估用 -->', "<routes>", header.rstrip("\n")] if header else [f'<!-- 2026-10-27 换车型探针 ({var}): 36 辆, 50 s 起每 100 s 一辆, 路线取自 car_weekday_18h (经 >=2 个 RL 路口, 两两不同), 三个文件车辆 id/路线/出发相同只换车型; 仅评估用 -->', "<routes>"]
    for k, (vid, edges, tls, lane) in enumerate(chosen):
        lines.append(f'  <vehicle id="probe_{k+1:02d}" type="{vtype}" depart="{50 + 100 * k:.1f}" departLane="0" departSpeed="max">\n    <route edges="{" ".join(edges)}"/>\n  </vehicle>')
    lines.append("</routes>")
    open(f"{NETDIR}/probe_{var}_18h.rou.xml", "w").write("\n".join(l for l in lines if l) + "\n")
json.dump([dict(id=f"probe_{k+1:02d}", depart=50 + 100 * k, src_car=c[0], n_edges=len(c[1]), rl_junctions=[short(t) for t in c[2]]) for k, c in enumerate(chosen)], open("analysis/data/priority_evidence_2026-10-27/probe_routes.json", "w"), indent=1, ensure_ascii=False)
print("候选路线", len(cands), "| 选 36 条; 经过 RL 路口数分布", collections.Counter(len(set(c[2])) for c in chosen), "| 各路口被覆盖次数:", {short(t): n for t, n in cover.most_common()})
