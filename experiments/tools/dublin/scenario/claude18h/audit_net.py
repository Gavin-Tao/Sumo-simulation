"""任务 0: NET18B 核验 (相位、历史问题回归、信号组一致、车道权限、College Green 布局、入口出口)。输出各项问题计数, 0 = 通过。
用法: python audit_net.py [--net N --meta M]"""
import argparse, collections, itertools, json, math, sys, xml.etree.ElementTree as ET
import sumolib
import common18 as C

CORES = {0: (("N", "S"), ("T",)), 1: (("E", "W"), ("T",)), 2: (("N", "S"), ("R",)), 3: (("E", "W"), ("R",)),
         4: (("N",), "LTR"), 5: (("S",), "LTR"), 6: (("E",), "LTR"), 7: (("W",), "LTR")}

def audit(net_path, meta_path):
    net = sumolib.net.readNet(str(net_path), withInternal=True)
    meta = json.load(open(meta_path))["tls"]
    root = ET.parse(str(net_path)).getroot()
    progs = {t.get("id"): [p.get("state") for p in t.findall("phase")] for t in root.findall("tlLogic")}
    P = collections.defaultdict(list)
    for tid, t in meta.items():
        links = {int(i): g for i, g in t["links"].items()}
        lab = lambda i: links[i][0]["approach"] + "-" + links[i][0]["turn"]
        def foes(i, j):
            if {c["from_edge"] for c in links[i]} & {c["from_edge"] for c in links[j]}:
                return False
            return any(a["junction"] == b["junction"] and net.getNode(a["junction"]).areFoes(a["junction_index"], b["junction_index"]) for a in links[i] for b in links[j])
        def merge(i, j):
            return bool({(c["to_edge"], c["to_lane"]) for c in links[i]} & {(c["to_edge"], c["to_lane"]) for c in links[j]})
        valid = {a: x["state"] for a, x in t["actions"].items() if x.get("valid")}
        # 槽标签一致 (并槽)
        for i, g in links.items():
            if len({(c["approach"], c["turn"]) for c in g}) != 1:
                P["slot_label_mixed"].append((tid, i))
        ap = collections.defaultdict(set)
        for g in links.values():
            for c in g:
                ap[c["approach"]].add(c["from_edge"])
        P["approach_multi_edge"] += [(tid, a) for a, es in ap.items() if len(es) > 1]
        netconn = {(c[0].getID(), c[1].getID(), c[2]) for c in net.getTLS(tid).getConnections()}
        metaconn = {(c["from_edge"] + "_" + str(c["from_lane"]), c["to_edge"] + "_" + str(c["to_lane"]), c["tl_index"]) for g in links.values() for c in g}
        if netconn != metaconn:
            P["net_meta_conn_mismatch"].append(tid)
        states = progs.get(tid, [])
        greens = [s for s in states if "y" not in s and s.count("r") + s.count("s") != len(s)]
        for s in states:
            if len(s) != t["n_state"]: P["state_len"].append(tid)
            if "g" in s: P["permissive_g"].append(tid)
        s2g = {int(a): gi for a, gi in t["std_to_green_index"].items()}
        for a, gi in s2g.items():
            canon = t["canonical"].get(str(a), a)
            if gi >= len(greens) or greens[gi] != t["actions"][f"a{canon}"]["state"]:
                P["action_program_mismatch"].append((tid, a))
        if sorted(i for i, v in enumerate(t["mask"]) if v) != sorted(int(a[1:]) for a in valid):
            P["mask_mismatch"].append(tid)
        for i in range(t["n_state"]):
            if i not in links and any(s[i] != "r" for s in states):
                P["gap_not_red"].append((tid, i))
        mv = collections.defaultdict(set)
        for i, g in links.items():
            mv[lab(i)].add(i)
        for a, s in valid.items():
            G = [i for i in links if s[i] == "G"]
            for i, j in itertools.combinations(G, 2):
                if foes(i, j): P["conflicting_green"].append((tid, a, lab(i), lab(j)))
                if merge(i, j): P["merging_green"].append((tid, a, lab(i), lab(j)))
            for i in links:
                if s[i] == "r" and not any(foes(i, j) or merge(i, j) for j in G):
                    P["wasted_green"].append((tid, a, lab(i)))
            for k, idx in mv.items():   # 信号组: 同一转向各车道同灯色
                if len({s[i] for i in idx}) > 1:
                    P["signal_group_split"].append((tid, a, k))
            sec, turns = CORES[int(a[1:])]
            if not any(any(c["approach"] in sec and c["turn"] in turns for c in links[i]) for i in G):
                P["no_core"].append((tid, a))
        served = {i for s in valid.values() for i in links if s[i] == "G"}
        P["never_green"] += [(tid, lab(i)) for i in links if i not in served]
        sets = [frozenset(i for i in links if s[i] == "G") for s in valid.values()]
        for x, y in itertools.combinations(sets, 2):
            if x == y: P["duplicate_phase"].append(tid)
        # 进口车道: 死车道 / 漏接信号 / 节点内未受控
        jn = {c["junction"] for g in links.values() for c in g}
        ctrl = {c[0].getID() for c in net.getTLS(tid).getConnections()}
        for j in jn:
            for e in net.getNode(j).getIncoming():
                if e.getFromNode().getID() in jn: continue
                for l in e.getLanes():
                    if not l.allows("passenger"): continue
                    if not l.getOutgoing(): P["dead_inlet_lane"].append(l.getID())
                    elif l.getID() not in ctrl: P["inlet_not_signalled"].append(l.getID())
        for g in links.values():
            for c in g:
                if net.getEdge(c["from_edge"]).getLength() < 20: P["inlet_lt_20m"].append((tid, c["from_edge"]))
                if c["dir"] in ("t", "T"): P["uturn"].append((tid, c["from_edge"]))
    # 车道权限 (只允许小汽车 = 构建遗留问题)
    for e in root.findall("edge"):
        if e.get("function") == "internal": continue
        for l in e.findall("lane"):
            if l.get("allow") == "passenger": P["passenger_only_lane"].append(l.get("id"))
    if root.get("lefthand") != "true": P["not_lefthand"].append(1)
    # College Green 布局
    n = net.getNode(C.CG)
    if n.getType() == "traffic_light": P["cg_still_signal"].append(1)
    if net.getEdge("304634278#0").getOutgoing(): P["cg_304634278_not_terminal"].append(1)
    entries = [e.getID() for e in net.getEdges(withInternal=False) if not e.getIncoming() and e.getLanes()[0].allows("passenger")]
    sinks = [e.getID() for e in net.getEdges(withInternal=False) if not e.getOutgoing() and e.getLanes()[0].allows("passenger")]
    info = dict(tls=len(meta), entries=len(entries), sinks=len(sinks))
    if len(meta) != 18: P["tls_count"].append(len(meta))
    if len(entries) != 15 or len(sinks) != 15: P["entry_sink_count"].append((len(entries), len(sinks)))
    return {k: v for k, v in P.items()}, info

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--net", default=str(C.NET)); ap.add_argument("--meta", default=str(C.META)); a = ap.parse_args()
    P, info = audit(a.net, a.meta)
    print(info)
    keys = ["slot_label_mixed", "approach_multi_edge", "net_meta_conn_mismatch", "state_len", "permissive_g", "action_program_mismatch", "mask_mismatch", "gap_not_red",
            "conflicting_green", "merging_green", "wasted_green", "signal_group_split", "no_core", "never_green", "duplicate_phase", "dead_inlet_lane", "inlet_not_signalled",
            "inlet_lt_20m", "uturn", "passenger_only_lane", "not_lefthand", "cg_still_signal", "cg_304634278_not_terminal", "tls_count", "entry_sink_count"]
    for k in keys:
        v = P.get(k, []); print(f"{k}: {len(v)}" + (f"  {v[:6]}" if v else ""))
    C.dump(dict(info=info, issues=P), "audit_net.json")
