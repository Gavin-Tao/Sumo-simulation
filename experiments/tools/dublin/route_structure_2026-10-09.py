"""三个时段 (02h / 11h / 18h 改道第二版) × 三类车 的路线结构对比 (2026-10-09 用户问: 除了路网相同、流量不同, 路线上到底有什么区别)。
输出: experiments/analysis/data/dublin_route_structure_2026-10-09.json + 终端摘要。路口 = 经过的 RL 信号路口 (边的终点节点在 tlLogic 集合内)。"""
import os, sys, json, collections, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import sumolib
R = "nets/dublin"
WIN = {"02h": ("weekday_02h", "amb_weekday_02h_probe11h"), "11h": ("weekday_11h", "amb_weekday_11h"), "18h": ("weekday_18h_reroute_v2", "amb_weekday_18h_probe11h")}
net = sumolib.net.readNet(f"{R}/dublin_enum.net.xml")
RL = {t.get("id") for t in ET.parse(f"{R}/dublin_enum.net.xml").getroot().findall("tlLogic")}
E = {e.getID(): e for e in net.getEdges()}

def junctions_of(edges):
    """路线经过的 RL 路口序列 (按顺序, 每条边的终点节点; 最后一条边的终点不算, 车在那里离网)."""
    return [E[a].getToNode().getID() for a in edges[:-1] if E[a].getToNode().getID() in RL]

def load(win):
    folder, amb = WIN[win]; out = {}
    for cls, f in (("car", f"car_weekday_{win}"), ("bus", f"bus_weekday_{win}"), ("ambulance", amb)):
        vs = []
        for v in ET.parse(f"{R}/{folder}/{f}.rou.xml").getroot().iter("vehicle"):
            edges = v.find("route").get("edges").split()
            vs.append(dict(id=v.get("id"), depart=float(v.get("depart")), edges=edges, length=sum(E[a].getLength() for a in edges), junctions=junctions_of(edges), entry=edges[0], exit=edges[-1]))
        out[cls] = vs
    return out

def summarize(win, data):
    s = {}
    for cls, vs in data.items():
        jc = collections.Counter(j for v in vs for j in v["junctions"])
        routes = collections.Counter(tuple(v["edges"]) for v in vs)
        s[cls] = dict(n=len(vs), distinct_routes=len(routes), mean_len=sum(v["length"] for v in vs) / len(vs), mean_rl_junctions=sum(len(v["junctions"]) for v in vs) / len(vs),
                      share_no_rl=sum(1 for v in vs if not v["junctions"]) / len(vs), junction_visits=dict(jc), junctions_touched=len(jc),
                      entries=dict(collections.Counter(v["entry"] for v in vs).most_common()), exits=dict(collections.Counter(v["exit"] for v in vs).most_common()),
                      top_routes=[(" ".join(k), n) for k, n in routes.most_common(5)])
    return s

def main():
    data = {w: load(w) for w in WIN}; summ = {w: summarize(w, d) for w, d in data.items()}
    # 车流路线集合在三个时段之间的重合 (按边序列)
    cross = {}
    for cls in ("car", "bus", "ambulance"):
        sets = {w: {tuple(v["edges"]) for v in data[w][cls]} for w in WIN}
        od = {w: {(v["entry"], v["exit"]) for v in data[w][cls]} for w in WIN}
        cross[cls] = {f"{a}∩{b}": dict(routes_a=len(sets[a]), routes_b=len(sets[b]), common_routes=len(sets[a] & sets[b]), od_a=len(od[a]), od_b=len(od[b]), common_od=len(od[a] & od[b]))
                      for a, b in (("02h", "11h"), ("11h", "18h"), ("02h", "18h"))}
    amb = {w: [dict(id=v["id"], depart=v["depart"], length=round(v["length"]), junctions=v["junctions"], edges=v["edges"]) for v in data[w]["ambulance"]] for w in WIN}
    json.dump(dict(summary=summ, cross=cross, ambulances=amb, rl_junctions=sorted(RL)), open("experiments/analysis/data/dublin_route_structure_2026-10-09.json", "w"), indent=1, ensure_ascii=False)
    print("时段  车类       车辆  不同路线  均长 m  经过RL路口数/车  不经过任何RL路口  触及RL路口数  入口数  出口数")
    for w in WIN:
        for cls in ("car", "bus", "ambulance"):
            x = summ[w][cls]; print(f"{w:4} {cls:9} {x['n']:6d} {x['distinct_routes']:8d} {x['mean_len']:7.0f} {x['mean_rl_junctions']:14.2f} {100*x['share_no_rl']:14.0f}% {x['junctions_touched']:12d} {len(x['entries']):7d} {len(x['exits']):7d}")
    print("\n路线集合跨时段重合 (相同边序列 / 相同起终点边):")
    for cls, d in cross.items():
        for k, v in d.items(): print(f"  {cls:9} {k}: 路线 {v['routes_a']} vs {v['routes_b']} 共有 {v['common_routes']} | 起终点对 {v['od_a']} vs {v['od_b']} 共有 {v['common_od']}")
    print("\n救护车:")
    for w in WIN:
        for a in amb[w]: print(f"  {w} {a['id']:12} depart {a['depart']:7.1f} 长 {a['length']:5d} m 经过 RL 路口 {len(a['junctions'])}: {' → '.join(j[:14] for j in a['junctions'])}")
    print("\n每小时各 RL 路口的经过量 (车类: 02h/11h/18h):")
    rows = []
    for j in sorted(RL):
        r = [j[:40]] + [summ[w][c]["junction_visits"].get(j, 0) for c in ("car", "bus", "ambulance") for w in WIN]; rows.append(r)
    print(f"{'路口':40} {'car 02/11/18':>18} {'bus 02/11/18':>16} {'amb 02/11/18':>12}")
    for r in rows: print(f"{r[0]:40} {r[1]:5d}/{r[2]:5d}/{r[3]:5d}   {r[4]:4d}/{r[5]:4d}/{r[6]:4d}   {r[7]:2d}/{r[8]:2d}/{r[9]:2d}")

if __name__ == "__main__": main()
