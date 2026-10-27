"""按合并边的改名表改写路由文件 (2026-10-28): 把被合并的旧边 id 换成合并后的 id, 连续重复的合并边压成一个; 核对改写后每辆车从 departLane 起不换道可走完 (车道一致)。
用法: python rewrite_routes_merged.py --net 新路网 --renames build_merge389332/merge_report.json --in a.rou.xml --out b.rou.xml [--in ... --out ...]"""
import argparse, json, xml.etree.ElementTree as ET, sumolib
ap = argparse.ArgumentParser(); ap.add_argument("--net", required=True); ap.add_argument("--renames", required=True); ap.add_argument("--in", dest="inp", action="append", required=True); ap.add_argument("--out", action="append", required=True); a = ap.parse_args()
ren = json.load(open(a.renames))["renames"]; net = sumolib.net.readNet(a.net)
def rewrite(edges):
    out = []
    for e in edges:
        e2 = ren.get(e, e)
        if not out or out[-1] != e2: out.append(e2)
    return out
def lane_ok(edges, dl):
    lanes = {dl}
    for e1, e2 in zip(edges, edges[1:]):
        nxt = {c.getToLane().getIndex() for c in net.getEdge(e1).getOutgoing().get(net.getEdge(e2), []) if c.getFromLane().getIndex() in lanes}
        if not nxt: return False
        lanes = nxt
    return True
for src, dst in zip(a.inp, a.out):
    tree = ET.parse(src); root = tree.getroot(); n_changed = 0; bad = 0; n = 0
    for rt in root.iter("route"):
        e = rt.get("edges").split(); e2 = rewrite(e)
        if e2 != e: rt.set("edges", " ".join(e2)); n_changed += 1
    for v in root.iter("vehicle"):
        r = v.find("route"); edges = (r.get("edges").split() if r is not None else None)
        if edges is None: continue
        n += 1; dl = v.get("departLane")
        if dl is not None and dl.isdigit() and not lane_ok(edges, int(dl)): bad += 1
    tree.write(dst, encoding="UTF-8", xml_declaration=True)
    print(f"{src} → {dst}: 改写路线 {n_changed} 条, 车辆 {n}, 车道不一致 {bad}")
