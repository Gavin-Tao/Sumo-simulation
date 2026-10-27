"""把一串只被纯几何节点隔开的边合并成一条边 (2026-10-28, 用户要求: 389332 东边 11 + 21 + 49 m 三段并成一条进口道, 观测范围变长)。
做法: netconvert 导出 plain XML (nod/edg/con/tll) → 删掉链中间的节点, 用链两端的 from/to 和拼接的形状生成一条边 (id 沿用紧挨路口那一段, 这样
路口的信号连接与 TLS 程序不用改), 链内部的连接删掉, 进出链的连接改名 → netconvert 从 plain 重建 (--lefthand) → 核对: 其余连接逐条不变。
--chain "a b c": 按行驶方向列出的边, 最后一段紧挨目标路口 (进口链, 合并边 id = c); --chain-out "c' b' a'": 反向离开路口的链 (合并边 id = c', 第一段紧挨路口)。
用法: python merge_edge_chain.py --src nets/dublin/dublin_8action_demoted.net.xml --out nets/dublin/build_merge389332/dublin_8action_merged.net.xml \\
        --remove-node 5504143189 --remove-node 550879544 --chain "52879957#1 -304634278#1 -304634278#0" --chain-out "304634278#0 304634278#1 4476043#0" """
import argparse, os, subprocess, sys, xml.etree.ElementTree as ET
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--remove-node", action="append", default=[]); ap.add_argument("--chain", action="append", default=[]); ap.add_argument("--chain-out", action="append", default=[]); a = ap.parse_args()
wd = os.path.dirname(os.path.abspath(a.out)); os.makedirs(wd, exist_ok=True); pre = os.path.join(wd, "plain")
r = subprocess.run(["netconvert", "-s", a.src, "--plain-output-prefix", pre, "--offset.disable-normalization"], capture_output=True, text=True)
if r.returncode != 0: sys.exit(r.stderr)
nod = ET.parse(pre + ".nod.xml"); edg = ET.parse(pre + ".edg.xml"); con = ET.parse(pre + ".con.xml"); tll = ET.parse(pre + ".tll.xml")
nodes = {n.get("id"): n for n in nod.getroot().iter("node")}; edges = {e.get("id"): e for e in edg.getroot().iter("edge")}
def pts(eid):
    e = edges[eid]
    if e.get("shape"): return [tuple(map(float, p.split(","))) for p in e.get("shape").split()]
    f, t = nodes[e.get("from")], nodes[e.get("to")]; return [(float(f.get("x")), float(f.get("y"))), (float(t.get("x")), float(t.get("y")))]
renames = {}                        # 旧边 id -> 合并边 id (用于连接改名与路由改写)
def merge(chain, keep_last):
    ids = chain.split(); keep = ids[-1] if keep_last else ids[0]
    lanes = {edges[i].get("numLanes") for i in ids}; assert len(lanes) == 1, f"车道数不一致 {chain}: {lanes}"
    shape = []
    for i in ids:
        p = pts(i); shape += (p if not shape else p[1:])
    first, last = edges[ids[0]], edges[ids[-1]]; new = edges[keep]
    new.set("from", first.get("from")); new.set("to", last.get("to")); new.set("shape", " ".join(f"{x:.2f},{y:.2f}" for x, y in shape)); new.set("spreadType", "center")
    for i in ids:
        if i != keep: edg.getroot().remove(edges[i]); renames[i] = keep
    return keep, ids
chains = [merge(c, True) for c in a.chain] + [merge(c, False) for c in a.chain_out]
for nid in a.remove_node: nod.getroot().remove(nodes[nid])
# 连接: 链内部的删掉; 进出链的改名
croot = con.getroot(); removed_internal = 0
for c in list(croot.iter("connection")):
    f, t = c.get("from"), c.get("to")
    chain_ids = {i for _, ids in chains for i in ids}
    if f in chain_ids and t in chain_ids: croot.remove(c); removed_internal += 1; continue
    if f in renames: c.set("from", renames[f])
    if t in renames: c.set("to", renames[t])
for el in tll.getroot().iter("connection"):
    if el.get("from") in renames: el.set("from", renames[el.get("from")])
    if el.get("to") in renames: el.set("to", renames[el.get("to")])
for tr, suf in ((nod, ".nod.xml"), (edg, ".edg.xml"), (con, ".con.xml"), (tll, ".tll.xml")): tr.write(pre + "_merged" + suf, encoding="UTF-8", xml_declaration=True)
r = subprocess.run(["netconvert", "-n", pre + "_merged.nod.xml", "-e", pre + "_merged.edg.xml", "-x", pre + "_merged.con.xml", "-i", pre + "_merged.tll.xml", "--lefthand", "--offset.disable-normalization",
                    "--no-turnarounds", "--junctions.corner-detail", "5", "-o", a.out], capture_output=True, text=True)
print(r.stderr.strip()[-1500:])
if r.returncode != 0: sys.exit("netconvert 重建失败")
def conns(path): return {(c.get("from"), c.get("to"), c.get("fromLane"), c.get("toLane")) for c in ET.parse(path).getroot().findall("connection") if not c.get("from", "").startswith(":")}
chain_ids = {i for _, ids in chains for i in ids}
before = {c for c in conns(a.src) if c[0] not in chain_ids and c[1] not in chain_ids}; after = {c for c in conns(a.out) if c[0] not in chain_ids and c[1] not in chain_ids}
print(f"合并: {[(k, ids) for k, ids in chains]}; 删节点 {a.remove_node}; 删链内连接 {removed_internal}; 其余连接 丢失 {len(before - after)} 新增 {len(after - before)}")
for c in sorted(before - after)[:8]: print("  丢失", c)
for c in sorted(after - before)[:8]: print("  新增", c)
import sumolib
net = sumolib.net.readNet(a.out)
for k, ids in chains: e = net.getEdge(k); print(f"  合并边 {k}: {e.getLength():.1f} m, 车道 {e.getLaneNumber()}, {e.getFromNode().getID()[:20]} → {e.getToNode().getID()[:20]}")
import json; json.dump(dict(renames=renames, chains=[(k, ids) for k, ids in chains], removed_nodes=a.remove_node), open(os.path.join(wd, "merge_report.json"), "w"), indent=1)
