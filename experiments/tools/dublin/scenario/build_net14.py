"""构建 14 信号路网 (2026-10-28, 用户决定: 降 4 个 RL 路口为让行; 可选把造成 10–11 m 短进口边的纯几何节点并入路口)。
源网 = nets/dublin/dublin_8action_demoted.net.xml (6 月降级后的 18 信号网, lefthand)。步骤:
  1. netconvert 节点补丁: 被降级节点 type=priority; --join 指定的节点组用 <join nodes="..."/> 并成一个路口 (短边被吸收进路口内部);
  2. 校验: 未涉及合并的连接 (from/to/fromLane/toLane) 逐条不变; 报告被吸收/消失的边 (路由文件要据此改写);
  3. 调 reindex_8std.py <新网> --split-slots --tag <tag> 生成 8STD 分槽网与 meta (nets/dublin/dublin_8std_<tag>.net.xml / _meta.json);
  4. 打印 14 个 TLS 的相位数与每个进口最短边长度。
用法: python build_net14.py --tag splitslot14 --demote 32236377 135109534 cluster_135109528_9101656 389332 [--join "389281 3506865299" --join "1448436 26912118" ...]"""
import argparse, os, subprocess, sys, json, xml.etree.ElementTree as ET
REPO = "/home/xiaowen/sumo-rl"
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=f"{REPO}/nets/dublin/dublin_8action_demoted.net.xml"); ap.add_argument("--tag", required=True)
ap.add_argument("--demote", nargs="+", required=True); ap.add_argument("--join", action="append", default=[], help='一组要并成一个路口的节点 id, 空格分隔, 可重复')
ap.add_argument("--workdir", default=f"{REPO}/nets/dublin/build_{{tag}}"); a = ap.parse_args()
wd = a.workdir.format(tag=a.tag); os.makedirs(wd, exist_ok=True)
patch = os.path.join(wd, "patch.nod.xml")
with open(patch, "w") as f:
    f.write("<nodes>\n")
    for d in a.demote: f.write(f'  <node id="{d}" type="priority"/>\n')
    for g in a.join: f.write(f'  <join nodes="{g}"/>\n')
    f.write("</nodes>\n")
out_net = os.path.join(wd, f"dublin_8action_{a.tag}.net.xml")
res = subprocess.run(["netconvert", "-s", a.src, "-n", patch, "--offset.disable-normalization", "-o", out_net], capture_output=True, text=True)
print(res.stderr.strip()[-2000:])
if res.returncode != 0: sys.exit("netconvert failed")
def conns(path):
    return {(c.get("from"), c.get("to"), c.get("fromLane"), c.get("toLane")) for c in ET.parse(path).getroot().findall("connection") if not c.get("from", "").startswith(":")}
def edges(path): return {e.get("id") for e in ET.parse(path).getroot().findall("edge") if e.get("function") != "internal"}
before, after = conns(a.src), conns(out_net); eb, ea = edges(a.src), edges(out_net)
joined_nodes = {n for g in a.join for n in g.split()}
gone = sorted(eb - ea); new = sorted(ea - eb)
print(f"边: 源 {len(eb)} 新 {len(ea)}; 消失的边 (被并进路口): {gone}; 新增的边: {new}")
diff_b = {c for c in before - after if c[0] in ea and c[1] in ea}; diff_a = {c for c in after - before if c[0] in eb and c[1] in eb}
print(f"连接: 源 {len(before)} 新 {len(after)}; 在未变动的边之间 丢失 {len(diff_b)} 新增 {len(diff_a)}")
for c in sorted(diff_b)[:10]: print("   丢失:", c)
for c in sorted(diff_a)[:10]: print("   新增:", c)
types = {j.get("id"): j.get("type") for j in ET.parse(out_net).getroot().iter("junction")}
print("TLS 数:", sum(1 for t in types.values() if t == "traffic_light"), "; 被降级节点现在:", {d: types.get(d) for d in a.demote})
if a.join: print("合并后节点:", [k for k in types if any(n in k for n in joined_nodes) and k not in joined_nodes][:10])
# 8STD 重索引
os.chdir(f"{REPO}/experiments")
res2 = subprocess.run([sys.executable, "tools/dublin/reindex_8std.py", out_net, "--split-slots", "--tag", a.tag], capture_output=True, text=True)
print(res2.stdout.strip()[-1500:]); print(res2.stderr.strip()[-800:])
if res2.returncode != 0: sys.exit("reindex failed")
net8 = f"{REPO}/nets/dublin/dublin_8std_{a.tag}.net.xml"; meta8 = f"{REPO}/nets/dublin/dublin_8std_{a.tag}_meta.json"
import sumolib
net = sumolib.net.readNet(net8); meta = json.load(open(meta8))["tls"]
print(f"\n8STD 网 {net8}: TLS {len(meta)}")
for tid, t in meta.items():
    nph = sum(1 for x in t["actions"].values() if x.get("valid")); arms = {}
    for grp in t["links"].values():
        for c in grp: arms[c["approach"]] = min(arms.get(c["approach"], 1e9), net.getEdge(c["from_edge"]).getLength())
    print(f"  {tid[:40]:40s} 相位 {nph}  进口最短边 {{{', '.join(f'{k}: {v:.0f}' for k, v in sorted(arms.items()))}}}")
json.dump(dict(src=a.src, demote=a.demote, join=a.join, net=net8, meta=meta8, edges_gone=gone, edges_new=new), open(os.path.join(wd, "build_report.json"), "w"), indent=1)
