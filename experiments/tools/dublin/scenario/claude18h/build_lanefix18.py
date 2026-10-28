"""结构性空车道修复 (用户 2026-10-28 要求 "解决这个问题"): cluster_21620852 东进口第 0 车道 (37721356#0_0) 只能由
cluster_1998326407 南进口第 0 车道直行喂车, 但上游节点 658464 处 2294384#0 第 0 车道同时连向 4472590#0 第 0/1 车道, SUMO 把直行车全分到第 1 车道。
修复: 只改非 RL 节点 658464 的一条连接 — 2294384#0 第 0 车道 → 4472590#0 第 1 车道 改为 2294384#0 第 1 车道 → 4472590#0 第 1 车道
(第 0 车道 → 第 0 车道 [左转+直行]; 第 1 车道 → 第 1 [直行] + 第 2 [右转] 车道, 常见展宽方式)。
沿用 build_net18_nocg.py 的 plain 文件与 netconvert 参数, 再 8STD 重索引; 原路网文件不动, 输出 *_lanefix。核对: 18 个路口动作表与信号程序逐项一致, RL 进出车道长度/形状不变。"""
import json, shutil, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
import sumolib
import common18 as C
SRC = C.NETS / "build_corridor15/net18_nocg"; OUT = C.WORK / "lanefix"; OUT.mkdir(parents=True, exist_ok=True)
TAG = "splitslot18_southclosed_corridor_nocg_lanefix"
NETCONVERT = "/home/xiaowen/anaconda3/envs/sumo_rl/lib/python3.9/site-packages/sumo/bin/netconvert"

def main():
    for ext in ("nod", "edg", "con", "tll"):
        shutil.copy2(SRC / f"corridor18_nocg.{ext}.xml", OUT / f"lanefix.{ext}.xml")
    con = ET.parse(OUT / "lanefix.con.xml"); root = con.getroot(); changed = 0
    for c in root.findall("connection"):
        if c.get("from") == "2294384#0" and c.get("to") == "4472590#0" and c.get("fromLane") == "0" and c.get("toLane") == "1":
            c.set("fromLane", "1"); changed += 1
    assert changed == 1, changed
    con.write(OUT / "lanefix.con.xml", encoding="UTF-8", xml_declaration=True)
    raw = OUT / "lanefix.net.xml"
    r = subprocess.run([NETCONVERT, "-n", OUT / "lanefix.nod.xml", "-e", OUT / "lanefix.edg.xml", "-x", OUT / "lanefix.con.xml", "-i", OUT / "lanefix.tll.xml",
                        "--lefthand", "--offset.disable-normalization", "--no-turnarounds", "--junctions.corner-detail", "5", "-o", raw], capture_output=True, text=True)
    (OUT / "build.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, r.stderr[-1500:]
    r = subprocess.run([sys.executable, "-B", str(C.REPO / "experiments/tools/dublin/reindex_8std.py"), str(raw), "--split-slots", "--tag", TAG],
                       cwd=str(C.REPO / "experiments"), capture_output=True, text=True)
    (OUT / "reindex.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    new_net = C.NETS / f"dublin_8std_{TAG}.net.xml"; new_meta = C.NETS / f"dublin_8std_{TAG}_meta.json"
    # ---- 核对
    mo = json.load(open(C.META))["tls"]; mn = json.load(open(new_meta))["tls"]
    diff_meta = [t for t in mo if json.dumps(mo[t], sort_keys=True) != json.dumps(mn.get(t), sort_keys=True)]
    no = sumolib.net.readNet(str(C.NET)); nn = sumolib.net.readNet(str(new_net))
    rl_lanes = set()
    for t in mo:
        for c in no.getTLS(t).getConnections(): rl_lanes |= {c[0].getID(), c[1].getID()}
    geo = [l for l in sorted(rl_lanes) if abs(no.getLane(l).getLength() - nn.getLane(l).getLength()) > 0.01 or no.getLane(l).getShape() != nn.getLane(l).getShape()]
    conn = lambda n: {(c.getFrom().getID(), c.getFromLane().getIndex(), c.getTo().getID(), c.getToLane().getIndex()) for e in n.getEdges() for l in e.getLanes() for c in l.getOutgoing()}
    co, cn = conn(no), conn(nn)
    rep = dict(changed_connection="2294384#0_0→4472590#0_1 改为 2294384#0_1→4472590#0_1", meta_diff_tls=diff_meta, rl_lane_geometry_changes=geo,
               connections_removed=sorted(map(list, co - cn)), connections_added=sorted(map(list, cn - co)),
               net=str(new_net.relative_to(C.REPO)), meta=str(new_meta.relative_to(C.REPO)), net_sha=C.sha256(new_net), meta_sha=C.sha256(new_meta))
    json.dump(rep, open(OUT / "lanefix_report.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
