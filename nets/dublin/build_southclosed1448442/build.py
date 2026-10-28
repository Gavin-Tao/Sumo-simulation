"""Build isolated 1448442 south-branch-closed variants; never overwrite the sources.

Run with the sumo_rl Python environment. No training is started.
"""
from pathlib import Path
import collections
import hashlib
import heapq
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import sumolib

REPO = Path(__file__).resolve().parents[3]
NETS = REPO / "nets/dublin"
WORK = Path(__file__).resolve().parent
SOURCE_NET = NETS / "dublin_8std_splitslot15.net.xml"
SOURCE_META = NETS / "dublin_8std_splitslot15_meta.json"
TAG = "splitslot15_southclosed"
NEW_NET = NETS / f"dublin_8std_{TAG}.net.xml"
NEW_META = NETS / f"dublin_8std_{TAG}_meta.json"
SCENES = ("weekday_pairs50_15", "weekday_18h_15")
CLOSED = {"14152304#2", "-14152304#3"}
RENAMES = {"532427444#3": "532427444#7", "532427444#4": "532427444#7",
           "-532427444#7": "-532427444#3", "-532427444#6": "-532427444#3"}
CHAINS = (("532427444#3", "532427444#4", "532427444#7"),
          ("-532427444#7", "-532427444#6", "-532427444#3"))
INTERNAL_PAIRS = {pair for chain in CHAINS for pair in zip(chain, chain[1:])}
REMOVED_NODES = ("135109542", "3506865299")


def remap_links(links):
    result = json.loads(json.dumps(links))
    for group in result.values():
        for c in group:
            for k in ("from_edge", "to_edge"):
                c[k] = RENAMES.get(c[k], c[k])
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    p = subprocess.run([str(a) for a in args], cwd=REPO, capture_output=True, text=True)
    with (WORK / "build.log").open("a") as f:
        f.write("COMMAND: " + " ".join(map(str, args)) + "\n" + p.stdout + p.stderr + "\n")
    if p.returncode:
        raise RuntimeError(p.stdout + p.stderr)
    return p.stdout


def connection_set(root):
    return {(c.get("from"), c.get("to"), c.get("fromLane"), c.get("toLane"))
            for c in root.findall("connection") if not c.get("from", "").startswith(":")}


def compress(edges):
    out = []
    for edge in edges:
        edge = RENAMES.get(edge, edge)
        if not out or edge != out[-1]:
            out.append(edge)
    return out


def lane_path(net, edges, departure_lane):
    lanes = {departure_lane}
    for fr, to in zip(edges, edges[1:]):
        if not net.hasEdge(fr) or not net.hasEdge(to):
            return False
        lanes = {c.getToLane().getIndex() for i in lanes
                 for c in net.getEdge(fr).getLane(i).getOutgoing()
                 if c.getTo().getID() == to and c.getToLane().allows("passenger")}
        if not lanes:
            return False
    return bool(lanes)


def repair_route(net, edges, departure_lane):
    """Lane-connected route with the same OD and departure lane.

    Prefer original edges, then distance. No lane changes are needed.
    """
    preferred = set(edges) - CLOSED
    start = (edges[0], departure_lane)
    assert net.hasEdge(start[0]) and net.hasEdge(edges[-1]), edges
    best = {start: (0, 0.0)}
    parent = {start: None}
    queue = [(0, 0.0, start)]
    final = None
    while queue:
        added, distance, current = heapq.heappop(queue)
        if best[current] != (added, distance):
            continue
        edge, lane = current
        if edge == edges[-1]:
            final = current
            break
        for c in net.getEdge(edge).getLane(lane).getOutgoing():
            target = c.getToLane()
            if not target.allows("passenger"):
                continue
            nxt = (c.getTo().getID(), target.getIndex())
            cost = (added + (nxt[0] not in preferred), distance + target.getLength())
            if nxt not in best or cost < best[nxt]:
                best[nxt] = cost
                parent[nxt] = current
                heapq.heappush(queue, (*cost, nxt))
    if final is None:
        raise ValueError(f"No lane-connected replacement: {edges[0]}:{departure_lane} -> {edges[-1]}")
    path = []
    while final is not None:
        path.append(final[0])
        final = parent[final]
    path.reverse()
    assert lane_path(net, path, departure_lane), path
    return path


def projected_position(old_lane, new_lane, at_end=False):
    """Project an old endpoint onto the merged lane in SUMO distance units."""
    point = old_lane.getShape()[-1 if at_end else 0]
    shape = new_lane.getShape()
    best = (math.inf, 0.0)
    covered = 0.0
    for a, b in zip(shape, shape[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        q = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length**2))
        distance = math.hypot(point[0] - a[0] - q * dx, point[1] - a[1] - q * dy)
        best = min(best, (distance, covered + q * length))
        covered += length
    return best[1] / covered * new_lane.getLength()


def extend_terminal(net, edges, departure_lane):
    """Continue a newly internal terminal through the downstream TLS, no lane change.

    Choose the shortest lane-connected continuation ending at a non-TLS node;
    the original route prefix and departure stay unchanged.
    """
    lanes = {departure_lane}
    for fr, to in zip(edges, edges[1:]):
        lanes = {c.getToLane().getIndex() for i in lanes
                 for c in net.getEdge(fr).getLane(i).getOutgoing() if c.getTo().getID() == to}
    heap = [(0.0, (edges[-1], i), ()) for i in lanes]
    heapq.heapify(heap); seen = set()
    while heap:
        distance, current, suffix = heapq.heappop(heap)
        if current in seen:
            continue
        seen.add(current)
        edge, lane = current
        if suffix and net.getEdge(edge).getToNode().getType() != "traffic_light":
            result = edges + list(suffix)
            assert lane_path(net, result, departure_lane)
            return result
        for c in net.getEdge(edge).getLane(lane).getOutgoing():
            target = c.getToLane()
            if target.allows("passenger") and c.getTo().getID() not in edges:
                nxt = (c.getTo().getID(), target.getIndex())
                heapq.heappush(heap, (distance + target.getLength(), nxt, suffix + (nxt[0],)))
    raise ValueError(f"Cannot extend terminal: {edges}")


def main():
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    destinations = [NETS / (s + "_southclosed") for s in SCENES]
    assert not any(p.exists() for p in [NEW_NET, NEW_META, *destinations]), "Variant already exists; do not overwrite it."
    originals = [SOURCE_NET, SOURCE_META]
    for scene in SCENES:
        originals.extend(p for p in (NETS / scene).rglob("*") if p.is_file())
    source_hashes = {str(p.relative_to(REPO)): digest(p) for p in originals}
    source_root = ET.parse(SOURCE_NET).getroot()
    old_net = sumolib.net.readNet(str(SOURCE_NET))

    with tempfile.TemporaryDirectory(prefix="plain_", dir=WORK) as td:
        prefix = Path(td) / "source"
        run([sumolib.checkBinary("netconvert"), "-s", SOURCE_NET, "--plain-output-prefix", prefix, "--offset.disable-normalization"])
        trees = {s: ET.parse(str(prefix) + s) for s in (".nod.xml", ".edg.xml", ".con.xml", ".tll.xml")}
        nodes = {n.get("id"): n for n in trees[".nod.xml"].getroot()}
        edges = {e.get("id"): e for e in trees[".edg.xml"].getroot()}
        for chain in CHAINS:
            first, last = edges[chain[0]], edges[chain[-1]]
            assert all(edges[a].get("to") == edges[b].get("from") for a,b in zip(chain,chain[1:]))
            assert len({edges[i].get("numLanes") for i in chain}) == 1
            keep = RENAMES.get(chain[0], chain[0])
            e = edges[keep]
            node_ids = [first.get("from")] + [edges[i].get("to") for i in chain]
            e.set("from", node_ids[0]); e.set("to", node_ids[-1])
            e.set("shape", " ".join(f"{nodes[n].get('x')},{nodes[n].get('y')}" for n in node_ids))
            e.set("spreadType", "right")
        for edge in CLOSED | set(RENAMES):
            trees[".edg.xml"].getroot().remove(edges[edge])
        for node in REMOVED_NODES:
            trees[".nod.xml"].getroot().remove(nodes[node])
        internal_pairs = INTERNAL_PAIRS
        for suffix in (".con.xml", ".tll.xml"):
            root = trees[suffix].getroot()
            for c in list(root.findall("connection")):
                fr, to = c.get("from"), c.get("to")
                if fr in CLOSED or to in CLOSED or (fr, to) in internal_pairs:
                    root.remove(c)
                    continue
                if fr in RENAMES:
                    c.set("from", RENAMES[fr])
                if to in RENAMES:
                    c.set("to", RENAMES[to])
        for suffix, tree in trees.items():
            tree.write(WORK / ("modified" + suffix), encoding="UTF-8", xml_declaration=True)

    raw_net = WORK / "closed_merged.net.xml"
    run([sumolib.checkBinary("netconvert"), "-n", WORK / "modified.nod.xml", "-e", WORK / "modified.edg.xml",
         "-x", WORK / "modified.con.xml", "-i", WORK / "modified.tll.xml", "--lefthand", "--offset.disable-normalization",
         "--no-turnarounds", "--junctions.corner-detail", "5", "-o", raw_net])
    expected = {(RENAMES.get(fr, fr), RENAMES.get(to, to), fl, tl)
                for fr, to, fl, tl in connection_set(source_root)
                if fr not in CLOSED and to not in CLOSED and (fr, to) not in INTERNAL_PAIRS}
    assert connection_set(ET.parse(raw_net).getroot()) == expected, "Unexpected connectivity change"
    run([sys.executable, "-B", REPO / "experiments/tools/dublin/reindex_8std.py", raw_net, "--split-slots", "--tag", TAG])
    new_meta = json.loads(NEW_META.read_text())
    old_meta = json.loads(SOURCE_META.read_text())
    assert set(new_meta["tls"]) == set(old_meta["tls"])
    for tid in old_meta["tls"]:
        for key in ("mask", "canonical", "std_to_green_index", "actions"):
            assert old_meta["tls"][tid][key] == new_meta["tls"][tid][key], (tid, key)
        assert remap_links(old_meta['tls'][tid]['links']) == new_meta['tls'][tid]['links'], (tid, 'links')
    new_net = sumolib.net.readNet(str(NEW_NET))
    report = dict(source_hashes=source_hashes, closed_edges=sorted(CLOSED), removed_nodes=REMOVED_NODES, renames=RENAMES,
                  net=str(NEW_NET.relative_to(REPO)), meta=str(NEW_META.relative_to(REPO)), scenes={},
                  south_inlet_lengths={l.getID(): l.getLength() for l in new_net.getEdge("532427444#7").getLanes()},
                  south_outlet_lengths={l.getID(): l.getLength() for l in new_net.getEdge("-532427444#3").getLanes()},
                  action_semantics_unchanged=True)

    for scene, dest in zip(SCENES, destinations):
        shutil.copytree(NETS / scene, dest)
        (dest / "README.md").rename(dest / "SOURCE_README.md")
        if (dest / "calib").exists():
            (dest / "calib").rename(dest / "source_calib")
        xmls = {p.name: ET.parse(p) for p in dest.glob("*.rou.xml")}
        route_defs = {r.get("id"): r for t in xmls.values() for r in t.getroot().findall("route") if r.get("id")}
        original_routes = {key: r.get("edges").split() for key, r in route_defs.items()}
        named_lanes = collections.defaultdict(set)
        for t in xmls.values():
            for v in t.getroot().findall("vehicle"):
                if v.get("route"):
                    named_lanes[v.get("route")].add(int(v.get("departLane", "0")))
        edits = []; file_counts = {}; position_edits = []; terminal_edits = []
        for key, r in route_defs.items():
            old = original_routes[key]; revised = compress(old)
            assert len(named_lanes[key]) <= 1, key
            dl = next(iter(named_lanes[key]), 0)
            if CLOSED & set(revised) or not lane_path(new_net, revised, dl):
                revised = repair_route(new_net, revised, dl)
                edits.append(dict(file="named_route", id=key, old=old, new=revised))
            r.set("edges", " ".join(revised))
        for filename, tree in xmls.items():
            n = 0
            for v in tree.getroot().findall("vehicle"):
                n += 1
                dl = int(v.get("departLane", "0"))
                route = v.find("route")
                if route is None:
                    old = original_routes[v.get("route")]
                    revised = route_defs[v.get("route")].get("edges").split()
                else:
                    old = route.get("edges").split(); revised = compress(old)
                    if CLOSED & set(revised) or not lane_path(new_net, revised, dl):
                        revised = repair_route(new_net, revised, dl)
                        edits.append(dict(file=filename, id=v.get("id"), old=old, new=revised))
                    route.set("edges", " ".join(revised))
                if old[-1] in RENAMES and new_net.getEdge(revised[-1]).getToNode().getType() == 'traffic_light':
                    assert route is not None, 'Named terminal route needs explicit extension'
                    before = revised[:]
                    revised = extend_terminal(new_net, revised, dl)
                    route.set('edges', ' '.join(revised))
                    terminal_edits.append(dict(file=filename, id=v.get('id'), old_terminal=old[-1],
                                               new_terminal=revised[-1], added_edges=revised[len(before):]))
                assert lane_path(new_net, revised, dl), (filename, v.get("id"))
                assert revised[0] == RENAMES.get(old[0], old[0])
                assert revised[-1] == RENAMES.get(old[-1], old[-1]) or terminal_edits[-1]['id'] == v.get('id')
                # Preserve positions for traffic that originally began/ended within the now-merged southbound edge.
                if old[0] in RENAMES:
                    assert v.get("departPos") is None
                    offset = projected_position(old_net.getLane(old[0] + f"_{dl}"), new_net.getLane(revised[0] + f"_{dl}"))
                    v.set("departPos", f"{offset + 5.1:.3f}")
                    position_edits.append(dict(file=filename, id=v.get("id"), departPos=v.get("departPos")))
            file_counts[filename] = n
            tree.write(dest / filename, encoding="UTF-8", xml_declaration=True)
        for p in dest.glob("*.sumocfg"):
            cfg = ET.parse(p); cfg.getroot().find("input/net-file").set("value", "../" + NEW_NET.name)
            cfg.write(p, encoding="UTF-8", xml_declaration=True)
        # Previous reports describe the source network, not this variant.
        for name in ("background_report.json", "pairs_report.json"):
            if (dest / name).exists():
                (dest / name).rename(dest / ("source_" + name))
        info = dict(source=scene, vehicles_by_file=file_counts, detours=edits, preserved_position_edits=position_edits,
                    extended_terminals=terminal_edits)
        (dest / "modification_report.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        report[scene] = info
        report["scenes"][scene] = dict(path=str(dest.relative_to(REPO)), vehicles_by_file=file_counts,
                                      detours=len(edits), position_edits=len(position_edits), extended_terminals=len(terminal_edits))
        (dest / "README.md").write_text(
            f"# {dest.name}\n\n"
            f"从当前 `{scene}` 独立复制; 原场景和原路网不改。\n\n"
            "## 改造\n\n"
            "封闭 135109542 的侧路连接 (`14152304#2`、`-14152304#3`), 删除该节点, "
            "同时删除仅有直行连接的 3506865299, 南北主路各合并为一条边, 直连 1448442 与 389281; "
            "两个路口均保留独立信号及原有动作语义。\n\n"
            f"共用路网: `../{NEW_NET.name}`; 配套 meta: `../{NEW_META.name}`。"
            "训练时 net_file、action_meta_file、cfg_file 和 route_file 必须一起使用本副本的路径。"
            "本次没有创建训练配置或启动训练。\n\n"
            f"南进口车道长度: `{report['south_inlet_lengths']}`; 南出口: `{report['south_outlet_lengths']}`。"
            "RL 直接读取合并后的完整进口及其实际长度; 不修改观测维数或奖励公式。\n\n"
            "## 路线\n\n"
            "车辆 ID、类型、数量、出发时间、出发车道保留。旧边名按合并关系替换; "
            "穿过封闭连接的路线改走保持同一起终边、指定出发车道可连续通过的替代路径。"
            "原来在被合并路段中间出发的背景车使用 departPos 保留对应位置。"
            "pairs50 有 290 辆背景车的旧终点因此落入 RL 进口, 已沿可连续通过的车道延长至下游非 RL 节点; "
            "这会增加下游需求, 与旧场景不能视为仅几何变化。18h 没有该类终点, 起终边保留。"
            "详细差异见 modification_report.json, 包含备用未见路线与三类孪生探针。\n\n"
            "## 来源与验证\n\n"
            "SOURCE_README.md、source_calib 和 source_*report.json 是原版本快照, "
            "其中旧路径、需求和标定数值不能作为本副本的新结果。"
            "生成与输入哈希记录在 ../build_southclosed1448442/build_report.json。\n",
            encoding="UTF-8")
    assert all(digest(REPO / p) == h for p, h in source_hashes.items()), "Source files changed"
    (WORK / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("net", "south_inlet_lengths", "south_outlet_lengths", "scenes")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
