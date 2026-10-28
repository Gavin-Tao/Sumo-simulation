"""写场景文件 (nets/dublin/weekday_18h_nocg_claude/)。所有车辆: 车型同参数、禁止换道 (vtypes_same.add.xml)、写明 departLane、departSpeed=max、路线车道连续。"""
import collections, json, random, shutil, xml.etree.ElementTree as ET
import common18 as C

VT_SRC = C.NETS / "weekday_18h_15_southclosed_corridor/vtypes_same.add.xml"

def write_routes(path, vehicles, comment):
    root = ET.Element("routes"); root.append(ET.Comment(comment))
    for v in sorted(vehicles, key=lambda x: (x["depart"], x["id"])):
        e = ET.SubElement(root, "vehicle", id=v["id"], type=v["type"], depart=f"{v['depart']:.1f}", departLane=str(v["lane"]), departSpeed="max")
        ET.SubElement(e, "route", edges=" ".join(v["edges"]))
    ET.indent(root, space="    "); ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)

def place(t, taken):
    """在入口车道已占用时刻 taken 中, 把 t 向后挪到与所有已占用时刻相距 ≥ 3 s。"""
    while any(abs(t - x) < C.MIN_HEADWAY for x in taken): t = round(t + 0.5, 1)
    return t

def main():
    C.check_hashes(); bg = C.load("background.json"); buses = C.load("buses.json"); amb = C.load("ambulances.json")
    out = C.OUT; out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VT_SRC, out / "vtypes_same.add.xml")
    routes = bg["routes"]
    cars = [dict(id=c["id"], type="car", depart=c["depart"], lane=routes[c["route"]]["lane"], edges=routes[c["route"]]["edges"], entry=c["entry"], key=c["key"]) for c in bg["cars"]]
    for L in C.LEVELS:
        sub = [c for c in cars if c["key"] < L]
        write_routes(out / f"background_cars_{int(L * 100):03d}.rou.xml", sub, f"claude18h 背景小汽车 {L:.1f} 档 (嵌套抽稀, 车辆 id 与 1.0 档一致); 平台期流率按 demand_report.json")
    car_taken = collections.defaultdict(list)
    for c in cars: car_taken[c["entry"]].append(c["depart"])
    rt = amb["route"]; amb_entry = rt["lanes"][0]
    def bus_tt(k):
        if str(k) in buses["timetables"]: return buses["timetables"][str(k)]
        rng = random.Random(1000 + k)
        return {b["id"]: round(min(3500.0, max(0.0, b["depart"] + rng.uniform(-60, 60))), 1) for b in buses["buses"]}
    moved = collections.Counter(); files = {}
    for k in range(1, 14):
        taken = {l: list(v) for l, v in car_taken.items()}
        amb_times = amb["timetables"][str(k)] if k <= C.N_TT else amb["eval_timetables"][str(k)]
        av = []
        for i, t in enumerate(amb_times):
            t2 = place(t, taken.setdefault(amb_entry, [])); moved["amb"] += t2 != t; taken[amb_entry].append(t2)
            av.append(dict(id=f"amb_{i + 1:02d}", type="ambulance", depart=t2, lane=rt["lane"], edges=rt["edges"]))
        tt = bus_tt(k); bv = []
        for b in sorted(buses["buses"], key=lambda b: tt[b["id"]]):
            t2 = place(tt[b["id"]], taken.setdefault(b["entry"], [])); moved["bus"] += t2 != tt[b["id"]]; taken[b["entry"]].append(t2)
            bv.append(dict(id=b["id"], type="bus", depart=t2, lane=b["lane"], edges=b["edges"]))
        tag = "train" if k <= 5 else ("calib" if k == 6 else ("test" if k == 7 else "eval"))
        write_routes(out / f"bus_tt{k}.rou.xml", bv, f"claude18h GTFS 18 点公交, 时刻表 {k} ({tag}): ±60 s 抖动")
        write_routes(out / f"amb_tt{k}.rou.xml", av, f"claude18h 救护车, 时刻表 {k} ({tag}): 每集 3 辆, 一条设计路线")
        write_routes(out / f"bus_tt{k}_as_car.rou.xml", [dict(v, type="car") for v in bv], f"换车型孪生 (只供评估): 公交时刻表 {k} 以 car 身份")
        write_routes(out / f"amb_tt{k}_as_car.rou.xml", [dict(v, type="car") for v in av], f"换车型孪生 (只供评估): 救护车时刻表 {k} 以 car 身份")
        files[k] = tag
    # 评估: 频率补充测试 (每集约 20 辆) 与未见路线
    taken = {l: list(v) for l, v in car_taken.items()}
    fp = []
    for i, t in enumerate(amb["freq_probe_departs"]):
        t2 = place(t, taken.setdefault(amb_entry, [])); taken[amb_entry].append(t2)
        fp.append(dict(id=f"ambprobe_{i + 1:02d}", type="ambulance", depart=t2, lane=rt["lane"], edges=rt["edges"]))
    write_routes(out / "amb_freqprobe_eval.rou.xml", fp, "只供评估: 救护车频率补充测试 (每集约 20 辆, 间隔 ≥100 s); 与 bus_tt7 一起用")
    write_routes(out / "amb_freqprobe_eval_as_car.rou.xml", [dict(v, type="car") for v in fp], "只供评估: 上一文件的小汽车孪生")
    un = []
    for i, x in enumerate(amb["unseen_timetable"]):
        u = amb["unseen_routes"][x["route"]]; t2 = place(x["depart"], taken.setdefault(u["lanes"][0], [])); taken[u["lanes"][0]].append(t2)
        un.append(dict(id=f"ambunseen_{i + 1:02d}", type="ambulance", depart=t2, lane=u["lane"], edges=u["edges"]))
    write_routes(out / "amb_unseen_eval.rou.xml", un, "只供评估: 3 条未见路线 (不与训练路线共享 RL 进口边) 的救护车")
    write_routes(out / "amb_unseen_eval_as_car.rou.xml", [dict(v, type="car") for v in un], "只供评估: 上一文件的小汽车孪生")
    # sumocfg (训练与标定一致: 不忽略路口阻塞, 开路口碰撞检查)
    cfg = ET.Element("configuration"); inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value="../" + C.NET.name); ET.SubElement(inp, "additional-files", value="vtypes_same.add.xml")
    ET.SubElement(inp, "route-files", value="background_cars_100.rou.xml,bus_tt1.rou.xml,amb_tt1.rou.xml")
    tm = ET.SubElement(cfg, "time"); ET.SubElement(tm, "begin", value="0"); ET.SubElement(tm, "end", value="3600")
    pr = ET.SubElement(cfg, "processing"); ET.SubElement(pr, "time-to-teleport", value="-1"); ET.SubElement(pr, "ignore-junction-blocker", value="-1")
    ET.SubElement(pr, "collision.check-junctions", value="true"); ET.SubElement(pr, "collision.action", value="warn")
    rp = ET.SubElement(cfg, "report"); ET.SubElement(rp, "no-step-log", value="true"); ET.SubElement(rp, "duration-log.statistics", value="true")
    ET.indent(cfg, space="    "); ET.ElementTree(cfg).write(out / "scenario.sumocfg", encoding="UTF-8", xml_declaration=True)
    C.dump(dict(timetables=files, moved_for_headway=dict(moved), n_cars={f"{L:.1f}": sum(1 for c in cars if c["key"] < L) for L in C.LEVELS}, n_bus=len(buses["buses"]), amb_per_episode=C.AMB_PER_EP), "write_report.json")
    print("写出", out, "; 小汽车", {f"{L:.1f}": sum(1 for c in cars if c["key"] < L) for L in C.LEVELS}, "; 为保证间隔挪动", dict(moved))

if __name__ == "__main__":
    main()
