"""状态/奖励逐步独立重算 (复用 nets/dublin/build_corridor15/validate_net18.py 的校验逻辑, 路网设置同 validate_net18_nocg.py),
路由换成本场景 (背景 1.0 档 + 公交 tt1 + 救护车 tt1, 合并为一个文件)。原校验要求所有车到达终点; 本场景集末才出发的车开不完属正常,
因此只把 "到达数" 单独报告, 其余断言 (状态、奖励、统计器、碰撞、瞬移、换道、车身重叠) 照常检查。
用法: python validate18.py fixed|random 123"""
import json, sys, xml.etree.ElementTree as ET
import common18 as C
sys.path.insert(0, str(C.NETS / "build_corridor15"))
import sumolib
import validate_net18 as diagnostic
from build_net18_nocg import CG

def make_routes(label):
    out = C.WORK / "validation"; out.mkdir(parents=True, exist_ok=True)
    root = ET.Element("routes"); vs = []
    for f in ["background_cars_100.rou.xml", "bus_tt1.rou.xml", "amb_tt1.rou.xml"]:
        vs += list(ET.parse(C.OUT / f).getroot().iter("vehicle"))
    for v in sorted(vs, key=lambda v: float(v.get("depart"))): root.append(v)
    p = out / f"merged_{label}.rou.xml"; ET.ElementTree(root).write(p, encoding="UTF-8", xml_declaration=True)
    return p, len(vs)

def main():
    policy = sys.argv[1] if len(sys.argv) > 1 else "fixed"; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 123
    C.check_hashes(); dest = C.WORK / "validation"; dest.mkdir(parents=True, exist_ok=True)
    diagnostic.NET, diagnostic.META, diagnostic.OUT = C.NET, C.META, dest
    diagnostic.net = sumolib.net.readNet(str(C.NET)); diagnostic.meta = json.loads(C.META.read_text())["tls"]
    diagnostic.NEW = {"659799", "659805", "9101555", CG}; diagnostic.NORTH_JOIN = CG
    diagnostic.EXPANDED = {"668344588": ("1448436", 47.69)}
    try:
        diagnostic.main(policy, seed, steps=720, route_factory=make_routes, require_no_lane_changes=True)
        status = "全部断言通过"
    except AssertionError as e:
        status = f"断言: {str(e)[:300]}"
    rep = json.loads((dest / f"{policy}_{seed}_report.json").read_text())
    ok = rep["decisions"] == 720 and not rep["collisions"] and not rep["teleports"] and not rep["tracker_missing"] and not rep["lane_changes"] and not rep["body_overlap_samples"]
    summary = dict(policy=policy, seed=seed, decisions=rep["decisions"], state_checks=rep["state_checks"], reward_checks=rep["reward_checks"], tracker_checks=rep["tracker_checks"],
                   collisions=rep["collisions"], teleports=rep["teleports"], tracker_missing=rep["tracker_missing"], lane_changes=rep["lane_changes"],
                   body_overlap=len(rep["body_overlap_samples"]), planned=rep["planned"], arrived=rep["arrived"], remaining_at_end=len(rep.get("remaining", [])),
                   pending_at_end=len(rep.get("pending", [])), status=status, verdict="通过" if ok else "不通过")
    json.dump(summary, open(dest / f"summary_{policy}_{seed}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
