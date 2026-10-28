"""任务 7: 状态/奖励逐步独立重算 (复用 build_corridor15/validate_net18.py 的校验逻辑, 同 claude18h/validate18.py), 路由换成本场景
(背景 1.0 档 + GTFS 公交以 car 身份 + 配对车辆第 1 份); 另在第 1 份与第 1′ 份上、531 与 351 下各起新进程, 逐步核对观测档位与奖励权重 (calib18 --check-priority)。
结果写 build/validation/。用法: python validate50.py"""
import json, os, subprocess, sys, xml.etree.ElementTree as ET
import common50 as P

PY = sys.executable; D = P.WORK / "validation"


def make_routes(label):
    D.mkdir(parents=True, exist_ok=True)
    root = ET.Element("routes"); vs = []
    for f in list(P.BG_DEFAULT) + ["pairs_tt1.rou.xml"]:
        vs += list(ET.parse(P.OUT / f).getroot().iter("vehicle"))
    for v in sorted(vs, key=lambda v: float(v.get("depart"))): root.append(v)
    p = D / f"merged_{label}.rou.xml"; ET.ElementTree(root).write(p, encoding="UTF-8", xml_declaration=True)
    return p, len(vs)


def one(policy, seed):
    """在子进程里调用 validate_net18 (它改模块全局量), 与 validate18.py 相同的设置。"""
    sys.path.insert(0, str(P.NETS / "build_corridor15"))
    import sumolib, validate_net18 as diagnostic
    from build_net18_nocg import CG
    diagnostic.NET, diagnostic.META, diagnostic.OUT = P.NET, P.META, D
    diagnostic.net = sumolib.net.readNet(str(P.NET)); diagnostic.meta = json.loads(P.META.read_text())["tls"]
    diagnostic.NEW = {"659799", "659805", "9101555", CG}; diagnostic.NORTH_JOIN = CG
    diagnostic.EXPANDED = {"668344588": ("1448436", 47.69)}
    try:
        diagnostic.main(policy, seed, steps=720, route_factory=make_routes, require_no_lane_changes=True); status = "全部断言通过"
    except AssertionError as e:
        status = f"断言: {str(e)[:300]}"
    rep = json.loads((D / f"{policy}_{seed}_report.json").read_text())
    ok = rep["decisions"] == 720 and not rep["collisions"] and not rep["teleports"] and not rep["tracker_missing"] and not rep["lane_changes"] and not rep["body_overlap_samples"]
    summ = dict(policy=policy, seed=seed, decisions=rep["decisions"], state_checks=rep["state_checks"], reward_checks=rep["reward_checks"], tracker_checks=rep["tracker_checks"],
                collisions=rep["collisions"], teleports=rep["teleports"], tracker_missing=rep["tracker_missing"], lane_changes=rep["lane_changes"],
                body_overlap=len(rep["body_overlap_samples"]), planned=rep["planned"], arrived=rep["arrived"], status=status, verdict="通过" if ok else "不通过")
    json.dump(summ, open(D / f"summary_{policy}_{seed}.json", "w"), indent=1, ensure_ascii=False)


def main():
    if len(sys.argv) == 3: one(sys.argv[1], int(sys.argv[2])); return
    P.check_inputs(); D.mkdir(parents=True, exist_ok=True); env = dict(os.environ, OMP_NUM_THREADS="1")
    procs = [subprocess.Popen([PY, __file__, pol, "123"], cwd=str(P.TOOLS), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env) for pol in ("fixed", "random")]
    man = str(P.OUT / "pairs_manifest.json")
    for tab in ("531", "351"):
        for suf in ("", "_mirror"):
            routes = ",".join([str(P.OUT / f) for f in P.BG_DEFAULT] + [str(P.OUT / f"pairs_tt1{suf}.rou.xml")])
            procs.append(subprocess.Popen([PY, str(P.TOOLS18 / "calib18.py"), "--policy", "prio", "--seed", "123", "--routes", routes, "--priority-table", tab,
                                           "--manifest", man, "--check-priority", "--out", str(D / f"levels_{tab}{suf or '_orig'}")], cwd=str(P.TOOLS18),
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env))
    for p in procs: p.wait()
    out = dict(validation={pol: json.load(open(D / f"summary_{pol}_123.json")) for pol in ("fixed", "random")}, levels={})
    for tab in ("531", "351"):
        for suf in ("_orig", "_mirror"):
            c = json.load(open(D / f"levels_{tab}{suf}.json"))["PRIO_CHECK"]
            out["levels"][f"{tab}{suf}"] = {k: c[k] for k in ("obs_checked", "obs_mismatch", "reward_checked", "reward_max_abs_diff", "ctrl_table", "expected_table")}
    P.dump(out, "validation.json")
    for k, v in out["validation"].items(): print(k, {x: v[x] for x in ("decisions", "state_checks", "reward_checks", "tracker_checks", "collisions", "teleports", "tracker_missing", "lane_changes", "body_overlap", "verdict")})
    for k, v in out["levels"].items(): print(k, v)


if __name__ == "__main__":
    main()
