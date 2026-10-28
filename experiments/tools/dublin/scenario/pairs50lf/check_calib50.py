"""任务 5 的核对: calib18 扩展后 (1) 不给新参数时与改动前逐项相同; (2) 知道车型的控制器只依据数值档位 (对调车型 + 对调规格 → 逐车相同);
(3) --priority-table 同时作用于控制器、观测与奖励。全部在 18h 定稿第 1 份、种子 123 上跑, 结果写 build/calib18_checks.json。"""
import collections, json, os, subprocess, sys, xml.etree.ElementTree as ET
import common50 as P

PY = sys.executable; CAL = str(P.TOOLS18 / "calib18.py"); D = P.WORK / "checks"
R18 = ["background_cars_100.rou.xml", "bus_tt1.rou.xml", "amb_tt1.rou.xml"]
SWAP = {"bus": "ambulance", "ambulance": "bus"}


def swapped_files():
    """18h 第 1 份的公交、救护车文件, 只把 type 在 bus 与 ambulance 之间对调 (id、路线、时刻不变); 同时写清单。"""
    D.mkdir(parents=True, exist_ok=True); man = {}
    for f in R18[1:]:
        t = ET.parse(P.SRC18 / f)
        for v in t.getroot().iter("vehicle"):
            man[v.get("id")] = dict(type=v.get("type")); v.set("type", SWAP[v.get("type")])
        t.write(D / f.replace(".rou.xml", "_swapped.rou.xml"), encoding="UTF-8", xml_declaration=True)
    json.dump(man, open(D / "manifest18.json", "w"))
    return [str(P.SRC18 / R18[0])] + [str(D / f.replace(".rou.xml", "_swapped.rou.xml")) for f in R18[1:]]


def run(name, args):
    cmd = [PY, CAL, "--seed", "123", "--out", str(D / name)] + args
    return name, subprocess.Popen(cmd, cwd=str(P.TOOLS18), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS="1"))


def diff(a, b, path=""):
    """两个 JSON 结构的第一处不同 (None = 完全相同)。"""
    if type(a) != type(b): return f"{path}: 类型 {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: return f"{path}/{k}: 只在一边"
            d = diff(a[k], b[k], f"{path}/{k}")
            if d: return d
        return None
    if isinstance(a, list):
        if len(a) != len(b): return f"{path}: 长度 {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff(x, y, f"{path}[{i}]")
            if d: return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def strip_types(recs, swap=False):
    """逐车记录去掉车型 (对调前后车型本来就不同); 相遇里的车型按对调映射回来。"""
    out = {}
    for vid, r in recs.items():
        r = dict(r); r.pop("type")
        r["visits"] = [dict(x, enc=sorted(SWAP.get(e, e) if swap else e for e in x["enc"])) for x in r["visits"]]
        out[vid] = r
    return out


def main():
    P.check_inputs()
    base = ",".join(R18); sw = ",".join(swapped_files()); man = str(D / "manifest18.json")
    jobs = [run("fixed", ["--policy", "fixed", "--routes", base]), run("served", ["--policy", "served", "--routes", base]),
            run("prio531", ["--policy", "prio", "--routes", base])]
    for tab, rts, nm in (("531", base, "p531_orig"), ("351", sw, "p351_swap"), ("111", base, "p111_orig"), ("111", sw, "p111_swap")):
        jobs.append(run(nm, ["--policy", "prio", "--routes", rts, "--priority-table", tab, "--manifest", man, "--out-pairs", str(D / f"{nm}_pairs.json")]))
    for tab in ("531", "351", "111"):
        jobs.append(run(f"plumb{tab}", ["--policy", "prio", "--routes", base, "--priority-table", tab, "--check-priority"]))
    for nm, p in jobs:
        if p.wait() != 0: raise RuntimeError(f"{nm} 运行失败")
    W18 = P.SRC18 / "build"
    ref = {"fixed": W18 / "calib_check/fixed_123.json", "served": W18 / "calib_check/served_123.json", "prio531": W18 / "rank3/cur_prio_123.json"}
    res = {"default_unchanged": {}, "numeric_only": {}, "plumbing": {}}
    for k, f in ref.items():
        d = diff(json.load(open(f)), json.load(open(D / f"{k}.json")))
        res["default_unchanged"][k] = dict(identical=d is None, diff=d, reference=str(f))
    for a_, b_, key in (("p531_orig", "p351_swap", "prio531_vs_swapped351"), ("p111_orig", "p111_swap", "prio111_vs_swapped111")):
        ra = json.load(open(D / f"{a_}_pairs.json")); rb = json.load(open(D / f"{b_}_pairs.json"))
        d = diff(strip_types(ra), strip_types(rb, swap=True))
        ja, jb = json.load(open(D / f"{a_}.json")), json.load(open(D / f"{b_}.json"))
        net_d = diff({k: ja[k] for k in ("C1", "C8")}, {k: jb[k] for k in ("C1", "C8")})
        res["numeric_only"][key] = dict(identical=d is None and net_d is None, diff=d or net_d, n_vehicles=len(ra))
    for tab in ("531", "351", "111"):
        res["plumbing"][tab] = json.load(open(D / f"plumb{tab}.json"))["PRIO_CHECK"]
        res["plumbing"][tab]["expected_table"] = {t: P.PRIORITY_TABLES[tab][t] for t in ("car", "bus", "ambulance")}
        res["plumbing"][tab]["ctrl_table"] = {t: int(v) for t, v in res["plumbing"][tab]["ctrl_table"].items()}
    P.dump(res, "calib18_checks.json")
    for sec, v in res.items():
        for k, x in v.items(): print(sec, k, {kk: vv for kk, vv in x.items() if kk in ("identical", "diff", "n_vehicles", "obs_checked", "obs_mismatch", "reward_checked", "reward_max_abs_diff", "ctrl_table")})


if __name__ == "__main__":
    main()
