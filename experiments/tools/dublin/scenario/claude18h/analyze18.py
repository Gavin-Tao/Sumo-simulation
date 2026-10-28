"""逐路口体检 + 结构性基准差距 (不分车型的控制器下三类车的每次访问等待)。读 build/calib_check/*.json (calib18 带 C15 输出)。
输出 build/junction_check.json 与 build/junction_check.md。
口径: 每次访问 = 一辆车一次经过一个 RL 路口的进口车道; 等待 = 该次访问中的累计停车等待 (与训练统计器同源); 恒绿转向上的访问单列 (wandb per-visit 指标不含恒绿车道)。"""
import collections, glob, json, statistics
import common18 as C

def main():
    meta = C.load_meta(); D = C.load("demand_report.json"); AM = C.load("ambulances.json"); B = C.load("buses.json")
    ag = C.always_green(meta)
    runs = {p: [json.load(open(f)) for f in sorted(glob.glob(str(C.WORK / f"calib_check/{p}_*.json")))] for p in ("served", "fixed")}
    out = {"baseline": {}, "junctions": {}}
    for pol, rs in runs.items():
        agg = collections.defaultdict(lambda: [0, 0.0, 0])   # n, sum wait, zero-wait
        aggj = collections.defaultdict(lambda: [0, 0.0]); aggm = collections.defaultdict(lambda: [0, 0.0])
        for d in rs:
            for key, (n, s, z) in d["C15"]["by_type_ctrl"].items(): a = agg[key]; a[0] += n; a[1] += s; a[2] += z
            for key, (n, s) in d["C15"]["by_tid_type"].items(): a = aggj[key]; a[0] += n; a[1] += s
            for key, (n, s) in d["C15"]["by_mv_type"].items(): a = aggm[key]; a[0] += n; a[1] += s
        out["baseline"][pol] = {k: dict(visits=v[0], mean_wait=round(v[1] / max(1, v[0]), 2), zero_share=round(v[2] / max(1, v[0]), 3)) for k, v in agg.items()}
        lane_agg = collections.defaultdict(lambda: [0, 0.0]); seeds = collections.defaultdict(list)
        for d in rs:
            for ty, (n, s_) in d["C16"]["by_type"].items(): lane_agg[ty][0] += n; lane_agg[ty][1] += s_; seeds[ty].append(round(s_ / max(1, n), 2))
        out["baseline"][pol + "_lane"] = {ty: dict(visits=v[0], mean_wait=round(v[1] / max(1, v[0]), 2), per_seed=seeds[ty]) for ty, v in lane_agg.items()}
        out["baseline"][pol + "_by_tid"] = {k: dict(visits=v[0], mean_wait=round(v[1] / max(1, v[0]), 2)) for k, v in aggj.items()}
        out["baseline"][pol + "_by_mv"] = {k: dict(visits=v[0], mean_wait=round(v[1] / max(1, v[0]), 2)) for k, v in aggm.items()}
    # 逐路口体检
    lanes_of = collections.defaultdict(set)
    for t, m in meta.items():
        for g in m["links"].values():
            for c in g: lanes_of[t].add(c["from_edge"] + "_" + str(c["from_lane"]))
    amb_tids = {t for t, _ in AM["route"]["traversals"]}
    for t in meta:
        rs = runs["served"]
        jr = dict(
            demand={k.split("|")[1]: round(v["realized_total"]) for k, v in D["movements"].items() if k.startswith(t + "|")},
            always_green=sorted(m for (tt, m) in ag if tt == t), zero=[m for tt, m in C.ZERO_EXPECTED if tt == t],
            lane_load_cap={l: (round(D["lane_capacity"][l]["load"]), round(D["lane_capacity"][l]["cap"])) for l in sorted(lanes_of[t]) if l in D["lane_capacity"]},
            queue_step_share=[round(d["C10"][t], 3) for d in rs],
            c14_growth=[{e: v["grow"] for e, v in d["C14"].items() if v["tid"] == t} for d in rs][:1],
            refill_fast=[sum(v["refill_lt60"] for k, v in d["C5"]["by_slot"].items() if k.startswith(t + "|")) for d in rs],
            clears=[sum(v["clears"] for k, v in d["C5"]["by_slot"].items() if k.startswith(t + "|")) for d in rs],
            blocked=[sum(v for k, v in d["C3"]["blocked"].items() if k.startswith(t + "|")) for d in rs],
            opp={ty: [d["C15"]["opp_by_tid_type"].get(f"{t}|{ty}", [0, 0]) for d in rs] for ty in ("bus", "ambulance")},
            on_amb_route=t in amb_tids,
            wait_served={ty: out["baseline"]["served_by_tid"].get(f"{t}|{ty}") for ty in ("car", "bus", "ambulance")},
            wait_fixed={ty: out["baseline"]["fixed_by_tid"].get(f"{t}|{ty}") for ty in ("car", "bus", "ambulance")})
        out["junctions"][t] = jr
    C.dump(out, "junction_check.json")
    # 摘要打印
    for pol in ("served", "fixed"):
        print(f"[{pol}] 每次访问平均等待 (s):", {k: (v["mean_wait"], v["visits"]) for k, v in sorted(out["baseline"][pol].items())})
    print("逐路口 (排队驱动): 小汽车/公交/救护车 平均等待 (s, 访问数)")
    for t, j in out["junctions"].items():
        w = j["wait_served"]; f = lambda x: f"{x['mean_wait']:.1f}({x['visits']})" if x else "—"
        print(f"  {t.replace('cluster_','c')[:16]:16s} car {f(w['car']):>12s} bus {f(w['bus']):>12s} amb {f(w['ambulance']):>10s} | 排队步 {j['queue_step_share']} 堵死 {j['blocked']} 快速再满 {j['refill_fast']}/{j['clears']}")

if __name__ == "__main__":
    main()
