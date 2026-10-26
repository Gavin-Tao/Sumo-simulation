"""全网静态审计 (2026-10-26, 状态/奖励修正实施计划任务 9)。只读分析, 不训练。

对一个配置的路网 + meta, 逐路口核对:
  1. 观测类实际用的槽表 (容量 / 出口车道 / lane_occ 供车车道) 与"遍历全部连接"重建的表是否一致 (旧口径下应只在多连接索引路口不一致);
  2. 同一信号索引下各连接的 meta 槽标签是否一致 (turnmap_from_meta 会报错);
  3. 每个绿相位里同一槽的各索引灯色是否一致 (不一致 = is_green 对槽内部分车辆为错);
  4. meta 转向标签与路网 netconvert 标注 (dir_turn) 不同的连接 (命名合并, 信息项);
  5. 进口边上不受本信号控制的连接 (走这些连接的车会被丢弃);
  6. 可选 --steps N: 跑 N 个决策步 (贪心策略), 统计被丢弃 (槽 -1) 的车辆观测与未映射的边对。
用法: python experiments/tools/dublin/state_audit_full.py <exp> [--net splitslot] [--all-connections] [--steps 0] [--out report.json]"""
import argparse, collections, json, os, sys, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
APP = ("N", "E", "S", "W"); TURN = ("L", "T", "R")
name = lambda s: APP[s // 3] + "-" + TURN[s % 3]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--net", choices=["base", "splitslot"], default="base")
    ap.add_argument("--all-connections", action="store_true"); ap.add_argument("--steps", type=int, default=0); ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out"); a = ap.parse_args()
    import ckpt_env, yaml
    from exp_params import turnmap_from_meta
    ov = {}
    if a.net == "splitslot":
        cfg0 = yaml.safe_load(open(ckpt_env.find_cfg(a.exp)))
        ov.update(net_file="../nets/dublin/dublin_8std_splitslot.net.xml", action_meta_file="../nets/dublin/dublin_8std_splitslot_meta.json",
                  cfg_file=cfg0["cfg_file"].replace(".sumocfg", "_8std_splitslot.sumocfg"))
    if a.all_connections:
        ov.update(obs_all_connections=True, all_connections=True)
    B = ckpt_env.build(a.exp, seed=a.seed, cfg_override=ov); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]; sumo = env.sumo
    meta = json.load(open(cfg["action_meta_file"]))["tls"]
    net = ET.parse(os.path.join(os.path.dirname(os.path.abspath(cfg["action_meta_file"])), os.path.basename(cfg["net_file"]))).getroot()
    conns = collections.defaultdict(list)
    for c in net.iter("connection"):
        if not c.get("from").startswith(":"):
            conns[c.get("from")].append(c)
    flags = collections.defaultdict(list); summary = {}
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]; of = ts.observation_fn
        try:
            turnmap = turnmap_from_meta(meta[tid])
        except ValueError as e:
            flags[tid].append(f"标签不一致: {e}"); continue
        links = sumo.trafficlight.getControlledLinks(tid)
        feeds, cap, outs = collections.defaultdict(set), collections.defaultdict(float), collections.defaultdict(set)
        for i, grp in enumerate(links):
            if not grp or i not in turnmap:
                continue
            s = APP.index(turnmap[i][0]) * 3 + TURN.index(turnmap[i][1])
            if len(grp) > 1:        # 开了 all_connections 后多连接索引只是信息, 否则是漏登记的来源
                flags[tid].append(("信息: " if a.all_connections else "") + f"索引 {i} 挂 {len(grp)} 条连接 (槽 {name(s)}): " + ", ".join(f"{c[0]}→{c[1]}" for c in grp))
            for conn in grp:
                feeds[conn[0]].add(s); outs[s].add(conn[1])
        for fl, ss in feeds.items():
            for s in ss:
                cap[s] += max(sumo.lane.getLength(fl) / 7.5, 1.0)
        for s in sorted(cap):
            if abs(of._slot_cap[s] - max(cap[s], 1.0)) > 1e-6:
                flags[tid].append(f"槽 {name(s)} 容量 观测用 {of._slot_cap[s]:.2f} 全连接 {cap[s]:.2f}")
            if set(of._slot_out.get(s, ())) != outs[s]:
                flags[tid].append(f"槽 {name(s)} 出口车道 观测用 {sorted(of._slot_out.get(s, ()))} 全连接 {sorted(outs[s])}")
            want_in = sorted(l for l, ss in feeds.items() if s in ss)
            if sorted(of._slot_in_lanes.get(s, ())) != want_in:
                flags[tid].append(f"槽 {name(s)} lane_occ 车道 观测用 {sorted(of._slot_in_lanes.get(s, ()))} 全连接 {want_in}")
        slot_idx = collections.defaultdict(list)
        for i, s in of._link_slot.items():
            slot_idx[s].append(i)
        for k, ph in enumerate(ts.green_phases):
            for s, idxs in slot_idx.items():
                st = {ph.state[i] in "Gg" for i in idxs}
                if len(st) > 1:
                    flags[tid].append(f"相位 {k} 槽 {name(s)} 索引 {idxs} 灯色 {[ph.state[i] for i in idxs]} 不一致 (is_green 对部分车辆为错)")
        for i, cs in meta[tid]["links"].items():
            for c in cs:
                if c["turn"] != c.get("dir_turn", c["turn"]):
                    flags[tid].append(f"信息: 索引 {i} 路网标注 {c['dir']} 按 meta 记为 {c['approach']}-{c['turn']} (几何 {c.get('dir_turn')})")
        in_edges = {l.rsplit("_", 1)[0] for l in ts.lanes}
        unc = [f"{e}_{c.get('fromLane')}→{c.get('to')}" for e in in_edges for c in conns[e] if c.get("tl") != tid]
        if unc:
            flags[tid].append(f"进口边上不归本信号控制的连接 {len(unc)} 条: " + "; ".join(unc[:4]))
        summary[tid] = dict(n_idx=len(of._link_slot), n_slots=len(cap), n_lanes=len(ts.lanes), n_flags=len(flags[tid]),
                            n_problem=sum(1 for f in flags[tid] if not f.startswith("信息")))
    drop = collections.Counter(); seen = collections.Counter(); pairs = collections.Counter()
    for step in range(a.steps):
        states, _, done, _ = env.step(action={t: act(t, states[t])[0] for t in env.ts_ids})
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]; of = ts.observation_fn
            for lane in ts.lanes:
                e = of._edge(lane)
                for vid in sumo.lane.getLastStepVehicleIDs(lane):
                    seen[tid] += 1
                    if of._vehicle_slot(vid, e) < 0:
                        drop[tid] += 1; r = sumo.vehicle.getRoute(vid); k = sumo.vehicle.getRouteIndex(vid)
                        pairs[(tid, e, r[k + 1] if k + 1 < len(r) else None)] += 1
        if done["__all__"]:
            break
    env.close()
    print(f"exp{a.exp} net={a.net} all_connections={a.all_connections}")
    print(f"{'路口':<32} 索引 槽 进口道 问题 信息  丢弃/观测")
    for tid in env.ts_ids:
        s = summary.get(tid, {}); print(f"{tid[:32]:<32} {s.get('n_idx', 0):3d} {s.get('n_slots', 0):2d} {s.get('n_lanes', 0):3d} {s.get('n_problem', 0):4d} {s.get('n_flags', 0) - s.get('n_problem', 0):4d}  {drop[tid]}/{seen[tid]}")
    for tid, fl in flags.items():
        for f in fl:
            print(f"  {tid[:32]}: {f}")
    if pairs:
        print("未映射边对 (路口, 进口边, 下一条边): 次数", pairs.most_common(10))
    if a.out:
        json.dump(dict(exp=a.exp, net=a.net, all_connections=a.all_connections, summary=summary, flags=dict(flags),
                       dropped={t: [drop[t], seen[t]] for t in env.ts_ids}, unmapped=[[list(k), v] for k, v in pairs.most_common(20)]),
                  open(a.out, "w"), indent=1, ensure_ascii=False)
    n_problem = sum(s.get("n_problem", 0) for s in summary.values())
    print(f"问题项合计 {n_problem} (信息项不计)")


if __name__ == "__main__":
    main()
