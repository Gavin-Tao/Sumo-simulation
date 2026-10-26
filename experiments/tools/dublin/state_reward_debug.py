"""状态/奖励逐步独立重算核对 (2026-10-26, 状态/奖励修正实施计划任务 9)。只读分析, 不训练。

用训练配置构造环境与贪心策略 (ckpt_env.build), 跑一集; 每个决策步在 SUMO 快照上**独立**重算 PriorityMovement 的每一组特征
(头部、is_green、进口 count/queue/mean_awt、下游 count/queue、lane_occ、槽内最短绿) 与奖励 (legacy 跨车道记账 / visit 统计器),
与环境输出逐维比对; 同时统计车辆跨 RL/非 RL 路口的转换、重访、共享车道归属、多连接索引车道覆盖、被丢弃车辆、统计器缺条目。
重算用自己的表 (从 getControlledLinks + meta 标签建), 不读观测类的内部表。

用法:
  python experiments/tools/dublin/state_reward_debug.py <exp> [--seed 123] [--steps 720] [--net splitslot] [--basis visit]
        [--all-connections] [--no-memo] [--strict] [--out report.json]
  --net splitslot     换成 dublin_8std_splitslot 网/meta/sumocfg (18h 改道第二版)
  --basis visit       obs_awt_basis=visit + reward_wait_basis=visit
  --all-connections   obs_all_connections + all_connections
  --no-memo           obs_slot_memo=false
退出码: 任一组最大绝对差 > 1e-5 → 1。"""
import argparse, collections, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
APP = ("N", "E", "S", "W"); TURN = ("L", "T", "R"); LEVELS = (1, 2, 3, 4, 5); LOG = math.log1p(3600.0)
TOL = 1e-5


def slot_of(ap, tn):
    return APP.index(ap) * 3 + TURN.index(tn)


class Tables:
    """独立的静态表 (与观测类同一规则, 但自己从 SUMO + meta 建)。"""
    def __init__(self, sumo, ts, turnmap, all_connections, veh_len=5.0, min_gap=2.5):
        links = sumo.trafficlight.getControlledLinks(ts.id)
        self.link_slot, self.by_edges, feeds, self.slot_out = {}, {}, collections.defaultdict(set), collections.defaultdict(set)
        for i, grp in enumerate(links):
            if not grp or i not in turnmap:
                continue
            s = slot_of(*turnmap[i]); self.link_slot[i] = s
            for conn in (grp if all_connections else grp[:1]):
                fl, tl = conn[0], conn[1]
                self.by_edges[(fl.rsplit("_", 1)[0], tl.rsplit("_", 1)[0])] = s
                feeds[fl].add(s); self.slot_out[s].add(tl)
        cap = lambda l: max(sumo.lane.getLength(l) / (min_gap + veh_len), 1.0)
        self.lane_cap = {l: cap(l) for l in feeds}
        self.slot_cap = {s: max(sum(self.lane_cap[l] for l, ss in feeds.items() if s in ss), 1.0) for s in range(12)}
        self.slot_in = {s: [l for l, ss in feeds.items() if s in ss] for s in range(12)}
        self.out_cap = {s: max(sum(cap(t) for t in tls), 1.0) for s, tls in self.slot_out.items()}
        self.lane_slots = {l: ss for l, ss in feeds.items()}
        self.serves = {k: {self.link_slot[i] for i in self.link_slot if i < len(ph.state) and ph.state[i] in "Gg"}
                       for k, ph in enumerate(ts.green_phases)}
        self.lanes = list(ts.lanes)
        self.multi_idx_lanes = {conn[0] for grp in links if grp and len(grp) > 1 for conn in grp[1:]}   # 多连接索引的第二条及以后进口车道


class Ref:
    """逐步独立重算。"""
    def __init__(self, env, cfg, meta):
        from exp_params import turnmap_from_meta
        self.env, self.cfg, self.sumo = env, cfg, env.sumo
        self.fields = tuple(cfg.get("obs_fields", ("count", "queue", "mean_awt", "max_awt")))
        self.ds_fields = tuple(cfg.get("obs_downstream_fields", ("count", "queue"))) if cfg.get("obs_downstream") else ()
        self.lane_occ = bool(cfg.get("obs_lane_occ", False)); self.slot_mg = bool(cfg.get("obs_slot_min_green", False))
        self.header = bool(cfg.get("obs_header", True)); self.awt_log = bool(cfg.get("obs_awt_log", False))
        self.awt_cap = cfg.get("obs_awt_cap"); self.basis = cfg.get("obs_awt_basis", "global"); self.memo = bool(cfg.get("obs_slot_memo", True))
        self.rbasis = cfg.get("reward_wait_basis", "legacy"); self.reward_name = cfg.get("reward_fn")
        table = cfg.get("obs_priority_source", cfg.get("priority_source")) or {"ambulance": 5, "bus": 4, "car": 1}
        self.prio = dict(table); self.rtable = dict(cfg.get("priority_source") or table)
        self.T = {t: Tables(self.sumo, env.traffic_signals[t], turnmap_from_meta(meta[t]), bool(cfg.get("obs_all_connections", False))) for t in env.ts_ids}
        self.entry = {}; self.vid_slot = {}; self.recs = {}
        self.stats = collections.Counter(); self.first_seen = {}; self.visits = collections.defaultdict(int); self.on = set()
        assert not (cfg.get("obs_since_green") or cfg.get("obs_green_elapsed")), "本工具不重算 since_green/green_elapsed"

    def wait(self, vid, lane, t):
        if self.basis == "visit":
            return self.env.visit_tracker.get(t, vid, lane)
        acc = self.sumo.vehicle.getAccumulatedWaitingTime(vid)
        if self.basis == "global":
            return acc
        k = (vid, lane); e = self.entry.get(k)
        if e is None or acc < e:
            self.entry[k] = e = acc
        return acc - e

    def slot(self, vid, edge, T, t):
        key = (t, vid)                                  # 观测类的缓存是每路口一个实例, 这里同样按 (路口, 车辆) 记
        if self.memo and key in self.vid_slot:
            cur = self._current_slot(vid, edge, T)
            if cur != self.vid_slot[key]:
                self.stats["memo_slot_stale"] += 1       # 旧缓存与当前路线不一致 (重访/路线终点)
            return self.vid_slot[key]
        s = self._current_slot(vid, edge, T)
        if self.memo:
            self.vid_slot[key] = s
        return s

    def _current_slot(self, vid, edge, T):
        r = self.sumo.vehicle.getRoute(vid); k = self.sumo.vehicle.getRouteIndex(vid)
        nxt = r[k + 1] if k + 1 < len(r) else None
        return T.by_edges.get((edge, nxt), -1)

    def expected_state(self, t):
        env, sumo, T = self.env, self.sumo, self.T[t]; ts = env.traffic_signals[t]
        mg_ok = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0
        served = T.serves.get(ts.green_phase, set())
        obs = [mg_ok, ts.time_since_last_phase_change / 100.0] if self.header else []
        stats = {s: {p: [0, 0, 0.0, 0.0] for p in LEVELS} for s in range(12)}; lane_n = {}
        present_now = set()
        for lane in T.lanes:
            edge = lane.rsplit("_", 1)[0]; vids = sumo.lane.getLastStepVehicleIDs(lane); lane_n[lane] = len(vids)
            for vid in vids:
                present_now.add((vid, t))
                if (vid, t) not in self.on:
                    self.visits[(vid, t)] += 1
                    if vid not in self.first_seen:
                        acc0 = sumo.vehicle.getAccumulatedWaitingTime(vid); self.first_seen[vid] = acc0
                        self.stats["first_rl_sighting"] += 1; self.stats["first_rl_sighting_with_upstream_wait"] += acc0 > 0
                if lane in T.multi_idx_lanes:
                    self.stats["veh_on_second_lane_of_multi_index"] += 1
                s = self.slot(vid, edge, T, t)
                if len(T.lane_slots.get(lane, ())) > 1:
                    self.stats["veh_on_shared_lane"] += 1
                if s < 0:
                    self.stats["dropped_slot_minus1"] += 1; continue
                p = self.prio.get(sumo.vehicle.getTypeID(vid), 1); st = stats[s][p]
                w = self.wait(vid, lane, t); st[0] += 1
                if sumo.vehicle.getSpeed(vid) < 0.1:
                    st[1] += 1
                st[2] += w; st[3] = max(st[3], w)
        self.on = {k for k in self.on if k[1] != t} | present_now
        for s in range(12):
            obs.append(1.0 if s in served else 0.0); cap = T.slot_cap[s]
            for p in LEVELS:
                c, q, sw, mw = stats[s][p]; mean = sw / c if c else 0.0; mean_raw = mean
                if self.awt_cap is not None and not self.awt_log:
                    mean = min(mean, self.awt_cap); mw = min(mw, self.awt_cap)
                vals = {"count": c / cap, "queue": q / cap,
                        "mean_awt": math.log1p(mean) / LOG if self.awt_log else mean / 100.0,
                        "max_awt": math.log1p(mw) / LOG if self.awt_log else mw / 100.0,
                        "mean_awt_log": math.log1p(mean_raw) / LOG}
                obs += [vals[f] for f in self.fields]
            if self.ds_fields:
                cnt = {p: 0 for p in LEVELS}; qd = {p: 0 for p in LEVELS}
                for tl in T.slot_out.get(s, ()):
                    for vid in sumo.lane.getLastStepVehicleIDs(tl):
                        p = self.prio.get(sumo.vehicle.getTypeID(vid), 1); cnt[p] += 1
                        if sumo.vehicle.getSpeed(vid) < 0.1:
                            qd[p] += 1
                capo = T.out_cap.get(s, 1.0)
                for f in self.ds_fields:
                    src = cnt if f == "count" else qd; obs += [src[p] / capo for p in LEVELS]
            if self.lane_occ:
                obs.append(max([lane_n.get(l, 0) / T.lane_cap[l] for l in T.slot_in[s]] or [0.0]))
            if self.slot_mg:
                obs.append(mg_ok if s in served else 0.0)
        return np.asarray(obs, dtype=np.float64)

    def expected_reward(self, t):
        sumo, T = self.sumo, self.T[t]; tot, cnt = {}, {}
        for lane in T.lanes:
            for vid in sumo.lane.getLastStepVehicleIDs(lane):
                if self.rbasis == "visit":
                    lw = self.env.visit_tracker.get(t, vid, lane)
                else:
                    acc = sumo.vehicle.getAccumulatedWaitingTime(vid); vl = sumo.vehicle.getLaneID(vid)
                    if vid not in self.recs:
                        self.recs[vid] = {vl: acc}
                    else:
                        self.recs[vid][vl] = acc - sum(x for l, x in self.recs[vid].items() if l != vl)
                    lw = self.recs[vid][vl]
                w = 1.0 if self.reward_name == "avg-waiting-time" else float(self.rtable.get(sumo.vehicle.getTypeID(vid), 1))
                tot[w] = tot.get(w, 0.0) + lw; cnt[w] = cnt.get(w, 0) + 1
        if self.reward_name == "avg-waiting-time":
            n = sum(cnt.values()); return -(sum(tot.values()) / n) / 100.0 if n else 0.0
        return -sum(w * tot[w] / cnt[w] for w in tot) / 100.0


def group_layout(ref):
    """每槽内各维所属的组名 (用于按组汇总差异)。"""
    per_slot = ["is_green"] + [f"phi_{f}" for _ in LEVELS for f in ref.fields] + [f"psi_{f}" for f in ref.ds_fields for _ in LEVELS]
    per_slot += ["lane_occ"] if ref.lane_occ else []; per_slot += ["slot_min_green"] if ref.slot_mg else []
    return (["header"] * 2 if ref.header else []) + per_slot * 12


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--net", choices=["base", "splitslot"], default="base"); ap.add_argument("--basis", choices=["legacy", "visit"], default="legacy")
    ap.add_argument("--all-connections", action="store_true"); ap.add_argument("--no-memo", action="store_true"); ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out"); a = ap.parse_args()
    import ckpt_env
    ov = {}
    if a.net == "splitslot":
        ov.update(net_file="../nets/dublin/dublin_8std_splitslot.net.xml", action_meta_file="../nets/dublin/dublin_8std_splitslot_meta.json")
        cfg0 = __import__("yaml").safe_load(open(ckpt_env.find_cfg(a.exp)))
        ov["cfg_file"] = cfg0["cfg_file"].replace(".sumocfg", "_8std_splitslot.sumocfg")
    if a.basis == "visit":
        ov.update(obs_awt_basis="visit", reward_wait_basis="visit")
    if a.all_connections:
        ov.update(obs_all_connections=True, all_connections=True)
    if a.no_memo:
        ov["obs_slot_memo"] = False
    B = ckpt_env.build(a.exp, seed=a.seed, cfg_override=ov); env, act, states, cfg = B["env"], B["act"], B["states"], B["cfg"]
    if a.strict and env.visit_tracker is not None:
        env.visit_tracker.strict = True
    meta = json.load(open(cfg["action_meta_file"]))["tls"]; ref = Ref(env, cfg, meta); groups = group_layout(ref)
    maxd = collections.defaultdict(float); first_bad = []; rdiff = 0.0; n_steps = 0
    for step in range(a.steps):
        actions = {t: act(t, states[t])[0] for t in env.ts_ids}
        states, r, done, _ = env.step(action=actions); n_steps += 1
        for t in env.ts_ids:
            got = np.asarray(states[t], dtype=np.float64); exp = ref.expected_state(t)
            if got.shape != exp.shape:
                sys.exit(f"维数不一致 {t}: 环境 {got.shape} 重算 {exp.shape}")
            d = np.abs(got - exp)
            for g, x in zip(groups, d):
                maxd[g] = max(maxd[g], float(x))
            if d.max() > TOL and len(first_bad) < 8:
                i = int(d.argmax()); first_bad.append((env.sumo.simulation.getTime(), t[:24], i, groups[i], float(got[i]), float(exp[i])))
            rd = abs(float(r[t]) - ref.expected_reward(t)); rdiff = max(rdiff, rd)
        if done["__all__"]:
            break
    tr_missing = env.visit_tracker.missing if env.visit_tracker is not None else None
    env.close()
    revisits = sum(1 for k, n in ref.visits.items() if n > 1)
    report = dict(exp=a.exp, seed=a.seed, net=a.net, basis=a.basis, all_connections=a.all_connections, memo=not a.no_memo, steps=n_steps,
                  max_diff_by_group={g: maxd[g] for g in sorted(set(groups))}, reward_max_diff=rdiff, first_mismatches=first_bad,
                  stats=dict(ref.stats), revisits_vehicle_junction=revisits, tracker_missing=tr_missing)
    print(f"exp{a.exp} seed{a.seed} net={a.net} basis={a.basis} all_connections={a.all_connections} memo={not a.no_memo} 步 {n_steps}")
    print("  各组最大绝对差:", {g: f"{v:.2e}" for g, v in report["max_diff_by_group"].items()})
    print(f"  奖励最大绝对差 {rdiff:.2e} | 统计器缺条目 {tr_missing}")
    print("  统计:", dict(ref.stats), "| (车辆,路口) 重访数", revisits)
    for row in first_bad:
        print("  不一致:", row)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
    bad = max(maxd.values()) > TOL or rdiff > TOL
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
