"""任务 6: 救护车。一条设计路线 (每集固定 3 辆), 7 份时刻表 (1–5 训练 / 6 标定 / 7 最终测试),
评估用: 定稿后的额外时刻表 (8–13)、每集约 20 辆的频率补充测试、3 条未见路线探针。"""
import collections, json, random
import common18 as C

def classify(meta, anchors):
    ag = C.always_green(meta); prior = anchors["prior_old18h"]
    nact = {t: sum(1 for x in m["actions"].values() if x.get("valid")) for t, m in meta.items()}
    appr_tot = collections.Counter()
    for k, v in prior.items():
        t, mv = k.split("|"); appr_tot[(t, mv.split("-")[0])] += v["car"] + v["bus"]
    def minor2(t, mv):
        if nact[t] != 2 or (t, mv) in ag: return False
        ap = mv.split("-")[0]; others = [a for (tt, a) in appr_tot if tt == t and a != ap]
        return bool(others) and appr_tot[(t, ap)] < max(appr_tot[(t, a)] for a in others)
    return ag, nact, minor2

ENTRY_LOAD_MAX = 450.0   # 救护车入口车道平台期负荷上限 (辆/h), 避免插车延迟

def score_route(d, ag, nact, minor2, zero, rl_first_dist, entry_load=None):
    trav = [tuple(x) for x in d["traversals"]]
    comp = [m for m in trav if m not in ag and m not in zero]
    has_minor2 = any(minor2(t, mv) for t, mv in comp); has_multi = any(nact[t] >= 5 for t, _ in comp)
    load_ok = entry_load is None or entry_load.get(d["lanes"][0], 0.0) <= ENTRY_LOAD_MAX
    ok = (load_ok and d["entry_type"] == "boundary" and len(trav) >= 6 and len(comp) >= 5 and has_minor2 and has_multi and not d["yield_links"] and rl_first_dist >= 100)
    return ok, (len(comp), has_minor2 + has_multi, -len(d["yield_links"]), -d["length"]), comp

def first_rl_dist(r, meta, lanes):
    d = 0.0
    for l in lanes:
        d += r.net.getLane(l).getLength()
        if r.net.getLane(l).getEdge().getToNode().getID() in meta: return d
    return d

def timetable(seed, n=C.AMB_PER_EP, windows=C.AMB_WINDOWS, min_gap=300.0):
    rng = random.Random(seed)
    while True:
        ts = [round(rng.uniform(a, b), 1) for a, b in windows[:n]]
        if all(b - a >= min_gap for a, b in zip(ts, ts[1:])): return ts

def main():
    C.check_hashes(); meta = C.load_meta(); anchors = C.load("anchors_18h.json"); E = C.load("entries.json"); r = C.load_router()
    zero = {tuple(m) for m in E["zero"]} | {tuple(m) for m in E["zero_lane_loop"]}
    ag, nact, minor2 = classify(meta, anchors)
    pool = C.load("route_pool.json")["routes"]
    try: entry_load = C.load("demand_report.json")["entry_plateau_vph"]
    except FileNotFoundError: entry_load = None
    scored = []
    for d in pool:
        ok, key, comp = score_route(d, ag, nact, minor2, zero, first_rl_dist(r, meta, d["lanes"]), entry_load)
        if ok: scored.append((key, d, comp))
    scored.sort(key=lambda x: x[0], reverse=True)
    assert scored, "没有满足条件的救护车路线"
    choice = C.WORK / "amb_route_choice.json"     # 实测选线结果 (amb_select18.py) 优先
    if choice.exists():
        ch = json.load(open(choice))["edges"]
        hit = [x for x in scored if x[1]["edges"] == ch]
        if hit: scored = hit + [x for x in scored if x[1]["edges"] != ch]
    top = scored[:3]; key, route, comp = top[0]
    # 未见路线: 不与训练路线共享任何 RL 进口边, ≥4 个 RL 路口, 优先无让行; 三条之间也尽量不共享进口
    used_in = {l.rsplit("_", 1)[0] for x, y in zip(route["lanes"], route["lanes"][1:]) if (x, y) in r.movements for l in [x]}
    unseen = []
    cands = sorted(pool, key=lambda d: (len(d["yield_links"]), -len([m for m in map(tuple, d["traversals"]) if m not in ag])))
    for min_rl in (4, 3):        # 第一轮要求 ≥4 个 RL 路口, 凑不齐 3 条时第二轮放宽到 ≥3
        for d in cands:
            if len(unseen) == 3: break
            if d["entry_type"] != "boundary" or len(d["traversals"]) < min_rl or first_rl_dist(r, meta, d["lanes"]) < 100: continue
            if any(u["edges"] == d["edges"] for u in unseen): continue
            ins = {x.rsplit("_", 1)[0] for x, y in zip(d["lanes"], d["lanes"][1:]) if (x, y) in r.movements}
            if ins & used_in or any(ins & u["_ins"] for u in unseen): continue
            dd = dict(d); dd["_ins"] = ins; unseen.append(dd)
    for u in unseen: u["_ins"] = sorted(u["_ins"])
    tts = {k: timetable(5000 + k) for k in range(1, C.N_TT + 1)}
    eval_tts = {k: timetable(9000 + k) for k in range(8, 14)}            # 定稿后才使用的评估时刻表
    rng = random.Random(777); probe = []; t = 900.0
    while t <= 2700 and len(probe) < 20:
        probe.append(round(t + rng.uniform(0, 10), 1)); t += 100.0
    rng = random.Random(778); unseen_tt = []
    for i, u in enumerate(unseen):
        for k in range(5): unseen_tt.append(dict(route=i, depart=round(900 + 360 * k + 120 * i + rng.uniform(0, 30), 1)))
    out = dict(route=route, route_competing=[list(m) for m in comp], alternatives=[dict(score=list(k), edges=d["edges"], competing=[list(m) for m in c]) for k, d, c in top[1:]],
               n_candidates=len(scored), timetables=tts, eval_timetables=eval_tts, freq_probe_departs=probe, unseen_routes=unseen, unseen_timetable=unseen_tt,
               rule=dict(min_rl=6, min_competing=5, need_minor2=True, need_multiphase=True, yields=0, entry="boundary", first_rl_min_m=100))
    C.dump(out, "ambulances.json")
    print(f"(入口负荷{'已' if entry_load else '未'}纳入) 候选 {len(scored)} 条; 选中: 经过 RL {len(route['traversals'])} 个, 竞争转向 {len(comp)} 个, 长度 {route['length']:.0f} m, 让行 {len(route['yield_links'])}")
    print("  路线:", " ".join(route["edges"])); print("  转向:", [f"{t[:12]} {m}" for t, m in map(tuple, route["traversals"])])
    for k, d, c in top[1:]: print("  备选:", [f"{t[:10]} {m}" for t, m in map(tuple, d["traversals"])])
    print("未见路线:", [[f"{t[:10]} {m}" for t, m in map(tuple, u["traversals"])] for u in unseen])
    print("时刻表:", tts)

if __name__ == "__main__":
    main()
