"""读 force_lock_debug 的 json: 打印时间表 + 不变量核对。用法: python analyze_force.py file.json [every_sec]"""
import json, sys, collections
f = sys.argv[1]; every = float(sys.argv[2]) if len(sys.argv) > 2 else 120
d = json.load(open(f)); S = d["summary"]; rows = d["rows"]
print("== 摘要:", {k: S[k] for k in ("ckpt_episode", "force_std", "force_from", "force_until", "reward_max_diff", "tracker_missing", "remaining_vehicles", "teleports", "max_W_J", "reward_sum_J")})
print("   state 各组最大差:", {k: f"{v:.1e}" for k, v in S["state_max_diff_by_group"].items()}, "| own_vs_env", S["own_vs_env"], "| acc_identity", S["acc_identity"])
f0, f1 = S["force_from"], S["force_until"]
# ---- 不变量 1: 强制期内 is_green 只在被放行槽; 策略意愿统计 ----
want = collections.Counter(); hold_g = collections.Counter(); viol = 0
for r in rows:
    if r["forced"]:
        want[r["policy_k"]] += 1
        for sn, sd in r["slots"].items():
            if sd["g"]: hold_g[sn] += 1
# ---- 不变量 2: 被饿死车道的最大本次访问等待每步 +5 (同一车, 停着) ----
inc = collections.Counter(); prev = {}
for r in rows:
    for ln, L in r["lanes"].items():
        p = prev.get(ln)
        if p and L["vmax"] == p["vmax"] and L["halt"] > 0 and p["halt"] > 0 and r["t"] >= f0 + 60 and r["t"] < f1:
            inc[round(L["max_W"] - p["max_W"], 1)] += 1
        prev[ln] = L
print("== 强制期策略意愿 (policy_k 计数):", dict(want), "| 放行槽 is_green 计数:", dict(hold_g))
print("== 强制期被堵车道 max_W 每步增量分布:", dict(sorted(inc.items())))
# ---- 时间表 ----
hdr = "t    std k  Q(0/1/2/3/6/7)                          r_prev    nW300 | 槽(car: n,q,awt | occ | ds) | 车道(n/halt/maxW/acc)"
print("==", hdr)
last_t = -1e9
for r in rows:
    t = r["t"]
    if t - last_t < every and not (f0 - 10 <= t <= f0 + 10 or f1 - 10 <= t <= f1 + 10): continue
    last_t = t
    q = r["q"]; qs = " ".join(f"{k}:{v:6.3f}" for k, v in q.items()) if q else ""
    sl = []
    for sn, sd in r["slots"].items():
        car = sd.get("car"); bus = sd.get("bus")
        sl.append(f"{sn}{'*' if sd['g'] else ' '}car{car if car else '-'}{' bus'+str(bus) if bus else ''} occ{sd.get('occ')} ds{sd.get('ds_car')}")
    ln = []
    for l, L in r["lanes"].items():
        ln.append(f"{l[-12:]}:{L['n']}/{L['halt']}/{L['max_W']}/{L['vmax_acc']}")
    print(f"{t:5.0f} {r['std']}  {r['policy_k']} {'F' if r['forced'] else ' '} {qs} r={r['r_prev']} nW300={r['n_W300']} net={r['net_maxW']}")
    print("      槽:", " | ".join(sl))
    print("      道:", " | ".join(ln))
