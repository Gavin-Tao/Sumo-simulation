"""锁死扫描 (只读): 用某检查点贪心跑一集, 每决策步记每路口 (时刻, 当前 std 相位, 策略选的 std, 奖励, 本次访问最大等待, 在场车数, >=60s 车数)。
用法: python scan_locks.py <exp> <ckpt> <seed> <out.json>"""
import sys, os, json, collections
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
exp, ckpt, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
B = ckpt_env.build(exp, ckpt=ckpt, seed=seed)
env, act, states = B["env"], B["act"], B["states"]
ts_ids = list(env.ts_ids); tr = env.visit_tracker
if tr is None:      # 旧口径配置没有统计器: 挂一个只做测量的 (不影响环境的状态/奖励)
    from sumo_rl.environment.visit_wait import VisitWaitTracker
    tr = VisitWaitTracker(env.sumo, {t: env.traffic_signals[t].lanes for t in ts_ids}); _o0 = env._sumo_step
    def _h0(_o=_o0, _m=tr): _o(); _m.step()
    env._sumo_step = _h0
series = {t: [] for t in ts_ids}
done = {"__all__": False}
while not done["__all__"]:
    acts, ks = {}, {}
    for t in ts_ids:
        a, k, q = act(t, states[t]); acts[t] = a; ks[t] = int(k)
    states, r, done, _ = env.step(action=acts)
    now = float(env.sumo.simulation.getTime())
    mx = collections.defaultdict(float); n = collections.Counter(); n60 = collections.Counter()
    for (t, vid, edge), w in tr.W.items():
        n[t] += 1
        if w > mx[t]: mx[t] = w
        if w >= 60: n60[t] += 1
    for t in ts_ids:
        ts = env.traffic_signals[t]
        series[t].append((now, int(ts.std_action_map[ts.green_phase]), ks[t], round(float(r[t]), 4), round(mx[t], 1), n[t], n60[t]))
nveh = env.sumo.vehicle.getIDCount(); arrived = env.sumo.simulation.getArrivedNumber()
env.close()
summary = {}
for t in ts_ids:
    arr = series[t]; maxw = max(x[4] for x in arr); tmax = [x[0] for x in arr if x[4] == maxw][0]
    ph = collections.Counter(x[1] for x in arr)
    summary[t] = dict(max_w=maxw, t_max=tmax, reward_sum=round(sum(x[3] for x in arr), 2), steps_w120=sum(1 for x in arr if x[4] >= 120),
                      phase_share={str(k): round(v / len(arr), 3) for k, v in sorted(ph.items())})
lock = [t for t, s in summary.items() if s["max_w"] >= 150]
json.dump(dict(exp=exp, ckpt=ckpt, ckpt_episode=B["ckpt_episode"], seed=seed, lock=lock, n_remaining=nveh, summary=summary, series={t: series[t] for t in lock}), open(out, "w"))
tot = sum(s["reward_sum"] for s in summary.values())
print(f"exp{exp} ep{B['ckpt_episode']} seed{seed} reward_sum={tot:.1f} remaining={nveh} lock={[(t[:22], summary[t]['max_w'], summary[t]['t_max']) for t in lock]}")
