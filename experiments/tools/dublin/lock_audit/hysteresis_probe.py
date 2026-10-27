"""零重训探针: 切换滞回 —— 只有当 Q(最优) − Q(保持当前相位) > δ 时才切换, 否则保持。输出全网停车/visit、切换次数 (全网与若干路口)。"""
import os, sys, json, collections, numpy as np
os.environ["OMP_NUM_THREADS"]="1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import ckpt_env
exp, ck, seed, delta = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
B = ckpt_env.build(exp, ckpt=ck, seed=seed); env, act, states = B["env"], B["act"], B["states"]; sumo = env.sumo
def act2(t, s):
    a_env, k, q = act(t, s); ts = env.traffic_signals[t]; cur = int(ts.std_action_map[ts.green_phase])
    if k != cur and np.isfinite(q[cur]) and (q[k] - q[cur]) < delta: return ts.green_phase
    return a_env
tr = env.visit_tracker; seen=set(); halt=collections.Counter(); sw=collections.Counter(); last={}; maxW=collections.Counter()
done={"__all__": False}
while not done["__all__"]:
    states, r, done, _ = env.step(action={t: act2(t, states[t]) for t in env.ts_ids})
    for k,w in tr.W.items(): seen.add(k); maxW[k[0]]=max(maxW[k[0]], w)
    for t in env.ts_ids:
        ts=env.traffic_signals[t]
        for l in ts.lanes: halt[t]+=sumo.lane.getLastStepHaltingNumber(l)*5
        if t in last and last[t]!=ts.green_phase: sw[t]+=1
        last[t]=ts.green_phase
env.close(); visits=collections.Counter(k[0] for k in seen); tot_v=sum(visits.values()); tot_s=sum(halt.values())
sh=lambda t: t.split("_")[0]+("_"+t.split("_")[1] if t.startswith("cluster") else "")
focus={sh(t): (sw[t], round(halt[t]/max(visits[t],1),2)) for t in env.ts_ids if any(x in t for x in ("1448442","21620852","1998326407","26165090","659802","1448436"))}
print(f"seed{seed} δ={delta}: 全网停车/visit {tot_s/tot_v:.3f}, 切换总数 {sum(sw.values())}, 最大等待 {max(maxW.values()):.0f} s | 路口 (切换, 停车/visit): {focus}")
