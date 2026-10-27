import os, sys, json, collections
os.environ["OMP_NUM_THREADS"]="1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import ckpt_env
exp, ck, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); J="cluster_1448439_2464758361"
B = ckpt_env.build(exp, ckpt=ck, seed=seed); env, act, states = B["env"], B["act"], B["states"]; sumo = env.sumo; ts = env.traffic_signals[J]
meta = json.load(open(B["cfg"]["action_meta_file"]))["tls"][J]; ls = {int(i): g[0]["approach"]+"-"+g[0]["turn"] for i,g in meta["links"].items()}
lanes = list(ts.lanes); on = {}; exits = []   # (exit_time, movement, vid)
phase_log = []   # (t, std)
def rec():
    now = sumo.simulation.getTime(); phase_log.append((now, int(ts.std_action_map[ts.green_phase]), ts.is_yellow)); present = set()
    for l in lanes:
        for v in sumo.lane.getLastStepVehicleIDs(l):
            present.add(v)
            if v not in on:
                mv = None
                for x in sumo.vehicle.getNextTLS(v):
                    if x[0] == J: mv = ls.get(int(x[1])); break
                on[v] = mv
    for v in [v for v in on if v not in present]:
        exits.append((now, on.pop(v), v))
_o = env._sumo_step
def h(): _o(); rec()
env._sumo_step = h
done = {"__all__": False}
while not done["__all__"]:
    states, r, done, _ = env.step(action={t: act(t, states[t])[0] for t in env.ts_ids})
env.close()
# a2 窗口 (含随后 2 s 黄灯)
win = []; cur = None
for t, std, yel in phase_log:
    if std == 2 and cur is None: cur = [t, t]
    elif std == 2: cur[1] = t
    elif cur is not None: win.append((cur[0], cur[1] + 2)); cur = None
if cur: win.append((cur[0], cur[1] + 2))
tot = collections.Counter(mv for _, mv, _ in exits)
for a, b in win:
    c = collections.Counter(mv for t, mv, _ in exits if a <= t <= b)
    print(f"种子 {seed}: a2 窗口 {a:.0f}–{b:.0f} s ({b-a:.0f} s): 期间通过 {dict(c)}")
print(f"种子 {seed}: 全集通过 {dict(tot)}")
