"""541 倒挂根因诊断 — 用 exp145/147 真实 checkpoint + 真实 rollout 测 4 个假设:
  H1 过拟合 (Q 值校准: predicted Q vs realized return, 分组)
  H2 PER (已有数据定论, 此处不测)
  H3 amb 梯度被 bus 主宰 (分组梯度范数 + 方向冲突 cosine)
  H4 邻居 state/reward 对梯度的影响 (saliency 分块 + sigmoid(β))
  行为探针: amb 在场时 policy 选不选 amb 相位 (145 vs 147 对比)
"""
import sys, os
sys.path.insert(0, '/home/xiaowen/sumo-rl')
os.environ['WANDB_MODE'] = 'disabled'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.observations import PriorityBCAObservationFunction
from sumo_rl.environment.rewards import REWARD_REGISTRY

EXP_DIR = '/home/xiaowen/sumo-rl/experiments'
CKPT = {
    '5-2-1 (exp145)': f'{EXP_DIR}/models/exp145_coeff_1x3_521_avg_waiting_NS20bus_1amb_U_tuned',
    '5-4-1 (exp147)': f'{EXP_DIR}/models/exp147_coeff_1x3_541_avg_waiting_NS20bus_1amb_U_tuned',
}
NEIGHBOR_MAP = {"t": [None,None,None,"e"], "e": [None,None,"t","J0"], "J0": [None,None,"e",None]}
GAMMA = 0.99
N_EP = 8          # rollout episodes per policy
DEVICE = torch.device('cpu')

class Qnet(nn.Module):
    def __init__(self, sd, hd, ad):
        super().__init__()
        self.fc1 = nn.Linear(sd, hd)
        self.fc2 = nn.Linear(hd, ad)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def load_ckpt(model_dir):
    import glob
    f = sorted(glob.glob(f'{model_dir}/*/ckpt_ep02000.pth'))[-1]
    ck = torch.load(f, map_location='cpu', weights_only=False)
    q = Qnet(385, 128, 4)
    q.load_state_dict(ck['policy_state_dict'])
    q.eval()
    beta_raw = ck['beta']
    return q, beta_raw

# ── env ──────────────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open(f'{EXP_DIR}/configs/exp147_coeff_1x3_541_avg_waiting_NS20bus_1amb_U_tuned.yaml'))
env = SumoEnvironment(
    net_file=os.path.join(EXP_DIR, cfg['net_file']),
    route_file=os.path.join(EXP_DIR, cfg['route_file']),
    cfg_file=os.path.join(EXP_DIR, cfg['cfg_file']),
    observation_class=PriorityBCAObservationFunction,
    reward_fn=REWARD_REGISTRY['5-4-1-avg-waiting-time'],
    num_seconds=1000, delta_time=5, yellow_time=2, min_green=5,
    use_gui=False, use_max_green=True, max_green=50,
)
ts_ids = env.ts_ids
ts0 = ts_ids[0]

# obs 块结构
ts_obj = env.traffic_signals[ts0]
n_phase = ts_obj.num_green_phases
n_in  = len(ts_obj.lanes)
n_out = len(ts_obj.out_lanes)
own_dim = n_phase + 1 + 3*(n_in+n_out)
assert own_dim == 77, f"own_dim={own_dim}"
B = {}  # own-obs 内的块索引
i0 = 0
for name, ln in [("phase",n_phase),("min_green",1),
                 ("car_inc",n_in),("car_out",n_out),
                 ("bus_inc",n_in),("bus_out",n_out),
                 ("amb_inc",n_in),("amb_out",n_out)]:
    B[name] = (i0, i0+ln); i0 += ln

def amb_present_own(aug):   # amb 在自己 incoming
    s, e = B["amb_inc"]
    return aug[s:e].sum() > 0
def amb_present_nb(aug):    # amb 只在邻居 obs 里
    nb = aug[own_dim:].reshape(4, own_dim)
    s, e = B["amb_inc"]
    return nb[:, s:e].sum() > 0

_ = env.reset(123)   # init sumo so trafficlight API is available

# ── 相位映射 (programmatic) ─────────────────────────────────────────────────
# amb route: n_t→t_e (t), t_e→E0 (e), E0→E3 (J0)
AMB_MOVE = {"t": ("n_t","t_e"), "e": ("t_e","E0"), "J0": ("E0","E3")}
BUS_MOVE = {"t": ("n_t","t_s"), "e": ("n_e","e_s"), "J0": ("n_J0","J0_s")}  # NS-through 猜测, 用 link 验证
amb_phase, bus_phase = {}, {}
for ts in ts_ids:
    links = env.sumo.trafficlight.getControlledLinks(ts)
    logics = env.sumo.trafficlight.getAllProgramLogics(ts)
    phases = [p.state for p in logics[-1].phases]
    green_phases = [s for s in phases if 'y' not in s.lower() or 'G' in s]
    # 用 traffic_signal 的 green phase 列表更准
    gp = [p.state for p in env.traffic_signals[ts].green_phases] if hasattr(env.traffic_signals[ts], 'green_phases') else None
    states = gp if gp else phases
    def phase_serving(frm, to):
        out = []
        for li, conns in enumerate(links):
            for (lane_in, lane_out, _) in conns:
                if lane_in.rsplit('_',1)[0] == frm and lane_out.rsplit('_',1)[0] == to:
                    for pi, st in enumerate(states):
                        if li < len(st) and st[li] == 'G' and pi not in out:
                            out.append(pi)
        return out
    a_f, a_t = AMB_MOVE[ts]
    amb_phase[ts] = phase_serving(a_f, a_t)
print(f"amb-serving green phases: {amb_phase}")

# ── aug obs ──────────────────────────────────────────────────────────────────
def get_aug(ts, states):
    own = np.asarray(states[ts], dtype=np.float32)
    parts = [own]
    for nb in NEIGHBOR_MAP[ts]:
        if nb is not None and nb in states:
            parts.append(np.asarray(states[nb], dtype=np.float32))
        else:
            parts.append(np.zeros(own_dim, dtype=np.float32))
    return np.concatenate(parts)

# ── rollout ──────────────────────────────────────────────────────────────────
def rollout(qnet, n_ep, label):
    """Greedy rollout. Returns transitions + behavior probe records."""
    trans = []   # (ts, aug, a, r, aug', done)
    probe = []   # (ts, aug, chosen_a, q_values)  当 amb 在该 ts own incoming
    for ep in range(n_ep):
        env.sumo_seed = 123 + ep
        states = env.reset(env.sumo_seed)
        done = {'__all__': False}
        ep_buf = {ts: [] for ts in ts_ids}
        while not done['__all__']:
            augs = {ts: get_aug(ts, states) for ts in ts_ids}
            acts = {}
            for ts in ts_ids:
                with torch.no_grad():
                    qv = qnet(torch.tensor(augs[ts]).unsqueeze(0)).squeeze(0).numpy()
                a = int(qv.argmax())
                acts[ts] = a
                if amb_present_own(augs[ts]):
                    probe.append((ts, a, qv.copy()))
            nstates, rewards, done, _ = env.step(action=acts)
            naugs = {ts: get_aug(ts, nstates) for ts in ts_ids}
            for ts in ts_ids:
                ep_buf[ts].append((augs[ts], acts[ts], rewards[ts], naugs[ts], float(done['__all__'])))
            states = nstates
        for ts in ts_ids:
            trans.extend([(ts,)+t for t in ep_buf[ts]])
    print(f"  [{label}] {len(trans)} transitions, {len(probe)} amb-present decision steps")
    return trans, probe

results = {}
for label, mdir in CKPT.items():
    q, beta_raw = load_ckpt(mdir)
    sig = torch.sigmoid(beta_raw).numpy()
    print(f"\n=== {label}  sigmoid(β) = {np.round(sig, 4)} ===")
    trans, probe = rollout(q, N_EP, label)
    results[label] = dict(q=q, beta=sig, trans=trans, probe=probe)

env.close()

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("行为探针: amb 在本路口 incoming 时, policy 选 amb-serving phase 的概率")
print("="*78)
for label, R in results.items():
    probe = R['probe']
    print(f"\n--- {label} ---")
    for ts in ts_ids:
        recs = [(a, qv) for (t, a, qv) in probe if t == ts]
        if not recs: continue
        aphs = amb_phase[ts]
        n_amb_chosen = sum(1 for a, _ in recs if a in aphs)
        # Q margin: Q(amb_phase) - max(other)
        margins = []
        for a, qv in recs:
            qa = max(qv[p] for p in aphs) if aphs else np.nan
            qo = max(qv[p] for p in range(4) if p not in aphs)
            margins.append(qa - qo)
        print(f"  {ts:>3}: amb 相位被选率 {n_amb_chosen}/{len(recs)} = {100*n_amb_chosen/len(recs):.0f}%   "
              f"Q(amb相位)-Q(最优其它) = {np.mean(margins):+.3f} ± {np.std(margins):.3f}")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("H3 梯度分析 (exp147): amb 组 vs 其它组的梯度范数 / 方向冲突")
print("="*78)
R = results['5-4-1 (exp147)']
q = R['q']
trans = R['trans']

groups = {"AMB-own": [], "AMB-nb-only": [], "OTHER": []}
for (ts, s, a, r, sn, d) in trans:
    if amb_present_own(s): groups["AMB-own"].append((s,a,r,sn,d))
    elif amb_present_nb(s): groups["AMB-nb-only"].append((s,a,r,sn,d))
    else: groups["OTHER"].append((s,a,r,sn,d))
for g, v in groups.items():
    print(f"  {g:<12}: {len(v):5d} transitions ({100*len(v)/len(trans):.1f}%)")

def batch_tensors(batch):
    s  = torch.tensor(np.stack([b[0] for b in batch]))
    a  = torch.tensor([b[1] for b in batch]).view(-1,1)
    r  = torch.tensor([b[2] for b in batch], dtype=torch.float32).view(-1,1)
    sn = torch.tensor(np.stack([b[3] for b in batch]))
    d  = torch.tensor([b[4] for b in batch], dtype=torch.float32).view(-1,1)
    return s,a,r,sn,d

def group_grad(q, batch):
    """mean gradient vector + per-sample TD stats"""
    s,a,r,sn,d = batch_tensors(batch)
    q.zero_grad()
    qa = q(s).gather(1, a)
    with torch.no_grad():
        tgt = r + GAMMA * q(sn).max(1)[0].view(-1,1) * (1-d)
    td = (tgt - qa).detach()
    loss = F.mse_loss(qa, tgt)
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in q.parameters()])
    return g.clone(), td.abs().squeeze().numpy(), loss.item()

rng = np.random.default_rng(0)
gvecs, tds = {}, {}
for gname, gdata in groups.items():
    if len(gdata) < 8: continue
    idx = rng.choice(len(gdata), min(512, len(gdata)), replace=False)
    batch = [gdata[i] for i in idx]
    g, td, loss = group_grad(q, batch)
    gvecs[gname] = g; tds[gname] = td
    print(f"\n  {gname}: |TD| mean={td.mean():.4f} median={np.median(td):.4f} max={td.max():.2f}, "
          f"loss={loss:.4f}, ||grad||={g.norm():.5f}")

print(f"\n  梯度方向 cosine (组间):")
names = list(gvecs.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        c = F.cosine_similarity(gvecs[names[i]], gvecs[names[j]], dim=0).item()
        print(f"    cos({names[i]}, {names[j]}) = {c:+.3f}")

# 混合 batch 里 amb 的实际贡献占比 (按 |TD|² 加权, MSE 梯度正比 TD)
n_amb, n_oth = len(groups["AMB-own"]), len(groups["OTHER"])
td_amb_sq = (tds["AMB-own"]**2).mean() if "AMB-own" in tds else 0
td_oth_sq = (tds["OTHER"]**2).mean()
share = n_amb*td_amb_sq / (n_amb*td_amb_sq + n_oth*td_oth_sq)
print(f"\n  混合 batch 里 AMB-own 对 loss 的贡献占比 ≈ {100*share:.1f}%  "
      f"(样本占比 {100*n_amb/(n_amb+n_oth):.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("H1 Q 值校准 (exp147): predicted Q vs realized return (greedy rollout)")
print("="*78)
# 按 (ts, episode) 重组算 return — trans 是顺序 append 的, 每 ep 每 ts 200 步
trans_by = {}
for (ts, s, a, r, sn, d) in trans:
    trans_by.setdefault(ts, []).append((s,a,r,d))
calib = {"AMB-own": [], "OTHER": []}
for ts, seq in trans_by.items():
    # 按 done 切 episode
    ep, eps_list = [], []
    for t in seq:
        ep.append(t)
        if t[3] > 0.5: eps_list.append(ep); ep = []
    if ep: eps_list.append(ep)
    for ep in eps_list:
        G = 0.0; returns = [0.0]*len(ep)
        for i in range(len(ep)-1, -1, -1):
            G = ep[i][2] + GAMMA*G
            returns[i] = G
        ss = torch.tensor(np.stack([e[0] for e in ep]))
        aa = torch.tensor([e[1] for e in ep]).view(-1,1)
        with torch.no_grad():
            qq = q(ss).gather(1, aa).squeeze().numpy()
        for i, e in enumerate(ep):
            key = "AMB-own" if amb_present_own(e[0]) else "OTHER"
            calib[key].append((qq[i], returns[i]))
for k, v in calib.items():
    if not v: continue
    qs = np.array([x[0] for x in v]); gs = np.array([x[1] for x in v])
    print(f"  {k:<10}: n={len(v):5d}  Q_pred={qs.mean():8.2f}  G_real={gs.mean():8.2f}  "
          f"bias(Q-G)={np.mean(qs-gs):+7.2f}  |err|={np.mean(np.abs(qs-gs)):6.2f}")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*78)
print("H4 saliency (exp147): amb 在场状态下 |∂Q/∂input| 分块")
print("="*78)
amb_states = [s for (ts, s, a, r, sn, d) in trans if amb_present_own(s)]
if amb_states:
    S = torch.tensor(np.stack(amb_states[:256]), requires_grad=True)
    qmax = q(S).max(1)[0].sum()
    qmax.backward()
    sal = S.grad.abs().mean(0).numpy()   # [385]
    own_sal = sal[:own_dim]
    nb_sal  = sal[own_dim:].reshape(4, own_dim)
    print(f"  own 块总 saliency:      {own_sal.sum():.5f}")
    for i, dname in enumerate(["up","down","left","right"]):
        print(f"  nb-{dname:<5} 块总 saliency: {nb_sal[i].sum():.5f}")
    print(f"\n  own 内部分块 (per-dim mean):")
    for name, (s0, e0) in B.items():
        seg = own_sal[s0:e0]
        print(f"    {name:<10}: total={seg.sum():.5f}  per-dim={seg.mean():.6f}")
print("\n完成")
