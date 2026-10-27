"""用某检查点在评估种子上跑一集 (贪心), 存全部 (s,a,r,s') 转移; 然后用该网络 (目标网=在线网) 按训练同样的 TD 目标 (double, clip[-50,0], huber)
算每个样本的损失, 比较 '正常' 与 '长等待' 样本; 再抽 256 批算批均损失与梯度范数 (含/不含长等待样本)。
用法: python batch_loss_probe.py <exp> --ckpt P --seed 123 --out prefix"""
import argparse, os, sys, json, numpy as np, torch, torch.nn.functional as F
os.environ["OMP_NUM_THREADS"] = "1"; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ckpt_env
ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--ckpt"); ap.add_argument("--seed", type=int, default=123); ap.add_argument("--out", required=True); a = ap.parse_args()
B = ckpt_env.build(a.exp, ckpt=a.ckpt, seed=a.seed); env, act, states, agent = B["env"], B["act"], B["states"], B["agent"]
import yaml; cfg = B["cfg"]
S_, A_, R_, S2_, T_, J_ = [], [], [], [], [], []
done = {"__all__": False}
while not done["__all__"]:
    acts, ks = {}, {}
    for t in env.ts_ids:
        a_env, k, q = act(t, states[t]); acts[t] = a_env; ks[t] = int(k)
    prev = {t: np.asarray(states[t], dtype=np.float32) for t in env.ts_ids}
    states, r, done, _ = env.step(action=acts)
    for t in env.ts_ids:
        S_.append(prev[t]); A_.append(ks[t]); R_.append(float(r[t])); S2_.append(np.asarray(states[t], dtype=np.float32)); T_.append(float(env.sumo.simulation.getTime())); J_.append(t)
env.close()
S_ = torch.tensor(np.stack(S_)); S2_ = torch.tensor(np.stack(S2_)); A_ = torch.tensor(A_).view(-1, 1); R_ = torch.tensor(R_).view(-1, 1)
net = agent.q_net; gamma = cfg.get("gamma", 0.95); cmin, cmax = cfg.get("target_clip_min", -50.0), cfg.get("target_clip_max", 0.0)
import json as _j
meta = _j.load(open(cfg["action_meta_file"]))["tls"]; masks = torch.tensor(np.stack([np.array(meta[t]["mask"], dtype=bool) for t in J_]))
with torch.no_grad():
    q_sa = net(S_).gather(1, A_)
    nq = net(S2_).masked_fill(~masks, -1e9); na = nq.argmax(1, keepdim=True); q_next = net(S2_).gather(1, na)
    tgt = (R_ + gamma * q_next).clamp(min=cmin, max=cmax)
    per = F.smooth_l1_loss(q_sa, tgt, reduction="none").squeeze(1).numpy(); td = (tgt - q_sa).squeeze(1).numpy()
R = R_.squeeze(1).numpy()
def q(x, p): return float(np.quantile(x, p))
rep = dict(exp=a.exp, ckpt_episode=B["ckpt_episode"], n=len(R), reward=dict(zero_frac=float((R == 0).mean()), p50=q(R, .5), p10=q(R, .1), p01=q(R, .01), min=float(R.min()), n_below_1=int((R < -1).sum()), n_below_5=int((R < -5).sum())),
           per_sample_loss=dict(median=float(np.median(per)), mean=float(per.mean()), p99=q(per, .99), max=float(per.max()), share_top1pct=float(np.sort(per)[-len(per)//100:].sum() / per.sum())),
           td=dict(abs_median=float(np.median(np.abs(td))), abs_p99=q(np.abs(td), .99), abs_max=float(np.abs(td).max())))
normal = np.flatnonzero(R >= -1.0); bad = np.flatnonzero(R < -1.0)
rep["lock_samples"] = dict(n=int(len(bad)), loss_mean=float(per[bad].mean()) if len(bad) else None, loss_max=float(per[bad].max()) if len(bad) else None, td_abs_mean=float(np.abs(td[bad]).mean()) if len(bad) else None)
rng = np.random.default_rng(0)
def batch_stats(idx_pool, n_batches=50, force_bad=0):
    L, G = [], []
    for _ in range(n_batches):
        idx = rng.choice(idx_pool, 256 - force_bad, replace=False)
        if force_bad: idx = np.concatenate([idx, rng.choice(bad, force_bad, replace=False)])
        idx = torch.tensor(idx)
        net.zero_grad(); qs = net(S_[idx]).gather(1, A_[idx]); loss = F.smooth_l1_loss(qs, tgt[idx]); loss.backward()
        g = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters() if p.grad is not None)).item(); L.append(loss.item()); G.append(g)
    return dict(loss_mean=float(np.mean(L)), loss_max=float(np.max(L)), grad_mean=float(np.mean(G)), grad_max=float(np.max(G)))
rep["batch_normal_only"] = batch_stats(normal)
rep["batch_random_all"] = batch_stats(np.arange(len(R)))
if len(bad): rep["batch_with_1_lock"] = batch_stats(normal, force_bad=1); rep["batch_with_3_lock"] = batch_stats(normal, force_bad=min(3, len(bad)))
json.dump(rep, open(a.out + ".json", "w"), indent=1); print(json.dumps(rep, indent=None))
