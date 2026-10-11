"""Q 对头部 vs 排队的敏感度 (只读): 用一个策略跑 1x1 一集收集状态, 对每个状态分别扰动头部 (相位已持续时间 0/0.25/0.5) 与排队特征 (×0.5/×1.5), 看 Q 的变化幅度与 argmax 翻转率。"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/home/xiaowen/sumo-rl/experiments/tools/dublin"); import ckpt_env
exp, ckpt = sys.argv[1], sys.argv[2]
B = ckpt_env.build(exp, ckpt, 123); env, act, states, agent = B["env"], B["act"], B["states"], B["agent"]; t = list(env.ts_ids)[0]
S = []; done = {"__all__": False}; i = 0
while not done["__all__"]:
    if i % 4 == 0: S.append(np.array(states[t], dtype=np.float32))
    states, _, done, _ = env.step(action={t: act(t, states[t])[0]}); i += 1
env.close()
S = np.stack(S); H = 2; of = env.traffic_signals[t].observation_fn; fields = of.fields; nl = 5; per = 1 + len(fields) * nl + (1 if of.include_since_green else 0) + (1 if of.include_green_elapsed else 0)
qi = fields.index("queue"); qidx = [H + s * per + 1 + p * len(fields) + qi for s in range(12) for p in range(nl)]
idx = agent._idx[t]; pm, rel, exist, mask = agent._tensors(torch.tensor([idx]))
def Q(X):
    with torch.no_grad(): q = agent.q_net(torch.tensor(X), pm.expand(len(X), -1, -1), rel.expand(len(X), -1, -1), exist.expand(len(X), -1)).numpy()
    q[:, ~mask[0].numpy()] = np.nan; return q
q0 = Q(S); a0 = np.nanargmax(q0, 1); scale = np.nanmean(np.abs(q0))
def report(name, X):
    q = Q(X); d = np.nanmean(np.abs(q - q0)) / scale; flip = np.mean(np.nanargmax(q, 1) != a0); return f"{name}: |ΔQ|/|Q| {d:.3f}, argmax 翻转 {100*flip:.0f}%"
out = []
for el in (0.0, 0.25, 0.5):
    X = S.copy(); X[:, 1] = el; X[:, 0] = 1.0 if el >= 0.07 else 0.0; out.append(report(f"头部 elapsed={el:.2f}", X))
for k in (0.5, 1.5):
    X = S.copy(); X[:, qidx] = np.clip(X[:, qidx] * k, 0, 1); out.append(report(f"排队 ×{k}", X))
print(f"[{exp}] 状态数 {len(S)} |Q| 均 {scale:.3f} | " + " | ".join(out))
