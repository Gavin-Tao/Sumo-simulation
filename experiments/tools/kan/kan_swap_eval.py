"""K4 — closed-loop evaluation of distilled G/S plugged into the frozen
FRAP aggregation, plus per-step action agreement vs the original network.

Q_hat(p) = Σ_{m∈p} G_hat(x_m) + Σ_{n∉p} max_{m∈p, conflict} S_hat(x_m,x_n,rel)
(the aggregation is the same frozen math as FRAPQNet.forward — only the two
scalar functions are replaced by their distilled versions).

Usage (repo root):
  python experiments/tools/kan/kan_swap_eval.py \
      --config experiments/configs/exp211_dublin11h_531_enumfrap.yaml \
      --models experiments/analysis/kan_data/fit \
      --seeds 123 [--policy kan|net] [--num-seconds 3600]
Outputs agreement %, per-seed eval reward for the executed policy.
Read-only analysis tooling — trains nothing, touches no config.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml

from extract_frap_targets import build  # env construction (same protocol)
from kan_distill import load_bundle

NEG = -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--models", required=True, help="dir with model_{g,s_merge,s_crossing}.pt")
    ap.add_argument("--seeds", default="123")
    ap.add_argument("--policy", choices=["kan", "net"], default="kan")
    ap.add_argument("--num-seconds", type=int, default=None)
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    models_dir = os.path.abspath(args.models)
    os.chdir(os.path.join(_REPO, "experiments"))
    cfg = yaml.safe_load(open(cfg_path))
    if args.num_seconds:
        cfg["num_seconds"] = args.num_seconds
    ckpt = args.ckpt or sorted(glob.glob(
        os.path.join("models", cfg["name"], "*", "best.pth")))[-1]

    g_model, g_b = load_bundle(os.path.join(models_dir, "model_g.pt"))
    s_models = {}
    for rel, name in ((2, "s_merge"), (3, "s_crossing")):
        p = os.path.join(models_dir, f"model_{name}.pt")
        if os.path.exists(p):
            s_models[rel], _ = load_bundle(p)
    dims = g_b["dims"]
    print(f"ckpt={ckpt}  G dims={len(dims)}  S models={sorted(s_models)}")

    from frap_glue import load_enum_tables, build_frap_agent
    env = build(cfg)
    tables = load_enum_tables(cfg["enum_meta_file"])

    def reset(seed):
        env.reset(int(seed))
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.observation_fn.rebind_movements(tables["turnmap"][tid])
            ts.observation_space = ts.observation_fn.observation_space()
        return {tid: env.traffic_signals[tid].observation_fn()
                for tid in env.ts_ids}

    reset(cfg["seed"])
    agent = build_frap_agent(cfg, tables, env, "cpu")
    agent.q_net.load_state_dict(torch.load(
        ckpt, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval()
    net, hd, sd = agent.q_net, agent.q_net.header_dim, agent.q_net.slot_dim

    # static per-junction conflict-pair index lists (precomputed once) so the
    # per-step S evaluation is ONE batched forward per relation class —
    # per-pair looped forwards would dominate runtime regardless of backend.
    pairs_by_ts = {}
    for i in range(len(agent._ids)):
        _, rel, exist, _ = agent._tensors(torch.tensor([i]))
        rel, exist = rel[0], exist[0]
        pr = {2: [], 3: []}
        for m in range(12):
            for n in range(12):
                r = int(rel[m, n])
                if r >= 2 and exist[m] > 0 and exist[n] > 0 and r in s_models:
                    pr[r].append((m, n))
        pairs_by_ts[i] = pr

    def kan_q(slots_t, i):
        """Distilled Q over the junction's padded menu (mirrors FRAPQNet)."""
        pm, rel, exist, mask = agent._tensors(torch.tensor([i]))
        pm, rel, exist, mask = pm[0], rel[0], exist[0], mask[0]
        with torch.no_grad():
            g = g_model(slots_t[:, dims])[:, 0]                    # (12,)
            s = torch.full((12, 12), NEG)
            for r, pairs in pairs_by_ts[i].items():
                if not pairs:
                    continue
                X = torch.stack([torch.cat([slots_t[m, dims], slots_t[n, dims]])
                                 for m, n in pairs])
                vals = s_models[r](X)[:, 0]                        # one batch
                for (m, n), v in zip(pairs, vals):
                    s[m, n] = v
        K = pm.shape[0]
        q = torch.full((K,), NEG)
        for p in range(int(mask.sum())):
            members = pm[p] > 0
            q_self = g[members].sum()
            q_sup = 0.0
            for n in range(12):
                if exist[n] <= 0 or members[n]:
                    continue
                duels = [s[m, n] for m in range(12)
                         if members[m] and int(rel[m, n]) >= 2]
                if duels:
                    d = max(duels)
                    if d > NEG / 2:
                        q_sup += d
            q[p] = q_self + q_sup
        return q, mask

    stats = {"agree": 0, "total": 0, "rewards": {}}
    for seed in args.seeds.split(","):
        states = reset(seed)
        done = {"__all__": False}
        ep_rew = 0.0
        while not done["__all__"]:
            acts = {}
            for i, ts in enumerate(agent._ids):
                x = np.asarray(states[ts], dtype=np.float32)
                xt = torch.tensor(x).unsqueeze(0)
                slots_t = torch.tensor(x[hd:].reshape(12, sd))
                pm, rel, exist, mask = agent._tensors(torch.tensor([i]))
                with torch.no_grad():
                    q_net_v = net(xt, pm, rel, exist).masked_fill(~mask, NEG)
                a_net = int(q_net_v.argmax().item())
                qk, mk = kan_q(slots_t, i)
                a_kan = int(qk.masked_fill(~mk.bool(), NEG).argmax().item())
                stats["agree"] += int(a_net == a_kan)
                stats["total"] += 1
                acts[ts] = a_kan if args.policy == "kan" else a_net
            states, r, done, _ = env.step(action=acts)
            ep_rew += float(np.mean(list(r.values())))
        stats["rewards"][seed] = ep_rew
        print(f"seed {seed}: executed={args.policy} ep_reward={ep_rew:.2f} "
              f"agreement so far={stats['agree']/max(stats['total'],1):.1%}")
    env.close()
    stats["agreement"] = stats["agree"] / max(stats["total"], 1)
    out = os.path.join(models_dir, f"swap_eval_{args.policy}.json")
    json.dump(stats, open(out, "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
