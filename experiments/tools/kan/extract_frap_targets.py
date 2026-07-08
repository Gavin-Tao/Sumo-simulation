"""K1 — collect KAN-distillation targets from a trained FRAP-enum checkpoint.

For every decision step and junction, records:
  G rows: (x_m 27-dim raw slot features)          -> g_head(enc(x_m))
  S rows: (x_m, x_n, rel in {2 merge, 3 crossing}) -> duel score s_mn
  CTX rows: (obs, tls_idx, greedy action)  — for later action-agreement eval.
The aggregation Q(p)=Σg+Σmax s is frozen math, so G and S fully determine
the controller (design: KAN distillation refinement 2026-07-03).

Runs the standard eval protocol (greedy, masked) plus optional ε-rollouts
for state coverage. Read-only analysis: trains nothing, touches no config.

Usage (repo root):
  python experiments/tools/kan/extract_frap_targets.py \
      --config experiments/configs/exp211_dublin11h_531_enumfrap.yaml \
      --seeds 123,2000,2001 --eps-seeds 3000 --out experiments/analysis/kan_data
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import sys

os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments"))

import numpy as np
import torch
import yaml

FEATURE_NAMES = (["is_green"]
                 + [f"{f}_l{p}" for p in range(1, 6)
                    for f in ("cnt", "que", "awt")]
                 + [f"down_cnt_l{p}" for p in range(1, 6)]
                 + [f"down_que_l{p}" for p in range(1, 6)]
                 + ["lane_occ"])          # exp211 perphase slot layout (27)


def build(cfg):
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    from sumo_rl.environment.priority_map import load_priority_table
    # Key resolution mirrors train.py: obs_priority_source > priority_source >
    # None (-> built-in default table); awt_cap/basis only passed when present
    # (obs class defaults None/global). Lets the extractor run on any enum_frap
    # config, not just exp211-style ones with every key spelled out.
    obs_prio = (cfg.get("obs_priority_source")
                if "obs_priority_source" in cfg else cfg.get("priority_source"))
    obs_kwargs = dict(
        fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
        priority_source=obs_prio,
        include_downstream=bool(cfg.get("obs_downstream", False)),
        downstream_fields=tuple(cfg.get("obs_downstream_fields", ())),
        include_lane_occ=bool(cfg.get("obs_lane_occ", False)))
    if "obs_awt_cap" in cfg:
        obs_kwargs["awt_cap"] = float(cfg["obs_awt_cap"])
    if "obs_awt_basis" in cfg:
        obs_kwargs["awt_basis"] = cfg["obs_awt_basis"]
    obs_class = functools.partial(obsmod.PriorityMovementObservationFunction,
                                  **obs_kwargs)
    reward_fn = make_priority_avg_waiting_reward(
        load_priority_table(cfg.get("priority_source")))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
        num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
        max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
        single_agent=False, yellow_time=cfg["yellow_time"],
        delta_time=cfg["delta_time"], reward_fn=reward_fn,
        observation_class=obs_class, sumo_seed=cfg["seed"], sumo_warnings=False)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="default: latest best.pth")
    ap.add_argument("--seeds", default="123", help="greedy eval seeds, comma")
    ap.add_argument("--eps-seeds", default="", help="ε=0.2 coverage seeds")
    ap.add_argument("--out", default="experiments/analysis/kan_data")
    ap.add_argument("--num-seconds", type=int, default=None, help="override (smoke)")
    ap.add_argument("--cap", type=int, default=500_000, help="max rows per array")
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    os.chdir(os.path.join(_REPO, "experiments"))   # config paths convention
    cfg = yaml.safe_load(open(cfg_path))
    if args.num_seconds:
        cfg["num_seconds"] = args.num_seconds
    ckpt = args.ckpt or sorted(glob.glob(
        os.path.join("models", cfg["name"], "*", "best.pth")))[-1]
    print("checkpoint:", ckpt)

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

    reset(cfg["seed"])                       # spaces ready before agent build
    agent = build_frap_agent(cfg, tables, env, "cpu")
    agent.q_net.load_state_dict(torch.load(
        ckpt, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval()
    net, hd, sd = agent.q_net, agent.q_net.header_dim, agent.q_net.slot_dim
    assert sd == len(FEATURE_NAMES), (sd, len(FEATURE_NAMES))

    G_X, G_y = [], []
    S_Xm, S_Xn, S_rel, S_y = [], [], [], []
    CTX_obs, CTX_ts, CTX_act = [], [], []
    runs = [(s, 0.0) for s in args.seeds.split(",") if s] + \
           [(s, 0.2) for s in args.eps_seeds.split(",") if s]
    rng = np.random.RandomState(0)
    for seed, eps in runs:
        states = reset(seed)
        done = {"__all__": False}
        print(f"run seed={seed} eps={eps}")
        while not done["__all__"]:
            acts = {}
            for i, ts in enumerate(agent._ids):
                x = np.asarray(states[ts], dtype=np.float32)
                xt = torch.tensor(x).unsqueeze(0)
                ii = torch.tensor([i])
                pm, rel, exist, mask = agent._tensors(ii)
                with torch.no_grad():
                    d = net.encode(xt)
                    g = net.g_head(d).squeeze(-1)[0]          # (12,)
                    s = net.duel_scores(d, rel)[0]            # (12,12)
                    q = net(xt, pm, rel, exist).masked_fill(~mask, -1e9)
                a = (int(np.random.choice(np.flatnonzero(mask[0].numpy())))
                     if rng.rand() < eps else int(q.argmax().item()))
                acts[ts] = a
                if len(G_X) < args.cap:
                    slots = x[hd:].reshape(12, sd)
                    ex = exist[0].numpy() > 0
                    for m in np.flatnonzero(ex):
                        G_X.append(slots[m]); G_y.append(float(g[m]))
                    r = rel[0].numpy()
                    for m in np.flatnonzero(ex):
                        for n in np.flatnonzero(ex):
                            if r[m, n] >= 2:
                                S_Xm.append(slots[m]); S_Xn.append(slots[n])
                                S_rel.append(int(r[m, n]))
                                S_y.append(float(s[m, n]))
                    CTX_obs.append(x); CTX_ts.append(i); CTX_act.append(a)
            states, _, done, _ = env.step(action=acts)
    env.close()

    os.makedirs(args.out, exist_ok=True)
    G_X = np.asarray(G_X, np.float32)
    amb_flag = G_X[:, 13] > 0                     # cnt_l5 offset = 1 + 4*3
    np.savez_compressed(os.path.join(args.out, "frap_targets.npz"),
        g_X=G_X, g_y=np.asarray(G_y, np.float32),
        s_Xm=np.asarray(S_Xm, np.float32), s_Xn=np.asarray(S_Xn, np.float32),
        s_rel=np.asarray(S_rel, np.int8), s_y=np.asarray(S_y, np.float32),
        ctx_obs=np.asarray(CTX_obs, np.float32),
        ctx_ts=np.asarray(CTX_ts, np.int16), ctx_act=np.asarray(CTX_act, np.int16))
    manifest = {"config": cfg_path, "ckpt": ckpt, "runs": runs,
                "feature_names": FEATURE_NAMES,
                "n_g": len(G_y), "n_s": len(S_y), "n_ctx": len(CTX_act),
                "amb_rows_g": int(amb_flag.sum()),
                "s_rel_counts": {int(k): int(v) for k, v in
                                 zip(*np.unique(np.asarray(S_rel), return_counts=True))}}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1)[:600])


if __name__ == "__main__":
    main()
