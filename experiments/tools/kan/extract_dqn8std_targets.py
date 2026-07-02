"""K1(8std) — collect factorized-distillation targets from a masked-8std DQN
checkpoint (exp205/208 family: PriorityMovement LEGACY obs + std masks).

The structure test (user formulation 2026-07-03):
    Q_k ?≈ Σ_m A_km · g(φ_m) + b·1[k == current]
where A (8×12, per junction) = which movements each std action serves —
a KNOWN constant from the 8std meta, so the decomposition needs no FRAP
teacher. Fitting on Q-DIFFERENCES removes the monolithic net's V(s) baseline.

Records per junction-step: slot features (12×26), Q vector (8), valid mask,
current std action, junction index; plus the per-junction A stack.

Usage (repo root):
  python experiments/tools/kan/extract_dqn8std_targets.py \
      --config experiments/configs/exp208_dublin11h_...yaml \
      --seeds 123,2000 --eps-seeds 3000 --out experiments/analysis/kan8_data
Read-only analysis: trains nothing, touches no config.
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

# legacy B slot layout (NO is_green bit): 5 levels × (cnt,que,awt) + downstream
FEATURE_NAMES = ([f"{f}_l{p}" for p in range(1, 6) for f in ("cnt", "que", "awt")]
                 + [f"down_cnt_l{p}" for p in range(1, 6)]
                 + [f"down_que_l{p}" for p in range(1, 6)]
                 + ["lane_occ"])          # 26
HEADER = 9                                 # 8 std one-hot + min_green_ok
SLOTS = [(a, t) for a in ("N", "E", "S", "W") for t in ("L", "T", "R")]
SLOT_IDX = {s: k for k, s in enumerate(SLOTS)}


def load_meta_tables(path):
    meta = json.load(open(path))["tls"]
    ts_mask, std2green, green2std, turnmap, A = {}, {}, {}, {}, {}
    for tid, t in meta.items():
        ts_mask[tid] = np.array(t["mask"], dtype=bool)
        links = {int(i): c for i, c in t["links"].items()}
        turnmap[tid] = {i: (c[0]["approach"], c[0]["turn"])
                        for i, c in links.items()}
        link_slot = {i: SLOT_IDX[(c[0]["approach"], c[0]["turn"])]
                     for i, c in links.items()}
        s2g = np.full(8, -1, dtype=int)
        for a, gi in t["std_to_green_index"].items():
            s2g[int(a)] = gi
        std2green[tid] = s2g
        g2s = np.full(int(s2g.max()) + 1, -1, dtype=int)
        for a in range(8):
            if s2g[a] >= 0:
                g2s[s2g[a]] = a
        green2std[tid] = g2s
        Am = np.zeros((8, 12), dtype=np.float32)
        for a in range(8):
            info = t["actions"][f"a{a}"]
            if info.get("valid"):
                st = info["state"]
                for i in links:
                    if i < len(st) and st[i] == "G":
                        Am[a, link_slot[i]] = 1.0
        A[tid] = Am
    return ts_mask, std2green, green2std, turnmap, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seeds", default="123")
    ap.add_argument("--eps-seeds", default="")
    ap.add_argument("--out", default="experiments/analysis/kan8_data")
    ap.add_argument("--num-seconds", type=int, default=None)
    ap.add_argument("--cap", type=int, default=300_000)
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    os.chdir(os.path.join(_REPO, "experiments"))
    cfg = yaml.safe_load(open(cfg_path))
    if args.num_seconds:
        cfg["num_seconds"] = args.num_seconds
    ckpt = args.ckpt or sorted(glob.glob(
        os.path.join("models", cfg["name"], "*", "best.pth")))[-1]
    print("checkpoint:", ckpt)

    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    from sumo_rl.environment.priority_map import load_priority_table
    from sumo_rl.agents.dqn_agent_txw import DQN

    obs_class = functools.partial(
        obsmod.PriorityMovementObservationFunction,
        fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
        priority_source=cfg["priority_source"],
        include_downstream=bool(cfg.get("obs_downstream", False)),
        downstream_fields=tuple(cfg.get("obs_downstream_fields", ())),
        include_lane_occ=bool(cfg.get("obs_lane_occ", False)),
        awt_cap=float(cfg["obs_awt_cap"]), awt_basis=cfg["obs_awt_basis"])
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
        num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
        max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
        single_agent=False, yellow_time=cfg["yellow_time"],
        delta_time=cfg["delta_time"],
        reward_fn=make_priority_avg_waiting_reward(
            load_priority_table(cfg["priority_source"])),
        observation_class=obs_class, sumo_seed=cfg["seed"], sumo_warnings=False)

    ts_mask, std2green, green2std, turnmap, A = \
        load_meta_tables(cfg["action_meta_file"])
    ts_ids = None

    def reset(seed):
        env.reset(int(seed))
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.std_action_map = green2std[tid]
            ts.observation_fn.rebind_movements(turnmap[tid])
            ts.observation_space = ts.observation_fn.observation_space()
        return {tid: env.traffic_signals[tid].observation_fn()
                for tid in env.ts_ids}

    states = reset(cfg["seed"])
    ts_ids = list(env.ts_ids)
    od = len(next(iter(states.values())))
    sd, rem = divmod(od - HEADER, 12)
    assert rem == 0 and sd == len(FEATURE_NAMES), (od, sd)
    agent = DQN(starting_state=tuple([0.0] * od), state_space=od,
        hidden_dim=cfg.get("hidden_dim", 128), action_space=8,
        learning_rate=1e-3, gamma=0.95, epsilon=0.0, target_update=10,
        capacity=100, mini_size=10**9, batch_size=1, eps_start=0, eps_end=0,
        eps_decay=1, device="cpu")
    agent.q_net.load_state_dict(torch.load(
        ckpt, map_location="cpu", weights_only=False)["policy_state_dict"])
    agent.q_net.eval()

    X_slots, Q, M, CUR, TSI = [], [], [], [], []
    runs = [(s, 0.0) for s in args.seeds.split(",") if s] + \
           [(s, 0.2) for s in args.eps_seeds.split(",") if s]
    rng = np.random.RandomState(0)
    for seed, eps in runs:
        states = reset(seed)
        done = {"__all__": False}
        print(f"run seed={seed} eps={eps}")
        while not done["__all__"]:
            acts = {}
            for i, ts in enumerate(ts_ids):
                x = np.asarray(states[ts], dtype=np.float32)
                with torch.no_grad():
                    q = agent.q_net(torch.tensor(x).unsqueeze(0))[0].numpy()
                mask = ts_mask[ts]
                a = (int(np.random.choice(np.flatnonzero(mask)))
                     if rng.rand() < eps
                     else int(np.where(mask, q, -np.inf).argmax()))
                acts[ts] = int(std2green[ts][a])
                if len(Q) < args.cap:
                    X_slots.append(x[HEADER:].reshape(12, sd))
                    Q.append(q)
                    M.append(mask)
                    CUR.append(int(x[:8].argmax()))   # legacy one-hot header
                    TSI.append(i)
            states, _, done, _ = env.step(action=acts)
    env.close()

    os.makedirs(args.out, exist_ok=True)
    Xs = np.asarray(X_slots, np.float32)
    amb_any = (Xs[:, :, 12] > 0).any(axis=1)          # cnt_l5 offset = 4*3
    np.savez_compressed(os.path.join(args.out, "dqn8std_targets.npz"),
        slots=Xs, q=np.asarray(Q, np.float32), mask=np.asarray(M, bool),
        cur=np.asarray(CUR, np.int8), ts_idx=np.asarray(TSI, np.int16),
        A=np.stack([A[t] for t in ts_ids]))
    manifest = {"config": cfg_path, "ckpt": ckpt, "runs": runs,
                "feature_names": FEATURE_NAMES, "ts_ids": ts_ids,
                "header_dim": HEADER, "slot_dim": sd,
                "n": len(Q), "amb_rows": int(amb_any.sum())}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    print(json.dumps({k: manifest[k] for k in ("ckpt", "n", "amb_rows")}, indent=1))


if __name__ == "__main__":
    main()
