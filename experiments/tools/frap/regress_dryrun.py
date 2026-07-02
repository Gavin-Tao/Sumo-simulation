"""Regression dry-run: N decision steps of a config's env/obs/action stack.

Digest covers (per step): sorted action dict, obs dims, rounded rewards.
Purpose: prove that train.py-adjacent code changes leave old configs'
env+obs+masked-action stack bit-identical (FRAP_ENUM_PLAN Task 0/6).
Run from repo root:  python experiments/tools/frap/regress_dryrun.py <cfg.yaml>
Deterministic on CPU for a fixed config (fixed seeds, greedy policy).
"""
import os, sys, json, hashlib, functools, argparse

os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
# config paths (net/route/cfg/meta) are relative to experiments/ — same cwd
# convention as train.py; main() chdirs there (after resolving the cfg arg)
# so the harness works from anywhere.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import random  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    cfg_path = os.path.abspath(a.cfg)
    os.chdir(os.path.join(_REPO, "experiments"))
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from sumo_rl.agents.dqn_agent_txw import DQN
    obs_class = getattr(obsmod, {
        "PriorityMovement": "PriorityMovementObservationFunction",
        "PriorityBCA": "PriorityBCAObservationFunction",
    }[cfg["observation_class"]])
    obs_kwargs = {}
    for src, dst in [("obs_fields", "fields"), ("obs_phase_state", "phase_state"),
                     ("priority_source", "priority_source"), ("obs_downstream", "include_downstream"),
                     ("obs_downstream_fields", "downstream_fields"), ("obs_lane_occ", "include_lane_occ"),
                     ("obs_awt_cap", "awt_cap"), ("obs_awt_basis", "awt_basis"),
                     ("obs_slot_stats", "slot_stats")]:
        if src in cfg:
            v = cfg[src]
            obs_kwargs[dst] = tuple(v) if isinstance(v, list) else v
    if obs_kwargs:
        obs_class = functools.partial(obs_class, **obs_kwargs)
    reward_fn = cfg["reward_fn"]
    if reward_fn == "priority-avg-waiting":
        from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
        from sumo_rl.environment.priority_map import load_priority_table
        reward_fn = make_priority_avg_waiting_reward(load_priority_table(cfg.get("priority_source")))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
        num_seconds=cfg.get("num_seconds", 1000), min_green=cfg.get("min_green", 5),
        max_green=cfg.get("max_green", 50), use_max_green=cfg.get("use_max_green", False),
        single_agent=False, yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5), reward_fn=reward_fn,
        observation_class=obs_class, sumo_seed=cfg.get("seed", 0), sumo_warnings=False)
    ts_mask, std2green, green2std, ts_turnmap = {}, {}, {}, {}
    meta_file = cfg.get("action_meta_file")
    if meta_file:
        meta = json.load(open(meta_file))["tls"]
        for tid, t in meta.items():
            ts_mask[tid] = np.array(t["mask"], dtype=bool)
            ts_turnmap[tid] = {int(i): (c[0]["approach"], c[0]["turn"]) for i, c in t["links"].items()}
            s2g = np.full(8, -1, dtype=int)
            for k, gi in t["std_to_green_index"].items():
                s2g[int(k)] = gi
            std2green[tid] = s2g
            g2s = np.full(int(s2g.max()) + 1, -1, dtype=int)
            for k in range(8):
                if s2g[k] >= 0:
                    g2s[s2g[k]] = k
            green2std[tid] = g2s
    states = env.reset(int(cfg.get("seed", 0)))
    if meta_file:
        for tid in env.ts_ids:
            ts = env.traffic_signals[tid]
            ts.std_action_map = green2std[tid]
            if hasattr(ts.observation_fn, "rebind_movements"):
                ts.observation_fn.rebind_movements(ts_turnmap[tid])
            ts.observation_space = ts.observation_fn.observation_space()
        states = {tid: env.traffic_signals[tid].observation_fn() for tid in env.ts_ids}
    od = len(next(iter(states.values())))
    agent = DQN(starting_state=tuple([0.0] * od), state_space=od,
        hidden_dim=cfg.get("hidden_dim", 64),
        action_space=(8 if meta_file else env.action_space.n), learning_rate=1e-3,
        gamma=0.95, epsilon=0.0, target_update=10, capacity=100, mini_size=10**9,
        batch_size=1, eps_start=0, eps_end=0, eps_decay=1, device="cpu")
    h = hashlib.sha256()
    done = {"__all__": False}
    for _ in range(a.steps):
        if done["__all__"]:
            break
        if meta_file:
            acts = {ts: int(std2green[ts][agent.take_action(states[ts], mask=ts_mask[ts])])
                    for ts in env.ts_ids}
        else:
            acts = {ts: agent.take_action(states[ts]) for ts in env.ts_ids}
        states, r, done, _ = env.step(action=acts)
        h.update(json.dumps([sorted(acts.items()),
                             sorted((k, len(v)) for k, v in states.items()),
                             sorted((k, round(float(v), 6)) for k, v in r.items())]).encode())
    env.close()
    print(f"DIGEST {h.hexdigest()}")


if __name__ == "__main__":
    main()
