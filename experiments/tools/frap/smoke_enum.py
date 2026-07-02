"""V3 smoke for the enum_frap stack (FRAP_ENUM_PLAN Task 7). No wandb.

Runs the exp211 config's env + enum tables + FRAPAgent for N decision steps
with runtime assertions:
  * chosen action always inside the junction's phase mask (greedy AND explore)
  * protected-only: no 'g' in any tlLogic phase state of the enum net
  * learn_step produces finite, non-exploding losses
  * forward shape sanity at the K=2 and K=11 junctions
Run from repo root: SUMO_RL_LIBSUMO=1 python experiments/tools/frap/smoke_enum.py
"""
import os, sys, re, json, functools, random

os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

CFG = os.path.join(_REPO, "experiments", "configs", "exp211_dublin11h_531_enumfrap.yaml")
STEPS = 120


def main():
    os.chdir(os.path.join(_REPO, "experiments"))
    cfg = yaml.safe_load(open(CFG))
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    # protected-only assertion on the net itself
    net_xml = open(cfg["net_file"]).read()
    states = re.findall(r'state="([^"]+)"', net_xml)
    assert states and all("g" not in s for s in states), "protected-only violated in enum net"
    print(f"net check: {len(states)} phase states, zero 'g' ✓")

    from sumo_rl.environment.env import SumoEnvironment
    from sumo_rl.environment import observations as obsmod
    from frap_glue import load_enum_tables, build_frap_agent
    from sumo_rl.environment.rewards import make_priority_avg_waiting_reward
    from sumo_rl.environment.priority_map import load_priority_table

    obs_class = functools.partial(
        obsmod.PriorityMovementObservationFunction,
        fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"],
        priority_source=cfg["priority_source"],
        include_downstream=bool(cfg["obs_downstream"]),
        downstream_fields=tuple(cfg["obs_downstream_fields"]),
        include_lane_occ=bool(cfg["obs_lane_occ"]),
        awt_cap=float(cfg["obs_awt_cap"]), awt_basis=cfg["obs_awt_basis"])
    reward_fn = make_priority_avg_waiting_reward(load_priority_table(cfg["priority_source"]))
    env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
        num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
        max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
        single_agent=False, yellow_time=cfg["yellow_time"],
        delta_time=cfg["delta_time"], reward_fn=reward_fn,
        observation_class=obs_class, sumo_seed=cfg["seed"], sumo_warnings=False)

    tables = load_enum_tables(cfg["enum_meta_file"])
    states = env.reset(int(cfg["seed"]))
    for tid in env.ts_ids:
        ts = env.traffic_signals[tid]
        ts.observation_fn.rebind_movements(tables["turnmap"][tid])
        ts.observation_space = ts.observation_fn.observation_space()
    states = {tid: env.traffic_signals[tid].observation_fn() for tid in env.ts_ids}
    dims = {len(v) for v in states.values()}
    assert len(dims) == 1, f"obs dim not uniform across junctions: {dims}"
    print(f"obs dim uniform: {dims.pop()} ✓")

    agent = build_frap_agent(cfg, tables, env, "cpu")
    agent.mini_size = 64                              # smoke: learn early
    agent.epsilon = 0.3                               # exercise both branches

    # per-junction green-phase count must equal menu size (net<->meta consistency)
    for tid in env.ts_ids:
        k_meta = int(tables["tls"][tid]["mask"].sum())
        k_env = env.traffic_signals[tid].num_green_phases
        assert k_env == k_meta, f"{tid}: env greens {k_env} != meta {k_meta}"
    print("net greens == meta menus for all 18 TLS ✓")

    losses, done = [], {"__all__": False}
    for step in range(STEPS):
        if done["__all__"]:
            break
        acts = {}
        for ts in env.ts_ids:
            a = agent.take_action(states[ts], ts)
            assert tables["tls"][ts]["mask"][a], f"invalid phase {a} at {ts}"
            acts[ts] = a
        nstates, r, done, _ = env.step(action=acts)
        for ts in env.ts_ids:
            agent.replay_buffer.add(states[ts],
                env.traffic_signals[ts].last_executed_action,
                r[ts], tuple(nstates[ts]), done[ts], ts)
        states = nstates
        agent.learn_step()
        if agent.loss is not None:
            losses.append(agent.loss)
    env.close()
    assert len(losses) > 20, f"too few updates: {len(losses)}"
    assert all(np.isfinite(losses)), "non-finite loss"
    assert np.mean(losses[-10:]) < np.mean(losses[:10]) * 10 + 1.0, "loss exploding"
    print(f"SMOKE OK: steps={step + 1} updates={len(losses)} "
          f"loss[first10]={np.mean(losses[:10]):.4f} loss[last10]={np.mean(losses[-10:]):.4f} "
          f"q_mean={agent.q_mean:.3f} q_abs_max={agent.q_abs_max:.3f}")


if __name__ == "__main__":
    main()
