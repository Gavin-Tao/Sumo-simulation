"""No-wandb smoke of exp220 (enum_frap on exp217 1x1 scenario), cribbed from
tools/frap/smoke_enum.py with exp217-style obs/reward defaults."""
import os, sys, re, functools, random
os.environ.setdefault("SUMO_RL_LIBSUMO", "1")
_REPO = "/home/xiaowen/sumo-rl"
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "experiments"))
os.chdir(os.path.join(_REPO, "experiments"))
import numpy as np, torch, yaml
cfg = yaml.safe_load(open("configs/exp220_1x1_531_NS20bus_enumfrap_cqm.yaml"))
torch.manual_seed(0); random.seed(0); np.random.seed(0)
net_xml = open(cfg["net_file"]).read()
sts = re.findall(r'state="([^"]+)"', net_xml)
assert sts and all("g" not in s for s in sts), "protected-only violated"
print(f"net check: {len(sts)} phase states, zero 'g' OK")
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment import observations as obsmod
from frap_glue import load_enum_tables, build_frap_agent
obs_kwargs = dict(fields=tuple(cfg["obs_fields"]), phase_state=cfg["obs_phase_state"])
for src, dst in [("priority_source","priority_source"),("obs_awt_cap","awt_cap"),("obs_awt_basis","awt_basis")]:
    if src in cfg: obs_kwargs[dst] = cfg[src]
obs_class = functools.partial(obsmod.PriorityMovementObservationFunction, **obs_kwargs)
env = SumoEnvironment(net_file=cfg["net_file"], route_file=cfg["route_file"],
    cfg_file=cfg["cfg_file"], out_csv_name=None, use_gui=False,
    num_seconds=cfg["num_seconds"], min_green=cfg["min_green"],
    max_green=cfg["max_green"], use_max_green=cfg["use_max_green"],
    single_agent=False, yellow_time=cfg["yellow_time"], delta_time=cfg["delta_time"],
    reward_fn=cfg["reward_fn"], observation_class=obs_class,
    sumo_seed=cfg["seed"], sumo_warnings=False)
tables = load_enum_tables(cfg["enum_meta_file"])
env.reset(int(cfg["seed"]))
for tid in env.ts_ids:
    ts = env.traffic_signals[tid]
    ts.observation_fn.rebind_movements(tables["turnmap"][tid])
    ts.observation_space = ts.observation_fn.observation_space()
states = {tid: env.traffic_signals[tid].observation_fn() for tid in env.ts_ids}
print("obs dim:", {len(v) for v in states.values()})
agent = build_frap_agent(cfg, tables, env, "cpu")
agent.mini_size = 64; agent.epsilon = 0.3
for tid in env.ts_ids:
    k_meta = int(tables["tls"][tid]["mask"].sum())
    k_env = env.traffic_signals[tid].num_green_phases
    assert k_env == k_meta, f"{tid}: env {k_env} != meta {k_meta}"
print("net greens == meta menu OK")
losses, done = [], {"__all__": False}
for step in range(120):
    if done["__all__"]: break
    acts = {}
    for t in env.ts_ids:
        a = agent.take_action(states[t], t)
        assert tables["tls"][t]["mask"][a], f"invalid phase {a} at {t}"
        acts[t] = a
    nstates, r, done, _ = env.step(action=acts)
    for t in env.ts_ids:
        agent.replay_buffer.add(states[t], env.traffic_signals[t].last_executed_action,
                                r[t], tuple(nstates[t]), done[t], t)
    states = nstates
    agent.learn_step()
    if agent.loss is not None: losses.append(agent.loss)
env.close()
assert len(losses) > 20 and all(np.isfinite(losses)), "loss problem"
print(f"SMOKE OK: steps={step+1} updates={len(losses)} "
      f"loss[first10]={np.mean(losses[:10]):.4f} loss[last10]={np.mean(losses[-10:]):.4f}")
