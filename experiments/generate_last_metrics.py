"""Generate {best,last}_metrics[_warmup<N>].json by re-evaluating the chosen ckpt.

Matches the conditions under which the original best_metrics.json was produced
during training (num_seconds, eval_seed, delta_time, excluded_lanes), with the
addition of an optional warmup window.

Writes to:
  experiments/models/<exp>/<TS>/<kind>_metrics[_warmup<N>].json

  kind=last: re-evaluates the highest-episode ckpt_ep*.pth
  kind=best: re-evaluates best.pth (overwrites/parallels training-time best_metrics.json)
  suffix:    "_warmup<N>" if --warmup N (steps) > 0, else ""

Usage:
  python experiments/generate_last_metrics.py                   # kind=last, warmup=0
  python experiments/generate_last_metrics.py --kind best       # best.pth, warmup=0
  python experiments/generate_last_metrics.py --warmup 20       # last + 20-step warmup
  python experiments/generate_last_metrics.py --kind best --warmup 20 exp126
"""

from __future__ import annotations
import os
import sys
import json
import glob
import argparse
from pathlib import Path

import yaml
import torch

# Project root + SUMO setup (same as eval_unified.py)
os.environ["SUMO_RL_LIBSUMO"] = "0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Reuse eval_unified.py's helpers
from eval_unified import (
    OBS_REGISTRY, load_agent, pick_action,
)
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.environment.metrics import EpisodeMetricsCollector

EXP_DIR = Path(__file__).resolve().parent
# chdir so cfg's "../nets/..." paths resolve correctly (same as train.py / plot_best_metrics.py)
os.chdir(EXP_DIR)
DEFAULT_EXPS = [f"exp{i}" for i in (122, 123, 124, 125, 126, 127, 128, 129,
                                     133, 134, 135, 136, 137)]


def infer_agent_type(model_dir_name: str) -> str:
    """colightorig in name → orico; else → dqn."""
    return "orico" if "colightorig" in model_dir_name else "dqn"


def latest_run_dir(exp_prefix: str) -> Path | None:
    matches = sorted(glob.glob(str(EXP_DIR / "models" / f"{exp_prefix}_*" / "*")))
    matches = [m for m in matches if Path(m).is_dir()]
    return Path(matches[-1]) if matches else None


def find_last_ckpt(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("ckpt_ep*.pth"))
    return ckpts[-1] if ckpts else None


def find_best_ckpt(run_dir: Path) -> Path | None:
    p = run_dir / "best.pth"
    return p if p.exists() else None


def run_one_eval(cfg: dict, ckpt_path: Path, agent_type: str,
                 device: torch.device, warmup_steps: int = 0) -> dict:
    """Run a single greedy eval episode matching training-time eval conditions.

    If warmup_steps > 0, the first `warmup_steps` env.step calls happen but
    EpisodeMetricsCollector.collect_step is not called during them, so initial
    network-loading transients don't enter the averages.

    Returns EpisodeMetricsCollector.summary() (nested dict).
    """
    obs_class    = OBS_REGISTRY[cfg["observation_class"]]
    neighbor_map = cfg.get("neighbor_map", {})
    n_heads      = cfg.get("n_heads", 2)
    eval_seed    = int(cfg.get("eval_seed", cfg.get("seed", 0)))

    env = SumoEnvironment(
        net_file=cfg["net_file"],
        route_file=cfg["route_file"],
        cfg_file=cfg["cfg_file"],
        out_csv_name=None,
        use_gui=False,
        num_seconds=cfg.get("num_seconds", 1000),
        min_green=cfg.get("min_green", 5),
        max_green=cfg.get("max_green", 50),
        use_max_green=cfg.get("use_max_green", False),
        single_agent=cfg.get("single_agent", False),
        yellow_time=cfg.get("yellow_time", 2),
        delta_time=cfg.get("delta_time", 5),
        reward_fn=cfg["reward_fn"],
        observation_class=obs_class,
        sumo_seed=eval_seed,
    )

    initial_states = env.reset(eval_seed)
    obs_dim        = env.observation_space.shape[0]
    ts_lane_map    = {ts: env.traffic_signals[ts].signal_controlled_lanes
                      for ts in env.ts_ids}
    always_green   = set().union(*(env.traffic_signals[ts].always_green_lanes
                                    for ts in env.ts_ids))

    agent = load_agent(agent_type, str(ckpt_path), cfg, obs_dim,
                       env.action_space.n, device, n_heads)

    mc = EpisodeMetricsCollector(
        ts_lane_map, delta_time=env.delta_time, excluded_lanes=always_green,
    )
    done: dict = {"__all__": False}
    step = 0
    while not done["__all__"]:
        if step >= warmup_steps:
            mc.collect_step(env.sumo)
        actions = {ts: pick_action(agent_type, agent, ts,
                                   initial_states, neighbor_map, obs_dim)
                   for ts in env.ts_ids}
        initial_states, _, done, _ = env.step(action=actions)
        step += 1
    mc.collect_step(env.sumo)
    mc.finalize(env.sumo)
    env.close()
    return mc.summary()


def process_exp(exp_prefix: str, device: torch.device,
                kind: str = "last", warmup_steps: int = 0,
                overwrite: bool = False) -> bool:
    run_dir = latest_run_dir(exp_prefix)
    if run_dir is None:
        print(f"[{exp_prefix}] no run dir — skipped")
        return False

    suffix = f"_warmup{warmup_steps}" if warmup_steps > 0 else ""
    out_name = f"{kind}_metrics{suffix}.json"
    out_path = run_dir / out_name
    if out_path.exists() and not overwrite:
        print(f"[{exp_prefix}] {out_name} already exists — skipped (use --overwrite)")
        return False

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        print(f"[{exp_prefix}] config.yaml missing in {run_dir} — skipped")
        return False

    if kind == "last":
        ckpt_path = find_last_ckpt(run_dir)
    elif kind == "best":
        ckpt_path = find_best_ckpt(run_dir)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    if ckpt_path is None:
        print(f"[{exp_prefix}] no ckpt found for kind={kind} — skipped")
        return False

    # Parse episode number. Two cases:
    #   - "ckpt_epNNNNN.pth" → parse N from filename
    #   - "best.pth"         → load ckpt and read its "episode" key (saved by train.py)
    try:
        ep_num = int(ckpt_path.stem.split("ep")[-1])
    except ValueError:
        try:
            _ckpt = torch.load(str(ckpt_path), map_location="cpu")
            ep_num = int(_ckpt.get("episode", -1))
            del _ckpt
        except Exception:
            ep_num = -1

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    agent_type = infer_agent_type(run_dir.parent.name)
    delta_time = cfg.get("delta_time", 5)
    print(f"[{exp_prefix}] {agent_type} kind={kind} ckpt={ckpt_path.name} "
          f"warmup={warmup_steps} steps ({warmup_steps * delta_time}s)")

    summary = run_one_eval(cfg, ckpt_path, agent_type, device,
                           warmup_steps=warmup_steps)

    payload = {
        "_meta": {
            "kind":           kind,
            "episode":        ep_num,
            "ckpt_filename":  ckpt_path.name,
            "agent_type":     agent_type,
            "eval_seed":      int(cfg.get("eval_seed", cfg.get("seed", 0))),
            "num_seconds":    cfg.get("num_seconds", 1000),
            "warmup_steps":   warmup_steps,
            "warmup_seconds": warmup_steps * delta_time,
        },
        "metrics": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[{exp_prefix}] → {out_path}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exps", nargs="*", default=DEFAULT_EXPS,
                   help="experiment prefixes (default: exp122..exp137)")
    p.add_argument("--kind", choices=("best", "last"), default="last",
                   help="best.pth vs last ckpt_ep*.pth (default: last)")
    p.add_argument("--warmup", type=int, default=0,
                   help="warmup steps to skip before metric collection (default: 0)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index; -1 for CPU")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate even if output file exists")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}"
                          if args.gpu >= 0 and torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    n_ok = 0
    for exp in args.exps:
        try:
            if process_exp(exp, device, kind=args.kind,
                           warmup_steps=args.warmup,
                           overwrite=args.overwrite):
                n_ok += 1
        except Exception as e:
            print(f"[{exp}] ERROR: {e}")
            import traceback; traceback.print_exc()
    suffix = f"_warmup{args.warmup}" if args.warmup > 0 else ""
    print(f"\nDone — wrote {args.kind}_metrics{suffix}.json for "
          f"{n_ok}/{len(args.exps)} experiments")


if __name__ == "__main__":
    main()
