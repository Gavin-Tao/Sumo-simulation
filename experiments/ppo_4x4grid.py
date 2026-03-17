import os
import sys


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from pathlib import Path

# locate project root (folder that contains the 'sumo_rl' package)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# climb up until we find sumo_rl folder (robust across run locations)
probe = PROJECT_ROOT
while not (probe / "sumo_rl").exists() and probe.parent != probe:
    probe = probe.parent
PROJECT_ROOT = probe
sys.path.insert(0, str(PROJECT_ROOT))
import sumo_rl


if __name__ == "__main__":
    # ray.init()
    ray.init()


    env_name = "4x4grid"

    # default paths (relative to project root)
    nets_dir = PROJECT_ROOT / "nets" / "4x4-Lucas"
    default_net = str(nets_dir / "4x4.net.xml")
    default_route = str(nets_dir / "4x4c1c2c1c2.rou.xml")
    default_cfg = str(nets_dir / "4x4.sumocfg")
    default_out = str(PROJECT_ROOT / "outputs" / "4x4grid" / "ppo")

    # allow overrides via environment variables for portability
    net_file = os.environ.get("NET_FILE", default_net)
    route_file = os.environ.get("ROUTE_FILE", default_route)
    cfg_file = os.environ.get("CFG_FILE", default_cfg)
    out_csv_name = os.environ.get("OUT_CSV", default_out)
    use_gui = os.environ.get("USE_GUI", "false").lower() in ("1", "true", "yes")
    num_seconds = int(os.environ.get("NUM_SECONDS", "1000"))

    # ensure output directory exists
    Path(out_csv_name).parent.mkdir(parents=True, exist_ok=True)

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file=net_file,
                route_file=route_file,
                cfg_file = cfg_file,
                out_csv_name=out_csv_name,
                use_gui=use_gui,
                num_seconds=num_seconds,
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    )

    # outputs dir: 可通过环境变量 RLLIB_RESULTS 指定，否则放到项目下的 ray_results
    default_results = PROJECT_ROOT / "ray_results" / env_name
    local_dir = Path(os.environ.get("RLLIB_RESULTS", str(default_results))).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        local_dir=str(local_dir),
        config=config.to_dict(),
    )
