import argparse
import os
import sys
from datetime import datetime


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
    
sys.path.insert(0, 'E:\\txw\\SUMO-RL') #第一优先在这个路径下去寻找包

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.ql_agent import QLAgent
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
from sumo_rl.environment.observations import PressLightObservationFunction
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import ray
import sumo_rl

from torch.utils.tensorboard import SummaryWriter

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
#设置SummaryWriter的路径
WRITER_PATH = "./logs/" + TIMESTAMP
#实例化tensorboard的类SummaryWriter
tb_writer = SummaryWriter(log_dir = WRITER_PATH)


if __name__ == "__main__":
    ray.init()

    env_name = "syc"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file="nets/syc/syc.net.xml",
                route_file="nets/syc/syc.rou.xml",
                out_csv_name="outputs/syc/ppo",
                use_gui=False,
                num_seconds=80000,
                use_max_green = True,
                sumo_seed=0, #固定住seed
                #single_agent= True, #设置成True貌似TL会报错。
                observation_class = PressLightObservationFunction,
                reward_fn = "pressure",
                delta_time = 20
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
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        local_dir="E:\\txw\\SUMO-RL\\ray_results\\" + env_name,
        config=config.to_dict(),
    )
