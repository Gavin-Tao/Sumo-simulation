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
from stable_baselines3.dqn.dqn import DQN
from torch.utils.tensorboard import SummaryWriter

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
#设置SummaryWriter的路径
WRITER_PATH = "./logs/" + TIMESTAMP
#实例化tensorboard的类SummaryWriter
tb_writer = SummaryWriter(log_dir = WRITER_PATH)


if __name__ == "__main__":
   
    env = SumoEnvironment(
        net_file="nets/syc/syc.net.xml",
        route_file="nets/syc/syc.rou.xml",
        out_csv_name="outputs/syc/dqn",
        use_gui=True,
        num_seconds=1000,
        use_max_green = True,
        sumo_seed=0, #固定住seed
        #single_agent= True, #设置成True貌似TL会报错。
        observation_class = PressLightObservationFunction,
        reward_fn = "pressure",
        delta_time = 5,
        single_agent=True
    )

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=10000)