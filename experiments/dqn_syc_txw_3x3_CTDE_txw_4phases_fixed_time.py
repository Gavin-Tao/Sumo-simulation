import argparse
import os
import sys
from datetime import datetime
import torch
import random
import numpy as np

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
    
sys.path.insert(0, 'E:\\txw\\SUMO-RL') #第一优先在这个路径下去寻找包

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.dqn_agent_txw import DQN
from sumo_rl.environment.observations import PressLightObservationFunction

from torch.utils.tensorboard import SummaryWriter
import math
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
#设置SummaryWriter的路径
WRITER_PATH = "./logs/3x3/txw_4phases/" + TIMESTAMP
#实例化tensorboard的类SummaryWriter
tb_writer = SummaryWriter(log_dir = WRITER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = 200
checkpoint_interval = 5
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="nets/syc/3x3/intersection_heavy_2.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-use_max_green", dest="use_max_green", default=False, help="False: use pre-defined green duration as max; True: use max_green.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=1000, required=False, help="Number of simulation seconds.\n")  #这里我设置了1000s
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split(".")[0]
    out_csv = f"outputs/syc/{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}"

    env = SumoEnvironment(
        net_file="nets/syc/3x3/intersection_new_txw.net.xml",
        route_file=args.route,
        cfg_file = "nets/syc/3x3/syc_new_txw.sumocfg",
        out_csv_name=out_csv,
        use_gui=True,
        num_seconds=args.seconds,
        min_green=args.min_green, 
        max_green=args.max_green,
        use_max_green = True,
        sumo_seed=seed, #固定住seed
        #single_agent= True, #设置成True貌似TL会报错。
        observation_class = PressLightObservationFunction,
        reward_fn = "pressure",
        delta_time = 5,
        single_agent=False,
    )

    for run in range(1, args.runs + 1):
        initial_states = env.reset(env.sumo_seed)
        
        for ts in env.ts_ids:
            last_ts_id = ts
        
        dqn_agent = DQN(
                starting_state=tuple(initial_states[last_ts_id]), #初始化DQN agent，所以随便给一个starting_state就行，因为take_action那里会重新给state
                state_space=env.observation_space.shape[0],
                hidden_dim=64,
                action_space=env.action_space.n,
                learning_rate=0.01,
                gamma=0.99,
                epsilon=0.1,
                target_update=10,
                capacity=10000,
                mini_size=500,
                batch_size=256,
                eps_start=0.5,
                eps_end=0.01,
                eps_decay=1000,
                device=device,
            )
        
        step_counter = 0
        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset(env.sumo_seed)
                
            for step in range(0, env.sim_max_time):
                env._sumo_step()
                info = env._compute_info()
                env.sim_step_counter+=1
                
                
                #把metrics分别存进tensorboard
                metrics_keys = info.keys()
                for metrics_key in metrics_keys:
                    value = info[metrics_key]
                    tb_writer.add_scalar(metrics_key, value, step_counter) 
                
              
        
                        
        env.txw_save_csv(out_csv, run)
        env.close()
     
        