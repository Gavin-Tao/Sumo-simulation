import argparse
import os
import sys
from datetime import datetime
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

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
HIDDEN_NUMBER = 512
WRITE_TB = False
#设置SummaryWriter的路径
WRITER_PATH = f"./logs/1x1/4_phases/mixed_scenarios/{HIDDEN_NUMBER}/" + TIMESTAMP
STORE_PATH = f"./models/1x1/mixed_scenarios/{HIDDEN_NUMBER}/checkpoint/"+ TIMESTAMP
#实例化tensorboard的类SummaryWriter
tb_writer = SummaryWriter(log_dir = WRITER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = 120000
checkpoint_interval = 4000
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
    
  
# Define scenarios and their associated parameters
scenarios = [
    {
        "name": "heavy_low",
        "route_file": "nets/syc/1x1/Heavy_low/heavy_low.rou.xml",
        "net_file": "nets/syc/1x1/Heavy_low/syc_4phases.net.xml",
        "cfg_file": "nets/syc/1x1/Heavy_low/syc_4phases.sumocfg",
        "min_green": 10,
        "max_green": 50
    },
    {
        "name": "low_heavy",
        "route_file": "nets/syc/1x1/Low_heavy/low_heavy.rou.xml",
        "net_file": "nets/syc/1x1/Low_heavy/syc_4phases.net.xml",
        "cfg_file": "nets/syc/1x1/Low_heavy/syc_4phases_low_heavy.sumocfg",
        "min_green": 10,
        "max_green": 50
    },
    {
        "name": "heavy_heavy",
        "route_file": "nets/syc/1x1/Equal_entries_500/equal_entries_500.rou.xml",
        "net_file": "nets/syc/1x1/Equal_entries_500/syc_4phases.net.xml",
        "cfg_file": "nets/syc/1x1/Equal_entries_500/syc_4phases_equal_entries_500.sumocfg",
        "min_green": 10,
        "max_green": 50
    }
]


# Initialize scenario counters
scenario_counts = {scenario["name"]: 0 for scenario in scenarios}

def update_scenario(env, scenarios):
    """
    Update the environment with the parameters of a randomly selected scenario.
    """
    selected_scenario = random.choice(scenarios)
    scenario_counts[selected_scenario["name"]]  += 1
    env._net = selected_scenario["net_file"]
    env._route = selected_scenario["route_file"]
    env._cfg = selected_scenario["cfg_file"]
    env.min_green = selected_scenario["min_green"]
    env.max_green = selected_scenario["max_green"]
    print("This is local {} env.py".format(env._route))


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
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
        net_file=scenarios[0]["net_file"],  # Using the first scenario's net file
        route_file=scenarios[0]["route_file"],  # Using the first scenario's route file
        cfg_file=scenarios[0]["cfg_file"],  # Using the first scenario's config file
        out_csv_name=out_csv,
        use_gui=False,
        num_seconds=args.seconds,
        min_green=args.min_green, 
        max_green=args.max_green,
        use_max_green = True,
        sumo_seed = seed, #固定住seed
        #single_agent= True, #设置成True貌似TL会报错。
        observation_class = PressLightObservationFunction,
        reward_fn = "pressure",
        delta_time = 5,
    )

    for run in range(1, args.runs + 1):
        initial_states = env.reset(env.sumo_seed)
        for ts in env.ts_ids:
            a = initial_states[ts]
            b=tuple(a)
            c=1
        
        dqn_agents = {
            ts: DQN(
                starting_state=tuple(initial_states[ts]),
                state_space=env.observation_space.shape[0],
                hidden_dim=HIDDEN_NUMBER,
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
            for ts in env.ts_ids
        }
        step_counter = 0 #这是训练的step，一个train step=delta time 5s
        # a= dqn_agents[env.ts_ids[0]]
        for episode in range(1, episodes + 1):
            
            if episode != 1:
                update_scenario(env, scenarios) #更新env
                initial_states = env.reset(env.sumo_seed)
                for ts in initial_states.keys():
                    dqn_agents[ts].state = tuple(initial_states[ts])

            episode_return ={ts: 0 for ts in dqn_agents.keys()}
            infos = []
            done = {"__all__": False}
            
            while not done["__all__"]:
                
                actions = {ts: dqn_agents[ts].take_action(dqn_agents[ts].state) for ts in dqn_agents.keys()}
                
                s, r, done, info = env.step(action=actions)
                
                #把metrics分别存进tensorboard
                if WRITE_TB:
                    metrics_keys = info.keys()
                    for metrics_key in metrics_keys:
                        value = info[metrics_key]
                        tb_writer.add_scalar(metrics_key, value, step_counter)  
                
                    #把r字典中的reward分别存进tensorboard
                    for ts_id in env.ts_ids:
                        if r[ts_id] is not None:
                            a=r[ts_id]
                            tb_writer.add_scalar("reward_" + ts_id, r[ts_id], step_counter)
                            
                            episode_return[ts_id] += r[ts_id]
                            
                        if dqn_agents[ts_id].loss is not None:
                            b=dqn_agents[ts_id].loss
                            tb_writer.add_scalar("loss_" + ts_id, dqn_agents[ts_id].loss, step_counter)
                step_counter += 1
                

                for ts in initial_states.keys():
                    ts_state = dqn_agents[ts].state
                    ts_action = actions[ts]
                    #print(ts_action)
                    ts_reward = r[ts]
                    ts_next_state = tuple(s[ts])
                    ts_done = done[ts]
                    
                    dqn_agents[ts].replay_buffer.add(ts_state, ts_action, ts_reward, ts_next_state, ts_done)
                    
                    dqn_agents[ts].state = ts_next_state #s_t = s_t+1
                    
                    if dqn_agents[ts].replay_buffer.size() > dqn_agents[ts].mini_size:
                        b_s, b_a, b_r, b_ns, b_d = dqn_agents[ts].replay_buffer.sample(dqn_agents[ts].batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        dqn_agents[ts].epsilon = dqn_agents[ts].eps_end + (dqn_agents[ts].eps_start - dqn_agents[ts].eps_end) * \
                        math.exp(-1. * dqn_agents[ts].count / dqn_agents[ts].eps_decay)
                        dqn_agents[ts].update(transition_dict)
            # 记录了episode return
            if WRITE_TB:
                for ts_id in env.ts_ids:
                    tb_writer.add_scalar("episode_return" + ts_id, episode_return[ts_id], episode)
                               
            if dqn_agents[env.ts_ids[0]].start_train:
                if episode != 0 and episode % checkpoint_interval == 0:
                    print("----------------saving model at ", episode, "-----------------")
                    checkpoint = {
                        "policy_state_dict": dqn_agents[env.ts_ids[0]].q_net.state_dict(),
                        "target_state_dict": dqn_agents[env.ts_ids[0]].target_q_net.state_dict(),
                        # "policy_copy_state_dict": agent.policy_net_copy.state_dict(),
                        "optimizer_state_dic": dqn_agents[env.ts_ids[0]].optimizer.state_dict(),
                        # "z_optimizer_state_dict": agent.z_optimizer.state_dict(),
                        # "gamma_optimizer_state_dict": agent.gamma_optimizer.state_dict(),
                        "episode": episode,
                        # "agent_loss": agent.loss_list
                    }
                    if os.path.exists(STORE_PATH) is False:
                        os.makedirs(STORE_PATH)
                    ckpt_timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
                    path_checkpoint = os.path.join(STORE_PATH, "ckpt_{}_{}.pth".format(ckpt_timestamp, episode))
                    torch.save(checkpoint, path_checkpoint)
                    print("----------------Finished saving model at ", episode, "-----------------")
                  
            
            if env.metrics is not None:
                for metric in env.metrics:
                    env.list_metrics.append(metric)
                b=1

            # Log scenario counts to TensorBoard for each episode
            if WRITE_TB:
                for scenario_name, count in scenario_counts.items():
                    tb_writer.add_scalar(f"scenario_counts/{scenario_name}", count, episode)
                    
        env.txw_save_csv(out_csv, run)
        env.close()