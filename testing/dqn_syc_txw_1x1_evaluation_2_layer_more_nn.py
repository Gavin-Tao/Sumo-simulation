import argparse
import os
import sys
from datetime import datetime
import torch
import random
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
import csv

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
    
sys.path.insert(0, 'E:\\txw\\SUMO-RL') #第一优先在这个路径下去寻找包

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.dqn_agent_txw_2layers import DQN2
from sumo_rl.environment.observations import PressLightObservationFunction

from torch.utils.tensorboard import SummaryWriter
import math
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
HIDDEN_NUMBER = 512
#设置SummaryWriter的路径
WRITER_PATH = f"./logs/1x1/evaluation/mixed_model_heavy_low_test/{HIDDEN_NUMBER}/" + TIMESTAMP
# path_checkpoint = "./models/1x1/heavy_low/checkpoint/2024-03-25T15-24-58/ckpt_2024-03-25T16-35-19_2000.pth" # heavy_low model
# path_checkpoint = "./models/1x1/low_heavy/checkpoint/2024-03-25T17-33-11/ckpt_2024-03-25T18-51-19_2000.pth" # low_heavy model
# path_checkpoint = "./models/1x1/equal_entries_500/checkpoint/2024-03-25T17-47-30/ckpt_2024-03-25T19-35-46_2000.pth" # heavy_heavy model
path_checkpoint = r"E:\txw\sumo-rl\models\1x1\mixed_scenarios\checkpoint\two_layer\more_nn\2024-05-13T20-09-48-256\ckpt_2024-05-14T20-22-34_12000.pth" #mixed model


# 构建带有时间戳的文件路径
csv_folder_path = f"./testing/mixed_model_heavy_low_test/{HIDDEN_NUMBER}/" + TIMESTAMP + "/"

# 确保文件夹存在，如果不存在则创建
if not os.path.exists(csv_folder_path):
    os.makedirs(csv_folder_path)

# 拼接文件路径
csv_file_path = os.path.join(csv_folder_path, "mixed_heavy_low_train_evaluation.csv")


#实例化tensorboard的类SummaryWriter
tb_writer = SummaryWriter(log_dir = WRITER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = 5
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
#warm up后计算episode的平均值
warm_up_length = 900
# 定义要写入CSV文件的字段名和数据
fieldnames = [
    "Episode",
    "Produced_vehicles_N_S",
    "Produced_vehicles_W_E",
    "Throughput_N_S",
    "Throughput_W_E",
    "Total_Throughput",
    "Percentage_Throughput_N_S",
    "Percentage_Throughput_W_E",
    "Percentage_Total_Throughput",
    "Average_Speed_N_S",
    "Average_Speed_W_E",
    "Overall_Average_Speed",
    "Average_Waiting_Time_N_S",
    "Average_Waiting_Time_W_E",
    "Overall_Average_Waiting_Time",
    "Sum_Waiting_Time_N_S",
    "Sum_Waiting_Time_W_E",
    "Overall_Sum_Waiting_Time",
    "Stopped_Times_N_S",
    "Stopped_Times_W_E",
    "Total_Stopped_Times"
]



if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="nets/syc/1x1/Heavy_low/heavy_low.rou.xml", # heavy_low
        # default="nets/syc/1x1/Low_heavy/low_heavy.rou.xml", # low_heavy
        # default="nets/syc/1x1/Equal_entries_500/equal_entries_500.rou.xml", # equal 500
        # default="nets/syc/1x1/Equal_entries_140/equal_entries_140.rou.xml", # equal 140
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
        net_file="nets/syc/1x1/Heavy_low/syc_4phases.net.xml",  # heavy_low
        # net_file="nets/syc/1x1/Low_heavy/syc_4phases.net.xml", # low_heavy
        # net_file="nets/syc/1x1/Equal_entries_500/syc_4phases.net.xml", # equal 500
        # net_file="nets/syc/1x1/Equal_entries_140/syc_4phases.net.xml", # equal 140
        route_file=args.route,
        out_csv_name=out_csv,
        cfg_file = "nets/syc/1x1/Heavy_low/syc_4phases.sumocfg", # heavy_low
        # cfg_file = "nets/syc/1x1/Low_heavy/syc_4phases_low_heavy.sumocfg", # low_heavy
        # cfg_file = "nets/syc/1x1/Equal_entries_500/syc_4phases_equal_entries_500.sumocfg", # equal 500
        # cfg_file = "nets/syc/1x1/Equal_entries_140/syc_4phases_equal_entries_140.sumocfg", # equal 140
        use_gui=False,
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
            
        
        dqn_agent = DQN2(
                starting_state=tuple(initial_states[last_ts_id]), #初始化DQN agent，所以随便给一个starting_state就行，因为take_action那里会重新给state
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
        
        print("----------------loading model-----------------")
        
        checkpoint = torch.load(path_checkpoint)  # 加载

        dqn_agent.q_net.load_state_dict(checkpoint['policy_state_dict'])  # 加载模型可学习参数
        dqn_agent.target_q_net.load_state_dict(checkpoint['target_state_dict'])
        dqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dic'])  # 加载优化器参数

        print("----------------Finished loading model-----------------")
        
        step_counter = 0
        list_episode_throughput_per_direction = {"N_S": [], "W_E": []}
        list_episode_total_throughput = []
        
        list_episode_vehicles_produced = {"N_S": [], "W_E": []}
        list_episode_total_vehicles_produced = []
        list_episode_throughput_percentage = {"N_S": [], "W_E": [], "total": []}

        
        
        list_episode_average_speed_per_direction = {"N_S": [], "W_E": []}
        list_episode_overall_average_speed = []
        
        list_episode_average_waiting_time_per_direction = {"N_S": [], "W_E": []} 
        list_episode_overall_average_waiting_time = []
        
        list_episode_sum_waiting_time_per_direction = {"N_S": [], "W_E": []} 
        list_episode_sum_waiting_time = []
        
        list_episode_stopped_times_per_direction = {"N_S": [], "W_E": []} 
        list_episode_total_stopped_times = []
        
        env.evaluation = True
        
        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset(env.sumo_seed)


            infos = []
            done = {"__all__": False}
            episode_return ={ts: 0 for ts in env.ts_ids}
            
            while not done["__all__"]:
                
                actions = {}
                for ts in env.ts_ids:
                    
                    action_ts = dqn_agent.take_action(initial_states[ts])
                    actions[ts] = action_ts
                
                s, r, done, info = env.step(action=actions)
                """ 
                #计算路口和每个方向的metrics
                intersection_info = env._get_system_info()
                waiting_time_system = intersection_info["system_total_waiting_time"]
                mean_waiting_system = intersection_info["system_mean_waiting_time"]
                stopped_times_system = intersection_info["system_total_stopped"]
                mean_speed_system = intersection_info["system_mean_speed"]
                
                
                agent_info = env._get_per_agent_info()
                agent_waiting_time = agent_info["t_accumulated_waiting_time"]
                agent_stopped_times = agent_info["t_stopped"]
                # accumulate_waiting_time, average_waiting_time, stopped_times, sum_speed_per_direction, average_speed_per_direction, overall_average_speed, overall_average_waiting_time = get_info_per_direction(env,N_S_lanes,W_E_lanes)
                a=1
                #a=get_accumulated_waiting_time_per_direction(env)
                
                #把metrics分别存进tensorboard
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
                        
                    if dqn_agent.loss is not None:
                        
                        tb_writer.add_scalar("loss_" + ts_id, dqn_agent.loss, step_counter)
                step_counter += 1
                    """ 
                initial_states = s #s_t = s_t+1
            """ 
            # 记录了episode return
            for ts_id in env.ts_ids:
                tb_writer.add_scalar("episode_return" + ts_id, episode_return[ts_id], episode)        
            
            if env.metrics is not None:
                for metric in env.metrics:
                    env.list_metrics.append(metric)
                b=1  
             """
            #记录throughput
            episode_throughput_per_direction = env.throughput_per_direction
            list_episode_total_throughput.append(env.total_throughput)
            list_episode_throughput_per_direction["N_S"].append(episode_throughput_per_direction["N_S"])
            list_episode_throughput_per_direction["W_E"].append(episode_throughput_per_direction["W_E"])
            
            #记录生成了多少量车
            episode_produced_vehicles = env.number_produced_vehicles
            produced_N_S = episode_produced_vehicles["N_S"]
            produced_W_E = episode_produced_vehicles["W_E"]
            total_produced = produced_N_S + produced_W_E
            list_episode_vehicles_produced["N_S"].append(produced_N_S)
            list_episode_vehicles_produced["W_E"].append(produced_W_E)
            list_episode_total_vehicles_produced.append(total_produced)
            
            #记录throughput的百分比
            throughput_percentage_N_S = episode_throughput_per_direction["N_S"] / produced_N_S
            throughput_percentage_W_E = episode_throughput_per_direction["W_E"] / produced_W_E
            throughput_percentage_total = env.total_throughput / total_produced
            list_episode_throughput_percentage["N_S"].append(throughput_percentage_N_S)
            list_episode_throughput_percentage["W_E"].append(throughput_percentage_W_E)
            list_episode_throughput_percentage["total"].append(throughput_percentage_total)
            
            #记录average speed
            list_average_speed_N_S = env.list_average_speed_per_direction["N_S"]
            last_900_average_speed_N_S = list_average_speed_N_S[-warm_up_length:]
            episode_average_speed_N_S = sum(last_900_average_speed_N_S) / len(last_900_average_speed_N_S)
            list_episode_average_speed_per_direction["N_S"].append(episode_average_speed_N_S)
            
            list_average_speed_W_E = env.list_average_speed_per_direction["W_E"]
            last_900_average_speed_W_E = list_average_speed_W_E[-warm_up_length:]
            episode_average_speed_W_E = sum(last_900_average_speed_W_E) / len(last_900_average_speed_W_E)
            list_episode_average_speed_per_direction["W_E"].append(episode_average_speed_W_E)
            
            list_overall_average_speed = env.list_overall_average_speed
            last_900_average_speed_overall = list_overall_average_speed[-warm_up_length:]
            episode_overall_average_speed = sum(last_900_average_speed_overall) / len(last_900_average_speed_overall)
            list_episode_overall_average_speed.append(episode_overall_average_speed)
            
            #记录average waiting time
            list_average_waiting_time_N_S = env.list_average_waiting_time_per_direction["N_S"]
            last_900_average_waiting_time_N_S = list_average_waiting_time_N_S[-warm_up_length:]
            episode_average_waiting_time_N_S = sum(last_900_average_waiting_time_N_S) / len(last_900_average_waiting_time_N_S)
            list_episode_average_waiting_time_per_direction["N_S"].append(episode_average_waiting_time_N_S)
            
            list_average_waiting_time_W_E = env.list_average_waiting_time_per_direction["W_E"]
            last_900_average_waiting_time_W_E = list_average_waiting_time_W_E[-warm_up_length:]
            episode_average_waiting_time_W_E = sum(last_900_average_waiting_time_W_E) / len(last_900_average_waiting_time_W_E)
            list_episode_average_waiting_time_per_direction["W_E"].append(episode_average_waiting_time_W_E)
            
            list_overall_average_waiting_time = env.list_overall_average_waiting_time
            last_900_average_waiting_time_overall = list_overall_average_waiting_time[-warm_up_length:]
            episode_overall_average_waiting_time = sum(last_900_average_waiting_time_overall) / len(last_900_average_waiting_time_overall)
            list_episode_overall_average_waiting_time.append(episode_overall_average_waiting_time)
            
            #记录sum waiting time
            list_sum_waiting_time_N_S = env.list_sum_waiting_time_per_direction["N_S"]
            last_900_sum_waiting_time_N_S = list_sum_waiting_time_N_S[-warm_up_length:]
            episode_sum_waiting_time_N_S = sum(last_900_sum_waiting_time_N_S) / len(last_900_sum_waiting_time_N_S)
            list_episode_sum_waiting_time_per_direction["N_S"].append(episode_sum_waiting_time_N_S)
            
            list_sum_waiting_time_W_E = env.list_sum_waiting_time_per_direction["W_E"]
            last_900_sum_waiting_time_W_E = list_sum_waiting_time_W_E[-warm_up_length:]
            episode_sum_waiting_time_W_E = sum(last_900_sum_waiting_time_W_E) / len(last_900_sum_waiting_time_W_E)
            list_episode_sum_waiting_time_per_direction["W_E"].append(episode_sum_waiting_time_W_E)
            
            list_overall_sum_waiting_time = env.list_sum_overall_waiting_time
            last_900_sum_waiting_time_overall = list_overall_sum_waiting_time[-warm_up_length:]
            episode_sum_waiting_time = sum(last_900_sum_waiting_time_overall) / len(last_900_sum_waiting_time_overall)
            list_episode_sum_waiting_time.append(episode_sum_waiting_time)
            
            #记录stopped times
            list_stopped_times_N_S = env.list_stopped_times_per_direction["N_S"]
            last_900_stopped_times_N_S = list_stopped_times_N_S[-warm_up_length:]
            episode_stopped_times_N_S = sum(last_900_stopped_times_N_S) / len(last_900_stopped_times_N_S)
            list_episode_stopped_times_per_direction["N_S"].append(episode_stopped_times_N_S)
            
            list_stopped_times_W_E = env.list_stopped_times_per_direction["W_E"]
            last_900_stopped_times_W_E = list_stopped_times_W_E[-warm_up_length:]
            episode_stopped_times_W_E = sum(last_900_stopped_times_W_E) / len(last_900_stopped_times_W_E)
            list_episode_stopped_times_per_direction["W_E"].append(episode_stopped_times_W_E)
            
            list_total_stopped_time = env.list_total_stopped_times
            last_900_total_stopped_time = list_total_stopped_time[-warm_up_length:]
            episode_total_stopped_time = sum(last_900_total_stopped_time) / len(last_900_total_stopped_time)
            list_episode_total_stopped_times.append(episode_total_stopped_time)
            
        print("----------------------------------------------------------")
        print("Produced Vehicles (per direction):", list_episode_vehicles_produced)    
        print("Produced Vehicles (total):", list_episode_total_vehicles_produced)  
        print("Throughput (per direction):", list_episode_throughput_per_direction)    
        print("Throughput (total):", list_episode_total_throughput)   
        print("Throughput Percentage (NS, WE, total):", list_episode_throughput_percentage)
        print("Average Speed (per direction):",list_episode_average_speed_per_direction)    
        print("Average Speed (overall):",list_episode_overall_average_speed)    
        print("Average Waiting Time (per direction):",list_episode_average_waiting_time_per_direction)    
        print("Average Waiting Time (overall):",list_episode_overall_average_waiting_time)    
        print("Sum Waiting Time (per direction):",list_episode_sum_waiting_time_per_direction)    
        print("Sum Waiting Time (overall):",list_episode_sum_waiting_time)    
        print("Stopped Times (per direction):",list_episode_stopped_times_per_direction)    
        print("Stopped Times (overall):",list_episode_total_stopped_times)   
        
        mean_produced_vehicles_N_S = sum(list_episode_vehicles_produced["N_S"]) / len(list_episode_vehicles_produced["N_S"])
        mean_produced_vehicles_W_E = sum(list_episode_vehicles_produced["W_E"]) / len(list_episode_vehicles_produced["W_E"])
        mean_total_vehicles_produced = sum(list_episode_total_vehicles_produced) / len(list_episode_total_vehicles_produced)
        
        mean_throughput_N_S = sum(list_episode_throughput_per_direction["N_S"]) / len(list_episode_throughput_per_direction["N_S"])
        mean_throughput_W_E = sum(list_episode_throughput_per_direction["W_E"]) / len(list_episode_throughput_per_direction["W_E"])
        mean_total_throughput = sum(list_episode_total_throughput) / len(list_episode_total_throughput)
        
        mean_throughput_percentage_N_S = sum(list_episode_throughput_percentage["N_S"]) / len(list_episode_throughput_percentage["N_S"])
        mean_throughput_percentage_W_E = sum(list_episode_throughput_percentage["W_E"]) / len(list_episode_throughput_percentage["W_E"])
        mean_throughput_percentage_total = sum(list_episode_throughput_percentage["total"]) / len(list_episode_throughput_percentage["total"])

        mean_speed_N_S = sum(list_episode_average_speed_per_direction["N_S"]) / len(list_episode_average_speed_per_direction["N_S"])
        mean_speed_W_E = sum(list_episode_average_speed_per_direction["W_E"]) / len(list_episode_average_speed_per_direction["W_E"])
        mean_overall_speed = sum(list_episode_overall_average_speed) / len(list_episode_overall_average_speed)

        mean_waiting_time_N_S = sum(list_episode_average_waiting_time_per_direction["N_S"]) / len(list_episode_average_waiting_time_per_direction["N_S"])
        mean_waiting_time_W_E = sum(list_episode_average_waiting_time_per_direction["W_E"]) / len(list_episode_average_waiting_time_per_direction["W_E"])
        mean_overall_waiting_time = sum(list_episode_overall_average_waiting_time) / len(list_episode_overall_average_waiting_time)

        mean_sum_waiting_time_N_S = sum(list_episode_sum_waiting_time_per_direction["N_S"]) / len(list_episode_sum_waiting_time_per_direction["N_S"])
        mean_sum_waiting_time_W_E = sum(list_episode_sum_waiting_time_per_direction["W_E"]) / len(list_episode_sum_waiting_time_per_direction["W_E"])
        mean_overall_sum_waiting_time = sum(list_episode_sum_waiting_time) / len(list_episode_sum_waiting_time)

        mean_stopped_times_N_S = sum(list_episode_stopped_times_per_direction["N_S"]) / len(list_episode_stopped_times_per_direction["N_S"])
        mean_stopped_times_W_E = sum(list_episode_stopped_times_per_direction["W_E"]) / len(list_episode_stopped_times_per_direction["W_E"])
        mean_total_stopped_times = sum(list_episode_total_stopped_times) / len(list_episode_total_stopped_times)

        # 打印每个指标的均值
        print("-----------------------------------------------------")
        print("Mean Produced Vehicles (N_S):", mean_produced_vehicles_N_S)
        print("Mean Produced Vehicles (W_E):", mean_produced_vehicles_W_E)
        print("Mean Total Produced Vehicles:", mean_total_vehicles_produced)
        
        print("Mean Throughput (N_S):", mean_throughput_N_S)
        print("Mean Throughput (W_E):", mean_throughput_W_E)
        print("Mean Total Throughput:", mean_total_throughput)

        print("Mean Percentage of Produced Vehicles (N_S):", mean_throughput_percentage_N_S)
        print("Mean Percentage of Produced Vehicles (W_E):", mean_throughput_percentage_W_E)
        print("Mean Percentage of Produced Vehicles (total):", mean_throughput_percentage_total)
        
        print("Mean Average Speed (N_S):", mean_speed_N_S)
        print("Mean Average Speed (W_E):", mean_speed_W_E)
        print("Mean Overall Average Speed:", mean_overall_speed)

        print("Mean Average Waiting Time (N_S):", mean_waiting_time_N_S)
        print("Mean Average Waiting Time (W_E):", mean_waiting_time_W_E)
        print("Mean Overall Average Waiting Time:", mean_overall_waiting_time)

        print("Mean Sum Waiting Time (N_S):", mean_sum_waiting_time_N_S)
        print("Mean Sum Waiting Time (W_E):", mean_sum_waiting_time_W_E)
        print("Mean Overall Sum Waiting Time:", mean_overall_sum_waiting_time)

        print("Mean Stopped Times (N_S):", mean_stopped_times_N_S)
        print("Mean Stopped Times (W_E):", mean_stopped_times_W_E)
        print("Mean Total Stopped Times:", mean_total_stopped_times)
        
        # 计算各个指标的标准差和标准误差
        std_throughput_N_S = statistics.stdev(list_episode_throughput_per_direction["N_S"])
        std_throughput_W_E = statistics.stdev(list_episode_throughput_per_direction["W_E"])
        std_total_throughput = statistics.stdev(list_episode_total_throughput)

        sem_throughput_N_S = std_throughput_N_S / math.sqrt(len(list_episode_throughput_per_direction["N_S"]))
        sem_throughput_W_E = std_throughput_W_E / math.sqrt(len(list_episode_throughput_per_direction["W_E"]))
        sem_total_throughput = std_total_throughput / math.sqrt(len(list_episode_total_throughput))
        
        
        std_percentage_throughput_N_S = statistics.stdev(list_episode_throughput_percentage["N_S"])
        std_percentage_throughput_W_E = statistics.stdev(list_episode_throughput_percentage["W_E"])
        std_percentage_throughput_total = statistics.stdev(list_episode_throughput_percentage["total"])
        
        sem_percentage_throughput_N_S = std_percentage_throughput_N_S / math.sqrt(len(list_episode_throughput_percentage["N_S"]))
        sem_percentage_throughput_W_E = std_percentage_throughput_W_E / math.sqrt(len(list_episode_throughput_percentage["W_E"]))
        sem_percentage_throughput_total = std_percentage_throughput_total / math.sqrt(len(list_episode_throughput_percentage["total"]))

        std_speed_N_S = statistics.stdev(list_episode_average_speed_per_direction["N_S"])
        std_speed_W_E = statistics.stdev(list_episode_average_speed_per_direction["W_E"])
        std_speed_overall = statistics.stdev(list_episode_overall_average_speed)

        sem_speed_N_S = std_speed_N_S / math.sqrt(len(list_episode_average_speed_per_direction["N_S"]))
        sem_speed_W_E = std_speed_W_E / math.sqrt(len(list_episode_average_speed_per_direction["W_E"]))
        sem_speed_overall = std_speed_overall / math.sqrt(len(list_episode_overall_average_speed))

        std_waiting_time_N_S = statistics.stdev(list_episode_average_waiting_time_per_direction["N_S"])
        std_waiting_time_W_E = statistics.stdev(list_episode_average_waiting_time_per_direction["W_E"])
        std_waiting_time_overall = statistics.stdev(list_episode_overall_average_waiting_time)

        sem_waiting_time_N_S = std_waiting_time_N_S / math.sqrt(len(list_episode_average_waiting_time_per_direction["N_S"]))
        sem_waiting_time_W_E = std_waiting_time_W_E / math.sqrt(len(list_episode_average_waiting_time_per_direction["W_E"]))
        sem_waiting_time_overall = std_waiting_time_overall / math.sqrt(len(list_episode_overall_average_waiting_time))

        std_sum_waiting_time_N_S = statistics.stdev(list_episode_sum_waiting_time_per_direction["N_S"])
        std_sum_waiting_time_W_E = statistics.stdev(list_episode_sum_waiting_time_per_direction["W_E"])
        std_sum_waiting_time_overall = statistics.stdev(list_episode_sum_waiting_time)

        sem_sum_waiting_time_N_S = std_sum_waiting_time_N_S / math.sqrt(len(list_episode_sum_waiting_time_per_direction["N_S"]))
        sem_sum_waiting_time_W_E = std_sum_waiting_time_W_E / math.sqrt(len(list_episode_sum_waiting_time_per_direction["W_E"]))
        sem_sum_waiting_time_overall = std_sum_waiting_time_overall / math.sqrt(len(list_episode_sum_waiting_time))

        std_stopped_times_N_S = statistics.stdev(list_episode_stopped_times_per_direction["N_S"])
        std_stopped_times_W_E = statistics.stdev(list_episode_stopped_times_per_direction["W_E"])
        std_total_stopped_times = statistics.stdev(list_episode_total_stopped_times)

        sem_stopped_times_N_S = std_stopped_times_N_S / math.sqrt(len(list_episode_stopped_times_per_direction["N_S"]))
        sem_stopped_times_W_E = std_stopped_times_W_E / math.sqrt(len(list_episode_stopped_times_per_direction["W_E"]))
        sem_total_stopped_times = std_total_stopped_times / math.sqrt(len(list_episode_total_stopped_times))

        # 打印各个指标的标准差和标准误差
        print("-----------------------------------------------")
        print("Throughput STD (N_S):", std_throughput_N_S)
        print("Throughput SEM (N_S):", sem_throughput_N_S)
        print("Throughput STD (W_E):", std_throughput_W_E)
        print("Throughput SEM (W_E):", sem_throughput_W_E)
        print("Total Throughput STD:", std_total_throughput)
        print("Total Throughput SEM:", sem_total_throughput)
        
        print("Percentage Throughput STD (N_S):", std_percentage_throughput_N_S)
        print("Percentage Throughput SEM (N_S):", sem_percentage_throughput_N_S)
        print("Percentage Throughput STD (W_E):", std_percentage_throughput_W_E)
        print("Percentage Throughput SEM (W_E):", sem_percentage_throughput_W_E)
        print("Percentage Total Throughput STD:", std_percentage_throughput_total)
        print("Percentage Total Throughput SEM:", sem_percentage_throughput_total)

        print("Speed STD (N_S):", std_speed_N_S)
        print("Speed SEM (N_S):", sem_speed_N_S)
        print("Speed STD (W_E):", std_speed_W_E)
        print("Speed SEM (W_E):", sem_speed_W_E)
        print("Overall Speed STD:", std_speed_overall)
        print("Overall Speed SEM:", sem_speed_overall)

        print("Waiting Time STD (N_S):", std_waiting_time_N_S)
        print("Waiting Time SEM (N_S):", sem_waiting_time_N_S)
        print("Waiting Time STD (W_E):", std_waiting_time_W_E)
        print("Waiting Time SEM (W_E):", sem_waiting_time_W_E)
        print("Overall Waiting Time STD:", std_waiting_time_overall)
        print("Overall Waiting Time SEM:", sem_waiting_time_overall)

        print("Sum Waiting Time STD (N_S):", std_sum_waiting_time_N_S)
        print("Sum Waiting Time SEM (N_S):", sem_sum_waiting_time_N_S)
        print("Sum Waiting Time STD (W_E):", std_sum_waiting_time_W_E)
        print("Sum Waiting Time SEM (W_E):", sem_sum_waiting_time_W_E)
        print("Overall Sum Waiting Time STD:", std_sum_waiting_time_overall)
        print("Overall Sum Waiting Time SEM:", sem_sum_waiting_time_overall)

        print("Stopped Times STD (N_S):", std_stopped_times_N_S)
        print("Stopped Times SEM (N_S):", sem_stopped_times_N_S)
        print("Stopped Times STD (W_E):", std_stopped_times_W_E)
        print("Stopped Times SEM (W_E):", sem_stopped_times_W_E)
        print("Total Stopped Times STD:", std_total_stopped_times)
        print("Total Stopped Times SEM:", sem_total_stopped_times)
        # env.txw_save_csv(out_csv, run)
        env.close()
        

        # 打开CSV文件进行写入
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # 循环写入每个episode的数据
            for episode, (produced_v_N_S, produced_v_W_E, thru_N_S, thru_W_E, total_thru, percen_thru_N_S, percen_thru_W_E, percen_thru_total, speed_N_S, speed_W_E, overall_speed,
                        wait_N_S, wait_W_E, overall_wait, sum_wait_N_S, sum_wait_W_E, overall_sum_wait,
                        stopped_N_S, stopped_W_E, total_stopped) in enumerate(zip(
                            list_episode_vehicles_produced["N_S"],
                            list_episode_vehicles_produced["W_E"],
                            list_episode_throughput_per_direction["N_S"],
                            list_episode_throughput_per_direction["W_E"],
                            list_episode_total_throughput,
                            list_episode_throughput_percentage["N_S"],
                            list_episode_throughput_percentage["W_E"],
                            list_episode_throughput_percentage["total"],
                            list_episode_average_speed_per_direction["N_S"],
                            list_episode_average_speed_per_direction["W_E"],
                            list_episode_overall_average_speed,
                            list_episode_average_waiting_time_per_direction["N_S"],
                            list_episode_average_waiting_time_per_direction["W_E"],
                            list_episode_overall_average_waiting_time,
                            list_episode_sum_waiting_time_per_direction["N_S"],
                            list_episode_sum_waiting_time_per_direction["W_E"],
                            list_episode_sum_waiting_time,
                            list_episode_stopped_times_per_direction["N_S"],
                            list_episode_stopped_times_per_direction["W_E"],
                            list_episode_total_stopped_times)):
                
                writer.writerow({
                    "Episode": episode,
                    "Produced_vehicles_N_S": produced_v_N_S,
                    "Produced_vehicles_W_E": produced_v_W_E,
                    "Throughput_N_S": thru_N_S,
                    "Throughput_W_E": thru_W_E,
                    "Total_Throughput": total_thru,
                    "Percentage_Throughput_N_S": percen_thru_N_S,
                    "Percentage_Throughput_W_E": percen_thru_W_E,
                    "Percentage_Total_Throughput": percen_thru_total,
                    "Average_Speed_N_S": speed_N_S,
                    "Average_Speed_W_E": speed_W_E,
                    "Overall_Average_Speed": overall_speed,
                    "Average_Waiting_Time_N_S": wait_N_S,
                    "Average_Waiting_Time_W_E": wait_W_E,
                    "Overall_Average_Waiting_Time": overall_wait,
                    "Sum_Waiting_Time_N_S": sum_wait_N_S,
                    "Sum_Waiting_Time_W_E": sum_wait_W_E,
                    "Overall_Sum_Waiting_Time": overall_sum_wait,
                    "Stopped_Times_N_S": stopped_N_S,
                    "Stopped_Times_W_E": stopped_W_E,
                    "Total_Stopped_Times": total_stopped
                })

        print("Data has been written to", csv_file_path)
                
        # 创建箱型图
        plt.figure(figsize=(12, 8))

        # 绘制吞吐量（Throughput）的箱型图
        plt.subplot(2, 3, 1)
        plt.boxplot([list_episode_throughput_per_direction["N_S"], list_episode_throughput_per_direction["W_E"], list_episode_total_throughput])
        plt.xticks([1, 2, 3], ['N_S', 'W_E', 'Total'])
        plt.title('Throughput')

        # 绘制速度（Speed）的箱型图
        plt.subplot(2, 3, 2)
        plt.boxplot([list_episode_average_speed_per_direction["N_S"], list_episode_average_speed_per_direction["W_E"], list_episode_overall_average_speed])
        plt.xticks([1, 2, 3], ['N_S', 'W_E', 'Overall'])
        plt.title('Speed')

        # 绘制等待时间（Waiting Time）的箱型图
        plt.subplot(2, 3, 3)
        plt.boxplot([list_episode_average_waiting_time_per_direction["N_S"], list_episode_average_waiting_time_per_direction["W_E"], list_episode_overall_average_waiting_time])
        plt.xticks([1, 2, 3], ['N_S', 'W_E', 'Overall'])
        plt.title('Waiting Time')

        # 绘制总等待时间（Sum Waiting Time）的箱型图
        plt.subplot(2, 3, 4)
        plt.boxplot([list_episode_sum_waiting_time_per_direction["N_S"], list_episode_sum_waiting_time_per_direction["W_E"], list_episode_sum_waiting_time])
        plt.xticks([1, 2, 3], ['N_S', 'W_E', 'Overall'])
        plt.title('Sum Waiting Time')

        # 绘制停车次数（Stopped Times）的箱型图
        plt.subplot(2, 3, 5)
        plt.boxplot([list_episode_stopped_times_per_direction["N_S"], list_episode_stopped_times_per_direction["W_E"], list_episode_total_stopped_times])
        plt.xticks([1, 2, 3], ['N_S', 'W_E', 'Total'])
        plt.title('Stopped Times')
        
        plt.subplot(2, 3, 6)
        plt.boxplot([list_episode_throughput_percentage["N_S"], list_episode_throughput_percentage["W_E"], list_episode_throughput_percentage["total"]])
        plt.xticks([1, 2, 3], ['N_S', 'W_E', 'Total'])
        plt.title('Throughput Percentage')

        plt.tight_layout()
        plt.show()
                