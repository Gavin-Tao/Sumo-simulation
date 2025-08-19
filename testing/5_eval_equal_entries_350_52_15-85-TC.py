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

#########################delete in linux#################################
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
##############################################################################

####################change to another one in linux ##############################   
    
sys.path.insert(0, 'E:\\txw\\SUMO-RL') #第一优先在这个路径下去寻找包
# sys.path.insert(0, '/home/jovyan/sumo-rl')  # 替换为实际路径

###################################################################################

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.dqn_agent_txw import DQN
from sumo_rl.environment.observations import PriorityObservationFunction

from torch.utils.tensorboard import SummaryWriter
import math
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

path_checkpoint = "./testing/s5.pth"
if not os.path.exists(path_checkpoint):
    raise FileNotFoundError(f"Checkpoint path not found: {path_checkpoint}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = 1
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
#warm up后计算episode的平均值
warm_up_length = -900
# 定义要写入CSV文件的字段名和数据
# CSV header
fieldnames = [
    "Episode",
    # per-type metrics (car, truck, bus)
    "car_count","car_sum_wait","car_sum_speed","car_avg_wait","car_avg_speed","car_stopped","car_produced","car_throughput",
    "truck_count","truck_sum_wait","truck_sum_speed","truck_avg_wait","truck_avg_speed","truck_stopped","truck_produced","truck_throughput",
    "bus_count","bus_sum_wait","bus_sum_speed","bus_avg_wait","bus_avg_speed","bus_stopped","bus_produced","bus_throughput",
    # overall metrics
    "overall_total_count","overall_avg_wait","overall_avg_speed","overall_total_stopped","overall_total_produced","overall_total_throughput"
]

# 构建带有时间戳的文件路径
csv_folder_path = "./testing/" + TIMESTAMP + "/"

# 确保文件夹存在，如果不存在则创建
if not os.path.exists(csv_folder_path):
    os.makedirs(csv_folder_path)

# 拼接文件路径
csv_file_path = os.path.join(csv_folder_path, "CT_evaluation.csv")

#设置计算steps的长度，因为一开始需要warm up一下
warm_up = 900
types_ = ["car","truck","bus"]

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        # default="./nets/syc/1x1/Equal_entries_350_bus/equal_entries_350_bus.rou.xml", # CTB
        default="./nets/syc/1x1/Equal_entries_350_CT/15-85-Truck-Car/equal_entries_350_54_15-85-TC.rou.xml", # CT
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
        # net_file="./nets/syc/1x1/Equal_entries_350_bus/syc_4phases.net.xml",  # CTB
        net_file="./nets/syc/1x1/Equal_entries_350_CT/15-85-Truck-Car/syc_4phases.net.xml",  # CT
        # net_file="nets/syc/1x1/Low_heavy/syc_4phases.net.xml", # low_heavy
        # net_file="nets/syc/1x1/Equal_entries_500/syc_4phases.net.xml", # equal 500
        # net_file="nets/syc/1x1/Equal_entries_140/syc_4phases.net.xml", # equal 140
        route_file=args.route,
        out_csv_name=out_csv,
        # cfg_file = "./nets/syc/1x1/Equal_entries_350_bus/syc_4phases_equal_entries_350_bus.sumocfg", # CTB
        cfg_file = "./nets/syc/1x1/Equal_entries_350_CT/15-85-Truck-Car/syc_4phases_equal_entries_350_54_15-85-TC.sumocfg", # CT
        # cfg_file = "nets/syc/1x1/Low_heavy/syc_4phases_low_heavy.sumocfg", # low_heavy
        # cfg_file = "nets/syc/1x1/Equal_entries_500/syc_4phases_equal_entries_500.sumocfg", # equal 500
        # cfg_file = "nets/syc/1x1/Equal_entries_140/syc_4phases_equal_entries_140.sumocfg", # equal 140
        use_gui=True,
        num_seconds=args.seconds,
        min_green=args.min_green, 
        max_green=args.max_green,
        use_max_green = True,
        sumo_seed=seed, #固定住seed
        #single_agent= True, #设置成True貌似TL会报错。
        observation_class = PriorityObservationFunction,
        reward_fn = "52-priority-pressure",
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
        env.evaluation = True
        print("----------------loading model-----------------")
        
        checkpoint = torch.load(path_checkpoint,
                        map_location=device)

        dqn_agent.q_net.load_state_dict(checkpoint['policy_state_dict'])  # 加载模型可学习参数
        dqn_agent.target_q_net.load_state_dict(checkpoint['target_state_dict'])
        dqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dic'])  # 加载优化器参数

        print("----------------Finished loading model-----------------")
        
        step_counter = 0
        # Storage for per-episode CTB metrics
        
        per_type_lists = {
            t: {k: [] for k in ("count","sum_wait","sum_speed","avg_wait","avg_speed","stopped","produced","throughput")}
            for t in types_
        }
        overall_list = {k: [] for k in ("total_count","avg_wait","avg_speed","total_stopped","total_produced","total_throughput")}

        
        
        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset(env.sumo_seed)

            # 在每个 episode 新建“步级”临时列表
            step_per_type = {
                t: {k: [] for k in per_type_lists[t].keys()} 
                for t in types_
            }
            step_overall = {k: [] for k in overall_list.keys()}
            step_counter = 0
            
            
            infos = []
            done = {"__all__": False}
            episode_return ={ts: 0 for ts in env.ts_ids}
            
            while not done["__all__"]:
                
                actions = {}
                for ts in env.ts_ids:
                    
                    action_ts = dqn_agent.take_action(initial_states[ts])
                    actions[ts] = action_ts
                
                s, r, done, info = env.step(action=actions)
                
                initial_states = s #s_t = s_t+1
           
                ## 2) 记录第 step_counter 步的 CTB_Metrics
                step_counter += 1

            # 环境跑完一个 episode（1000步）之后：
            hist = env.step_history_CTB
            # 1) per-type 指标：avg_wait、avg_speed、stopped，用尾部数据算平均
            tail_stats = {}
            for t in types_:
                tail = {}
                for metric in ("avg_wait","avg_speed","stopped"):
                    lst = hist["per_type"][t][metric]
                    data = lst[warm_up:] if len(lst) > warm_up else lst
                    tail[metric] = sum(data)/len(data) if data else 0.0
                # produced & throughput：仅取最后一步
                prod_list = hist["per_type"][t]["produced"]
                thr_list  = hist["per_type"][t]["throughput"]
                last_prod = prod_list[-1] if prod_list else 0
                last_thr  = thr_list[-1]  if thr_list  else 0
                tail["produced"]       = last_prod
                tail["throughput"]     = last_thr
                tail["throughput_pct"] = last_thr / last_prod if last_prod>0 else 0.0

                tail_stats[t] = tail

            # 2) overall 指标处理
            ov = {}
            for metric, key in [("avg_wait","avg_wait"),
                                ("avg_speed","avg_speed"),
                                ("total_stopped","total_stopped")]:
                lst  = hist["overall"][key]
                data = lst[warm_up:] if len(lst) > warm_up else lst
                ov[metric] = sum(data)/len(data) if data else 0.0

            # overall produced: 最后一步的值
            prod_o = hist["overall"]["total_produced"]
            last_prod_o = prod_o[-1] if prod_o else 0
            # overall throughput: 最后一步的值
            thr_o = hist["overall"]["total_throughput"]
            ov["throughput"] = thr_o[-1] if thr_o else 0
            ov["produced"]   = last_prod_o
            ov["throughput_pct"] = ov["throughput"] / last_prod_o if last_prod_o > 0 else 0.0

            # 3) 打印本 episode 结果
            print(f"\n=== Episode {episode} Results (warm_up={warm_up}) ===")
            for t in types_:
                s = tail_stats[t]
                print(f"--- {t} ---")
                print(f"  avg_wait       : {s['avg_wait']:.2f}")
                print(f"  avg_speed      : {s['avg_speed']:.2f}")
                print(f"  stopped        : {s['stopped']:.2f}")
                print(f"  produced       : {s['produced']}")
                print(f"  throughput     : {s['throughput']}")
                print(f"  throughput_pct : {s['throughput_pct']:.2%}")
            print("--- overall ---")
            print(f"  avg_wait        : {ov['avg_wait']:.2f}")
            print(f"  avg_speed       : {ov['avg_speed']:.2f}")
            print(f"  total_stopped   : {ov['total_stopped']:.2f}")
            print(f"  total_produced  : {ov['produced']}")
            print(f"  total_throughput: {ov['throughput']}")
            print(f"  throughput_pct  : {ov['throughput_pct']:.2%}")

        env.close()