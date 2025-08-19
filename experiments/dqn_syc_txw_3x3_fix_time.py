import argparse
import os
import sys
from datetime import datetime
import torch

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
WRITER_PATH = "./logs/" + TIMESTAMP
#实例化tensorboard的类SummaryWriter
tb_writer = SummaryWriter(log_dir = WRITER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = 1

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        # default="nets/syc/3x3/800/intersection_new_800.net.xml",
        default="nets/syc/3x3/1000/intersection_new_1000.net.xml",
        # default="nets/syc/3x3/700/intersection_new_900.net.xml",
        # default="nets/syc/3x3/700/intersection_new_700.net.xml",
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
        net_file="nets/syc/3x3/1000/intersection_new_1000.net.xml",
        # net_file="nets/syc/3x3/900/intersection_new_900.net.xml",
        # net_file="nets/syc/3x3/700/intersection_new_700.net.xml",
        # net_file="nets/syc/3x3/800/intersection_new_800.net.xml",
        # net_file="nets/syc/3x3/intersection_new_actuated.net.xml",
        route_file="nets/syc/3x3/1000/intersection1000.rou.xml",
        # route_file="nets/syc/3x3/900/intersection900.rou.xml",
        #route_file="nets/syc/3x3/700/intersection700.rou.xml",
        # route_file="nets/syc/3x3/800/intersection800.rou.xml",
        cfg_file = "nets/syc/3x3/1000/syc_new_1000.sumocfg",
        #cfg_file = "nets/syc/3x3/900/syc_new_900.sumocfg",
        # cfg_file = "nets/syc/3x3/700/syc_new_700.sumocfg",
        # cfg_file = "nets/syc/3x3/800/syc_new_800.sumocfg",
        out_csv_name=out_csv,
        use_gui=False,
        num_seconds=args.seconds,
        min_green=args.min_green, 
        max_green=args.max_green,
        use_max_green = True,
        sumo_seed=0, #固定住seed
        #single_agent= True, #设置成True貌似TL会报错。
        observation_class = PressLightObservationFunction,
        reward_fn = "pressure",
        delta_time = 5,
        single_agent=False,
        fixed_ts = True,
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
            for ts in env.ts_ids
        }
        step_counter = 0
        
        #####################################################################
        junction_edges_dict = {}
        incoming_vehicle_ids = []
        throughput_count = {}
        for ts_ in env.ts_ids:
            # 获取路口控制的车道（lanes）
            lanes_ = env.sumo.trafficlight.getControlledLanes(ts_)

            # 从车道中提取边
            edges_ = set()
            for lane_ in lanes_:
                lane_edges_ = env.sumo.lane.getEdgeID(lane_)
                edges_.add(lane_edges_)
                # 将路口ID和边的集合存储到字典中
            junction_edges_dict[ts_] = edges_        
            throughput_count[ts_] = 0 
        
        vehicles_id_last_edge_index = {}
        
        ###############################################################################################
        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                
            # 这里要修改仿真的长度，因为会影响step计数。
            for step in range(0, env.sim_max_time * 10):
                ############################################
                for ts in env.ts_ids:
                    # 获取路口控制的车道（lanes）
                    lanes = env.sumo.trafficlight.getControlledLanes(ts)
                    incoming_vehicle_ids = []
                    for lane in lanes:
                        vehicles_on_lane = env.sumo.lane.getLastStepVehicleIDs(lane)
                       
                        for incoming_vehicle in vehicles_on_lane:
                            
                            current_edge_index = env.sumo.vehicle.getRouteIndex(incoming_vehicle)
                            if current_edge_index == 0:
                                vehicles_id_last_edge_index[incoming_vehicle] = 0
                            
                            if current_edge_index != vehicles_id_last_edge_index[incoming_vehicle]:
                                throughput_count[ts] += 1
                                vehicles_id_last_edge_index[incoming_vehicle] = current_edge_index
                                b=1
                        
                            
                            c=1
                ##############################################
                
                
                env._sumo_step()
                info = env._compute_info()
                env.sim_step_counter+=1
                
            if env.metrics is not None:
                for metric in env.metrics:
                    env.list_metrics.append(metric)
                b=1
        ######################################
        average_value = sum(throughput_count.values()) / len(throughput_count)
        print(sum(throughput_count.values()))
        print(throughput_count)
        print(average_value)      
        #####################################      
        env.txw_save_csv(out_csv, run)
        env.close()
