import argparse
import os
import sys
from datetime import datetime
import torch
import random
import numpy as np

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
# 设置 SummaryWriter 的路径
############################ -1----修改路径----------------------#############################
WRITER_PATH = "./logs/1x1/4_phases/equal_entries_350/CT/54/85-15-Truck-Car/" + TIMESTAMP
STORE_PATH  = "./models/1x1/4_phases/equal_entries_350/CT/54/85-15-Truck-Car/checkpoint/" + TIMESTAMP

# 如果目录不存在，则创建
for path in (WRITER_PATH, STORE_PATH):
    os.makedirs(path, exist_ok=True)

#实例化tensorboard的类SummaryWriter
tb_writer = SummaryWriter(log_dir = WRITER_PATH)

############### -2----修改GPU第几块----------------------#############################
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
torch.cuda.set_device(device)
print(device)
if device.type == 'cuda':
    print("Current CUDA device index:", torch.cuda.current_device())
    print("Current CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# 在 cuda:2 上分配一个 1024×1024 的随机张量
x = torch.randn(1024, 1024, device=device)

# 现在再打印就不是 0 了
print("Allocated:", torch.cuda.memory_allocated(device))
print("Reserved: ", torch.cuda.memory_reserved(device))

episodes = 2000
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
        default="./nets/syc/1x1/Equal_entries_350_CT/85-15-Truck-Car/equal_entries_350_54_85-15-TC.rou.xml", ############################ -3----修改rou文件的路径----------------------#############################
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
        net_file="./nets/syc/1x1/Equal_entries_350_CT/85-15-Truck-Car/syc_4phases.net.xml", ############################ -4----修改net文件的路径----------------------#############################
        route_file=args.route,
        cfg_file = "./nets/syc/1x1/Equal_entries_350_CT/85-15-Truck-Car/syc_4phases_equal_entries_350_54_85-15-TC.sumocfg", ############################ -5----修改cfg文件的路径----------------------#############################
        out_csv_name=out_csv,
        use_gui=False,
        num_seconds=args.seconds,
        min_green=args.min_green, 
        max_green=args.max_green,
        use_max_green = True,
        sumo_seed = seed, #固定住seed
        #single_agent= True, #设置成True貌似TL会报错。
        observation_class = PriorityObservationFunction, ############################ -6----修改observation----------------------#############################
        reward_fn = "priority-pressure", ############################ --7---修改reward函数----------------------#############################
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
        step_counter = 0 #这是训练的step，一个train step=delta time 5s
        # a= dqn_agents[env.ts_ids[0]]
        print("torch success")
        print(torch.cuda.memory_allocated(device))
        print(torch.cuda.memory_reserved(device))
        for episode in range(1, episodes + 1):
            print("This is episode:", episode)
            if episode != 1:
                initial_states = env.reset(env.sumo_seed)
                for ts in initial_states.keys():
                    dqn_agents[ts].state = tuple(initial_states[ts])

            infos = []
            done = {"__all__": False}
            episode_return ={ts: 0 for ts in dqn_agents.keys()}
            
            while not done["__all__"]:
                
                actions = {ts: dqn_agents[ts].take_action(dqn_agents[ts].state) for ts in dqn_agents.keys()}
                
                s, r, done, info = env.step(action=actions)
                
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
                        
                    if dqn_agents[ts_id].loss is not None:
                        b=dqn_agents[ts_id].loss
                        tb_writer.add_scalar("loss_" + ts_id, dqn_agents[ts_id].loss, step_counter)
                step_counter += 1

                
                # 缓存写入
                if step_counter % 100 == 0:  # 每100步 flush，一旦大训练也能实时更新
                    tb_writer.flush()


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
            for ts_id in env.ts_ids:
                tb_writer.add_scalar("episode_return" + ts_id, episode_return[ts_id], episode)

            # 每个 episode 结束后立即 flush
            tb_writer.flush()

            
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

                    
        env.txw_save_csv(out_csv, run)
        env.close()
        
    # 训练结束，关闭 SummaryWriter，确保所有数据写入磁盘
    tb_writer.close()