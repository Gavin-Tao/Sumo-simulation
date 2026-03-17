import argparse
import os
import sys
from datetime import datetime
import torch
import random
import numpy as np


print("DEBUG CWD:", os.getcwd())

# 用libsumo
os.environ["SUMO_RL_LIBSUMO"] = "1"
# 或用traci
# os.environ["SUMO_RL_LIBSUMO"] = "0"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from pathlib import Path  
# locate project root (folder that contains the 'sumo_rl' package)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# climb up until we find sumo_rl folder (robust across run locations)
probe = PROJECT_ROOT
while not (probe / "sumo_rl").exists() and probe.parent != probe:
    probe = probe.parent
PROJECT_ROOT = probe
sys.path.insert(0, str(PROJECT_ROOT))


from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.dqn_agent_txw import DQN
from sumo_rl.environment.observations import PressLightObservationFunction

import wandb
import math
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = 5000
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
        default="../nets/syc/1x1/Eq_350_BC_Wstr_Estr-Wleft_Eleft/Eq_350_100-0-100-0-Bus-Car/Eq_350_100-0-100-0-Bus-Car.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
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
    wandb.init(project="sumo-rl-1x1", name=f"5min_26_py_presslight_delta10_{experiment_time}")

    env = SumoEnvironment(
        net_file="../nets/syc/1x1/Eq_350_BC_Wstr_Estr-Wleft_Eleft/Eq_350_100-0-100-0-Bus-Car/syc_4phases.net.xml",
        route_file=args.route,
        cfg_file = "../nets/syc/1x1/Eq_350_BC_Wstr_Estr-Wleft_Eleft/Eq_350_100-0-100-0-Bus-Car/Eq_350_100-0-100-0-Bus-Car.sumocfg",
        out_csv_name=out_csv,
        use_gui=False,
        num_seconds=args.seconds,
        min_green=args.min_green, 
        max_green=args.max_green,
        use_max_green = True,
        sumo_seed=seed, #固定住seed
        #single_agent= True, #设置成True貌似TL会报错。
        observation_class = PressLightObservationFunction,
        reward_fn = "pressure",
        delta_time = 10,
        single_agent=False,
    )

    for run in range(1, args.runs + 1):
        initial_states = env.reset(env.sumo_seed)
        #这里可能有bug？因为如果env.ts_ids的顺序每次不一样的话，last_ts_id就不一样，导致initial_states[last_ts_id]每次给DQN agent初始化的时候不一样了。虽然我觉得这个应该不会有太大影响，因为DQN agent在take_action那里会重新给state的。
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


            infos = []
            done = {"__all__": False}
            
            while not done["__all__"]:
                
                actions = {}
                for ts in env.ts_ids:
                    
                    action_ts = dqn_agent.take_action(initial_states[ts])
                    actions[ts] = action_ts
                
                s, r, done, info = env.step(action=actions)
                
                metrics_keys = info.keys()
                log_dict = {metrics_key: info[metrics_key] for metrics_key in metrics_keys}
                for ts_id in env.ts_ids:
                    if r[ts_id] is not None:
                        log_dict["reward_" + ts_id] = r[ts_id]
                    if dqn_agent.loss is not None:
                        log_dict["loss_" + ts_id] = dqn_agent.loss
                wandb.log(log_dict, step=step_counter)
                step_counter += 1

                for ts in env.ts_ids:
                    ts_state = initial_states[ts]
                    ts_action = env.traffic_signals[ts].last_executed_action
                    ts_reward = r[ts]
                    ts_next_state = tuple(s[ts])
                    ts_done = done[ts]
                    
                    dqn_agent.replay_buffer.add(ts_state, ts_action, ts_reward, ts_next_state, ts_done)
                    
                initial_states = s #s_t = s_t+1
                    
                if dqn_agent.replay_buffer.size() > dqn_agent.mini_size:
                    b_s, b_a, b_r, b_ns, b_d = dqn_agent.replay_buffer.sample(dqn_agent.batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                        }
                    dqn_agent.epsilon = dqn_agent.eps_end + (dqn_agent.eps_start - dqn_agent.eps_end) * \
                    math.exp(-1. * dqn_agent.count / dqn_agent.eps_decay)
                    dqn_agent.update(transition_dict)
            
            if env.metrics is not None:
                for metric in env.metrics:
                    env.list_metrics.append(metric)
                b=1
            
            if dqn_agent.start_train:
                if episode != 0 and episode % checkpoint_interval == 0:
                    print("----------------saving model at ", episode, "-----------------")
                    checkpoint = {
                        "policy_state_dict": dqn_agent.q_net.state_dict(),
                        "target_state_dict": dqn_agent.target_q_net.state_dict(),
                        # "policy_copy_state_dict": agent.policy_net_copy.state_dict(),
                        "optimizer_state_dic": dqn_agent.optimizer.state_dict(),
                        # "z_optimizer_state_dict": agent.z_optimizer.state_dict(),
                        # "gamma_optimizer_state_dict": agent.gamma_optimizer.state_dict(),
                        "episode": episode,
                        # "agent_loss": agent.loss_list
                    }
                    script_name = os.path.splitext(os.path.basename(__file__))[0]
                    model_dir = f"./models/{script_name}/{TIMESTAMP}/checkpoint"
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    ckpt_timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
                    path_checkpoint = f"{model_dir}/ckpt_{ckpt_timestamp}_{episode}.pth"
                    torch.save(checkpoint, path_checkpoint)
                    print("----------------Finished saving model at ", episode, "-----------------")
        
                        
        env.txw_save_csv(out_csv, run)
        env.close()
    wandb.finish()
     
        