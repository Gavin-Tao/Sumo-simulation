"""SUMO Environment for Traffic Signal Control."""
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import sumolib

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
import pandas as pd
LIBSUMO = 0
if LIBSUMO: 
    import libsumo as traci
else:
    import traci
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .observations import DefaultObservationFunction, ObservationFunction
from .traffic_signal import TrafficSignal

from datetime import datetime






def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEnvironment(gym.Env):
    """SUMO Environment for Traffic Signal Control.

    Class that implements a gym.Env interface for traffic signal control using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The duration in seconds of the simulation. Default: 20000
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict): String with the name of the reward function used by the agents, a reward function, or dictionary with reward functions assigned to individual traffic lights by their keys.
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file: str,
        route_file: str,
        cfg_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        single_agent: bool = False,
        use_max_green: bool = False,
        reward_fn: Union[str, Callable, dict] = "diff-waiting-time",
        observation_class: ObservationFunction = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = 0,
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the environment."""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file #TODO 加了cfg
        self._route = route_file
        self._cfg = cfg_file
        self.use_gui = use_gui
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time." #如果条件为假，则会引发 AssertionError 异常，其中包含指定的错误消息。

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.use_max_green = use_max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None
        self.evaluation = False

        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net], label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label)

        self.ts_ids = list(conn.trafficlight.getIDList())
        self.observation_class = observation_class

        # 看reward是不是一系列定义的字典.
        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.use_max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    conn,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            # 初始化信号灯
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.use_max_green,
                    self.begin_time,
                    self.reward_fn,
                    conn,
                )
                for ts in self.ts_ids
            }

        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
        self.sim_step_counter = 0
        self.list_metrics = []
        self.throughput_per_direction = {"N_S": 0, "W_E": 0}
        self.throughput_N_S ={}
        self.throughput_W_E = {}
        self.total_throughput = 0
        self.list_average_speed_per_direction = {"N_S": [], "W_E": []}
        self.list_overall_average_speed = []
        self.list_average_waiting_time_per_direction = {"N_S": [], "W_E": []}
        self.list_overall_average_waiting_time = []
        self.list_sum_waiting_time_per_direction = {"N_S": [], "W_E": []}
        self.list_sum_overall_waiting_time = []
        self.list_stopped_times_per_direction = {"N_S": [], "W_E": []}
        self.list_total_stopped_times = []
        self.generated_vehicle_ids_N_S = set()  # 用于存储N_S方向生成的车辆ID
        self.generated_vehicle_ids_W_E = set()  # 用于存储W_E方向生成的车辆ID
        self.number_produced_vehicles = {"N_S": 0, "W_E": 0}
        ##########CTB Metrics################
        self.total_number_produced_vehicles = {"car": 0, "bus": 0, "truck": 0}
        self.total_generated_vehicle_ids = {"car": set(), "bus": set(), "truck": set()}
        self.total_throughput_per_type = {"car": set(), "truck": set(), "bus": set()}
        self.CTB_Metrics = {}
        self.step_history_CTB = {
            "per_type": {
                "car": {
                    "count": [], "sum_wait": [], "sum_speed": [],
                    "avg_wait": [], "avg_speed": [],
                    "stopped": [], "produced": [], "throughput": []
                },
                "truck": {
                    "count": [], "sum_wait": [], "sum_speed": [],
                    "avg_wait": [], "avg_speed": [],
                    "stopped": [], "produced": [], "throughput": []
                },
                "bus": {
                    "count": [], "sum_wait": [], "sum_speed": [],
                    "avg_wait": [], "avg_speed": [],
                    "stopped": [], "produced": [], "throughput": []
                },
            },
            "overall": {
                "total_count": [], "avg_wait": [], "avg_speed": [],
                "total_stopped": [], "total_produced": [], "total_throughput": []
            }
        }
        print("This is local env.py")
        

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary, #TODO 我加了cfg
            "-c",
            self._cfg,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)

        if self.episode != 0:
            self.close()
            # self.txw_save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.use_max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    self.sumo,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.use_max_green,
                    self.begin_time,
                    self.reward_fn,
                    self.sumo,
                )
                for ts in self.ts_ids
            }

        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            a=self._compute_observations()
            return self._compute_observations()

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return self.sumo.simulation.getTime()

    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()
        self.sim_step_counter += self.delta_time
        
        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        """Set the next green phase for the traffic signals.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step_counter}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info


    def _compute_observations(self):
        self.observations.update(
            {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        self.rewards.update(
            {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.ts_ids[0]].action_space

    def observation_spaces(self, ts_id: str):
        """Return the observation space of a traffic signal."""
        return self.traffic_signals[ts_id].observation_space

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        """Return the action space of a traffic signal."""
        return self.traffic_signals[ts_id].action_space

    def evaluation_metrics(self):
       
        N_S_lanes = ["n_t_0", "n_t_1", "n_t_2", 
            "t_s_0", "t_s_1", "t_s_2", 
            "s_t_0", "s_t_1", "s_t_2", 
            "t_n_0", "t_n_1", "t_n_2"]

        W_E_lanes = ["w_t_0", "w_t_1", "w_t_2",
                    "t_e_0", "t_e_1", "t_e_2",
                    "e_t_0", "e_t_1", "e_t_2",
                    "t_w_0", "t_w_1", "t_w_2"]

        N_S_incoming_lanes = ["n_t_0", "n_t_1", "n_t_2", 
            "s_t_0", "s_t_1", "s_t_2"]
        
        W_E_incoming_lanes = ["w_t_0", "w_t_1", "w_t_2",
                    "e_t_0", "e_t_1", "e_t_2"]
        
        sum_waiting_time_per_direction = {"N_S": 0.0, "W_E": 0.0}
        sum_speed_per_direction = {"N_S": 0.0, "W_E": 0.0}
        
        
        average_speed_per_direction = {"N_S": 0.0, "W_E": 0.0}
        average_waiting_time_per_direction = {"N_S": 0.0, "W_E": 0.0}
        stopped_times_per_direction = {"N_S": 0.0, "W_E": 0.0}
        veh_count = {"N_S": 0, "W_E": 0}
        
        
    
        lanes_vehicles_list = []
        # 这里只考虑了车道上的车，没有考虑交叉口中的
        #N-S方向
        for n_s_lane in N_S_lanes:
            n_s_veh_list = self.sumo.lane.getLastStepVehicleIDs(n_s_lane)
            veh_count["N_S"] += len(n_s_veh_list)
            # 计算waiting time, average speed
            for n_s_veh in n_s_veh_list:
                #把车加入到车道车辆列表中
                lanes_vehicles_list.append(n_s_veh)
                
                #计算waiting time,函数返回是瞬时的的waiting time
                n_s_wait_time = self.sumo.vehicle.getWaitingTime(n_s_veh)
                sum_waiting_time_per_direction["N_S"] += n_s_wait_time
                
                #计算speed
                n_s_speed = self.sumo.vehicle.getSpeed(n_s_veh)
                sum_speed_per_direction["N_S"] += n_s_speed
        
            # 计算stopped times
            n_s_lane_stopped_times = self.sumo.lane.getLastStepHaltingNumber(n_s_lane)
            stopped_times_per_direction["N_S"] += n_s_lane_stopped_times
                    
            
        # 计算W-E方向上的累积等待时间
        for w_e_lane in W_E_lanes:
            w_e_veh_list = self.sumo.lane.getLastStepVehicleIDs(w_e_lane)
            veh_count["W_E"] += len(w_e_veh_list)
            # 计算waiting time, average speed
            for w_e_veh in w_e_veh_list:
                
                lanes_vehicles_list.append(w_e_veh)
                
                w_e_wait_time = self.sumo.vehicle.getWaitingTime(w_e_veh)
                sum_waiting_time_per_direction["W_E"] += w_e_wait_time
                
                
                w_e_speed = self.sumo.vehicle.getSpeed(w_e_veh)
                sum_speed_per_direction["W_E"] += w_e_speed
                
            # 计算stopped times
            w_e_lane_stopped_times = self.sumo.lane.getLastStepHaltingNumber(w_e_lane)
            stopped_times_per_direction["W_E"] += w_e_lane_stopped_times
        
        # 考虑交叉口中的车,用所有车的ID减去车道上的车
        all_vehicles_list = self.sumo.vehicle.getIDList()
        # 获取交叉口中的车辆ID列表
        intersection_vehicles_list = list(set(all_vehicles_list) - set(lanes_vehicles_list))
        #更新车辆数量，更新speed list等 + 计算throughput（车辆在交叉口意味着throughput+=1，通过记录在交叉口的车辆ID来计算）
        #不需要更新waiting time和stopped times，因为交叉口不会等待和停车
        for intersection_vehicle_id in intersection_vehicles_list:
            route = self.sumo.vehicle.getRoute(intersection_vehicle_id)
            
            if route and len(route) > 0:
                direction = route[0]
                if direction in ['n_t', 's_t']:
                    n_s_intersection_speed = self.sumo.vehicle.getSpeed(intersection_vehicle_id)
                    sum_speed_per_direction["N_S"] += n_s_intersection_speed
                    veh_count["N_S"] += 1
                    self.throughput_N_S[intersection_vehicle_id] = 1
                    a=1
                    
                elif direction in ['w_t', 'e_t']:
                    w_e_intersection_speed = self.sumo.vehicle.getSpeed(intersection_vehicle_id)
                    sum_speed_per_direction["W_E"] += w_e_intersection_speed
                    veh_count["W_E"] += 1
                    self.throughput_W_E[intersection_vehicle_id]=1
                    b=1
        
        for lane in N_S_incoming_lanes:
            vehicle_ids = self.sumo.lane.getLastStepVehicleIDs(lane)
            self.generated_vehicle_ids_N_S.update(vehicle_ids)

        for lane in W_E_incoming_lanes:
            vehicle_ids = self.sumo.lane.getLastStepVehicleIDs(lane)
            self.generated_vehicle_ids_W_E.update(vehicle_ids)

        self.number_produced_vehicles["W_E"] = len(self.generated_vehicle_ids_W_E)
        self.number_produced_vehicles["N_S"] = len(self.generated_vehicle_ids_N_S)
        
        self.throughput_per_direction["N_S"] = len(self.throughput_N_S)
        self.throughput_per_direction["W_E"] = len(self.throughput_W_E)
        self.total_throughput = self.throughput_per_direction["N_S"] + self.throughput_per_direction["W_E"]
        
        average_waiting_time_per_direction = {
        direction: sum_waiting_time_per_direction[direction] / veh_count[direction] if veh_count[direction] > 0 else 0
        for direction in ["N_S", "W_E"]
        }
        
        average_speed_per_direction = {
            direction: sum_speed_per_direction[direction] / veh_count[direction] if veh_count[direction] > 0 else 0
            for direction in ["N_S", "W_E"]
        }
        
        # Calculate overall average speed
        total_count = veh_count["N_S"] + veh_count["W_E"]
        
        total_stopped_times = stopped_times_per_direction["N_S"] + stopped_times_per_direction["W_E"]
        overall_average_speed = (sum(sum_speed_per_direction.values()) / total_count) if total_count > 0 else 0
        
        sum_overall_waiting_time = sum(sum_waiting_time_per_direction.values())
       
        
        overall_average_waiting_time = (sum(sum_waiting_time_per_direction.values()) / total_count) if total_count > 0 else 0
       
        info = self._get_system_info()
        
        #把每一个step的平均速度，等待时间存在list中，warm up之后取这个list平均值作为这一个1000step的evaluation的值
        self.list_average_speed_per_direction["N_S"].append(average_speed_per_direction["N_S"])
        self.list_average_speed_per_direction["W_E"].append(average_speed_per_direction["W_E"])
        self.list_overall_average_speed.append(overall_average_speed)
        
        self.list_average_waiting_time_per_direction["N_S"].append(average_waiting_time_per_direction["N_S"])
        self.list_average_waiting_time_per_direction["W_E"].append(average_waiting_time_per_direction["W_E"])
        self.list_overall_average_waiting_time.append(overall_average_waiting_time)
        
        self.list_sum_waiting_time_per_direction["N_S"].append(sum_waiting_time_per_direction["N_S"])
        self.list_sum_waiting_time_per_direction["W_E"].append(sum_waiting_time_per_direction["W_E"])
        self.list_sum_overall_waiting_time.append(sum_overall_waiting_time)
        
        self.list_stopped_times_per_direction["N_S"].append(stopped_times_per_direction["N_S"])
        self.list_stopped_times_per_direction["W_E"].append(stopped_times_per_direction["W_E"])
        self.list_total_stopped_times.append(total_stopped_times)
        a=1
    
    def evaluation_metrics_CTB(self):
        # 定义所有车道（用于统计当前步所有车辆）
        lanes = [
        "n_t_0","n_t_1","n_t_2","t_s_0","t_s_1","t_s_2",
        "s_t_0","s_t_1","s_t_2","t_n_0","t_n_1","t_n_2",
        "w_t_0","w_t_1","w_t_2","t_e_0","t_e_1","t_e_2",
        "e_t_0","e_t_1","e_t_2","t_w_0","t_w_1","t_w_2"
        ]
        incoming = ["n_t_0","n_t_1","n_t_2","s_t_0","s_t_1","s_t_2",
                    "w_t_0","w_t_1","w_t_2","e_t_0","e_t_1","e_t_2"]

        # 初始化按车型累加器，记录每一步产生的所有车辆的加和信息
        types_ = ["car","truck","bus"]
        sum_wait    = {t:0.0 for t in types_}
        sum_speed   = {t:0.0 for t in types_}
        count_veh   = {t:0   for t in types_}
        sum_stopped = {t:0   for t in types_}
        
        # ---- 在这里初始化 lane_ids ----
        lane_ids = set() #用于记录在进出车道上的车辆id，用于后续算交叉口中的车辆id

         # 车道统计：等待、速度、停车
        for lane in lanes:
            vehs  = self.sumo.lane.getLastStepVehicleIDs(lane)
            # 把本车道车辆 ID 加到 lane_ids
            lane_ids.update(vehs)

            for vid in vehs:
                t = self.sumo.vehicle.getTypeID(vid)
                if t not in types_:
                    continue
                a= self.sumo.vehicle.getWaitingTime(vid)
                c=self.sumo.vehicle.getAccumulatedWaitingTime(vid)
                b= self.sumo.vehicle.getSpeed(vid)
                sum_wait[t]    += self.sumo.vehicle.getAccumulatedWaitingTime(vid)
                sum_speed[t]   += self.sumo.vehicle.getSpeed(vid)
                count_veh[t]   += 1
                # 停滞判断 直接检测这辆车是不是停着（速度接近 0），如果是就 +1
                if self.sumo.vehicle.getSpeed(vid) < 0.1:
                    sum_stopped[t] += 1
                # 因为车辆肯定从进车道产生，所有我只要遍历过了进车道，就可以记录一共产生过多少车
                # 如果在入口车道上，则认为“新生成”，加入去重集合
             
                self.total_generated_vehicle_ids[t].add(vid)



        # 交叉口车辆：统计通过量和速度，用所有车辆id-进出车道上的id
        all_ids  = set(self.sumo.vehicle.getIDList())       
        inters   = all_ids - lane_ids
        
        for vid in inters:
            t = self.sumo.vehicle.getTypeID(vid)
            if t not in types_:
                continue
            
            sum_speed[t] += self.sumo.vehicle.getSpeed(vid)
            count_veh[t] += 1
            self.total_throughput_per_type[t].add(vid) #把进入过路口的车辆都加入到set里，通过set的长度来判度throughput
            
            # 交叉口这里不会出现停车和等待，保险起见还是加了
            c=self.sumo.vehicle.getAccumulatedWaitingTime(vid)
            b= self.sumo.vehicle.getSpeed(vid)
            sum_wait[t] += self.sumo.vehicle.getAccumulatedWaitingTime(vid)
            # 停滞判断 直接检测这辆车是不是停着（速度接近 0），如果是就 +1
            if self.sumo.vehicle.getSpeed(vid) < 0.1:
                sum_stopped[t] += 1
            self.total_generated_vehicle_ids[t].add(vid)

        # 在统计完所有车道后，更新总计数字 ——  
        for t in types_:
            self.total_number_produced_vehicles[t] = len(self.total_generated_vehicle_ids[t])
            
        # 汇总整体指标
        # 6) 计算整体指标
        total_count      = sum(count_veh.values()) #当前step有多少车
        total_wait_sum   = sum(sum_wait.values()) #当前step所有车的waiting time求和
        total_speed_sum  = sum(sum_speed.values()) #当前step所有车的speed求和
        throughput = {t: len(self.total_throughput_per_type[t]) for t in types_}
        total_throughput = sum(throughput.values()) #从开始到现在未知的throughput
        total_stopped    = sum(sum_stopped.values()) #当前step有几个停的车
        total_produced   = sum(self.total_number_produced_vehicles.values()) #从开始到现在产生了多少个车

        overall = {
            "total_count"     : total_count,
            "avg_wait"        : (total_wait_sum  / total_count)     if total_count else 0.0,
            "avg_speed"       : (total_speed_sum / total_count)     if total_count else 0.0,
            "total_stopped"   : total_stopped,
            "total_produced"  : total_produced,
            "total_throughput": total_throughput,
        }

        # 7) 构建 per_type 结果
        per_type = {}
        for t in types_:
            c = count_veh[t]
            per_type[t] = {
                "count"          : c,
                "sum_wait"       : sum_wait[t],
                "sum_speed"      : sum_speed[t],
                "avg_wait"       : (sum_wait[t]  / c) if c else 0.0,
                "avg_speed"      : (sum_speed[t] / c) if c else 0.0,
                "stopped"        : sum_stopped[t],
                "produced" : self.total_number_produced_vehicles[t],
                "throughput"     : throughput[t],
            }

        # 8) 将本步数据 append 到 step_history_CTB
        for t, stats in per_type.items():
            buf = self.step_history_CTB["per_type"][t]
            buf["count"].append(     stats["count"])
            buf["sum_wait"].append(  stats["sum_wait"])
            buf["sum_speed"].append( stats["sum_speed"])
            buf["avg_wait"].append(  stats["avg_wait"])
            buf["avg_speed"].append( stats["avg_speed"])
            buf["stopped"].append(   stats["stopped"])
            buf["produced"].append(  stats["produced"])
            buf["throughput"].append(stats["throughput"])

        ob = self.step_history_CTB["overall"]
        ob["total_count"].append(     overall["total_count"])
        ob["avg_wait"].append(        overall["avg_wait"])
        ob["avg_speed"].append(       overall["avg_speed"])
        ob["total_stopped"].append(   overall["total_stopped"])
        ob["total_produced"].append(  overall["total_produced"])
        ob["total_throughput"].append(overall["total_throughput"])

        # 9) 最后写回 CTB_Metrics 保持兼容
        self.CTB_Metrics = {"per_type": per_type, "overall": overall}
        
    def _sumo_step(self):
        if self.evaluation:
            # self.evaluation_metrics()
            self.evaluation_metrics_CTB()
        self.sumo.simulationStep()

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids
        ]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        if self.sumo is None:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self.close()

    def render(self):
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame
        elif self.render_mode == "rgb_array":
            # img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_step}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file.

        Args:
            out_csv_name (str): Path to the output .csv file. E.g.: "results/my_results
            episode (int): Episode number to be appended to the output file name.
        """
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

    def txw_save_csv(self, out_csv_name, episode):
        if out_csv_name is not None:
            df = pd.DataFrame(self.list_metrics)
            directory = Path(out_csv_name).parent
            directory.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            final_file_name = f"{directory}/result_{current_time}_ep{episode}.csv"  # 简化文件名进行测试
            print("out file")
            try:
                df.to_csv(final_file_name, index=False)
            except Exception as e:
                print(f"Error saving file: {e}")
                print(f"Attempted file path: {final_file_name}")
    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        """Encode the state of the traffic signal into a hashable object."""
        phase = int(np.where(state[: self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        pressure_density = [self._discretize_pressure(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1 :]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + pressure_density)
    
    def txw_encode(self, state, ts_id):
        """Encode the state of the traffic signal into a hashable object."""
        phase = int(np.where(state[: self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        pressure_density = [self._discretize_pressure(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1 :]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + pressure_density)

    def _discretize_pressure(self, pressure):
        return pressure


class SumoEnvironmentPZ(AECEnv, EzPickle):
    """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo.

    For more information, see https://pettingzoo.farama.org/api/aec/.

    The arguments are the same as for :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        """Initialize the environment."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironment(**self._kwargs)

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        """Set the seed for the environment."""
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.compute_info()

    def compute_info(self):
        """Compute the info for the current step."""
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a) or k.startswith("system"):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        """Return the observation space for the agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for the agent."""
        return self.action_spaces[agent]

    def observe(self, agent):
        """Return the observation for the agent."""
        obs = self.env.observations[agent].copy()
        return obs

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.env.close()

    def render(self):
        """Render the environment."""
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        """Save metrics of the simulation to a .csv file."""
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        """Step the environment."""
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({})."
                "It is currently {}".format(agent, self.action_spaces[agent].n, action)
            )

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
