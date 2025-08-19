"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces
from collections import OrderedDict

class TrafficSignal:
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`sumo_rl.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff-waiting-time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`sumo_rl.environment.env.SumoEnvironment`.
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html). Should this be parameterized?
    MIN_GAP = 2.5



    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        use_max_green: bool,
        begin_time: int,
        reward_fn: Union[str, Callable],
        sumo,
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            ts_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            begin_time (int): The time in seconds when the traffic signal starts operating.
            reward_fn (Union[str, Callable]): The reward function. Can be a string with the name of the reward function or a callable function.
            sumo (Sumo): The Sumo instance.
        """
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.use_max_green = use_max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0 #上一次的reward值
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo
        

        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        #lanes是指incoming lanes
        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        c = self.sumo.trafficlight.getControlledLinks(self.id)
        d = c[0]
        f = d[0]
        e=d[0][1]
        a = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        b = set(self.out_lanes)
        
        #这里set会把顺序打乱
        # self.out_lanes = list(set(self.out_lanes))
        #如果想不打乱顺序，可以这样，但是目前还没有必要保留顺序：
        self.out_lanes = list(OrderedDict.fromkeys(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)
        
        print("This is the local ts.py")

    #这个地方-重新建立了相位（把原来的绿灯赋予新的最大时间+重新构建了黄灯相位。）把绿灯相位放在前面，然后加上他们之间互相transit时候的黄灯相位，这里黄灯相位是N*（N-1），N是绿灯相位数量
    # syc的场景构建的有问题
    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        # print(phases)
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # 认为没有全红相位.Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            duration = phase.duration
            if self.use_max_green:
                if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                    self.green_phases.append(self.sumo.trafficlight.Phase(self.max_green, state))
            else:
                if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                    self.green_phases.append(self.sumo.trafficlight.Phase(duration, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        #把绿色相位都放在了前面，然后是黄色相位
        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id) #原始net文件中定义的相位,还没有被替换
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases #替换原始net文件中定义的相位,下一步传给sumo,此时minDur和maxDur都是-1,代表不适用或者没有设定-->ToDo 可变duration????
        # print(logic.phases)
        #将构建的相位传给sumo并设定第一个相位是初始相位。
        self.sumo.trafficlight.setProgramLogic(self.id, logic) #传给sumo
        # c=self.sumo.trafficlight.getAllProgramLogics(self.id)
        # print("c:", c)
        #这一行setRedYellowGreenState执行完后，这里不知道为什么会有两个logics，第一个是设定的，第二个是online，貌似是sumo自己生成的.设定第一个相位是初始相位。
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)
        # b=self.sumo.trafficlight.getAllProgramLogics(self.id)
        # print("b", b)
        tl_logic = self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.id)
        # a=1

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase) #确保 new_phase 参数是一个整数
        # 当前的绿灯阶段是否已经是 new_phase，以及自上次阶段变化以来的时间是否少于规定的黄灯时间加上最小绿灯时间。如果任何一个条件为真，当前绿灯阶段将保持不变。
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
            a=1
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0
            a=1

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        return self.last_reward

    def _pressure_reward(self):
        # a=self.get_pressure()
        return self.get_pressure()

    def _priority_pressure_reward(self):
        # a=self.get_pressure()
        return self.get_priority_pressure()

    def _52_priority_pressure_reward(self):
        # a=self.get_pressure()
        return self.get_priority_pressure_52()
    # 我做了修改了  ################## 20250818
    def _51_priority_pressure_reward(self):
        # a=self.get_pressure()
        return self.get_priority_pressure_51()
    def _21_priority_pressure_reward(self):
        # a=self.get_pressure()
        return self.get_priority_pressure_21()

    def _41_priority_pressure_reward(self):
        # a=self.get_pressure()
        return self.get_priority_pressure_41()

    def _45_priority_pressure_reward(self):
        # a=self.get_pressure()
        return self.get_priority_pressure_45()
    
    def _CTB_priority_pressure_reward(self):
        # a=self.get_pressure()
        return self.get_CTB_priority_pressure()
    
    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            List[float]: List of accumulated waiting time of each intersection lane.
        """
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        # a=sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes)
        # b=sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)
        # c = a-b
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_priority_pressure(self, alpha: float = 1.0, beta: float = 1.25) -> float:
        """
        计算按私家车/货车加权后的压力：
          pressure = α*(#out_car - #in_car) + β*(#out_truck - #in_truck)
        私家车 weight=α，货车 weight=β。
        """
        in_car = in_truck = out_car = out_truck = 0

        # 累计进口车道上的私/货车数
        for lane in self.lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "truck":
                    in_truck += 1
                else:
                    in_car += 1

        # 累计出口车道上的私/货车数
        for lane in self.out_lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "truck":
                    out_truck += 1
                else:
                    out_car += 1

        # 计算加权压力
        delta_car   = out_car   - in_car
        delta_truck = out_truck - in_truck
        priority_pressure = alpha * delta_car + beta * delta_truck

        return priority_pressure
    
    def get_priority_pressure_52(self, alpha: float = 1.0, beta: float = 2.50) -> float:
        """
        计算按私家车/货车加权后的压力：
          pressure = α*(#out_car - #in_car) + β*(#out_truck - #in_truck)
        私家车 weight=α，货车 weight=β。
        """
        in_car = in_truck = out_car = out_truck = 0

        # 累计进口车道上的私/货车数
        for lane in self.lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "truck":
                    in_truck += 1
                else:
                    in_car += 1

        # 累计出口车道上的私/货车数
        for lane in self.out_lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "truck":
                    out_truck += 1
                else:
                    out_car += 1

        # 计算加权压力
        delta_car   = out_car   - in_car
        delta_truck = out_truck - in_truck
        priority_pressure = alpha * delta_car + beta * delta_truck

        return priority_pressure
    
    #这里我改了############################################## 20250818
    def get_priority_pressure_51(self, alpha: float = 1.0, beta: float = 5.0) -> float:
        """
        计算按私家车/公交车加权后的压力：
        pressure = α*(#out_car - #in_car) + β*(#out_bus - #in_bus)
        私家车 weight=α，公交车 weight=β。
        说明：除 type=="car" 外的车辆一律按 bus 计数（即 else 归为 bus）。
        """
        in_car = in_bus = out_car = out_bus = 0

        # 累计进口车道上的 car / bus 数
        for lane in self.lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "car":
                    in_car += 1
                else:
                    in_bus += 1

        # 累计出口车道上的 car / bus 数
        for lane in self.out_lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "car":
                    out_car += 1
                else:
                    out_bus += 1

        # 计算加权压力
        delta_car = out_car - in_car
        delta_bus = out_bus - in_bus
        priority_pressure = alpha * delta_car + beta * delta_bus

        return priority_pressure

    def get_priority_pressure_21(self, alpha: float = 1.0, beta: float = 2.0) -> float:
        """
        计算按私家车/公交车加权后的压力：
          pressure = α*(#out_car - #in_car) + β*(#out_bus - #in_bus)
        私家车 weight=α，公交车 weight=β。
        """
        in_car = in_bus = out_car = out_bus = 0

        # 累计进口车道上的私/公交车数
        for lane in self.lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "car":
                    in_car += 1
                else:
                    in_bus += 1

        # 累计出口车道上的私/公交车数
        for lane in self.out_lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "car":
                    out_car += 1
                else:
                    out_bus += 1

        # 计算加权压力
        delta_car   = out_car   - in_car
        delta_bus   = out_bus   - in_bus
        priority_pressure = alpha * delta_car + beta * delta_bus

        return priority_pressure


    def get_priority_pressure_41(self, alpha: float = 1.0, beta: float = 4.0) -> float:
            """
            计算按私家车/公交车加权后的压力：
            pressure = α*(#out_car - #in_car) + β*(#out_bus - #in_bus)
            私家车 weight=α，公交车 weight=β。
            """
            in_car = in_bus = out_car = out_bus = 0

            # 累计进口车道上的私/公交车数
            for lane in self.lanes:
                vids = self.sumo.lane.getLastStepVehicleIDs(lane)
                for vid in vids:
                    if self.sumo.vehicle.getTypeID(vid) == "car":
                        in_car += 1
                    else:
                        in_bus += 1

            # 累计出口车道上的私/公交车数
            for lane in self.out_lanes:
                vids = self.sumo.lane.getLastStepVehicleIDs(lane)
                for vid in vids:
                    if self.sumo.vehicle.getTypeID(vid) == "car":
                        out_car += 1
                    else:
                        out_bus += 1

            # 计算加权压力
            delta_car   = out_car   - in_car
            delta_bus   = out_bus   - in_bus
            priority_pressure = alpha * delta_car + beta * delta_bus

            return priority_pressure


    def get_priority_pressure_45(self, alpha: float = 1.25, beta: float = 1.0) -> float:
        """
        计算按私家车/货车加权后的压力：
          pressure = α*(#out_car - #in_car) + β*(#out_truck - #in_truck)
        私家车 weight=α，货车 weight=β。
        """
        in_car = in_truck = out_car = out_truck = 0

        # 累计进口车道上的私/货车数
        for lane in self.lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "truck":
                    in_truck += 1
                else:
                    in_car += 1

        # 累计出口车道上的私/货车数
        for lane in self.out_lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            for vid in vids:
                if self.sumo.vehicle.getTypeID(vid) == "truck":
                    out_truck += 1
                else:
                    out_car += 1

        # 计算加权压力
        delta_car   = out_car   - in_car
        delta_truck = out_truck - in_truck
        priority_pressure = alpha * delta_car + beta * delta_truck

        return priority_pressure

    # 通用算priority pressure，还需要再打磨一下
    def common_get_priority_pressure(self) -> float:
        in_weight = 0.0
        out_weight = 0.0
        # wdict = self.priority_weights  # e.g. {'car':1.25, 'truck':1.0, ...}
        wdict ={'car':1.25, 'truck':1.0}
        # helper：处理某个 lane，sign = +1（出）或 -1（进）
        def process(lane_id, sign):
            nonlocal in_weight, out_weight
            # 直接按类型拿数量
            by_type = self.sumo.lane.getLastStepVehicleNumberByType(lane_id)
            # 返回形如 {'car': 5, 'truck':3, ...}
            total = sum(wdict.get(t,1.0) * cnt for t, cnt in by_type.items())
            if sign > 0:
                return total
            else:
                return -total

        # 累计进口车道
        for lane in self.lanes:
            in_weight += process(lane, -1)
        # 累计出口车道
        for lane in self.out_lanes:
            out_weight += process(lane, +1)

        return out_weight + in_weight
    
    # def cc_get_priority_pressure(self) -> float:
    #     """
    #     计算按 BNF 配置的优先级权重后的压力：
    #     pressure = Σ_over_all_lanes[ Σ_over_vids(weight(type(vid))) ]_out
    #             - Σ_over_all_lanes[ Σ_over_vids(weight(type(vid))) ]_in
    #     """
    #     in_weight  = 0.0
    #     out_weight = 0.0

    #     # 累计进口车道上的加权数
    #     for lane in self.lanes:
    #         for vid in self.sumo.lane.getLastStepVehicleIDs(lane):
    #             vtype  = self.sumo.vehicle.getTypeID(vid)
    #             # 从字典里取默认权重，没找到就用 1.0
    #             w = self.priority_weights.get(vtype, 1.0)
    #             in_weight += w

    #     # 累计出口车道上的加权数
    #     for lane in self.out_lanes:
    #         for vid in self.sumo.lane.getLastStepVehicleIDs(lane):
    #             vtype  = self.sumo.vehicle.getTypeID(vid)
    #             w = self.priority_weights.get(vtype, 1.0)
    #             out_weight += w

    #     # 计算优先级压力
    #     priority_pressure = out_weight - in_weight
    #     return priority_pressure


 
    def get_CTB_priority_pressure(self,
                          alpha: float = 1.0,
                          beta: float = 1.25,
                          gamma: float = 2.5) -> float:
        """
        高性能加权压力计算，支持 car/truck/bus，也兼容只有 car 和 truck 的场景。
        单次循环遍历所有进/出车道，累加压力值。
        """
        sumo = self.sumo
        get_ids = sumo.lane.getLastStepVehicleIDs
        get_type = sumo.vehicle.getTypeID

        pressure = 0.0

        # 遍历两组车道：出方向权重为 +1，进方向权重为 -1
        for sign, lanes in ((1, self.out_lanes), (-1, self.lanes)):
            for lane in lanes:
                for vid in get_ids(lane):
                    vtype = get_type(vid)
                    if vtype == "truck":
                        pressure += beta * sign
                    elif vtype == "bus":
                        pressure += gamma * sign
                    else:
                        pressure += alpha * sign

        return pressure

    
    
    def get_out_lanes_density(self) -> List[float]:
        """Returns the density of the vehicles in the outgoing lanes of the intersection."""
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.out_lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_density(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        """Returns the total number of vehicles halting in the intersection."""
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
        "priority-pressure": _priority_pressure_reward,
        "52-priority-pressure":_52_priority_pressure_reward,
        "51-priority-pressure":_51_priority_pressure_reward,
        "21-priority-pressure":_21_priority_pressure_reward,
        "41-priority-pressure":_41_priority_pressure_reward,
        "45-priority-pressure":_45_priority_pressure_reward,
        "CTB_priority-pressure":_CTB_priority_pressure_reward
    }
