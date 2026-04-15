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
from .rewards import REWARD_REGISTRY

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
        self.last_executed_action = 0

        

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

        # Precompute which incoming lanes are signal-controlled (ever get red).
        # Lanes that are G/g in ALL green phases are always-green (e.g. free right turns).
        self.signal_controlled_lanes: list = self._compute_signal_controlled_lanes()
        self.always_green_lanes: list = [
            l for l in self.lanes if l not in self.signal_controlled_lanes
        ]

        print("✅TThis is the local ts.py")

    def _compute_signal_controlled_lanes(self) -> list:
        """Return lanes that receive red in at least one green phase.

        A lane whose position is G or g in every green phase is permanently
        green (free right turn) and is excluded from the returned list.
        Lanes controlled by fixed_ts mode are all considered signal-controlled.

        Uses the raw (possibly duplicated) controlled-lanes list for phase-string
        indexing, because self.lanes is deduplicated whereas phase.state length
        equals len(getControlledLanes()) which may be larger.
        """
        if self.env.fixed_ts or not self.green_phases:
            return list(self.lanes)

        # Raw lane list (with possible duplicates) — indices match phase.state chars.
        raw_lanes = self.sumo.trafficlight.getControlledLanes(self.id)
        controlled_set: set = set()
        for i, lane in enumerate(raw_lanes):
            ever_red = any(
                phase.state[i] in ('r', 's')
                for phase in self.green_phases
            )
            if ever_red:
                controlled_set.add(lane)

        # Return in self.lanes order (deduplicated), preserving original ordering.
        return [lane for lane in self.lanes if lane in controlled_set]

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
            duration = max(phase.duration, 10) #这里我想确保原始的绿灯是大于10s的，避免原始sumo phase文件里设置太小导致跟delta time,min green time潜在的冲突
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
        print(f"\n[_build_phases] TS={self.id} — green phases sent to SUMO:")
        for i, p in enumerate(self.green_phases):
            print(f"  green_phase[{i}]: {p.state}")
        self.sumo.trafficlight.setProgramLogic(self.id, logic) #传给sumo
        #这一行setRedYellowGreenState执行完后，这里不知道为什么会有两个logics，第一个是设定的，第二个是online，貌似是sumo自己生成的.设定第一个相位是初始相位。
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)
        tl_logic = self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.id)
        print(f"[_build_phases] TS={self.id} — SUMO runtime phases (after setProgramLogic):")
        for logic_entry in tl_logic:
            for i, p in enumerate(logic_entry.phases):
                if "y" not in p.state and p.state.count("r") + p.state.count("s") != len(p.state):
                    print(f"  runtime green phase[{i}]: {p.state}")
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
            if new_phase ==3:
                print("🔥 Action 3 selected at TS:", self.id)
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0
            a=1
        self.last_executed_action = self.green_phase
        a=1

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        return self.last_reward

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

    def get_lanes_density_by_type(self):
        """
        Returns two lists: car_density, bus_density for each incoming lane.
        density = 该类型车辆数 / 车道可容纳车辆数
        """
        car_density = []
        bus_density = []
        for lane in self.lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            length = self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))
            car_count = sum(1 for vid in vids if self.sumo.vehicle.getTypeID(vid) == "car")
            bus_count = sum(1 for vid in vids if self.sumo.vehicle.getTypeID(vid) == "bus")
            car_density.append(min(1, car_count / length if length > 0 else 0))
            bus_density.append(min(1, bus_count / length if length > 0 else 0))
        return car_density, bus_density

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

    def get_lanes_queue_by_type(self):
        """
        Returns two lists: car_queue, bus_queue for each incoming lane.
        queue = 该类型且速度<0.1的车辆数 / 车道可容纳车辆数
        """
        car_queue = []
        bus_queue = []
        for lane in self.lanes:
            vids = self.sumo.lane.getLastStepVehicleIDs(lane)
            length = self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane))
            car_halt = sum(1 for vid in vids if self.sumo.vehicle.getTypeID(vid) == "car" and self.sumo.vehicle.getSpeed(vid) < 0.1)
            bus_halt = sum(1 for vid in vids if self.sumo.vehicle.getTypeID(vid) == "bus" and self.sumo.vehicle.getSpeed(vid) < 0.1)
            car_queue.append(min(1, car_halt / length if length > 0 else 0))
            bus_queue.append(min(1, bus_halt / length if length > 0 else 0))
        return car_queue, bus_queue

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

    reward_fns = REWARD_REGISTRY
