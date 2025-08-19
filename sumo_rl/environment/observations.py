"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

class PressLightObservationFunction(ObservationFunction):
    """PressLight observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize PressLight observation function."""
        super().__init__(ts)
        print("This is local obs.py")

    def __call__(self) -> np.ndarray:
        """Return the PressLight observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        list_inc_vehicle_counts, list_out_vehicle_counts = self._get_vehicle_count_by_lanes(self.ts.id)
        # d= self.ts.out_lanes
    
        observation = np.array(phase_id + min_green + list_inc_vehicle_counts + list_out_vehicle_counts, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        a= len(self.ts.lanes) #12
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

    def _get_vehicle_count_by_lanes(self, ts_id):
        """
        获取一个信号灯路口的每个进出车道的车辆数量
        :param ts_id: 信号灯的ID
        :return: 一个字典，键是车道ID，值是该车道上的车辆数量
        """
        inc_vehicle_counts = {}  # 存储结果的字典

        # 获取信号灯控制的所有车道
        inc_lanes = self.ts.sumo.trafficlight.getControlledLanes(ts_id)

        # 对于每个车道，计算并存储车辆数量
        for lane_id in inc_lanes:
            inc_vehicle_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane_id)
            inc_vehicle_counts[lane_id] = inc_vehicle_count
        
        # print(inc_vehicle_counts)

        out_vehicle_counts = {}
        # outgoing_lanes = self._get_outgoing_lanes(ts_id)
        outgoing_lanes = self.ts.out_lanes

        # 对于每个出口车道，计算并存储车辆数量
        for lane_id in outgoing_lanes:
            out_vehicle_count = self.ts.sumo.lane.getLastStepVehicleNumber(lane_id)
            out_vehicle_counts[lane_id] = out_vehicle_count
        
        # print(out_vehicle_counts)

        #将字典转换为list
        list_inc_vehicle_counts = [inc_vehicle_counts[key] for key in inc_vehicle_counts]
        list_out_vehicle_counts = [out_vehicle_counts[key] for key in out_vehicle_counts]

        return list_inc_vehicle_counts, list_out_vehicle_counts
    
    def _get_outgoing_lanes(self, ts_id):
        """
        根据信号灯ID获取所有出口车道
        假设进入边缘和离开边缘遵循特定的命名约定（例如 n_t 表示北进入，t_n 表示北离开）
        """
        # Define the mapping based on your naming convention
        edge_mapping = {'n_t': 't_n', 'e_t': 't_e', 's_t': 't_s', 'w_t': 't_w'}
        
        # Get the controlled lanes and their edges
        controlled_lanes = self.ts.sumo.trafficlight.getControlledLanes(ts_id)
        outgoing_lanes = []

        # Iterate through each controlled lane to find corresponding outgoing lanes
        for lane_id in controlled_lanes:
            # Extract edge ID from lane ID
            edge_id = lane_id.split('_')[0] + '_' + lane_id.split('_')[1]
            # Find the corresponding outgoing edge based on the mapping
            if edge_id in edge_mapping:
                outgoing_edge = edge_mapping[edge_id]
                # Get all lanes for the outgoing edge
                for lane_index in range(self.ts.sumo.edge.getLaneNumber(outgoing_edge)):
                    outgoing_lane_id = f"{outgoing_edge}_{lane_index}"
                    if outgoing_lane_id not in outgoing_lanes:
                        outgoing_lanes.append(outgoing_lane_id)
        # print(outgoing_lanes)
        a= list(outgoing_lanes)
        return list(outgoing_lanes)  # Return unique outgoing lanes
    
    
class PriorityObservationFunction(ObservationFunction):
    """Priority observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize Priority observation function."""
        super().__init__(ts)
        print("This is modified PriorityObservationFunction with car/truck state.")

    def __call__(self) -> np.ndarray:
        """Return the PressLight observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        # Get separate car/truck counts
        inc_car, inc_truck, out_car, out_truck = self._get_vehicle_count_by_type(self.ts.id)
        # d= self.ts.out_lanes
    
        # Build observation vector
        observation = np.array(
            phase_id
            + min_green
            + inc_car
            + out_car
            + inc_truck
            + out_truck,
            dtype=np.float32
        )
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        num_phases = self.ts.num_green_phases
        num_lanes = len(self.ts.lanes)
        # state length = num_phases + 1 + 4 * num_lanes
        dim = num_phases + 1 + 4 * num_lanes
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.ones(dim, dtype=np.float32),
        )


    def _get_vehicle_count_by_type(self, ts_id):
            """
            Return counts of cars and trucks on incoming and outgoing lanes.
            先按 type=="car" 计数，其余车辆一律归为 truck。
            :return: inc_car, inc_truck, out_car, out_truck as lists ordered by lane order
            """
            inc_lanes = self.ts.sumo.trafficlight.getControlledLanes(ts_id)
            out_lanes = self.ts.out_lanes  # assume exists

            inc_car, inc_truck = [], []
            for lane in inc_lanes:
                vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
                cars = sum(1 for v in vids if self.ts.sumo.vehicle.getTypeID(v) == 'car')
                inc_car.append(cars)
                inc_truck.append(len(vids) - cars)

            out_car, out_truck = [], []
            for lane in out_lanes:
                vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
                cars = sum(1 for v in vids if self.ts.sumo.vehicle.getTypeID(v) == 'car')
                out_car.append(cars)
                out_truck.append(len(vids) - cars)

            return inc_car, inc_truck, out_car, out_truck


class CTBPriorityObservationFunction(ObservationFunction):
    """Priority observation with optimized vehicle-type counting for performance."""

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        
    def __call__(self) -> np.ndarray:
        ts = self.ts
        # one-hot encoding of current green phase
        phase_id = [1.0 if ts.green_phase == i else 0.0
                    for i in range(ts.num_green_phases)]
        # minimum green flag
        min_green = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0

        # fast count
        inc_car, inc_truck, inc_bus, out_car, out_truck, out_bus = self._get_vehicle_count_by_type(ts.id)

        # build and return observation
        obs = phase_id + [min_green] + inc_car + out_car + inc_truck + out_truck + inc_bus + out_bus
        return np.array(obs, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        num_phases = self.ts.num_green_phases
        num_lanes = len(self.ts.lanes)
        dim = num_phases + 1 + 6 * num_lanes
        return spaces.Box(
            low=np.zeros(dim, dtype=np.float32),
            high=np.ones(dim, dtype=np.float32),
        )

    def _get_vehicle_count_by_type(self, ts_id):
        """Efficient single-pass counting of cars, trucks, and buses per lane."""
        sumo = self.ts.sumo
        get_ids = sumo.lane.getLastStepVehicleIDs
        get_type = sumo.vehicle.getTypeID

        inc_lanes = sumo.trafficlight.getControlledLanes(ts_id)
        out_lanes = self.ts.out_lanes

        def count_types(lanes):
            cars, trucks, buses = [], [], []
            for lane in lanes:
                vids = get_ids(lane)
                c = t = b = 0
                for vid in vids:
                    vtype = get_type(vid)
                    if vtype == 'truck':
                        t += 1
                    elif vtype == 'bus':
                        b += 1
                    else:
                        c += 1
                cars.append(c)
                trucks.append(t)
                buses.append(b)
            return cars, trucks, buses

        inc_car, inc_truck, inc_bus = count_types(inc_lanes)
        out_car, out_truck, out_bus = count_types(out_lanes)
        return inc_car, inc_truck, inc_bus, out_car, out_truck, out_bus

# cursor写的，还没验证还不好用
class PolicyAwareObservationFunction(ObservationFunction):
    """基于BNF Policy的道路结构感知观察函数"""

    def __init__(self, ts: TrafficSignal, bnf_parser=None):
        super().__init__(ts)
        self.bnf_parser = bnf_parser
        self.lane_structure = self._analyze_lane_structure()
        self.policy_count = len([k for k in bnf_parser.bnf_rules.keys() if k.startswith('policy')]) if bnf_parser else 0
        
    def _analyze_lane_structure(self):
        """分析道路结构：每个方向的车道数量和类型"""
        structure = {
            'north_lanes': [],
            'south_lanes': [],
            'west_lanes': [],
            'east_lanes': [],
            'total_incoming_lanes': len(self.ts.lanes),
            'total_outgoing_lanes': len(self.ts.out_lanes)
        }
        
        # 根据车道ID分析方向
        for lane in self.ts.lanes:
            if 'n_' in lane:
                structure['north_lanes'].append(lane)
            elif 's_' in lane:
                structure['south_lanes'].append(lane)
            elif 'w_' in lane:
                structure['west_lanes'].append(lane)
            elif 'e_' in lane:
                structure['east_lanes'].append(lane)
        
        # 添加车道数量信息
        structure['north_count'] = len(structure['north_lanes'])
        structure['south_count'] = len(structure['south_lanes'])
        structure['west_count'] = len(structure['west_lanes'])
        structure['east_count'] = len(structure['east_lanes'])
        
        return structure

    def __call__(self) -> np.ndarray:
        ts = self.ts
        observation = []
        
        # 1. 基础信息：相位和时间
        phase_id = [1.0 if ts.green_phase == i else 0.0 
                   for i in range(ts.num_green_phases)]
        min_green = 1.0 if ts.time_since_last_phase_change >= ts.min_green + ts.yellow_time else 0.0
        observation.extend(phase_id)
        observation.append(min_green)
        
        # 2. 道路结构信息（归一化）
        max_lanes = 4  # 假设最大车道数
        structure_info = [
            self.lane_structure['north_count'] / max_lanes,
            self.lane_structure['south_count'] / max_lanes,
            self.lane_structure['west_count'] / max_lanes,
            self.lane_structure['east_count'] / max_lanes,
            self.lane_structure['total_incoming_lanes'] / (max_lanes * 4),
            self.lane_structure['total_outgoing_lanes'] / (max_lanes * 4)
        ]
        observation.extend(structure_info)
        
        # 3. 每条车道的详细信息
        lane_observations = self._get_lane_based_observations()
        observation.extend(lane_observations)
        
        # 4. 方向级别的汇总信息
        direction_observations = self._get_direction_based_observations()
        observation.extend(direction_observations)
        
        return np.array(observation, dtype=np.float32)

    def _get_lane_based_observations(self):
        """获取每条车道的观察信息"""
        lane_obs = []
        
        # 处理所有进入车道
        for lane in self.ts.lanes:
            # 获取车道上的所有车辆
            vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            total_vehicles = len(vids)
            
            # 按policy分组统计
            policy_counts = self._count_vehicles_by_policy(vids)
            
            # 车道容量归一化
            lane_capacity = self.ts.sumo.lane.getLength(lane) / 7.5  # 假设每车7.5米
            
            # 添加观察：总车辆数 + 各policy车辆数
            lane_obs.append(total_vehicles / max(lane_capacity, 1.0))
            
            # 添加每个policy的车辆数（归一化）
            for policy_num in range(1, self.policy_count + 1):
                policy_key = f'policy{policy_num}'
                count = policy_counts.get(policy_key, 0)
                lane_obs.append(count / max(lane_capacity, 1.0))
            
            # 添加等待车辆数
            waiting_vehicles = self.ts.sumo.lane.getLastStepHaltingNumber(lane)
            lane_obs.append(waiting_vehicles / max(lane_capacity, 1.0))
            
            # 添加平均速度
            if vids:
                avg_speed = np.mean([self.ts.sumo.vehicle.getSpeed(vid) for vid in vids])
                lane_obs.append(avg_speed / 13.89)  # 归一化到50km/h
            else:
                lane_obs.append(0.0)
        
        return lane_obs

    def _get_direction_based_observations(self):
        """获取按方向汇总的观察信息"""
        direction_obs = []
        
        directions = ['north_lanes', 'south_lanes', 'west_lanes', 'east_lanes']
        
        for direction in directions:
            lanes = self.lane_structure[direction]
            
            if not lanes:
                # 如果该方向没有车道，填充零
                direction_obs.extend([0.0] * (2 + self.policy_count))
                continue
            
            total_vehicles = 0
            total_waiting = 0
            policy_totals = {f'policy{i}': 0 for i in range(1, self.policy_count + 1)}
            
            # 汇总该方向所有车道的信息
            for lane in lanes:
                vids = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
                total_vehicles += len(vids)
                total_waiting += self.ts.sumo.lane.getLastStepHaltingNumber(lane)
                
                # 按policy统计
                policy_counts = self._count_vehicles_by_policy(vids)
                for policy, count in policy_counts.items():
                    if policy in policy_totals:
                        policy_totals[policy] += count
            
            # 计算该方向的总容量
            direction_capacity = sum(self.ts.sumo.lane.getLength(lane) / 7.5 for lane in lanes)
            
            # 归一化并添加到观察
            direction_obs.append(total_vehicles / max(direction_capacity, 1.0))
            direction_obs.append(total_waiting / max(direction_capacity, 1.0))
            
            # 添加各policy的车辆数
            for policy_num in range(1, self.policy_count + 1):
                policy_key = f'policy{policy_num}'
                count = policy_totals.get(policy_key, 0)
                direction_obs.append(count / max(direction_capacity, 1.0))
        
        return direction_obs

    def _count_vehicles_by_policy(self, vehicle_ids):
        """统计车辆按policy分组的数量"""
        policy_counts = {}
        
        if not self.bnf_parser:
            return policy_counts
        
        for vid in vehicle_ids:
            try:
                vehicle_type = self.ts.sumo.vehicle.getTypeID(vid)
                context = self._get_vehicle_context(vid)
                
                # 找到匹配的policy
                matching_policy = self._find_matching_policy(vehicle_type, context)
                
                if matching_policy:
                    if matching_policy not in policy_counts:
                        policy_counts[matching_policy] = 0
                    policy_counts[matching_policy] += 1
            except:
                # 如果获取车辆信息失败，跳过
                continue
        
        return policy_counts

    def _find_matching_policy(self, vehicle_type: str, context: dict) -> str:
        """找到车辆匹配的policy"""
        if not self.bnf_parser:
            return None
        
        # 遍历所有policy规则，找到匹配的
        for policy_name, policy_def in self.bnf_parser.bnf_rules.items():
            if policy_name.startswith('policy') and self._matches_policy_rule(vehicle_type, context, policy_def):
                return policy_name
        
        return None

    def _matches_policy_rule(self, vehicle_type: str, context: dict, policy_def: str) -> bool:
        """检查车辆是否匹配特定的policy规则"""
        # 简化的匹配逻辑，您可以根据BNF语法完善
        vehicle_type_map = {
            'car': 'Car',
            'truck': 'Truck', 
            'bus': 'Bus',
            'ambulance': 'Ambulance'
        }
        
        bnf_vehicle_type = vehicle_type_map.get(vehicle_type.lower(), vehicle_type)
        
        # 检查车辆类型是否匹配
        if f'"{bnf_vehicle_type}"' not in policy_def:
            return False
        
        # 这里可以添加更复杂的上下文匹配逻辑
        # 比如匹配task, location, time等
        
        return True

    def _get_vehicle_context(self, vehicle_id: str) -> dict:
        """获取车辆上下文信息"""
        ts = self.ts
        
        # 获取时间信息
        current_time = ts.sumo.simulation.getTime()
        current_hour = (current_time // 3600) % 24
        is_peak = 7 <= current_hour <= 9 or 17 <= current_hour <= 19
        
        # 获取位置信息
        try:
            lane_id = ts.sumo.vehicle.getLaneID(vehicle_id)
            route_id = ts.sumo.vehicle.getRouteID(vehicle_id)
        except:
            lane_id = ""
            route_id = ""
        
        return {
            'task': 'on',  # 简化假设
            'np': 'M',     # 简化假设
            'location': self._infer_location(lane_id, route_id),
            'time': 'Peak' if is_peak else 'non',
            'state': 'Normal'  # 简化假设
        }

    def _infer_location(self, lane_id: str, route_id: str) -> str:
        """根据车道ID和路线ID推断位置"""
        combined = (lane_id + route_id).lower()
        
        if 'hospital' in combined:
            return 'Hospital'
        elif 'center' in combined or 'city' in combined:
            return 'City Centre'
        elif 'industrial' in combined:
            return 'Industrial'
        elif 'residential' in combined:
            return 'Residential'
        elif 'parking' in combined:
            return 'Parking'
        else:
            return '-'

    def observation_space(self) -> spaces.Box:
        """定义观察空间"""
        # 计算观察向量的总维度
        num_phases = self.ts.num_green_phases
        num_lanes = len(self.ts.lanes)
        
        # 基础信息：相位(4) + 最小绿灯(1) = 5
        basic_dim = num_phases + 1
        
        # 道路结构信息：6个特征
        structure_dim = 6
        
        # 每条车道信息：总车辆数(1) + 各policy车辆数(policy_count) + 等待车辆数(1) + 平均速度(1)
        lane_features_per_lane = 3 + self.policy_count
        lane_dim = num_lanes * lane_features_per_lane
        
        # 方向汇总信息：4个方向 × (总车辆数(1) + 等待车辆数(1) + 各policy车辆数(policy_count))
        direction_features_per_direction = 2 + self.policy_count
        direction_dim = 4 * direction_features_per_direction
        
        total_dim = basic_dim + structure_dim + lane_dim + direction_dim
        
        return spaces.Box(
            low=np.zeros(total_dim, dtype=np.float32),
            high=np.ones(total_dim, dtype=np.float32),
        )
