import re
from typing import Dict, Callable

class BNFParser:
    """从BNF语法自动生成state和reward函数"""
    
    def __init__(self, bnf_file_path: str):
        self.bnf_rules = self.parse_bnf_file(bnf_file_path)
    
    def parse_bnf_file(self, file_path: str) -> Dict:
        """解析BNF文件并提取规则"""
        rules = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析BNF规则 (示例格式)
        # <state> ::= <phase_info> + <vehicle_count> + <waiting_time>
        # <reward> ::= -<total_waiting_time> + <throughput_bonus>
        
        lines = content.strip().split('\n')
        for line in lines:
            if '::=' in line:
                rule_name = line.split('::=')[0].strip().strip('<>')
                rule_definition = line.split('::=')[1].strip()
                rules[rule_name] = rule_definition
        
        return rules
    
    def generate_state_function(self) -> Callable:
        """根据BNF规则生成state函数"""
        state_rule = self.bnf_rules.get('state', '')
        
        def dynamic_state_function(ts):
            observation = []
            
            # 解析state规则并构建observation
            if 'phase_info' in state_rule:
                phase_id = [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)]
                observation.extend(phase_id)
            
            if 'vehicle_count' in state_rule:
                vehicle_counts = self._get_vehicle_counts(ts)
                observation.extend(vehicle_counts)
            
            if 'waiting_time' in state_rule:
                waiting_times = self._get_waiting_times(ts)
                observation.extend(waiting_times)
            
            return np.array(observation, dtype=np.float32)
        
        return dynamic_state_function
    
    def generate_reward_function(self) -> Callable:
        """根据BNF规则生成reward函数"""
        reward_rule = self.bnf_rules.get('reward', '')
        
        def dynamic_reward_function(ts):
            reward = 0.0
            
            # 解析reward规则并计算奖励
            if 'total_waiting_time' in reward_rule:
                waiting_penalty = sum(ts.get_accumulated_waiting_time_per_lane())
                if '-' in reward_rule and 'total_waiting_time' in reward_rule:
                    reward -= waiting_penalty
                else:
                    reward += waiting_penalty
            
            if 'throughput_bonus' in reward_rule:
                throughput = len(ts.get_departed_vehicles())  # 需要实现这个方法
                reward += throughput * 0.1  # 可以从BNF中解析权重
            
            return reward
        
        return dynamic_reward_function
