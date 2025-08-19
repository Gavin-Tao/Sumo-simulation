import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.insert(0, 'E:\\txw\\SUMO-RL') #第一优先在这个路径下去寻找包
from sumo_rl.agents.dqn_model import DQN_model
from sumo_rl.agents.replay_memory import ReplayMemory
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN_agent:
    ''' DQN算法 '''
    def __init__(self, starting_state, state_space, action_space, batch_size, learning_rate, gamma,
                 eps_start, eps_end, eps_decay, target_update_frequency, tau, replay_capacity, device):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        self.replay_capacity = replay_capacity
        self.device = device
        
        self.policy_net = DQN_model(self.state_space, self.action_space).to(self.device)  # Q网络
        # 目标网络
        self.target_net = DQN_model(self.state_space, self.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 使用Adam优化器
        self.optimizer = optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayMemory(self.replay_capacity)
        
        
        self.steps_done = 0
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def select_action(self, state):  # epsilon-贪婪策略采取动作
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                a=self.policy_net(state).max(1).indices.view(1,1)
                return self.policy_net(state).max(1).indices.view(1,1)
        else:
            action = np.random.choice(self.action_space)
            return torch.tensor([[action]], device=self.device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        
        #更新network参数
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)