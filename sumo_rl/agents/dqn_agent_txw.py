import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done, next_mask=None):
        """next_mask: optional 8-dim bool array (Dublin masked-action mode,
        M3) — the valid-action mask of THIS transition's junction, needed for
        the masked target max. Per run either ALL transitions carry a mask or
        none does (mixing would break unzip)."""
        if next_mask is None:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        cols = list(zip(*transitions))
        if len(cols) == 6:
            state, action, reward, next_state, done, next_mask = cols
            return (np.array(state), action, reward, np.array(next_state), done,
                    np.array(next_mask))
        state, action, reward, next_state, done = cols
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    
class DQN:
    ''' DQN算法 (默认 vanilla;可选 Double DQN / PER) '''
    def __init__(self, starting_state, state_space, hidden_dim, action_space, learning_rate, gamma,
                 epsilon, target_update, capacity, mini_size, batch_size, eps_start, eps_end, eps_decay, device,
                 use_double: bool = False,
                 use_per: bool = False,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_end: float = 1.0,
                 per_beta_steps: int = 100_000,
                 per_eps: float = 1e-6,
                 q_net_factory=None,
                 grad_clip: float = None):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        # q_net_factory: optional zero-arg callable returning an nn.Module with the same
        # contract as Qnet (flat (B, state_space) in → (B, action_space) Q-values out).
        # Lets train scripts swap the architecture (e.g. lane-token Transformer) without
        # touching the update/take_action/checkpoint logic.
        if q_net_factory is not None:
            self.q_net = q_net_factory().to(device)
            self.target_q_net = q_net_factory().to(device)
        else:
            self.q_net = Qnet(self.state_space, self.hidden_dim,
                              self.action_space).to(device)  # Q网络
            # 目标网络
            self.target_q_net = Qnet(self.state_space, self.hidden_dim,
                                     self.action_space).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.capacity = capacity

        # ── PER 开关 + 超参 (默认全 False/默认值 = vanilla 行为) ──
        self.use_per        = use_per
        self.per_alpha      = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end   = per_beta_end
        self.per_beta_steps = per_beta_steps
        self.per_eps        = per_eps

        if use_per:
            from sumo_rl.agents.prioritized_replay_buffer import PrioritizedReplayBuffer
            self.replay_buffer = PrioritizedReplayBuffer(self.capacity, alpha=per_alpha, eps=per_eps)
        else:
            self.replay_buffer = ReplayBuffer(self.capacity)
        self.mini_size = mini_size
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        # grad_clip: max grad-norm for clip_grad_norm_ before optimizer.step();
        # None (default) = no clipping = exact historical behaviour. Set (e.g. 1.0)
        # for architectures that need it (transformer Q-net at lr 1e-3).
        self.grad_clip = grad_clip
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.loss = None
        # per-update diagnostics (read by trainers for wandb; None until first update)
        self.grad_norm = None    # pre-clip total grad norm — only set when grad_clip is on
        self.q_mean = None       # mean of online-net Q(s,a) over the batch
        self.q_abs_max = None    # max |Q| over the batch — divergence shows here first
        self.start_train = False
        self.use_double = use_double  # 默认 False = 原 vanilla DQN 行为

    @property
    def current_beta(self) -> float:
        """PER 模式下的 importance-sampling β,从 per_beta_start 线性退火到 per_beta_end。"""
        if not self.use_per:
            return 1.0
        if self.count >= self.per_beta_steps:
            return self.per_beta_end
        frac = self.count / max(self.per_beta_steps, 1)
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    def take_action(self, state, mask=None):  # epsilon-贪婪策略采取动作
        """mask: optional bool array over the action space (M3, Dublin) —
        the ε branch samples only valid actions, the greedy branch sets
        invalid Q to -inf before argmax. Invalid actions are therefore never
        chosen and never enter the buffer."""
        if np.random.random() < self.epsilon:
            if mask is not None:
                action = int(np.random.choice(np.flatnonzero(mask)))
            else:
                action = np.random.randint(self.action_space)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            q = self.q_net(state)
            if mask is not None:
                q = q.masked_fill(~torch.as_tensor(mask, dtype=torch.bool,
                                                   device=self.device), -1e9)
            action = q.argmax().item()
        return action

    def update(self, transition_dict):
        
        self.start_train = True
        
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值 (M3: masked target max — never bootstrap from a
        # phantom action of a junction that doesn't have it)
        next_mask_t = None
        if 'next_masks' in transition_dict:
            next_mask_t = torch.as_tensor(np.asarray(transition_dict['next_masks']),
                                          dtype=torch.bool, device=self.device)
        if self.use_double:
            # Double DQN: online net 选动作, target net 给值
            with torch.no_grad():
                nq = self.q_net(next_states)
                if next_mask_t is not None:
                    nq = nq.masked_fill(~next_mask_t, -1e9)
                next_actions = nq.argmax(1, keepdim=True)
                max_next_q_values = self.target_q_net(next_states).gather(1, next_actions)
        else:
            # Vanilla DQN: target net 自取 max (原逻辑)
            tq = self.target_q_net(next_states)
            if next_mask_t is not None:
                tq = tq.masked_fill(~next_mask_t, -1e9)
            max_next_q_values = tq.max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标

        # ── Loss: PER 用 importance-sampling 权重加权;否则原 MSE ──
        if 'weights' in transition_dict:
            weights = torch.tensor(transition_dict['weights'],
                                   dtype=torch.float).view(-1, 1).to(self.device)
            td_errors_for_per = (q_targets - q_values).detach()  # 保留符号,后面取 abs
            elementwise_sq    = (q_values - q_targets) ** 2
            dqn_loss          = (weights * elementwise_sq).mean()
        else:
            dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.loss = dqn_loss.item()
        with torch.no_grad():
            self.q_mean    = q_values.mean().item()
            self.q_abs_max = q_values.abs().max().item()
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        if self.grad_clip is not None:
            # clip_grad_norm_ returns the PRE-clip total norm — free diagnostic
            self.grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.grad_clip))
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

        # ── PER: 用刚算的 TD-error 更新对应样本的优先级 ──
        if 'weights' in transition_dict and 'indices' in transition_dict:
            td_np = td_errors_for_per.squeeze(-1).cpu().numpy()
            # guarded by 'weights' in dict → buffer is PrioritizedReplayBuffer
            self.replay_buffer.update_priorities(  # type: ignore[attr-defined]
                transition_dict['indices'], td_np)