import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# 初始化权重
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        
        # 初始化最后一层权重为较小的值
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.mean.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.bias.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SAC_Eval:
    def __init__(self, state_dim, action_dim, action_space_shape):
        """用于评估和可视化的SAC版本"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Action dimension: {action_dim}")
        
        # 创建网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        
        # 动作范围是[-1, 1]
        self.action_scale = torch.FloatTensor([1.0] * action_dim).to(self.device)
        self.action_bias = torch.FloatTensor([0.0] * action_dim).to(self.device)

    def select_action(self, state, evaluate=True):
        """选择动作"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)
        
        with torch.no_grad():
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            action = action.reshape(-1)  # 确保动作是一维的
        
        return action.cpu().numpy()

    def load_state_dict(self, state_dict):
        """加载模型状态"""
        if isinstance(state_dict, dict) and 'actor_state_dict' in state_dict:
            # 确保模型在正确的设备上
            for k, v in state_dict['actor_state_dict'].items():
                state_dict['actor_state_dict'][k] = v.to(self.device)
            self.actor.load_state_dict(state_dict['actor_state_dict'])
        else:
            # 确保模型在正确的设备上
            for k, v in state_dict.items():
                state_dict[k] = v.to(self.device)
            self.actor.load_state_dict(state_dict)
