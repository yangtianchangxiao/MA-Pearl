import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal initialization for the layer weights"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data, gain=gain)
        nn.init.constant_(layer.bias.data, 0)

class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU activation"""
    def __init__(self, dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize with smaller weights for residual path
        orthogonal_init(self.fc1, gain=0.1)
        orthogonal_init(self.fc2, gain=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.ln2(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        return x + residual

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        
        # 输入层标准化
        self.input_norm = nn.LayerNorm(obs_dim)
        
        # 特征提取器
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1]) 
            for _ in range(2)
        ])
        
        # 输出层
        self.mean = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std = nn.Linear(hidden_dims[-1], act_dim)
        
        # 初始化输出层
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, 0)

    def forward(self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 输入标准化
        x = self.input_norm(obs)
        
        # 特征提取
        x = self.feature_extractor(x)
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
            
        # 输出层
        mu = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        # Sample action using reparameterization trick
        if deterministic:
            action = mu
        else:
            normal = torch.distributions.Normal(mu, std)
            action = normal.rsample()
        
        # Compute log probability if needed
        if with_logprob:
            log_prob = torch.distributions.Normal(mu, std).log_prob(action).sum(axis=-1)
            log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
        else:
            log_prob = None
            
        action = torch.tanh(action)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        
        # 输入层标准化
        self.input_norm = nn.LayerNorm(obs_dim + act_dim)
        
        # 特征提取器
        layers = []
        in_dim = obs_dim + act_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
            
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[-1])
            for _ in range(2)
        ])
        
        # 输出层
        self.output = nn.Linear(hidden_dims[-1], 1)
        nn.init.orthogonal_(self.output.weight, gain=0.01)
        nn.init.constant_(self.output.bias, 0)
        
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        x = self.input_norm(x)
        x = self.feature_extractor(x)
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
            
        return self.output(x)

class TempNetwork(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.ones(1) * np.log(init_temp))
        
    def forward(self) -> torch.Tensor:
        return self.log_temp.exp()
