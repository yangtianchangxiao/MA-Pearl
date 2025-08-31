"""
机械臂Graph Transformer专用分布模块
处理变长batch中的entropy计算和动作采样
适配SAC算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple, Optional
import math


class MaskedNormal:
    """
    支持mask的正态分布，用于变长batch
    处理不同DOF机械臂的动作分布
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            mean: [batch_size, max_action_dim] 动作均值
            std: [batch_size, max_action_dim] 动作标准差
            mask: [batch_size, max_action_dim] 有效动作mask
        """
        self.mean = mean
        self.log_std = torch.log(std.clamp(min=1e-6))
        self.std = std
        self.mask = mask
        
        # 创建底层Normal分布
        self.normal = Normal(mean, std)
    
    def sample(self) -> torch.Tensor:
        """采样动作"""
        sample = self.normal.sample()
        if self.mask is not None:
            sample = sample * self.mask.float()
        return sample
    
    def rsample(self) -> torch.Tensor:
        """重参数化采样 (用于SAC)"""
        sample = self.normal.rsample()
        if self.mask is not None:
            sample = sample * self.mask.float()
        return sample
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """计算对数概率"""
        log_prob = self.normal.log_prob(value)
        if self.mask is not None:
            # 只计算有效动作的对数概率
            log_prob = log_prob * self.mask.float()
            # 对每个样本求和
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob
    
    def entropy(self) -> torch.Tensor:
        """计算熵 - 处理变长情况"""
        # 正态分布的熵: 0.5 * log(2π * σ²) + 0.5
        entropy_per_dim = 0.5 * (math.log(2 * math.pi) + 2 * self.log_std + 1)
        
        if self.mask is not None:
            # 只计算有效动作维度的熵
            entropy_per_dim = entropy_per_dim * self.mask.float()
            # 对每个样本的有效维度求和
            total_entropy = entropy_per_dim.sum(dim=-1, keepdim=True)
        else:
            total_entropy = entropy_per_dim.sum(dim=-1, keepdim=True)
            
        return total_entropy
    
    def mode(self) -> torch.Tensor:
        """返回众数(均值)"""
        mode = self.mean
        if self.mask is not None:
            mode = mode * self.mask.float()
        return mode


class GraphSACDistribution(nn.Module):
    """
    Graph SAC专用分布层
    支持变长机械臂的连续动作分布
    """
    def __init__(
        self, 
        input_dim: int,
        max_action_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        
        self.max_action_dim = max_action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 均值网络
        self.mean_linear = nn.Linear(input_dim, max_action_dim)
        
        # 标准差网络
        self.log_std_linear = nn.Linear(input_dim, max_action_dim)
        
    def forward(self, graph_features: torch.Tensor, action_mask: torch.Tensor = None) -> MaskedNormal:
        """
        Args:
            graph_features: [batch_size, input_dim] 图特征
            action_mask: [batch_size, max_action_dim] 动作mask
            
        Returns:
            MaskedNormal: 支持mask的分布
        """
        # 计算均值和对数标准差
        mean = self.mean_linear(graph_features)
        log_std = self.log_std_linear(graph_features)
        
        # 限制标准差范围
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return MaskedNormal(mean, std, action_mask)


class TanhMaskedNormal:
    """
    Tanh变换的MaskedNormal，用于SAC
    将无界高斯分布映射到[-1, 1]
    """
    def __init__(self, masked_normal: MaskedNormal):
        self.masked_normal = masked_normal
        self.mask = masked_normal.mask
    
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: tanh变换后的动作
            raw_action: 变换前的原始动作
        """
        raw_action = self.masked_normal.rsample()
        action = torch.tanh(raw_action)
        
        if self.mask is not None:
            action = action * self.mask.float()
            raw_action = raw_action * self.mask.float()
            
        return action, raw_action
    
    def log_prob(self, action: torch.Tensor, raw_action: torch.Tensor = None) -> torch.Tensor:
        """
        计算tanh变换后的对数概率
        使用变量变换公式: log p(tanh(z)) = log p(z) - log|1 - tanh²(z)|
        """
        if raw_action is None:
            # 从tanh(x)反推x = arctanh(action)
            raw_action = 0.5 * torch.log((1 + action.clamp(-0.999, 0.999)) / 
                                         (1 - action.clamp(-0.999, 0.999)))
        
        # 原始分布的对数概率
        log_prob = self.masked_normal.log_prob(raw_action)
        
        # Tanh变换的雅可比行列式
        log_det_jacobian = torch.log(1 - action.pow(2) + 1e-6)
        
        if self.mask is not None:
            log_det_jacobian = log_det_jacobian * self.mask.float()
            log_det_jacobian = log_det_jacobian.sum(dim=-1, keepdim=True)
        else:
            log_det_jacobian = log_det_jacobian.sum(dim=-1, keepdim=True)
        
        return log_prob - log_det_jacobian
    
    def entropy(self) -> torch.Tensor:
        """
        Tanh变换分布的熵（近似）
        由于tanh变换复杂，使用原分布熵作为近似
        """
        return self.masked_normal.entropy()


class GraphSACPolicy(nn.Module):
    """
    Graph SAC策略网络
    支持变长机械臂动作生成
    """
    def __init__(
        self,
        graph_transformer,
        max_action_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        
        self.graph_transformer = graph_transformer
        self.max_action_dim = max_action_dim
        
        # 分布层
        self.distribution = GraphSACDistribution(
            input_dim=graph_transformer.graph_transformer.output_proj[-1].out_features,
            max_action_dim=max_action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max
        )
        
    def forward(self, data_batch, action_mask: torch.Tensor = None, deterministic: bool = False):
        """
        Args:
            data_batch: PyG Batch对象
            action_mask: [batch_size, max_action_dim] 动作mask
            deterministic: 是否使用确定性策略
            
        Returns:
            action: [batch_size, max_action_dim] 动作
            log_prob: [batch_size, 1] 对数概率
            entropy: [batch_size, 1] 熵
        """
        # 提取图特征
        _, graph_features, _ = self.graph_transformer.graph_transformer(data_batch)
        
        # 获得分布
        masked_normal = self.distribution(graph_features, action_mask)
        tanh_normal = TanhMaskedNormal(masked_normal)
        
        if deterministic:
            # 确定性模式：使用均值
            raw_action = masked_normal.mode()
            action = torch.tanh(raw_action)
            if action_mask is not None:
                action = action * action_mask.float()
            log_prob = tanh_normal.log_prob(action, raw_action)
        else:
            # 随机模式：采样
            action, raw_action = tanh_normal.sample()
            log_prob = tanh_normal.log_prob(action, raw_action)
        
        # 计算熵
        entropy = tanh_normal.entropy()
        
        return action, log_prob, entropy
    
    def evaluate_actions(self, data_batch, actions: torch.Tensor, action_mask: torch.Tensor = None):
        """
        评估给定动作的对数概率和熵
        用于策略更新
        """
        _, graph_features, _ = self.graph_transformer.graph_transformer(data_batch)
        masked_normal = self.distribution(graph_features, action_mask)
        tanh_normal = TanhMaskedNormal(masked_normal)
        
        log_prob = tanh_normal.log_prob(actions)
        entropy = tanh_normal.entropy()
        
        return log_prob, entropy


def create_action_mask(dof_list: list, max_dof: int) -> torch.Tensor:
    """
    为不同DOF的机械臂创建动作mask
    
    Args:
        dof_list: 每个机械臂的DOF数量列表
        max_dof: 最大DOF数量
        
    Returns:
        mask: [batch_size, max_dof] 动作mask
    """
    batch_size = len(dof_list)
    mask = torch.zeros(batch_size, max_dof)
    
    for i, dof in enumerate(dof_list):
        mask[i, :dof] = 1
        
    return mask


# 测试代码
if __name__ == "__main__":
    print("=== 机械臂分布模块测试 ===")
    
    # 创建测试数据
    batch_size = 3
    max_action_dim = 6
    feature_dim = 128
    
    # 模拟不同DOF的机械臂
    dof_list = [3, 6, 4]  # 第一个3DOF，第二个6DOF，第三个4DOF
    action_mask = create_action_mask(dof_list, max_action_dim)
    
    print(f"DOF列表: {dof_list}")
    print(f"Action mask形状: {action_mask.shape}")
    print(f"Action mask:\n{action_mask}")
    
    # 测试MaskedNormal
    graph_features = torch.randn(batch_size, feature_dim)
    distribution_layer = GraphSACDistribution(feature_dim, max_action_dim)
    
    # 获得分布
    masked_normal = distribution_layer(graph_features, action_mask)
    
    # 采样和计算
    action_sample = masked_normal.sample()
    log_prob = masked_normal.log_prob(action_sample)
    entropy = masked_normal.entropy()
    
    print(f"\n动作采样形状: {action_sample.shape}")
    print(f"对数概率形状: {log_prob.shape}")
    print(f"熵形状: {entropy.shape}")
    
    # 测试TanhMaskedNormal
    tanh_normal = TanhMaskedNormal(masked_normal)
    tanh_action, raw_action = tanh_normal.sample()
    tanh_log_prob = tanh_normal.log_prob(tanh_action, raw_action)
    
    print(f"\nTanh动作形状: {tanh_action.shape}")
    print(f"Tanh对数概率形状: {tanh_log_prob.shape}")
    
    # 验证mask效果
    print(f"\n=== Mask效果验证 ===")
    for i, dof in enumerate(dof_list):
        valid_actions = action_sample[i, :dof]
        invalid_actions = action_sample[i, dof:]
        print(f"机械臂{i+1} ({dof}DOF):")
        print(f"  有效动作: {valid_actions}")
        print(f"  无效动作(应为0): {invalid_actions}")
        print(f"  无效动作全为0: {torch.allclose(invalid_actions, torch.zeros_like(invalid_actions))}")
    
    print("\n✅ 分布模块测试成功!")