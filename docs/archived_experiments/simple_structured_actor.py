#!/usr/bin/env python3
"""
简单结构化Actor网络
保留核心洞察：3维节点特征 + 结构-空间分离
但用简单MLP替代复杂Graph attention
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from simplified_graph_demo import SimplifiedGraphState


class SimpleStructuredActorNetwork(nn.Module):
    """
    简单结构化Actor网络
    
    核心思路：
    - 保留3维节点特征 [joint1, joint2, length] 
    - 保留结构-空间分离设计
    - 用简单MLP替代Graph attention (大幅提速)
    """
    
    def __init__(
        self,
        action_dim: int,
        dof_range: Tuple[int, int] = (2, 4),
        hidden_dim: int = 128
    ):
        super().__init__()
        self.action_dim = action_dim
        self.dof_range = dof_range
        self.max_dof = max(dof_range) * 2
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 结构编码器：处理所有节点特征 (简单MLP)
        max_nodes = max(dof_range)
        self.structure_encoder = nn.Sequential(
            nn.Linear(max_nodes * 3, hidden_dim),  # 展平所有节点特征
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 空间编码器：处理Goal信息
        self.spatial_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # achieved(3) + desired(3)
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 结构-空间融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作输出
        self.action_mean_head = nn.Linear(hidden_dim, action_dim)
        self.action_std_head = nn.Linear(hidden_dim, action_dim)
        
        print(f"🪶 简单结构化Actor网络:")
        print(f"   节点特征: 3维 [joint1, joint2, length] (保留)")
        print(f"   处理方式: 简单MLP (替代Graph attention)")
        print(f"   结构-空间分离: 保留")
        print(f"   预期速度提升: 10-20x")
    
    def forward(self, her_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: HER format → 简单结构化处理
        """
        her_tensor = her_tensor.float()
        
        if her_tensor.dim() == 1:
            her_tensor = her_tensor.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size = her_tensor.shape[0]
        
        # 批处理
        structure_features = []
        spatial_features = []
        
        for i in range(batch_size):
            her_obs = her_tensor[i]
            graph_tensor = self._her_to_graph_state(her_obs)
            graph_state = SimplifiedGraphState.from_tensor(graph_tensor)
            
            # 1. 结构特征处理 (简单展平)
            structure_feat = self._process_structure_simple(graph_state.node_features)
            structure_features.append(structure_feat)
            
            # 2. 空间特征处理
            spatial_info = torch.cat([graph_state.achieved_goal, graph_state.desired_goal])
            spatial_feat = self.spatial_encoder(spatial_info)
            spatial_features.append(spatial_feat)
        
        # 组合
        structure_features = torch.stack(structure_features)
        spatial_features = torch.stack(spatial_features)
        
        # 融合
        combined = torch.cat([structure_features, spatial_features], dim=1)
        fused_features = self.fusion(combined)
        
        # 动作输出
        action_mean = self.action_mean_head(fused_features)
        action_log_std = self.action_std_head(fused_features)
        
        # 数值稳定性
        action_mean = torch.clamp(action_mean, -5, 5)
        action_log_std = torch.clamp(action_log_std, -10, 1)
        
        if single_sample:
            action_mean = action_mean.squeeze(0)
            action_log_std = action_log_std.squeeze(0)
        
        return action_mean, action_log_std
    
    def _process_structure_simple(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        简单结构处理：展平 + MLP (替代Graph attention)
        """
        max_nodes = max(self.dof_range)
        current_nodes = node_features.shape[0]
        
        # 零填充到最大节点数
        if current_nodes < max_nodes:
            padding = torch.zeros(max_nodes - current_nodes, 3, 
                                device=node_features.device, dtype=node_features.dtype)
            padded_features = torch.cat([node_features, padding], dim=0)
        else:
            padded_features = node_features[:max_nodes]  # 截断
        
        # 展平处理
        flattened = padded_features.flatten()  # [max_nodes * 3]
        
        # 简单MLP处理
        structure_feature = self.structure_encoder(flattened)
        
        return structure_feature
    
    def _her_to_graph_state(self, her_obs: torch.Tensor) -> torch.Tensor:
        """复用之前的转换逻辑"""
        joint_angles = her_obs[:self.max_dof].detach().cpu().numpy()
        achieved_goal = her_obs[self.max_dof:self.max_dof+3]
        desired_goal = her_obs[self.max_dof+3:self.max_dof+6]
        
        # 确定实际DOF
        non_zero_mask = np.abs(joint_angles) > 1e-6
        if np.any(non_zero_mask):
            last_non_zero = np.where(non_zero_mask)[0][-1] + 1
            actual_dof = min(last_non_zero, self.max_dof)
            actual_dof = (actual_dof + 1) // 2 * 2
        else:
            actual_dof = 2
        
        n_segments = actual_dof // 2
        
        # 创建节点特征
        node_features_list = []
        for i in range(n_segments):
            joint1 = joint_angles[i * 2]
            joint2 = joint_angles[i * 2 + 1] if i * 2 + 1 < actual_dof else 0.0
            length = 0.21
            
            node_feature = torch.tensor([joint1, joint2, length], dtype=torch.float32)
            node_features_list.append(node_feature)
        
        node_features = torch.stack(node_features_list)
        
        # 简化：不需要邻接矩阵了！
        adjacency_matrix = torch.eye(n_segments, dtype=torch.float32)
        
        graph_state = SimplifiedGraphState(
            node_features=node_features.to(self.device),
            adjacency_matrix=adjacency_matrix.to(self.device),
            achieved_goal=achieved_goal.to(self.device),
            desired_goal=desired_goal.to(self.device)
        )
        
        return graph_state.to_tensor()
    
    def sample_action(self, her_tensor: torch.Tensor, get_log_prob: bool = False):
        """SAC兼容接口"""
        action_mean, action_log_std = self.forward(her_tensor)
        action_std = action_log_std.exp()
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()
        
        if get_log_prob:
            log_prob = dist.log_prob(action)
            if action.dim() > 1:
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob
        else:
            return action


def test_simple_structured_speed():
    """测试简单结构化网络速度"""
    print("⚡ 测试简单结构化网络速度")
    print("=" * 50)
    
    actor = SimpleStructuredActorNetwork(action_dim=6, dof_range=(2, 3), hidden_dim=128).cuda()
    her_obs = torch.randn(12).cuda()
    
    # Warmup
    for _ in range(10):
        _ = actor.forward(her_obs)
    
    # 测试速度
    import time
    start_time = time.time()
    
    for i in range(100):
        action_mean, action_log_std = actor.forward(her_obs)
    
    elapsed = time.time() - start_time
    print(f"100次前向传播 = {elapsed:.2f}秒")
    print(f"平均每次前向: {elapsed/100*1000:.2f}ms")
    print(f"vs Graph网络(2.42ms): {2.42/(elapsed/100*1000):.1f}x 速度提升!")
    
    # 测试功能
    action = actor.sample_action(her_obs)
    action_with_prob = actor.sample_action(her_obs, get_log_prob=True)
    print(f"✅ 功能测试: action{action.shape}, log_prob{action_with_prob[1].shape}")


if __name__ == "__main__":
    test_simple_structured_speed()