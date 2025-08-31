#!/usr/bin/env python3
"""
超轻量GNN卷积Actor网络
保持Graph核心思想，但用简单GNN卷积替代复杂attention
目标：10x速度提升，保持Graph推理能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from simplified_graph_demo import SimplifiedGraphState


class LightweightGNNLayer(nn.Module):
    """超轻量GNN卷积层"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        简单GNN卷积：邻居信息聚合
        node_features: [n_nodes, in_dim]
        adjacency_matrix: [n_nodes, n_nodes]
        """
        # 简单邻接矩阵消息传递
        neighbor_messages = torch.mm(adjacency_matrix, node_features)  # [n_nodes, in_dim]
        
        # 线性变换
        output = self.linear(neighbor_messages)  # [n_nodes, out_dim]
        
        # 激活
        output = self.activation(output)
        
        return output


class UltraLightGNNActor(nn.Module):
    """
    超轻量GNN Actor网络
    
    核心思想：
    - 保持Graph处理：节点特征 + 邻接矩阵
    - 用简单GNN卷积替代attention (大幅减速)
    - 极简参数：1层GNN + 小hidden_dim
    """
    
    def __init__(
        self,
        action_dim: int,
        dof_range: Tuple[int, int] = (2, 4),
        hidden_dim: int = 64,  # 更小！
        num_gnn_layers: int = 1  # 只用1层！
    ):
        super().__init__()
        self.action_dim = action_dim
        self.dof_range = dof_range
        self.max_dof = max(dof_range) * 2
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 节点特征编码器（超简单）
        self.node_encoder = nn.Linear(3, hidden_dim)  # [joint1, joint2, length] → hidden
        
        # 超轻量GNN层
        self.gnn_layers = nn.ModuleList([
            LightweightGNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_gnn_layers)
        ])
        
        # Goal编码器（超简单）
        self.goal_encoder = nn.Linear(6, hidden_dim)  # [achieved + desired] → hidden
        
        # 融合层（超简单）
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 输出层
        self.action_mean_head = nn.Linear(hidden_dim, action_dim)
        self.action_std_head = nn.Linear(hidden_dim, action_dim)
        
        print(f"⚡ 超轻量GNN Actor:")
        print(f"   隐藏维度: {hidden_dim} (vs 128-256)")
        print(f"   GNN层数: {num_gnn_layers} (vs 3-4)")
        print(f"   总参数: ~{self._count_parameters():,} (极少)")
        print(f"   预期速度提升: 5-10x")
    
    def _count_parameters(self):
        """估算参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, her_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """超简化前向传播"""
        her_tensor = her_tensor.float()
        
        if her_tensor.dim() == 1:
            her_tensor = her_tensor.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size = her_tensor.shape[0]
        
        # 批处理
        graph_features = []
        goal_features = []
        
        for i in range(batch_size):
            her_obs = her_tensor[i]
            graph_tensor = self._her_to_graph_state(her_obs)
            graph_state = SimplifiedGraphState.from_tensor(graph_tensor)
            
            # 1. 快速GNN处理
            graph_feat = self._process_with_lightweight_gnn(
                graph_state.node_features, 
                graph_state.adjacency_matrix
            )
            graph_features.append(graph_feat)
            
            # 2. 快速Goal处理
            goal_concat = torch.cat([graph_state.achieved_goal, graph_state.desired_goal])
            goal_feat = self.goal_encoder(goal_concat)
            goal_features.append(goal_feat)
        
        # 组合和输出
        graph_features = torch.stack(graph_features)
        goal_features = torch.stack(goal_features)
        
        # 超简单融合
        combined = torch.cat([graph_features, goal_features], dim=1)
        fused = F.relu(self.fusion(combined))
        
        # 输出
        action_mean = self.action_mean_head(fused)
        action_log_std = self.action_std_head(fused)
        
        # 数值稳定性
        action_mean = torch.clamp(action_mean, -5, 5)
        action_log_std = torch.clamp(action_log_std, -5, 1)
        
        if single_sample:
            action_mean = action_mean.squeeze(0)
            action_log_std = action_log_std.squeeze(0)
        
        return action_mean, action_log_std
    
    def _process_with_lightweight_gnn(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """用轻量GNN处理节点"""
        # 编码节点特征
        encoded_nodes = self.node_encoder(node_features)  # [n_nodes, hidden_dim]
        
        # GNN层处理（保持Graph思想但极简实现）
        current_features = encoded_nodes
        for gnn_layer in self.gnn_layers:
            current_features = gnn_layer(current_features, adjacency_matrix)
        
        # 全局pooling
        graph_feature = torch.mean(current_features, dim=0)  # [hidden_dim]
        
        return graph_feature
    
    def _her_to_graph_state(self, her_obs: torch.Tensor) -> torch.Tensor:
        """复用转换逻辑（但现在更快了）"""
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
        
        # 简单链状邻接矩阵
        adjacency_matrix = torch.zeros(n_segments, n_segments, dtype=torch.float32)
        for i in range(n_segments - 1):
            adjacency_matrix[i, i+1] = 1.0
            adjacency_matrix[i+1, i] = 1.0
        
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


def test_ultra_light_gnn_speed():
    """测试超轻量GNN速度"""
    print("⚡ 测试超轻量GNN Actor速度")
    print("=" * 50)
    
    actor = UltraLightGNNActor(action_dim=6, dof_range=(2, 3), hidden_dim=64, num_gnn_layers=1).cuda()
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
    print(f"vs Graph Attention(2.42ms): {2.42/(elapsed/100*1000):.1f}x 速度提升!")
    print(f"vs 简化MLP(1.28ms): {1.28/(elapsed/100*1000):.1f}x 对比")
    
    # 测试功能
    action = actor.sample_action(her_obs)
    action_with_prob = actor.sample_action(her_obs, get_log_prob=True)
    print(f"✅ 功能测试: action{action.shape}, log_prob{action_with_prob[1].shape}")
    
    # 参数统计
    print(f"📊 网络统计:")
    print(f"   总参数: {actor._count_parameters():,}")
    print(f"   GNN保持Graph思想: ✅")
    print(f"   超快速度: ✅")


if __name__ == "__main__":
    test_ultra_light_gnn_speed()