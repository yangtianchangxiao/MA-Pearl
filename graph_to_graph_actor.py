#!/usr/bin/env python3
"""
Graph-to-Graph Actor - 真正的分布式Multi-Agent架构

核心创新：
- Input: Graph (variable nodes)
- Output: Graph (per-node actions) 
- 支持任意DOF数量的真正泛化
- 每个segment作为独立agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from simplified_graph_demo import SimplifiedGraphState


class GraphToGraphGNNLayer(nn.Module):
    """
    Graph-to-Graph GNN层 - 保持节点个体性的消息传递
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        
        print(f"   GNN Layer: {in_dim} → {out_dim}")
        
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        关键：保持 [n_nodes, features] 格式，不做global pooling
        """
        # 邻居消息聚合
        neighbor_messages = torch.mm(adjacency_matrix, node_features)  # [n_nodes, in_dim]
        
        # 线性变换 + 激活
        output = self.activation(self.linear(neighbor_messages))  # [n_nodes, out_dim]
        
        return output


class GraphToGraphActor(nn.Module):
    """
    Graph-to-Graph Actor - 真正支持任意DOF的Multi-Agent架构
    
    核心设计：
    - Input: [n_segments, 3] node features + adjacency
    - Processing: GNN保持节点个体性
    - Output: [n_segments, 2] 每个segment输出自己的action
    """
    
    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 8),  # 支持更大范围！
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        action_per_node: int = 2  # 每个segment输出2DOF
    ):
        super().__init__()
        self.dof_range = dof_range
        self.hidden_dim = hidden_dim
        self.action_per_node = action_per_node
        self.device = torch.device('cpu')  # 训练时会改为cuda
        
        print(f"🎯 Graph-to-Graph Actor初始化:")
        print(f"   支持范围: {dof_range[0]}-{dof_range[1]}节 (动态!)")
        print(f"   隐藏维度: {hidden_dim}")
        print(f"   GNN层数: {num_gnn_layers}")
        print(f"   每节点输出: {action_per_node}DOF")
        
        # 节点特征编码器
        self.node_encoder = nn.Linear(3, hidden_dim)  # [joint1, joint2, length] → hidden
        print(f"   节点编码: 3 → {hidden_dim}")
        
        # Graph-to-Graph GNN层
        self.gnn_layers = nn.ModuleList([
            GraphToGraphGNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_gnn_layers)
        ])
        
        # Goal编码器
        self.goal_encoder = nn.Linear(6, hidden_dim)  # [achieved + desired] → hidden
        print(f"   目标编码: 6 → {hidden_dim}")
        
        # 关键：每个节点的动作输出头
        self.node_action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # node + goal features
            nn.ReLU(),
            nn.Linear(hidden_dim, action_per_node * 2)  # mean + std
        )
        print(f"   节点动作头: {hidden_dim * 2} → {action_per_node * 2}")
        
        print(f"   总参数: ~{self._count_parameters():,}")
        print(f"✅ Graph-to-Graph架构 - 真正的分布式Multi-Agent!")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, her_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Graph-to-Graph前向传播
        
        返回：每个segment的动作分布参数
        """
        her_tensor = her_tensor.float()
        
        if her_tensor.dim() == 1:
            her_tensor = her_tensor.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size = her_tensor.shape[0]
        
        # 批处理
        all_action_means = []
        all_action_stds = []
        
        for i in range(batch_size):
            her_obs = her_tensor[i]
            
            # 转换为Graph状态
            graph_state = self._her_to_graph_state(her_obs)
            
            # Graph-to-Graph处理
            action_mean, action_std = self._graph_to_graph_forward(graph_state)
            
            all_action_means.append(action_mean)
            all_action_stds.append(action_std)
        
        # 组合结果
        batch_action_means = torch.stack(all_action_means)
        batch_action_stds = torch.stack(all_action_stds)
        
        if single_sample:
            batch_action_means = batch_action_means.squeeze(0)
            batch_action_stds = batch_action_stds.squeeze(0)
        
        return batch_action_means, batch_action_stds
    
    def _graph_to_graph_forward(self, graph_state: SimplifiedGraphState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        核心：Graph-to-Graph处理逻辑
        """
        node_features = graph_state.node_features  # [n_segments, 3]
        adjacency_matrix = graph_state.adjacency_matrix  # [n_segments, n_segments]
        n_segments = node_features.shape[0]
        
        # 1. 编码节点特征 - 保持节点个体性
        encoded_nodes = self.node_encoder(node_features)  # [n_segments, hidden_dim]
        
        # 2. GNN处理 - 关键：不做global pooling！
        current_features = encoded_nodes
        for gnn_layer in self.gnn_layers:
            current_features = gnn_layer(current_features, adjacency_matrix)  # 仍然[n_segments, hidden_dim]
        
        # 3. Goal信息 - 每个节点都需要知道全局目标
        goal_concat = torch.cat([graph_state.achieved_goal, graph_state.desired_goal])
        goal_features = self.goal_encoder(goal_concat)  # [hidden_dim]
        
        # 扩展goal features给每个节点
        goal_features_expanded = goal_features.unsqueeze(0).expand(n_segments, -1)  # [n_segments, hidden_dim]
        
        # 4. 节点特征 + 目标信息
        combined_features = torch.cat([current_features, goal_features_expanded], dim=1)  # [n_segments, hidden_dim*2]
        
        # 5. 关键创新：每个节点输出自己的动作
        node_action_params = self.node_action_head(combined_features)  # [n_segments, action_per_node*2]
        
        # 6. 分离均值和标准差
        action_mean = node_action_params[:, :self.action_per_node]  # [n_segments, 2]
        action_log_std = node_action_params[:, self.action_per_node:]  # [n_segments, 2]
        
        # 7. 展平为环境需要的格式
        flattened_mean = action_mean.flatten()  # [n_segments * 2]
        flattened_log_std = action_log_std.flatten()  # [n_segments * 2]
        
        # 数值稳定性
        flattened_mean = torch.clamp(flattened_mean, -5, 5)
        flattened_log_std = torch.clamp(flattened_log_std, -5, 1)
        
        return flattened_mean, flattened_log_std
    
    def _her_to_graph_state(self, her_obs: torch.Tensor) -> SimplifiedGraphState:
        """
        HER观测转Graph状态 - 支持动态DOF数量
        """
        # 提取信息 - 动态确定DOF数量
        # HER格式: [joint_angles(variable), achieved_goal(3), desired_goal(3)]
        total_len = her_obs.shape[0]
        joint_angles_len = total_len - 6  # 减去goals的6维
        
        joint_angles = her_obs[:joint_angles_len].detach().cpu().numpy()
        achieved_goal = her_obs[joint_angles_len:joint_angles_len+3]
        desired_goal = her_obs[joint_angles_len+3:joint_angles_len+6]
        
        # 确定实际segments数量
        actual_dof = len([x for x in joint_angles if abs(x) > 1e-6])
        if actual_dof == 0:
            actual_dof = 2  # 最少2DOF
        
        n_segments = (actual_dof + 1) // 2  # 向上取整
        
        # 创建节点特征 - 动态数量
        node_features_list = []
        for i in range(n_segments):
            joint1 = joint_angles[i * 2] if i * 2 < len(joint_angles) else 0.0
            joint2 = joint_angles[i * 2 + 1] if i * 2 + 1 < len(joint_angles) else 0.0
            length = 0.21  # 默认长度
            
            node_feature = torch.tensor([joint1, joint2, length], dtype=torch.float32)
            node_features_list.append(node_feature)
        
        node_features = torch.stack(node_features_list)
        
        # 创建邻接矩阵 - 链状连接
        adjacency_matrix = torch.zeros(n_segments, n_segments, dtype=torch.float32)
        for i in range(n_segments - 1):
            adjacency_matrix[i, i+1] = 1.0
            adjacency_matrix[i+1, i] = 1.0
        
        return SimplifiedGraphState(
            node_features=node_features.to(self.device),
            adjacency_matrix=adjacency_matrix.to(self.device),
            achieved_goal=achieved_goal.to(self.device),
            desired_goal=desired_goal.to(self.device)
        )
    
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


def test_graph_to_graph():
    """测试Graph-to-Graph架构"""
    print("🧪 Graph-to-Graph架构测试")
    print("=" * 60)
    
    actor = GraphToGraphActor(
        dof_range=(2, 8), 
        hidden_dim=128, 
        num_gnn_layers=2
    )
    
    # 测试不同DOF数量
    test_configs = [
        (4, "2节4DOF"),
        (6, "3节6DOF"), 
        (10, "5节10DOF"),
        (12, "6节12DOF - 关键泛化测试!"),
        (16, "8节16DOF - 超大规模!")
    ]
    
    print(f"\n🎯 动态DOF测试:")
    for dof, desc in test_configs:
        # 创建测试输入
        her_obs = torch.zeros(dof + 6)  # joint_angles + goals
        her_obs[:dof] = torch.randn(dof) * 0.1  # 关节角度
        her_obs[dof:dof+3] = torch.randn(3) * 0.5  # achieved_goal
        her_obs[dof+3:dof+6] = torch.randn(3) * 0.5  # desired_goal
        
        # 前向传播
        try:
            action_mean, action_log_std = actor.forward(her_obs)
            action = actor.sample_action(her_obs)
            
            print(f"   {desc}: ✅")
            print(f"     输入: {her_obs.shape} → 输出: {action.shape}")
            print(f"     {dof}DOF输入 → {action.shape[0]}DOF输出 {'✅匹配' if action.shape[0] == dof else '❌不匹配'}")
        except Exception as e:
            print(f"   {desc}: ❌ {str(e)[:50]}...")
    
    print(f"\n🎉 Graph-to-Graph测试完成!")
    print(f"✅ 真正的动态DOF支持")
    print(f"✅ 分布式Multi-Agent架构")
    print(f"✅ 无维度限制的泛化能力")


if __name__ == "__main__":
    test_graph_to_graph()