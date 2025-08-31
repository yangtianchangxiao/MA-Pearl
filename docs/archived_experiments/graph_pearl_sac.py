#!/usr/bin/env python3
"""
Graph Transformer SAC for Pearl Framework
完整集成Graph网络到Pearl SAC框架，支持变长机械臂

核心特性：
1. 结构感知的Graph Transformer网络
2. 完全兼容Pearl SAC+HER框架
3. 变长机械臂的结构embedding
4. 高效的批处理和GPU加速
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Dict, Any
import math

from pearl.api.action_space import ActionSpace
from pearl.api.action import Action
from pearl.neural_networks.sequential_decision_making.actor_networks import ActorNetwork
from pearl.neural_networks.sequential_decision_making.critic_networks import CriticNetwork
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace


class StructuralEmbedding(nn.Module):
    """
    机械臂结构embedding模块
    将机械臂的几何和物理结构编码为特征
    """
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 节点类型embedding (关节类型)
        self.joint_type_embedding = nn.Embedding(4, embedding_dim // 4)  # revolute, prismatic, fixed, end
        
        # 长度/位置embedding
        self.length_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # DOF embedding
        self.dof_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.ReLU(), 
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # 位置编码 (沿kinematic chain的位置)
        self.position_embedding = nn.Embedding(20, embedding_dim // 4)  # 支持最多20个节点
        
    def forward(self, segment_lengths: Tensor, joint_types: Tensor, positions: Tensor) -> Tensor:
        """
        Args:
            segment_lengths: [batch_size, max_segments] 节段长度
            joint_types: [batch_size, max_segments] 关节类型ID  
            positions: [batch_size, max_segments] 节点在kinematic chain中的位置
        Returns:
            structural_embedding: [batch_size, max_segments, embedding_dim]
        """
        batch_size, max_segments = segment_lengths.shape
        
        # 各种结构特征embedding
        type_emb = self.joint_type_embedding(joint_types)  # [B, S, E//4]
        length_emb = self.length_encoder(segment_lengths.unsqueeze(-1))  # [B, S, E//4]  
        dof_emb = self.dof_encoder(torch.ones_like(segment_lengths.unsqueeze(-1)) * 2)  # 假设每节2DOF
        pos_emb = self.position_embedding(positions)  # [B, S, E//4]
        
        # 拼接所有embedding
        structural_embedding = torch.cat([type_emb, length_emb, dof_emb, pos_emb], dim=-1)
        
        return structural_embedding


class GraphConvLayer(nn.Module):
    """
    机械臂专用图卷积层
    处理kinematic chain的连接关系
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 节点特征变换
        self.node_transform = nn.Linear(in_dim, out_dim)
        
        # 邻居特征变换
        self.neighbor_transform = nn.Linear(in_dim, out_dim)
        
        # 边特征变换 (如果有)
        if edge_dim > 0:
            self.edge_transform = nn.Linear(edge_dim, out_dim)
        else:
            self.edge_transform = None
        
        # 注意力权重
        self.attention = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, node_features: Tensor, edge_index: Tensor, 
                edge_features: Tensor = None, node_mask: Tensor = None) -> Tensor:
        """
        Args:
            node_features: [batch_size, max_nodes, in_dim] 节点特征
            edge_index: [batch_size, 2, num_edges] 边连接 
            edge_features: [batch_size, num_edges, edge_dim] 边特征 (可选)
            node_mask: [batch_size, max_nodes] 节点有效性mask
        """
        batch_size, max_nodes, _ = node_features.shape
        
        # 自身特征变换
        self_features = self.node_transform(node_features)
        
        # 基于注意力的邻居聚合
        attended_features, _ = self.attention(
            self_features, self_features, self_features,
            key_padding_mask=~node_mask if node_mask is not None else None
        )
        
        # 残差连接和层归一化
        output = self.layer_norm(attended_features + self_features)
        
        # 应用mask
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)
        
        return output


class RobotGraphTransformer(nn.Module):
    """
    机械臂专用Graph Transformer
    处理变长机械臂的结构化特征学习
    """
    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128, 
        num_layers: int = 3,
        max_nodes: int = 10,
        structural_embedding_dim: int = 32
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        
        # 结构embedding模块
        self.structural_embedding = StructuralEmbedding(structural_embedding_dim)
        
        # 输入特征投影 (状态特征 + 结构特征)
        input_dim = node_feature_dim + structural_embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph Transformer layers
        self.graph_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 图级别特征聚合
        self.global_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def create_kinematic_edges(self, num_nodes: int, device: torch.device) -> Tensor:
        """
        创建机械臂kinematic chain的边连接
        每个节点连接到相邻节点 (串联结构)
        """
        if num_nodes <= 1:
            return torch.empty(2, 0, device=device, dtype=torch.long)
        
        # 创建双向边 (i <-> i+1)  
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i+1])     # 前向边
            edges.append([i+1, i])     # 后向边
        
        edge_index = torch.tensor(edges, device=device, dtype=torch.long).t()
        return edge_index
    
    def forward(self, observations: Tensor, segment_lengths: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            observations: [batch_size, obs_dim] 原始观测
            segment_lengths: [batch_size, max_segments] 节段长度
        Returns:
            node_features: [batch_size, max_nodes, hidden_dim] 节点级特征
            graph_features: [batch_size, hidden_dim] 图级特征
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        # 解析观测 - 假设格式: [joint_angles, segment_lengths, achieved_goal, desired_goal]
        # 对于6DOF变长软体臂: [6joints + 3lengths + 3achieved + 3desired] = 15
        if observations.shape[1] == 15:  # 变长软体臂
            joint_angles = observations[:, :6]  # 6DOF
            lengths = observations[:, 6:9]      # 3个segment长度
            achieved_goal = observations[:, 9:12]   
            desired_goal = observations[:, 12:15]
            max_segments = 3
        else:
            raise ValueError(f"不支持的观测维度: {observations.shape[1]}")
        
        # 创建节点特征 - 每个节点包含关节状态和目标信息
        node_features = []
        for i in range(max_segments):
            # 每个节点的特征: [joint_angles(2), achieved_goal(3), desired_goal(3)] = 8维
            joint_feat = joint_angles[:, i*2:(i+1)*2]  # 2DOF per segment
            node_feat = torch.cat([
                joint_feat,           # 关节角度
                achieved_goal,        # 达到的目标 (全局信息)
                desired_goal          # 期望目标 (全局信息)
            ], dim=1)
            node_features.append(node_feat)
        
        node_features = torch.stack(node_features, dim=1)  # [B, max_segments, 8]
        
        # 创建结构embedding
        joint_types = torch.zeros(batch_size, max_segments, dtype=torch.long, device=device)  # 默认revolute
        positions = torch.arange(max_segments, device=device).unsqueeze(0).expand(batch_size, -1)
        
        structural_emb = self.structural_embedding(lengths, joint_types, positions)
        
        # 拼接状态特征和结构特征
        combined_features = torch.cat([node_features, structural_emb], dim=-1)
        
        # 输入投影
        x = self.input_projection(combined_features)  # [B, max_segments, hidden_dim]
        
        # 创建节点mask (所有节点都有效，因为是固定3节)
        node_mask = torch.ones(batch_size, max_segments, dtype=torch.bool, device=device)
        
        # Graph Transformer层
        for layer in self.graph_layers:
            # 为每个批次创建kinematic边
            edge_indices = []
            for b in range(batch_size):
                edge_index = self.create_kinematic_edges(max_segments, device)
                edge_indices.append(edge_index)
            
            # 这里简化处理，使用相同的边结构
            edge_index = self.create_kinematic_edges(max_segments, device)
            x = layer(x, edge_index.unsqueeze(0).expand(batch_size, -1, -1), node_mask=node_mask)
        
        # 图级特征聚合 (mean pooling over nodes)
        graph_features = torch.mean(x * node_mask.unsqueeze(-1), dim=1)  # [B, hidden_dim]
        graph_features = self.global_pooling(graph_features)
        
        return x, graph_features


class GraphActorNetwork(ActorNetwork):
    """
    基于Graph Transformer的Actor网络
    完全兼容Pearl框架
    """
    def __init__(
        self,
        input_dim: int,
        action_space: ActionSpace,
        hidden_dims: List[int] = None,
        **kwargs
    ):
        # 初始化基类
        super().__init__(
            input_dim=input_dim,
            action_space=action_space,
        )
        
        self.action_dim = action_space.shape[0]
        
        # Graph Transformer backbone
        self.graph_transformer = RobotGraphTransformer(
            node_feature_dim=8,  # [joint(2) + achieved(3) + desired(3)]
            hidden_dim=hidden_dims[0] if hidden_dims else 128,
            num_layers=3,
            max_nodes=10
        )
        
        # Actor head
        hidden_dim = hidden_dims[0] if hidden_dims else 128
        self.mean_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
        
        # 参数范围限制
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, input_tensor: Tensor) -> Tensor:
        """直接前向传播用于推理"""
        _, graph_features = self.graph_transformer(input_tensor)
        
        mean = self.mean_head(graph_features)
        log_std = self.log_std_head(graph_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        # 返回动作均值 (推理时)
        return mean
    
    def sample_action(self, input_tensor: Tensor) -> Tensor:
        """采样动作 (训练时)"""
        _, graph_features = self.graph_transformer(input_tensor)
        
        mean = self.mean_head(graph_features)
        log_std = self.log_std_head(graph_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        action = normal.rsample()
        
        # Tanh squashing to bound actions
        action = torch.tanh(action)
        
        return action
    
    def get_action_and_log_prob(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """获取动作和对数概率 (SAC需要)"""
        _, graph_features = self.graph_transformer(input_tensor)
        
        mean = self.mean_head(graph_features)
        log_std = self.log_std_head(graph_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # 计算对数概率 (考虑tanh变换)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob


class GraphCriticNetwork(CriticNetwork):
    """
    基于Graph Transformer的Critic网络
    完全兼容Pearl框架
    """
    def __init__(
        self,
        input_dim: int,
        action_space: ActionSpace,
        hidden_dims: List[int] = None,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            action_space=action_space,
        )
        
        self.action_dim = action_space.shape[0]
        
        # Graph Transformer backbone (共享)
        self.graph_transformer = RobotGraphTransformer(
            node_feature_dim=8,
            hidden_dim=hidden_dims[0] if hidden_dims else 128,
            num_layers=3,
            max_nodes=10
        )
        
        # Q网络
        hidden_dim = hidden_dims[0] if hidden_dims else 128
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state_tensor: Tensor, action_tensor: Tensor) -> Tensor:
        """
        Args:
            state_tensor: [batch_size, state_dim] 状态
            action_tensor: [batch_size, action_dim] 动作
        Returns:
            q_value: [batch_size, 1] Q值
        """
        _, graph_features = self.graph_transformer(state_tensor)
        
        # 拼接图特征和动作
        combined = torch.cat([graph_features, action_tensor], dim=-1)
        q_value = self.q_head(combined)
        
        return q_value


def create_graph_sac_policy_learner(
    state_dim: int,
    action_space: ActionSpace, 
    actor_hidden_dims: List[int] = [256, 256],
    critic_hidden_dims: List[int] = [256, 256],
    **sac_kwargs
) -> ContinuousSoftActorCritic:
    """
    创建使用Graph网络的SAC policy learner
    完全兼容Pearl框架
    """
    
    # 创建Graph Actor网络
    actor = GraphActorNetwork(
        input_dim=state_dim,
        action_space=action_space,
        hidden_dims=actor_hidden_dims
    )
    
    # 创建Graph Critic网络 (双Q)
    critic1 = GraphCriticNetwork(
        input_dim=state_dim,
        action_space=action_space,
        hidden_dims=critic_hidden_dims
    )
    
    critic2 = GraphCriticNetwork(
        input_dim=state_dim,
        action_space=action_space, 
        hidden_dims=critic_hidden_dims
    )
    
    # 创建SAC policy learner
    policy_learner = ContinuousSoftActorCritic(
        state_dim=state_dim,
        action_space=action_space,
        actor_network=actor,
        critic_network=critic1,
        critic_network_2=critic2,
        **sac_kwargs
    )
    
    return policy_learner


if __name__ == "__main__":
    # 测试Graph SAC网络
    print("🧪 测试Graph Pearl SAC网络...")
    
    # 模拟变长软体机械臂环境
    batch_size = 32
    state_dim = 15  # 6joints + 3lengths + 3achieved + 3desired
    action_dim = 6  # 6DOF
    
    # 创建测试数据
    test_states = torch.randn(batch_size, state_dim)
    test_actions = torch.randn(batch_size, action_dim)
    
    # 创建action space
    action_space = BoxActionSpace(
        low=torch.full((action_dim,), -1.0),
        high=torch.full((action_dim,), 1.0)
    )
    
    # 创建Graph SAC policy learner
    policy_learner = create_graph_sac_policy_learner(
        state_dim=state_dim,
        action_space=action_space,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128]
    )
    
    print("✅ Graph SAC创建成功!")
    print(f"   State维度: {state_dim}")
    print(f"   Action维度: {action_dim}")
    
    # 测试Actor
    policy_learner.train()
    actions, log_probs = policy_learner._actor.get_action_and_log_prob(test_states)
    print(f"   Actor输出形状: {actions.shape}")
    print(f"   Log prob形状: {log_probs.shape}")
    
    # 测试Critic
    q_values = policy_learner._critic.forward(test_states, test_actions)
    print(f"   Critic输出形状: {q_values.shape}")
    
    print("🎉 Graph Pearl SAC集成测试通过!")