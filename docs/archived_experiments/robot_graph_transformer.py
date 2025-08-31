"""
机械臂专用Graph Transformer - 优化版本
基于原始gnn_transformer.py，针对机械臂场景简化和优化

核心特性：
1. 支持可变节点数量（不同DOF的机械臂）
2. 机械臂关节类型embedding
3. 保留mask机制兼容批处理
4. 简化的消息传递和注意力机制
5. 与Pearl SAC框架兼容
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union, Optional

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops, to_dense_batch, softmax
from torch_geometric.typing import OptPairTensor, Adj, OptTensor

import math
from graph_utils import init, orthogonal_init, create_mlp


class RobotJointEmbedding(MessagePassing):
    """
    机械臂关节专用嵌入层
    支持不同类型的关节：软体关节、硬体关节、末端执行器等
    """
    def __init__(
        self,
        joint_feature_dim: int,      # 关节特征维度 (angle, position等)
        joint_types: int = 4,        # 关节类型数量 (软体/硬体/末端/基座)
        embedding_dim: int = 64,     # 嵌入维度
        hidden_dim: int = 128,       # 隐藏层维度
        edge_dim: int = 2,           # 边特征维度 (length, constraint)
        add_self_loop: bool = True,
    ):
        super().__init__(aggr="mean")  # 使用mean聚合，对机械臂更合适
        
        self.joint_feature_dim = joint_feature_dim
        self.joint_types = joint_types
        self.embedding_dim = embedding_dim
        self.add_self_loop = add_self_loop
        
        # 关节类型嵌入 (软体=0, 硬体=1, 末端=2, 基座=3)
        self.joint_type_embedding = nn.Embedding(joint_types, embedding_dim)
        
        # 特征投影层
        input_dim = joint_feature_dim + embedding_dim + edge_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None):
        """
        x: [num_nodes, joint_feature_dim + 1]  # 最后一维是joint_type
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim]
        """
        if self.add_self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j: Tensor, edge_attr: OptTensor):
        # 分离关节特征和类型
        joint_features = x_j[:, :-1]  # [num_nodes, joint_feature_dim]
        joint_types = x_j[:, -1].long()  # [num_nodes]
        
        # 获取类型嵌入
        type_embedding = self.joint_type_embedding(joint_types)  # [num_nodes, embedding_dim]
        
        # 拼接特征
        if edge_attr is not None:
            node_features = torch.cat([joint_features, type_embedding, edge_attr], dim=1)
        else:
            node_features = torch.cat([joint_features, type_embedding], dim=1)
            
        return self.feature_proj(node_features)


class RobotTransformerConv(TransformerConv):
    """
    机械臂专用Transformer卷积层
    简化的注意力机制，针对链式图结构优化
    """
    def __init__(self, in_channels, out_channels, heads=4, concat=True, 
                 dropout=0.1, edge_dim=None, bias=True):
        super().__init__(in_channels, out_channels, heads, concat, 
                        beta=False, dropout=dropout, edge_dim=edge_dim, bias=bias)
    
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: int) -> Tensor:
        
        # 边特征融入key
        if self.lin_edge is not None and edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr
        
        # 计算注意力分数
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 应用注意力到value
        out = value_j
        if edge_attr is not None:
            out = out + edge_attr
            
        return out * alpha.view(-1, self.heads, 1)


class RobotGraphTransformer(nn.Module):
    """
    机械臂专用Graph Transformer网络
    支持可变DOF，保留mask机制
    """
    def __init__(
        self,
        joint_feature_dim: int = 4,      # 关节特征维度 [angle, pos_x, pos_y, length]
        joint_types: int = 4,            # 关节类型数量
        embedding_dim: int = 64,         # 类型嵌入维度
        hidden_dim: int = 128,           # 隐藏层维度
        num_heads: int = 4,              # 注意力头数
        num_layers: int = 3,             # Transformer层数
        edge_dim: int = 2,               # 边特征维度
        output_dim: int = 128,           # 输出维度
        dropout: float = 0.1,
        global_pool: str = "mean",       # 全局池化方法
    ):
        super().__init__()
        
        self.joint_feature_dim = joint_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.global_pool = global_pool
        
        # 关节嵌入层
        self.joint_embedding = RobotJointEmbedding(
            joint_feature_dim=joint_feature_dim,
            joint_types=joint_types,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim
        )
        
        # Transformer层序列
        self.transformer_layers = nn.ModuleList()
        in_dim = hidden_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim
            layer = RobotTransformerConv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=num_heads,
                concat=True,  # 拼接多头输出
                dropout=dropout,
                edge_dim=edge_dim
            )
            self.transformer_layers.append(layer)
            in_dim = out_dim * num_heads  # 拼接后的维度
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data_batch: Batch) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            data_batch: PyG Batch对象，包含x, edge_index, edge_attr, batch
            
        Returns:
            node_features: [total_nodes, output_dim] 节点特征
            graph_features: [batch_size, output_dim] 图级特征  
            mask: [batch_size, max_nodes] 节点mask
        """
        x, edge_index, edge_attr, batch = data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        
        # 1. 关节嵌入
        x = self.joint_embedding(x, edge_index, edge_attr)
        
        # 2. Transformer层
        for layer in self.transformer_layers:
            x = F.relu(layer(x, edge_index, edge_attr))
        
        # 3. 输出投影
        node_features = self.output_proj(x)
        
        # 4. 全局池化获得图级特征
        if self.global_pool == "mean":
            from torch_geometric.nn import global_mean_pool
            graph_features = global_mean_pool(node_features, batch)
        elif self.global_pool == "max":
            from torch_geometric.nn import global_max_pool
            graph_features = global_max_pool(node_features, batch)
        elif self.global_pool == "add":
            from torch_geometric.nn import global_add_pool
            graph_features = global_add_pool(node_features, batch)
        else:
            raise ValueError(f"不支持的池化方法: {self.global_pool}")
        
        # 5. 生成密集表示和mask (用于变长支持)
        node_features_dense, mask = to_dense_batch(node_features, batch)
        
        return node_features_dense, graph_features, mask


class RobotGraphSAC(nn.Module):
    """
    基于Graph Transformer的机械臂SAC网络
    兼容Pearl SAC框架
    """
    def __init__(
        self,
        joint_feature_dim: int = 4,
        action_dim: int = 6,
        joint_types: int = 4,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Graph Transformer主干
        self.graph_transformer = RobotGraphTransformer(
            joint_feature_dim=joint_feature_dim,
            joint_types=joint_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=hidden_dim
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic网络 (双Q)
        self.critic1 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward_actor(self, data_batch: Batch) -> Tensor:
        """Actor前向传播"""
        _, graph_features, _ = self.graph_transformer(data_batch)
        return self.actor(graph_features)
    
    def forward_critic(self, data_batch: Batch, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """Critic前向传播"""
        _, graph_features, _ = self.graph_transformer(data_batch)
        state_action = torch.cat([graph_features, actions], dim=1)
        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)
        return q1, q2


def create_robot_graph(joint_angles: np.ndarray, 
                      segment_lengths: np.ndarray = None,
                      joint_types: np.ndarray = None) -> Data:
    """
    将机械臂状态转换为图表示
    
    Args:
        joint_angles: [n_joints] 关节角度
        segment_lengths: [n_segments] 段长度 (可选)
        joint_types: [n_joints] 关节类型 (可选，默认全部为软体关节)
    
    Returns:
        Data: PyG图数据对象
    """
    n_joints = len(joint_angles)
    
    # 默认关节类型 (0=软体关节)
    if joint_types is None:
        joint_types = np.zeros(n_joints, dtype=np.long)
    
    # 默认段长度
    if segment_lengths is None:
        segment_lengths = np.ones(n_joints) * 0.21
    
    # 计算关节位置 (简化的运动学)
    positions = np.zeros((n_joints, 2))
    cumulative_angle = 0.0
    current_pos = np.array([0.0, 0.0])
    
    for i in range(n_joints):
        cumulative_angle += joint_angles[i]
        if i < len(segment_lengths):
            length = segment_lengths[i]
        else:
            length = segment_lengths[-1]
            
        current_pos += length * np.array([np.cos(cumulative_angle), np.sin(cumulative_angle)])
        positions[i] = current_pos
    
    # 节点特征: [angle, pos_x, pos_y, length, joint_type]
    node_features = []
    for i in range(n_joints):
        length = segment_lengths[i] if i < len(segment_lengths) else segment_lengths[-1]
        features = [joint_angles[i], positions[i, 0], positions[i, 1], length, joint_types[i]]
        node_features.append(features)
    
    # 边索引 (链式连接)
    edge_index = []
    edge_attr = []
    
    for i in range(n_joints - 1):
        # 双向边
        edge_index.extend([[i, i+1], [i+1, i]])
        
        # 边特征: [length, constraint_type]
        length = segment_lengths[i] if i < len(segment_lengths) else segment_lengths[-1]
        edge_attr.extend([[length, 0], [length, 0]])  # constraint_type=0表示固定连接
    
    # 转换为tensor
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.empty((0, 2), dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# 使用示例和测试
if __name__ == "__main__":
    # 创建测试数据
    joint_angles_3dof = np.array([0.1, 0.2, 0.3])
    joint_angles_6dof = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    segment_lengths = np.array([0.21, 0.18, 0.25])
    
    # 创建图数据
    graph_3dof = create_robot_graph(joint_angles_3dof, segment_lengths[:3])
    graph_6dof = create_robot_graph(joint_angles_6dof, np.tile(segment_lengths, 2))
    
    # 创建批次 (不同DOF的机械臂)
    batch = Batch.from_data_list([graph_3dof, graph_6dof])
    
    # 创建网络
    model = RobotGraphSAC(
        joint_feature_dim=4,  # angle, pos_x, pos_y, length
        action_dim=6,
        hidden_dim=128
    )
    
    # 测试前向传播
    print("=== 机械臂Graph Transformer测试 ===")
    print(f"批次信息:")
    print(f"  图数量: {batch.num_graphs}")
    print(f"  总节点数: {batch.num_nodes}")
    print(f"  节点特征维度: {batch.x.shape}")
    print(f"  边数量: {batch.edge_index.shape[1]}")
    
    # Actor输出
    actions = model.forward_actor(batch)
    print(f"\nActor输出形状: {actions.shape}")
    
    # Critic输出
    test_actions = torch.randn(batch.num_graphs, 6)
    q1, q2 = model.forward_critic(batch, test_actions)
    print(f"Critic输出形状: Q1={q1.shape}, Q2={q2.shape}")
    
    print("\n✅ Graph Transformer测试成功!")