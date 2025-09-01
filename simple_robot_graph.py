"""
简化版机械臂Graph Transformer
不依赖torch-scatter和torch-sparse，使用纯PyTorch实现
适用于机械臂强化学习
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Dict
import math


class SimpleGraphConv(nn.Module):
    """
    简化的图卷积层
    不依赖PyG的MessagePassing，使用纯PyTorch实现
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 节点自身特征变换
        self.node_proj = nn.Linear(in_dim, out_dim, bias=bias)
        # 邻居特征变换  
        self.neighbor_proj = nn.Linear(in_dim, out_dim, bias=bias)
        
    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        """
        Args:
            x: [num_nodes, in_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_weight: [num_edges] 边权重 (可选)
        """
        num_nodes = x.size(0)
        
        # 节点自身变换
        self_features = self.node_proj(x)
        
        # 邻居特征聚合
        neighbor_features = self.neighbor_proj(x)
        
        # 手动实现消息传递
        aggregated = torch.zeros_like(self_features)
        
        if edge_index.numel() > 0:
            src_nodes = edge_index[0]  # 源节点
            dst_nodes = edge_index[1]  # 目标节点
            
            # 聚合邻居特征
            for i in range(edge_index.size(1)):
                src, dst = src_nodes[i], dst_nodes[i]
                weight = 1.0 if edge_weight is None else edge_weight[i]
                aggregated[dst] += neighbor_features[src] * weight
        
        return self_features + aggregated


class SimpleGraphAttention(nn.Module):
    """
    简化的图注意力层
    实现类似GraphTransformer的注意力机制
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim必须能被num_heads整除"
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, edge_index: Tensor, node_mask: Tensor = None) -> Tensor:
        """
        Args:
            x: [num_nodes, in_dim] 节点特征
            edge_index: [2, num_edges] 边索引  
            node_mask: [num_nodes] 节点mask
        """
        num_nodes = x.size(0)
        
        # 投影到查询、键、值
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)  
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # 缩放因子
        scale = math.sqrt(self.head_dim)
        
        # 计算注意力权重
        attn_weights = torch.zeros(num_nodes, num_nodes, self.num_heads, device=x.device)
        
        if edge_index.numel() > 0:
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            # 计算边上的注意力
            for i in range(edge_index.size(1)):
                src, dst = src_nodes[i], dst_nodes[i]
                # q[dst] * k[src] 
                attn_score = (q[dst] * k[src]).sum(dim=-1) / scale  # [num_heads]
                attn_weights[dst, src] = attn_score
        
        # 添加自环（节点到自身的连接）
        for i in range(num_nodes):
            attn_score = (q[i] * k[i]).sum(dim=-1) / scale
            attn_weights[i, i] = attn_score
        
        # 应用节点mask
        if node_mask is not None:
            mask_expanded = node_mask.view(-1, 1, 1).expand(-1, num_nodes, self.num_heads)
            attn_weights = attn_weights.masked_fill(~mask_expanded, -1e9)
        
        # Softmax归一化
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值
        out = torch.zeros_like(v)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if attn_weights[i, j].sum().item() > 0:
                    out[i] += attn_weights[i, j].unsqueeze(-1) * v[j]
        
        # 重塑和投影输出
        out = out.view(num_nodes, -1)  # [num_nodes, out_dim]
        out = self.out_proj(out)
        
        return out


class RobotGraphLayer(nn.Module):
    """
    机械臂图神经网络层
    结合卷积和注意力
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.graph_conv = SimpleGraphConv(in_dim, out_dim)
        self.graph_attn = SimpleGraphAttention(in_dim, out_dim, num_heads, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        
    def forward(self, x: Tensor, edge_index: Tensor, node_mask: Tensor = None) -> Tensor:
        # 图卷积 + 残差连接
        conv_out = self.graph_conv(x, edge_index)
        
        # 如果输入输出维度不同，需要投影
        if x.size(-1) != conv_out.size(-1):
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(x.size(-1), conv_out.size(-1)).to(x.device)
            x_proj = self.input_proj(x)
        else:
            x_proj = x
            
        conv_out = self.norm1(conv_out + x_proj)
        
        # 图注意力 + 残差连接
        attn_out = self.graph_attn(conv_out, edge_index, node_mask)
        attn_out = self.norm2(attn_out + conv_out)
        
        # 前馈网络 + 残差连接
        ffn_out = self.ffn(attn_out)
        output = attn_out + ffn_out
        
        return output


class SimpleRobotGraphTransformer(nn.Module):
    """
    简化版机械臂Graph Transformer
    支持变长机械臂，纯PyTorch实现
    """
    def __init__(
        self,
        joint_feature_dim: int = 5,   # [angle, pos_x, pos_y, length, joint_type]
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_nodes: int = 10,          # 最大节点数
    ):
        super().__init__()
        
        self.joint_feature_dim = joint_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_nodes = max_nodes
        
        # 输入投影
        self.input_proj = nn.Linear(joint_feature_dim, hidden_dim)
        
        # Graph Transformer层
        self.layers = nn.ModuleList([
            RobotGraphLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features: Tensor, edge_index: Tensor, 
                node_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            node_features: [batch_size, max_nodes, feature_dim] 节点特征
            edge_index: [batch_size, 2, max_edges] 边索引
            node_mask: [batch_size, max_nodes] 节点mask
            
        Returns:
            node_out: [batch_size, max_nodes, hidden_dim] 节点输出
            graph_out: [batch_size, hidden_dim] 图级输出
        """
        batch_size, max_nodes, feature_dim = node_features.shape
        
        # 输入投影
        x = self.input_proj(node_features)  # [batch_size, max_nodes, hidden_dim]
        
        # 批量处理
        batch_node_outputs = []
        batch_graph_outputs = []
        
        for b in range(batch_size):
            # 获取当前批次的有效节点
            if node_mask is not None:
                valid_mask = node_mask[b]
                valid_nodes = valid_mask.sum().int().item()
            else:
                valid_nodes = max_nodes
                valid_mask = torch.ones(max_nodes, device=node_features.device)
            
            if valid_nodes == 0:
                # 没有有效节点，返回零
                node_out = torch.zeros(max_nodes, self.hidden_dim, device=x.device)
                graph_out = torch.zeros(self.hidden_dim, device=x.device)
            else:
                # 获取有效节点的特征和边
                x_batch = x[b, :valid_nodes]  # [valid_nodes, hidden_dim]
                if edge_index is not None and edge_index.dim() == 3:
                    edge_batch = edge_index[b]
                else:
                    edge_batch = self._create_chain_edges(valid_nodes, x.device)
                
                # 应用Graph Transformer层
                for layer in self.layers:
                    x_batch = layer(x_batch, edge_batch, valid_mask[:valid_nodes])
                
                # 图级池化 (平均池化)
                graph_out = x_batch.mean(dim=0)  # [hidden_dim]
                
                # 填充到max_nodes
                node_out = torch.zeros(max_nodes, self.hidden_dim, device=x.device)
                node_out[:valid_nodes] = x_batch
            
            batch_node_outputs.append(node_out)
            batch_graph_outputs.append(graph_out)
        
        # 堆叠批次结果
        node_outputs = torch.stack(batch_node_outputs)  # [batch_size, max_nodes, hidden_dim]
        graph_outputs = torch.stack(batch_graph_outputs)  # [batch_size, hidden_dim]
        
        # 输出投影
        node_outputs = self.output_proj(node_outputs)
        
        return node_outputs, graph_outputs
    
    def _create_chain_edges(self, num_nodes: int, device: torch.device) -> Tensor:
        """创建链式边连接 (用于机械臂)"""
        if num_nodes <= 1:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        
        edges = []
        # 创建双向边 (i-j 和 j-i)
        for i in range(num_nodes - 1):
            edges.extend([[i, i+1], [i+1, i]])
        
        return torch.tensor(edges, dtype=torch.long, device=device).T


class SimpleRobotGraphSAC(nn.Module):
    """
    基于简化Graph Transformer的SAC网络
    """
    def __init__(
        self,
        joint_feature_dim: int = 5,
        max_action_dim: int = 9,
        hidden_dim: int = 128,
        max_nodes: int = 10,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.max_action_dim = max_action_dim
        self.max_nodes = max_nodes
        
        # Graph Transformer主干
        self.graph_transformer = SimpleRobotGraphTransformer(
            joint_feature_dim=joint_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_nodes=max_nodes
        )
        
        # Actor网络 (输出动作分布参数)
        self.actor_mean = nn.Linear(hidden_dim, max_action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, max_action_dim)
        
        # Critic网络 (双Q)
        self.critic1 = nn.Sequential(
            nn.Linear(hidden_dim + max_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(hidden_dim + max_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 参数限制
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward_actor(self, node_features: Tensor, edge_index: Tensor = None, 
                     node_mask: Tensor = None, action_mask: Tensor = None):
        """Actor前向传播"""
        _, graph_features = self.graph_transformer(node_features, edge_index, node_mask)
        
        # 计算动作分布参数
        mean = self.actor_mean(graph_features)
        log_std = self.actor_log_std(graph_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        # 应用动作mask
        if action_mask is not None:
            mean = mean * action_mask
            log_std = log_std * action_mask + (1 - action_mask) * self.log_std_min
        
        return mean, log_std
    
    def forward_critic(self, node_features: Tensor, actions: Tensor, 
                      edge_index: Tensor = None, node_mask: Tensor = None):
        """Critic前向传播"""
        _, graph_features = self.graph_transformer(node_features, edge_index, node_mask)
        
        # 拼接状态和动作
        state_action = torch.cat([graph_features, actions], dim=1)
        
        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)
        
        return q1, q2


def create_robot_graph_data(joint_angles: np.ndarray, segment_lengths: np.ndarray = None,
                           joint_types: np.ndarray = None) -> Dict[str, torch.Tensor]:
    """
    创建机械臂图数据 (简化版)
    
    Returns:
        dict包含: node_features, edge_index, node_mask
    """
    n_joints = len(joint_angles)
    
    if segment_lengths is None:
        segment_lengths = np.ones(n_joints) * 0.21
    
    if joint_types is None:
        joint_types = np.zeros(n_joints)  # 默认软体关节
    
    # 计算关节位置 (简化运动学)
    positions = np.zeros((n_joints, 2))
    cumulative_angle = 0.0
    current_pos = np.array([0.0, 0.0])
    
    for i in range(n_joints):
        cumulative_angle += joint_angles[i]
        length = segment_lengths[i] if i < len(segment_lengths) else segment_lengths[-1]
        current_pos += length * np.array([np.cos(cumulative_angle), np.sin(cumulative_angle)])
        positions[i] = current_pos
    
    # 节点特征: [angle, pos_x, pos_y, length, joint_type]
    node_features = []
    for i in range(n_joints):
        length = segment_lengths[i] if i < len(segment_lengths) else segment_lengths[-1]
        features = [joint_angles[i], positions[i, 0], positions[i, 1], length, joint_types[i]]
        node_features.append(features)
    
    # 边索引 (链式连接)
    edge_list = []
    for i in range(n_joints - 1):
        edge_list.extend([[i, i+1], [i+1, i]])  # 双向边
    
    # 转换为tensor
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_list, dtype=torch.long).T if edge_list else torch.empty(2, 0, dtype=torch.long)
    node_mask = torch.ones(n_joints, dtype=torch.bool)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,  
        'node_mask': node_mask
    }


# 测试代码
if __name__ == "__main__":
    print("=== 简化版机械臂Graph Transformer测试 ===")
    
    # 创建测试数据
    joint_angles_3dof = np.array([0.1, 0.2, 0.3])
    joint_angles_6dof = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    # 创建图数据
    data_3dof = create_robot_graph_data(joint_angles_3dof)
    data_6dof = create_robot_graph_data(joint_angles_6dof)
    
    print(f"3DOF图: {data_3dof['node_features'].shape[0]}个节点")
    print(f"6DOF图: {data_6dof['node_features'].shape[0]}个节点")
    
    # 创建批次数据 (填充到相同大小)
    max_nodes = 6
    batch_size = 2
    
    # 批量节点特征
    batch_node_features = torch.zeros(batch_size, max_nodes, 5)
    batch_node_features[0, :3] = data_3dof['node_features']
    batch_node_features[1, :6] = data_6dof['node_features']
    
    # 批量节点mask
    batch_node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    batch_node_mask[0, :3] = True
    batch_node_mask[1, :6] = True
    
    # 动作mask
    batch_action_mask = torch.zeros(batch_size, 9)
    batch_action_mask[0, :3] = 1  # 3DOF
    batch_action_mask[1, :6] = 1  # 6DOF
    
    print(f"\n批量数据形状:")
    print(f"  节点特征: {batch_node_features.shape}")
    print(f"  节点mask: {batch_node_mask.shape}")
    print(f"  动作mask: {batch_action_mask.shape}")
    
    # 创建模型
    model = SimpleRobotGraphSAC(
        joint_feature_dim=5,
        max_action_dim=9,
        hidden_dim=128,
        max_nodes=max_nodes
    )
    
    # 测试前向传播
    print(f"\n=== 前向传播测试 ===")
    
    # Actor输出
    mean, log_std = model.forward_actor(
        batch_node_features, 
        node_mask=batch_node_mask,
        action_mask=batch_action_mask
    )
    print(f"Actor输出:")
    print(f"  Mean形状: {mean.shape}")
    print(f"  Log std形状: {log_std.shape}")
    
    # Critic输出
    test_actions = torch.randn(batch_size, 9) * batch_action_mask
    q1, q2 = model.forward_critic(
        batch_node_features,
        test_actions,
        node_mask=batch_node_mask
    )
    print(f"Critic输出:")
    print(f"  Q1形状: {q1.shape}")
    print(f"  Q2形状: {q2.shape}")
    
    # 验证mask效果
    print(f"\n=== Mask效果验证 ===")
    print(f"3DOF样本动作均值: {mean[0]}")
    print(f"  前3维 (有效): {mean[0][:3]}")
    print(f"  后6维 (应为0): {mean[0][3:]}")
    print(f"  后6维全为0: {torch.allclose(mean[0][3:], torch.zeros(6))}")
    
    print(f"\n6DOF样本动作均值: {mean[1]}")
    print(f"  前6维 (有效): {mean[1][:6]}")
    print(f"  后3维 (应为0): {mean[1][6:]}")
    print(f"  后3维全为0: {torch.allclose(mean[1][6:], torch.zeros(3))}")
    
    print("\n✅ 简化版Graph Transformer测试成功!")