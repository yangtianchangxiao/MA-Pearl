#!/usr/bin/env python3
"""
简化的Graph设计演示
去掉节点中的位置信息，只保留关节角度和长度
位置信息完全通过achieved_goal和desired_goal提供
"""

import torch
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from graph_state_environment import GraphState


@dataclass
class SimplifiedGraphState:
    """简化的GraphState - 节点不包含位置信息"""
    node_features: torch.Tensor  # [n_nodes, 3] - [joint1, joint2, length]
    adjacency_matrix: torch.Tensor = None  # [n_nodes, n_nodes] 
    achieved_goal: torch.Tensor = None     # [3] - 当前末端位置
    desired_goal: torch.Tensor = None      # [3] - 目标末端位置
    
    def to_tensor(self) -> torch.Tensor:
        """序列化为tensor: [graph_data, achieved_goal(3), desired_goal(3)]"""
        
        # Graph部分序列化 
        node_features_flat = self.node_features.flatten()  # [n_nodes * 3]
        
        if self.adjacency_matrix is not None:
            adj_matrix_flat = self.adjacency_matrix.flatten()
            has_adjacency = 1.0
        else:
            adj_matrix_flat = torch.tensor([], dtype=torch.float32, device=self.node_features.device)
            has_adjacency = 0.0
        
        # 元数据：[n_nodes, node_feature_dim=3, has_adjacency]
        n_nodes = self.node_features.shape[0]
        metadata = torch.tensor([n_nodes, 3.0, has_adjacency], dtype=torch.float32, device=self.node_features.device)
        
        # 组合Graph tensor
        if adj_matrix_flat.numel() > 0:
            graph_tensor = torch.cat([node_features_flat, adj_matrix_flat, metadata])
        else:
            graph_tensor = torch.cat([node_features_flat, metadata])
        
        # 拼接Goals
        full_tensor = torch.cat([
            graph_tensor,
            self.achieved_goal,  # [3]
            self.desired_goal   # [3]
        ])
        
        return full_tensor
    
    @classmethod  
    def from_tensor(cls, full_tensor: torch.Tensor) -> 'SimplifiedGraphState':
        """从tensor反序列化"""
        
        # 提取goals
        achieved_goal = full_tensor[-6:-3]
        desired_goal = full_tensor[-3:]
        
        # 提取graph
        graph_tensor = full_tensor[:-6]
        
        # 解析元数据
        metadata = graph_tensor[-3:]
        n_nodes = int(metadata[0].item())
        node_feature_dim = int(metadata[1].item())  # 应该是3
        has_adjacency = metadata[2].item() > 0.5
        
        # 解析节点特征
        node_features_size = n_nodes * node_feature_dim  # n_nodes * 3
        node_features_flat = graph_tensor[:node_features_size]
        node_features = node_features_flat.reshape(n_nodes, node_feature_dim)
        
        # 解析邻接矩阵
        if has_adjacency:
            adj_matrix_size = n_nodes * n_nodes
            adj_matrix_flat = graph_tensor[node_features_size:node_features_size + adj_matrix_size]
            adjacency_matrix = adj_matrix_flat.reshape(n_nodes, n_nodes)
        else:
            adjacency_matrix = None
        
        return cls(
            node_features=node_features,
            adjacency_matrix=adjacency_matrix,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal
        )


class SimplifiedGraphActorNetwork(torch.nn.Module):
    """
    简化Graph网络 - 专门处理简化的node features
    
    核心思路：
    - Graph处理：joint_angles + segment_lengths → 结构理解
    - Goal处理：achieved_goal + desired_goal → 空间目标
    - 融合：结构理解 + 空间目标 → 动作决策
    """
    
    def __init__(self, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # 节点编码器（处理3维特征：joint1, joint2, length）
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_dim),  # 注意：输入维度是3！
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph注意力层
        self.graph_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Goal编码器（处理6维：achieved + desired）
        self.goal_encoder = torch.nn.Sequential(
            torch.nn.Linear(6, hidden_dim),
            torch.nn.ReLU(),  
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 融合层
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作输出
        self.action_mean = torch.nn.Linear(hidden_dim, action_dim)
        self.action_log_std = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, full_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        batch_size = full_tensor.shape[0] if full_tensor.dim() > 1 else 1
        if full_tensor.dim() == 1:
            full_tensor = full_tensor.unsqueeze(0)
        
        graph_features_list = []
        goal_features_list = []
        
        for i in range(batch_size):
            tensor = full_tensor[i]
            
            # 重建GraphState
            graph_state = SimplifiedGraphState.from_tensor(tensor)
            
            # 处理Graph结构（只有3维节点特征！）
            node_features = graph_state.node_features  # [n_nodes, 3]
            encoded_nodes = self.node_encoder(node_features)  # [n_nodes, hidden_dim]
            
            # Graph注意力
            attended_nodes, _ = self.graph_attention(
                encoded_nodes.unsqueeze(0),
                encoded_nodes.unsqueeze(0), 
                encoded_nodes.unsqueeze(0)
            )
            attended_nodes = attended_nodes.squeeze(0) + encoded_nodes  # 残差
            
            # Graph全局特征
            graph_feature = torch.mean(attended_nodes, dim=0)  # [hidden_dim]
            graph_features_list.append(graph_feature)
            
            # 处理Goals
            goals_combined = torch.cat([graph_state.achieved_goal, graph_state.desired_goal])  # [6]
            goal_feature = self.goal_encoder(goals_combined)  # [hidden_dim]
            goal_features_list.append(goal_feature)
        
        # Batch化
        graph_features = torch.stack(graph_features_list)  # [batch_size, hidden_dim]
        goal_features = torch.stack(goal_features_list)    # [batch_size, hidden_dim]
        
        # 融合
        combined = torch.cat([graph_features, goal_features], dim=1)  # [batch_size, hidden_dim*2]
        fused = self.fusion(combined)  # [batch_size, hidden_dim]
        
        # 动作输出
        action_mean = self.action_mean(fused)
        action_log_std = self.action_log_std(fused)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        return action_mean, action_log_std
    
    def sample_action(self, full_tensor: torch.Tensor, get_log_prob: bool = False):
        """采样动作"""
        action_mean, action_log_std = self.forward(full_tensor)
        action_std = action_log_std.exp()
        
        # 重参数化采样
        normal_dist = torch.distributions.Normal(action_mean, action_std)
        action = normal_dist.rsample()
        
        if get_log_prob:
            log_prob = normal_dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob
        else:
            return action


def test_simplified_design():
    """测试简化设计"""
    print("🧪 测试简化Graph设计")
    print("💡 节点特征：[joint1, joint2, length] - 去掉pos_x, pos_y, pos_z")
    print("🎯 空间信息：achieved_goal + desired_goal")
    print("=" * 60)
    
    # 创建测试数据
    for n_segments in [2, 3, 4]:
        print(f"\\n测试 {n_segments}节机械臂:")
        
        # 简化的节点特征：只有关节角度和长度
        node_features = torch.randn(n_segments, 3)  # [joint1, joint2, length]
        adjacency_matrix = torch.ones(n_segments, n_segments) - torch.eye(n_segments)  # 全连接-自连接
        achieved_goal = torch.randn(3)  # 当前末端位置
        desired_goal = torch.randn(3)   # 目标位置
        
        # 创建GraphState
        graph_state = SimplifiedGraphState(
            node_features=node_features,
            adjacency_matrix=adjacency_matrix,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal
        )
        
        print(f"   节点特征形状: {node_features.shape} (简化为3维)")
        print(f"   节点特征示例: {node_features[0].numpy()}")
        
        # 序列化测试
        tensor_state = graph_state.to_tensor()
        print(f"   序列化tensor长度: {tensor_state.shape[0]}")
        
        # 反序列化测试
        reconstructed = SimplifiedGraphState.from_tensor(tensor_state)
        print(f"   重建节点特征: {reconstructed.node_features.shape}")
        print(f"   重建goals: achieved{reconstructed.achieved_goal.shape}, desired{reconstructed.desired_goal.shape}")
        
        # 网络处理测试
        actor = SimplifiedGraphActorNetwork(action_dim=n_segments*2)
        action_mean, action_log_std = actor.forward(tensor_state.unsqueeze(0))
        print(f"   网络输出: mean{action_mean.shape}, log_std{action_log_std.shape}")
        
        # 验证重建精度
        reconstruction_error = torch.norm(node_features - reconstructed.node_features)
        print(f"   重建误差: {reconstruction_error.item():.6f}")
    
    print(f"\\n🎉 简化设计测试完成！")
    print(f"✅ 节点特征从6维简化到3维")
    print(f"✅ 位置信息完全由achieved_goal/desired_goal承载")
    print(f"✅ 网络必须从关节角度学会空间推理")
    print(f"✅ 更清晰的信息分离：结构 vs 空间")


if __name__ == "__main__":
    test_simplified_design()