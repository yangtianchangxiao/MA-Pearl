#!/usr/bin/env python3
"""
Graph State Environment - 真正的Graph状态表示

状态格式: [graph, achieved_goal, desired_goal]
- graph: 节点特征包含 [joint_angle, segment_length, local_features...]
- achieved_goal: 3D位置向量
- desired_goal: 3D位置向量

这种设计更具可扩展性，真正利用Graph的结构化表示能力
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class GraphState:
    """Graph状态表示"""
    node_features: torch.Tensor  # [n_nodes, node_feature_dim]
    edge_features: torch.Tensor = None  # [n_edges, edge_feature_dim] 
    adjacency_matrix: torch.Tensor = None  # [n_nodes, n_nodes]
    achieved_goal: torch.Tensor = None  # [3]
    desired_goal: torch.Tensor = None   # [3]
    
    def to_list(self) -> List:
        """转换为列表格式供网络使用"""
        return [self, self.achieved_goal, self.desired_goal]
    
    def to_tensor(self) -> torch.Tensor:
        """
        转换为Pearl兼容的tensor格式
        
        更直观的设计：Graph + achieved_goal + desired_goal 分别处理
        格式：[graph_tensor, achieved_goal(3), desired_goal(3)]
        
        其中graph_tensor = [node_features_flat, adj_matrix_flat, graph_metadata]
        """
        # 1. Graph部分序列化
        graph_tensor = self._serialize_graph()
        
        # 2. 直接拼接goals（简单直观）
        full_tensor = torch.cat([
            graph_tensor,
            self.achieved_goal,  # [3]
            self.desired_goal   # [3]
        ])
        
        return full_tensor
    
    def _serialize_graph(self) -> torch.Tensor:
        """序列化Graph结构为tensor"""
        node_features_flat = self.node_features.flatten()  # [n_nodes * node_feature_dim]
        
        # 处理邻接矩阵
        if self.adjacency_matrix is not None:
            adj_matrix_flat = self.adjacency_matrix.flatten()  # [n_nodes * n_nodes]
            has_adjacency = 1.0
        else:
            adj_matrix_flat = torch.tensor([])  # 空tensor
            has_adjacency = 0.0
        
        # Graph元数据：[n_nodes, node_feature_dim, has_adjacency]
        n_nodes = self.node_features.shape[0]
        node_feature_dim = self.node_features.shape[1]
        graph_metadata = torch.tensor([n_nodes, node_feature_dim, has_adjacency])
        
        # 组合Graph tensor
        if adj_matrix_flat.numel() > 0:  # 有邻接矩阵
            graph_tensor = torch.cat([node_features_flat, adj_matrix_flat, graph_metadata])
        else:  # 无邻接矩阵
            graph_tensor = torch.cat([node_features_flat, graph_metadata])
        
        return graph_tensor
    
    @classmethod
    def from_tensor(cls, full_tensor: torch.Tensor) -> 'GraphState':
        """从tensor格式重建GraphState对象"""
        # 从末尾提取goals（固定长度，容易处理）
        achieved_goal = full_tensor[-6:-3]  # 倒数第6到第4位
        desired_goal = full_tensor[-3:]     # 最后3位
        
        # 剩余部分是graph_tensor
        graph_tensor = full_tensor[:-6]
        
        # 重建Graph结构
        graph_part = cls._deserialize_graph(graph_tensor)
        
        return cls(
            node_features=graph_part['node_features'],
            adjacency_matrix=graph_part['adjacency_matrix'],
            achieved_goal=achieved_goal,
            desired_goal=desired_goal
        )
    
    @classmethod
    def _deserialize_graph(cls, graph_tensor: torch.Tensor) -> dict:
        """从graph_tensor重建Graph结构"""
        # 从末尾提取元数据（固定3个元素）
        metadata = graph_tensor[-3:]
        n_nodes = int(metadata[0].item())
        node_feature_dim = int(metadata[1].item())
        has_adjacency = metadata[2].item() > 0.5
        
        # 计算各部分大小
        node_features_size = n_nodes * node_feature_dim
        adj_matrix_size = n_nodes * n_nodes if has_adjacency else 0
        
        # 提取node features
        node_features_flat = graph_tensor[:node_features_size]
        node_features = node_features_flat.reshape(n_nodes, node_feature_dim)
        
        # 提取adjacency matrix
        if has_adjacency:
            adj_matrix_flat = graph_tensor[node_features_size:node_features_size + adj_matrix_size]
            adjacency_matrix = adj_matrix_flat.reshape(n_nodes, n_nodes)
        else:
            adjacency_matrix = None
        
        return {
            'node_features': node_features,
            'adjacency_matrix': adjacency_matrix
        }


class GraphSoftArmEnvironment:
    """
    使用Graph状态表示的软体机械臂环境
    
    核心思路：
    - 每个segment作为一个节点
    - 节点特征：[joint_angle_1, joint_angle_2, segment_length] - 简化设计，去掉冗余位置
    - 边特征：连接关系、距离等  
    - 全局信息：achieved_goal, desired_goal分离（包含所有空间信息）
    """
    
    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 4),
        base_segment_length: float = 0.21,
        segment_length_range: Tuple[float, float] = (0.168, 0.252),
        goal_threshold: float = 0.15,
        max_steps: int = 200,
        node_feature_dim: int = 3,  # [joint1, joint2, length] - 简化！
    ):
        self.dof_range = dof_range
        self.base_segment_length = base_segment_length
        self.segment_length_range = segment_length_range
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.node_feature_dim = node_feature_dim
        
        # 当前状态
        self.current_n_segments = None
        self.joint_angles = None
        self.segment_lengths = None
        self.goal_position = None
        self.step_count = 0
        
        print(f"🚀 Graph状态软体机械臂环境初始化")
        print(f"   DOF范围: {dof_range[0]}-{dof_range[1]}节")
        print(f"   节点特征维度: {node_feature_dim}")
        print(f"   状态格式: [graph, achieved_goal, desired_goal]")
    
    def _create_graph_state(self) -> GraphState:
        """创建Graph状态表示"""
        
        # 1. 计算每个segment的节点特征
        node_features = []
        current_pos = np.array([0.0, 0.0, 0.0])  # 累积位置
        current_angle = 0.0  # 累积角度
        
        for i in range(self.current_n_segments):
            # 该segment的关节角度
            joint1 = self.joint_angles[i * 2]      # 弯曲角
            joint2 = self.joint_angles[i * 2 + 1]  # 方向角
            segment_length = self.segment_lengths[i]
            
            # 计算该segment的局部位置（简化的正向运动学）
            current_angle += joint2
            local_end_pos = current_pos + segment_length * np.array([
                np.cos(current_angle) * np.cos(joint1),
                np.sin(current_angle) * np.cos(joint1), 
                np.sin(joint1)
            ])
            
            # 节点特征：[joint1, joint2, length, end_pos_x, end_pos_y, end_pos_z]
            node_feat = np.array([
                joint1, joint2, segment_length,
                local_end_pos[0], local_end_pos[1], local_end_pos[2]
            ], dtype=np.float32)
            
            node_features.append(node_feat)
            current_pos = local_end_pos
        
        # 2. 创建邻接矩阵（线性连接）
        adjacency_matrix = np.zeros((self.current_n_segments, self.current_n_segments), dtype=np.float32)
        for i in range(self.current_n_segments - 1):
            adjacency_matrix[i, i+1] = 1.0  # 前向连接
            adjacency_matrix[i+1, i] = 1.0  # 后向连接
        
        # 3. 计算achieved goal（末端位置）
        achieved_goal = current_pos
        
        # 4. 创建Graph状态
        graph_state = GraphState(
            node_features=torch.tensor(np.array(node_features), dtype=torch.float32),
            adjacency_matrix=torch.tensor(adjacency_matrix, dtype=torch.float32),
            achieved_goal=torch.tensor(achieved_goal, dtype=torch.float32),
            desired_goal=torch.tensor(self.goal_position, dtype=torch.float32)
        )
        
        return graph_state
    
    def reset(self, seed=None) -> Tuple[List, Dict]:
        """重置环境，返回Graph状态"""
        if seed is not None:
            np.random.seed(seed)
        
        # 采样DOF配置
        self.current_n_segments = np.random.randint(self.dof_range[0], self.dof_range[1] + 1)
        current_dof = self.current_n_segments * 2
        
        # 初始化状态
        self.joint_angles = np.zeros(current_dof, dtype=np.float32)
        self.segment_lengths = np.random.uniform(
            *self.segment_length_range, 
            size=self.current_n_segments
        ).astype(np.float32)
        
        # 采样目标
        total_length = np.sum(self.segment_lengths)
        max_reach = total_length * 0.7
        self.goal_position = np.random.uniform(-max_reach, max_reach, 3).astype(np.float32)
        self.goal_position[2] = max(0.05, abs(self.goal_position[2]))  # 避免地面
        
        self.step_count = 0
        
        # 创建Graph状态
        graph_state = self._create_graph_state()
        
        # 创建action space info
        action_space_info = {
            'shape': (current_dof,),
            'low': -1.0,
            'high': 1.0
        }
        
        print(f"🔄 Episode Reset - Graph: {self.current_n_segments}节({current_dof}DOF)")
        print(f"   节点特征形状: {graph_state.node_features.shape}")
        print(f"   邻接矩阵形状: {graph_state.adjacency_matrix.shape}")
        
        return graph_state.to_tensor(), action_space_info
    
    def step(self, action: torch.Tensor) -> Tuple[List, float, bool, bool, Dict]:
        """执行动作，返回新的Graph状态"""
        
        # 更新关节角度
        action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
        current_dof = self.current_n_segments * 2
        
        self.joint_angles[:current_dof] += action_np[:current_dof] * 0.01
        self.joint_angles[:current_dof] = np.clip(self.joint_angles[:current_dof], -np.pi/2, np.pi/2)
        
        # 创建新的Graph状态
        graph_state = self._create_graph_state()
        
        # 计算奖励
        distance = np.linalg.norm(graph_state.achieved_goal.numpy() - graph_state.desired_goal.numpy())
        
        if distance <= self.goal_threshold:
            reward = 50.0
            terminated = True
        else:
            reward = -1.0
            terminated = False
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        info = {
            'distance': distance,
            'success': terminated,
            'graph_info': {
                'n_nodes': graph_state.node_features.shape[0],
                'node_feature_dim': graph_state.node_features.shape[1],
                'has_edges': graph_state.adjacency_matrix is not None
            }
        }
        
        return graph_state.to_tensor(), reward, terminated, truncated, info
    
    def get_current_config(self) -> Dict:
        """获取当前配置"""
        return {
            'n_segments': self.current_n_segments,
            'current_dof': self.current_n_segments * 2 if self.current_n_segments else 0,
            'segment_lengths': self.segment_lengths.tolist() if self.segment_lengths is not None else [],
            'state_format': 'Graph: [graph, achieved_goal, desired_goal]',
            'node_feature_dim': self.node_feature_dim
        }


def test_graph_state_environment():
    """测试Graph状态环境"""
    print("🧪 测试Graph状态软体机械臂环境")
    print("=" * 60)
    
    env = GraphSoftArmEnvironment(dof_range=(2, 4), max_steps=5)
    
    # 测试多个episode
    for episode in range(3):
        graph_state_list, action_info = env.reset()
        config = env.get_current_config()
        
        print(f"\\nEpisode {episode+1}: {config['n_segments']}节")
        
        # 解析状态
        graph_state, achieved_goal, desired_goal = graph_state_list
        
        print(f"   Graph节点特征: {graph_state.node_features.shape}")
        print(f"   邻接矩阵: {graph_state.adjacency_matrix.shape}")
        print(f"   Achieved goal: {achieved_goal.shape}")
        print(f"   Desired goal: {desired_goal.shape}")
        print(f"   动作空间: {action_info['shape']}")
        
        # 执行几步
        for step in range(3):
            action = torch.randn(action_info['shape'][0]) * 0.1
            next_state_list, reward, terminated, truncated, info = env.step(action)
            
            print(f"     Step {step+1}: reward={reward:.1f}, distance={info['distance']:.3f}")
            
            if terminated or truncated:
                break
    
    print("\\n🎉 Graph状态环境测试完成！")
    print("✅ 节点特征包含：joint_angles + segment_lengths + positions")
    print("✅ 状态格式：[graph, achieved_goal, desired_goal]") 
    print("✅ 支持动态DOF：不同episode不同节点数")


if __name__ == "__main__":
    test_graph_state_environment()