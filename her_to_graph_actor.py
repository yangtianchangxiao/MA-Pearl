#!/usr/bin/env python3
"""
HER兼容的Graph Actor网络
接受HER格式输入，内部转换为GraphState格式处理
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from optimized_graph_network import OptimizedGraphActorNetwork
from simplified_graph_demo import SimplifiedGraphState


class HERGraphActorNetwork(nn.Module):
    """
    HER格式到Graph处理的Actor网络
    
    输入: HER格式 [joint_angles(max_dof), achieved_goal(3), desired_goal(3)]
    内部: 转换为GraphState进行Graph网络处理
    输出: 标准SAC兼容的action_mean, action_log_std
    """
    
    def __init__(
        self,
        action_dim: int,
        dof_range: Tuple[int, int] = (2, 4),
        hidden_dim: int = 128,
        num_graph_layers: int = 3,
        num_attention_heads: int = 4
    ):
        super().__init__()
        self.action_dim = action_dim
        self.dof_range = dof_range
        self.max_dof = max(dof_range) * 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Graph网络核心
        self.graph_actor = OptimizedGraphActorNetwork(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers,
            num_attention_heads=num_attention_heads
        ).to(self.device)
        
        print(f"🔗 HER-Graph Actor网络")
        print(f"   输入: HER格式 [joint_angles({self.max_dof}), goals(6)]")
        print(f"   处理: 转换为GraphState → Graph网络")
        print(f"   输出: 标准SAC动作分布")
    
    def forward(self, her_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: HER format → GraphState → Graph processing
        """
        # 确保float32类型
        her_tensor = her_tensor.float()
        
        # 处理batch维度
        if her_tensor.dim() == 1:
            her_tensor = her_tensor.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size = her_tensor.shape[0]
        
        # 转换每个样本到GraphState
        graph_tensors = []
        for i in range(batch_size):
            her_obs = her_tensor[i]
            graph_tensor = self._her_to_graph_state(her_obs)
            graph_tensors.append(graph_tensor)
        
        # 批处理GraphState tensors
        if batch_size == 1:
            graph_input = graph_tensors[0]
        else:
            # 对于batch，我们需要分别处理每个GraphState
            # 因为它们可能有不同的长度
            action_means = []
            action_log_stds = []
            
            for graph_tensor in graph_tensors:
                action_mean, action_log_std = self.graph_actor.forward(graph_tensor)
                action_means.append(action_mean)
                action_log_stds.append(action_log_std)
            
            action_mean = torch.stack(action_means)
            action_log_std = torch.stack(action_log_stds)
            
            if single_sample:
                action_mean = action_mean.squeeze(0)
                action_log_std = action_log_std.squeeze(0)
            
            return action_mean, action_log_std
        
        # 单样本处理
        action_mean, action_log_std = self.graph_actor.forward(graph_input)
        
        if single_sample:
            action_mean = action_mean.squeeze(0) if action_mean.dim() > 1 else action_mean
            action_log_std = action_log_std.squeeze(0) if action_log_std.dim() > 1 else action_log_std
        
        return action_mean, action_log_std
    
    def _her_to_graph_state(self, her_obs: torch.Tensor) -> torch.Tensor:
        """
        将HER格式转换为GraphState tensor
        
        输入: [joint_angles(max_dof), achieved_goal(3), desired_goal(3)]
        输出: GraphState tensor
        """
        # 提取组件
        joint_angles = her_obs[:self.max_dof].detach().cpu().numpy()
        achieved_goal = her_obs[self.max_dof:self.max_dof+3]
        desired_goal = her_obs[self.max_dof+3:self.max_dof+6]
        
        # 确定实际DOF（去除padding的零）
        non_zero_mask = np.abs(joint_angles) > 1e-6
        if np.any(non_zero_mask):
            last_non_zero = np.where(non_zero_mask)[0][-1] + 1
            actual_dof = min(last_non_zero, self.max_dof)
            # 确保是偶数（每个segment有2个关节）
            actual_dof = (actual_dof + 1) // 2 * 2
        else:
            actual_dof = 2  # 最小DOF
        
        n_segments = actual_dof // 2
        
        # 创建节点特征 [joint1, joint2, length]
        node_features_list = []
        for i in range(n_segments):
            joint1 = joint_angles[i * 2]
            joint2 = joint_angles[i * 2 + 1] if i * 2 + 1 < actual_dof else 0.0
            length = 0.21  # 默认segment长度
            
            node_feature = torch.tensor([joint1, joint2, length], dtype=torch.float32)
            node_features_list.append(node_feature)
        
        node_features = torch.stack(node_features_list)
        
        # 创建邻接矩阵（链状连接）
        adjacency_matrix = torch.zeros(n_segments, n_segments, dtype=torch.float32)
        for i in range(n_segments - 1):
            adjacency_matrix[i, i+1] = 1.0
            adjacency_matrix[i+1, i] = 1.0
        
        # 创建GraphState - 确保所有tensor在同一设备上
        graph_state = SimplifiedGraphState(
            node_features=node_features.to(self.device),
            adjacency_matrix=adjacency_matrix.to(self.device),
            achieved_goal=achieved_goal.to(self.device),
            desired_goal=desired_goal.to(self.device)
        )
        
        return graph_state.to_tensor()
    
    def sample_action(self, her_tensor: torch.Tensor, get_log_prob: bool = False):
        """SAC兼容的采样接口"""
        action_mean, action_log_std = self.forward(her_tensor)
        action_std = action_log_std.exp()
        
        # 重参数化采样
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


def test_her_graph_actor():
    """测试HER-Graph Actor网络"""
    print("🧪 测试HER-Graph Actor网络")
    print("=" * 50)
    
    # 创建网络
    actor = HERGraphActorNetwork(action_dim=8, dof_range=(2, 4))
    
    # 测试不同场景
    test_cases = [
        {
            "name": "2节机械臂(4DOF)",
            "her_obs": torch.tensor([0.1, -0.2, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0,  # joint_angles
                                   0.5, -0.3, 0.2,  # achieved_goal
                                   0.1, 0.2, 0.4], dtype=torch.float32)  # desired_goal
        },
        {
            "name": "4节机械臂(8DOF)",
            "her_obs": torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4, -0.2,  # joint_angles
                                   0.5, -0.3, 0.2,  # achieved_goal
                                   0.1, 0.2, 0.4], dtype=torch.float32)  # desired_goal
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        her_obs = test_case['her_obs']
        
        # 单样本测试
        action_mean, action_log_std = actor.forward(her_obs)
        print(f"   单样本输出: mean{action_mean.shape}, log_std{action_log_std.shape}")
        
        # 批处理测试
        batch_obs = her_obs.unsqueeze(0).repeat(3, 1)
        batch_mean, batch_log_std = actor.forward(batch_obs)
        print(f"   批处理输出: mean{batch_mean.shape}, log_std{batch_log_std.shape}")
        
        # 采样测试
        action = actor.sample_action(her_obs)
        action_with_prob = actor.sample_action(her_obs, get_log_prob=True)
        print(f"   采样动作: {action.shape}")
        print(f"   带概率采样: action{action_with_prob[0].shape}, log_prob{action_with_prob[1].shape}")
    
    print(f"\n✅ HER-Graph Actor网络测试完成!")
    print(f"✅ HER format → GraphState转换成功")
    print(f"✅ Graph网络处理成功")
    print(f"✅ SAC兼容接口成功")


if __name__ == "__main__":
    test_her_graph_actor()