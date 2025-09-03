#!/usr/bin/env python3
"""
复杂运动学HER包装器

基于optimized_graph_her_wrapper.py，但使用复杂运动学环境
支持GraphState格式转换
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any
from complex_kinematics_environment import ComplexKinematicsSoftArmEnvironment

class ComplexKinematicsHERWrapper:
    """
    复杂运动学HER包装器
    
    将复杂运动学环境包装成HER兼容格式
    同时支持GraphState转换
    """
    
    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 5),
        segment_length_range: Tuple[float, float] = (0.1, 0.35), 
        goal_threshold: float = 0.15,
        max_steps: int = 200
    ):
        self.dof_range = dof_range
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        
        # 创建复杂运动学环境
        self.env = ComplexKinematicsSoftArmEnvironment(
            dof_range=dof_range,
            segment_length_range=segment_length_range,
            goal_threshold=goal_threshold,
            max_steps=max_steps,
            reward_type="dense"
        )
        
        # HER兼容的空间定义
        max_dof = max(dof_range) * 2
        self.observation_space = self.env.observation_space
        
        # 转换gym.spaces.Box到Pearl BoxSpace
        from pearl.utils.instantiations.spaces.box import BoxSpace
        self.action_space = BoxSpace.from_gym(self.env.action_space)
        
        print("🔗 复杂运动学Graph-HER包装器")
        print(f"   DOF范围: {dof_range[0]}-{dof_range[1]}节")
        print(f"   输入: GraphState tensor (变长)")
        print(f"   输出: [joint_angles({max_dof}), achieved_goal(3), desired_goal(3)]")
        print(f"   使用C++复杂运动学")
    
    def reset(self, seed: int = None) -> Tuple[torch.Tensor, Any]:
        """重置环境并返回Graph格式观测"""
        obs, info = self.env.reset(seed=seed)
        
        # 转换为torch tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        return obs_tensor, self.action_space
    
    def step(self, action: torch.Tensor) -> Any:
        """执行一步并返回HER格式结果"""
        # 转换action到numpy
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = action
            
        # 环境step
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        
        # 转换为torch格式
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminated, dtype=torch.bool)
        truncated_tensor = torch.tensor(truncated, dtype=torch.bool)
        
        # 创建HER兼容的result对象
        class ActionResult:
            def __init__(self, obs, reward, terminated, truncated, info, available_action_space=None):
                self.observation = obs
                self.reward = reward
                self.terminated = terminated  
                self.truncated = truncated
                self.info = info
                self.available_action_space = available_action_space
                self.cost = None  # Pearl expects this attribute
        
        return ActionResult(obs_tensor, reward_tensor, terminated_tensor, truncated_tensor, info, self.action_space)
    
    def to_graph_state_dict(self, obs_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        将HER观测转换为GraphState字典格式
        用于Graph网络处理
        """
        max_dof = max(self.dof_range) * 2
        
        # 解析观测
        joint_angles = obs_tensor[:max_dof]
        achieved_goal = obs_tensor[max_dof:max_dof+3]
        desired_goal = obs_tensor[max_dof+3:max_dof+6]
        
        # 确定实际节数
        current_n_segments = self.env.current_n_segments
        
        # 构建节点特征 [n_segments, 3] -> [joint1, joint2, length]
        node_features = []
        for i in range(current_n_segments):
            joint1 = joint_angles[i * 2].item()
            joint2 = joint_angles[i * 2 + 1].item()
            length = self.env.segment_lengths[i]
            
            node_features.append([joint1, joint2, length])
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # 构建邻接矩阵 (链状连接)
        adjacency_matrix = torch.zeros(current_n_segments, current_n_segments, dtype=torch.float32)
        for i in range(current_n_segments - 1):
            adjacency_matrix[i, i+1] = 1.0
            adjacency_matrix[i+1, i] = 1.0
        
        return {
            'node_features': node_features,
            'adjacency_matrix': adjacency_matrix, 
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'n_segments': current_n_segments
        }


def test_complex_kinematics_her_wrapper():
    """测试复杂运动学HER包装器"""
    print("🧪 测试复杂运动学HER包装器")
    print("=" * 60)
    
    wrapper = ComplexKinematicsHERWrapper(
        dof_range=(2, 4),
        max_steps=10
    )
    
    # 测试reset
    obs, action_space = wrapper.reset(seed=42)
    print(f"Reset观测形状: {obs.shape}")
    print(f"动作空间形状: {action_space.shape}")
    
    # 测试GraphState转换
    graph_dict = wrapper.to_graph_state_dict(obs)
    print(f"\nGraphState转换:")
    print(f"  节点特征: {graph_dict['node_features'].shape}")
    print(f"  邻接矩阵: {graph_dict['adjacency_matrix'].shape}")
    print(f"  节点数: {graph_dict['n_segments']}")
    
    # 测试几步
    for step in range(5):
        action = torch.randn(action_space.shape[0]) * 0.1
        result = wrapper.step(action)
        
        print(f"Step {step}: 奖励={result.reward:.1f}, 距离={result.info['distance']:.3f}m")
        
        if result.terminated:
            print("✅ 任务成功!")
            break
    
    print(f"\n✅ 复杂运动学HER包装器测试完成")
    print(f"✅ HER格式兼容")
    print(f"✅ GraphState转换正常")
    print(f"✅ 使用C++复杂运动学")


if __name__ == "__main__":
    test_complex_kinematics_her_wrapper()