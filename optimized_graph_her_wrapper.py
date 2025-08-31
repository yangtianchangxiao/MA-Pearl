#!/usr/bin/env python3
"""
优化Graph环境的HER兼容包装器
将GraphState tensor转换为HER buffer期望的扁平格式
"""

import torch
import numpy as np
from typing import Tuple, Any
from pearl.api.environment import Environment
from pearl.api.action_result import ActionResult
from pearl.api.observation import Observation
from optimized_graph_environment import OptimizedGraphSoftArmEnvironment
from simplified_graph_demo import SimplifiedGraphState


class OptimizedGraphHERWrapper(Environment):
    """
    GraphState到HER格式的包装器
    
    输入: GraphState tensor (variable length)
    输出: [joint_angles, achieved_goal, desired_goal] (fixed length)
    """
    
    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 4),
        **kwargs
    ):
        self.env = OptimizedGraphSoftArmEnvironment(dof_range=dof_range, **kwargs)
        self.dof_range = dof_range
        self.max_dof = max(dof_range) * 2
        
        print(f"🔗 Graph-HER包装器: DOF {dof_range[0]}-{dof_range[1]}节")
        print(f"   输入: GraphState tensor (变长)")
        print(f"   输出: [joint_angles({self.max_dof}), achieved_goal(3), desired_goal(3)]")
    
    def reset(self, seed: int = None) -> Tuple[Observation, Any]:
        """重置并转换格式"""
        graph_obs, action_space = self.env.reset(seed)
        her_obs = self._convert_to_her_format(graph_obs)
        return her_obs, action_space
    
    def step(self, action: torch.Tensor) -> ActionResult:
        """执行动作并转换格式"""
        result = self.env.step(action)
        her_obs = self._convert_to_her_format(result.observation)
        
        return ActionResult(
            observation=her_obs,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            available_action_space=result.available_action_space
        )
    
    def _convert_to_her_format(self, graph_tensor: torch.Tensor) -> torch.Tensor:
        """将GraphState tensor转换为HER格式"""
        # 从GraphState tensor提取信息
        graph_state = SimplifiedGraphState.from_tensor(graph_tensor)
        
        # 提取joint angles
        current_dof = self.env.current_n_segments * 2
        joint_angles = self.env.joint_angles[:current_dof]
        
        # 补零到最大DOF
        padded_joint_angles = np.zeros(self.max_dof, dtype=np.float32)
        padded_joint_angles[:current_dof] = joint_angles
        
        # 组合HER格式: [joint_angles, achieved_goal, desired_goal]
        her_tensor = torch.cat([
            torch.tensor(padded_joint_angles, dtype=torch.float32),  # [max_dof]
            graph_state.achieved_goal,  # [3]
            graph_state.desired_goal    # [3]
        ])  # total: [max_dof + 6]
        
        return her_tensor
    
    @property
    def action_space(self):
        """代理action space"""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """返回HER兼容的观测空间"""
        from gymnasium.spaces import Box
        obs_dim = self.max_dof + 6  # joint_angles + achieved_goal + desired_goal
        return Box(low=-np.inf, high=np.inf, shape=(obs_dim,))


def test_her_wrapper():
    """测试HER包装器"""
    print("🧪 测试Graph-HER包装器")
    print("=" * 50)
    
    wrapper = OptimizedGraphHERWrapper(dof_range=(2, 4), max_steps=10)
    
    for episode in range(3):
        obs, action_space = wrapper.reset()
        print(f"\nEpisode {episode+1}:")
        print(f"   HER观测形状: {obs.shape}")
        print(f"   Joint angles: {obs[:8]}")  # 前8个是joint angles (max_dof)
        print(f"   Achieved goal: {obs[8:11]}")  # [8:11]是achieved_goal
        print(f"   Desired goal: {obs[11:14]}")  # [11:14]是desired_goal
        
        # 执行几步
        for step in range(3):
            action = torch.randn(action_space.shape[0]) * 0.1
            result = wrapper.step(action)
            
            print(f"     Step {step+1}: reward={result.reward.item():.2f}, "
                  f"obs_shape={result.observation.shape}")
            
            if result.terminated or result.truncated:
                break
    
    print(f"\n✅ Graph-HER包装器测试完成!")
    print(f"✅ GraphState → HER格式转换成功")
    print(f"✅ 固定观测维度: {wrapper.observation_space.shape}")


if __name__ == "__main__":
    test_her_wrapper()