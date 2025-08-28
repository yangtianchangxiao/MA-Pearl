# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
软体机械臂Pearl环境实现
继承Pearl抽象环境类，使用与3DOF相同的稀疏奖励格式
"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import os
import sys

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.api.reward import Reward
from pearl.api.space import Space
from pearl.utils.instantiations.spaces.box import BoxSpace
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace

# 导入软体臂C++/Python实现
soft_arm_path = os.path.join(os.path.dirname(__file__), 'soft_arm', 'robot_catch', 'env')
sys.path.append(soft_arm_path)
from robot_arm_python import RobotArm


class SoftArmReachEnvironment(Environment):
    """
    软体机械臂reaching环境，基于Pearl抽象环境类，支持任意节数配置
    
    State: [joint_angles(n_segments*2), achieved_goal(3), desired_goal(3)]
    Action: joint_velocity_deltas (continuous, n_segments*2 DOF)
    Reward: sparse (50.0 if within threshold, -1.0 otherwise) - 与3DOF保持一致
    
    Args:
        n_segments: 软体臂节数，每节2个自由度（弯曲角度+方向角）
        segment_length: 每段软体臂长度
        workspace_bounds: [min_x, max_x, min_y, max_y, min_z, max_z] for 3D workspace
        goal_threshold: distance threshold to consider goal reached
        max_steps: maximum steps per episode
    """

    def __init__(
        self,
        n_segments: int = 3,  # 软体臂节数，默认3节
        segment_length: float = 0.21,
        workspace_bounds: Optional[list[float]] = None,
        goal_threshold: float = 0.15,  # 相对3DOF的0.30，软体臂更精确
        max_steps: int = 200,
        # 向后兼容参数
        dof: Optional[int] = None,  # 如果提供，会被忽略，用n_segments*2计算
    ) -> None:
        self.n_segments = n_segments
        self.dof = n_segments * 2  # 每节2个自由度：弯曲角度+方向角
        self.spatial_dim = 3  # 3D空间
        self.segment_length = segment_length
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        
        # 向后兼容性检查
        if dof is not None and dof != self.dof:
            print(f"⚠️ dof参数({dof})被忽略，使用n_segments*2={self.dof}")
        
        # 工作空间边界 (3D) - 基于节数动态计算
        if workspace_bounds is None:
            # 软体臂实际可达工作空间：与节数成正比
            max_reach = n_segments * segment_length  # 理论最大伸展
            actual_safe_reach = max_reach * 0.8   # 80%安全边界，确保高可达性
            self.workspace_bounds = [
                -actual_safe_reach, actual_safe_reach,  # x: 对称，可达负X
                -actual_safe_reach, actual_safe_reach,  # y: 对称
                0.05, actual_safe_reach  # z: 避免地面碰撞，但可达较高位置
            ]
        else:
            self.workspace_bounds = workspace_bounds
        
        # 创建软体机械臂 - 使用可配置节数
        self.robot_arm = RobotArm(n_segments=n_segments, segment_length=segment_length)
        
        # 初始化状态
        self.joint_angles = np.zeros(self.dof, dtype=np.float32)
        self.goal_position = np.zeros(self.spatial_dim, dtype=np.float32)  
        self.step_count = 0
        
        # Action space: joint velocity deltas
        self._action_space = BoxActionSpace(
            low=torch.full((self.dof,), -1.0),
            high=torch.full((self.dof,), 1.0)
        )
        
        # 观测空间: [joint_angles(n_segments*2), achieved_goal(3), desired_goal(3)]
        state_dim = self.dof + self.spatial_dim + self.spatial_dim  # dof + 3 + 3
        self._observation_space = BoxSpace(
            low=np.full(state_dim, -np.inf),
            high=np.full(state_dim, np.inf)
        )
        
        print(f"✅ 软体臂Pearl环境初始化: {n_segments}节 {self.dof}DOF, 阈值={goal_threshold}, 工作空间=3D")

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property  
    def observation_space(self) -> Space:
        return self._observation_space

    def _forward_kinematics(self) -> np.ndarray:
        """获取当前末端执行器位置"""
        return np.array(self.robot_arm.get_ee_position(), dtype=np.float32)

    def _sample_goal(self) -> np.ndarray:
        """在工作空间内采样随机目标"""
        x = np.random.uniform(self.workspace_bounds[0], self.workspace_bounds[1])
        y = np.random.uniform(self.workspace_bounds[2], self.workspace_bounds[3])
        z = np.random.uniform(self.workspace_bounds[4], self.workspace_bounds[5])
        return np.array([x, y, z], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """完整状态: [joint_angles, achieved_goal, desired_goal] - 与3DOF格式一致"""
        achieved_goal = self._forward_kinematics()
        
        return np.concatenate([
            self.joint_angles,          # joint_angles: 当前关节角度 (6D)
            achieved_goal,              # achieved_goal: 当前end_effector位置 (3D)
            self.goal_position         # desired_goal: 目标end_effector位置 (3D)
        ], dtype=np.float32)

    def _compute_reward(self) -> float:
        """稀疏奖励: -1 per step + big success reward - 与3DOF保持一致"""
        current_end_pos = self._forward_kinematics()
        end_distance = np.linalg.norm(current_end_pos - self.goal_position)
        
        if end_distance <= self.goal_threshold:
            return 50.0  # Big success reward - terminate episode
        else:
            return -1.0  # Step penalty - encourages faster completion

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, ActionSpace]:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 重置软体机械臂到初始状态
        self.robot_arm.reset()
        self.step_count = 0
        
        # 随机初始关节配置 (小范围，避免极端姿态)
        self.joint_angles = np.random.uniform(-math.pi/8, math.pi/8, size=self.dof).astype(np.float32)
        
        # 设置机械臂到初始配置
        # 注意：robot_arm内部会处理关节限制
        for i, angle in enumerate(self.joint_angles):
            if hasattr(self.robot_arm, 'config_state'):
                self.robot_arm.config_state[i] = angle
        
        # 生成随机目标
        self.goal_position = self._sample_goal()
        
        observation = self._get_observation()
        return torch.tensor(observation), self.action_space

    def step(self, action: Action) -> ActionResult:
        # 确保action是正确格式
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.array(action)
        
        # 限制action范围
        action_np = np.clip(action_np, -1.0, 1.0)
        
        # 执行动作 (软体机械臂内部处理速度限制和关节限制)
        self.robot_arm.step(action_np, dt=0.02)
        self.step_count += 1
        
        # 更新关节状态
        if hasattr(self.robot_arm, 'config_state'):
            self.joint_angles = self.robot_arm.config_state.copy()
        
        # 计算奖励和终止条件
        reward = self._compute_reward()
        
        current_end_pos = self._forward_kinematics()
        end_distance = np.linalg.norm(current_end_pos - self.goal_position)
        
        terminated = (end_distance <= self.goal_threshold)
        truncated = (self.step_count >= self.max_steps)
        
        observation = self._get_observation()
        
        return ActionResult(
            observation=torch.tensor(observation),
            reward=torch.tensor(reward),
            terminated=torch.tensor(terminated),
            truncated=torch.tensor(truncated),
        )


# 兼容性别名，保持与gym风格的兼容
SoftArmReachEnv = SoftArmReachEnvironment


def test_soft_arm_pearl_env():
    """测试软体臂Pearl环境"""
    print("🧪 测试软体臂Pearl环境...")
    
    env = SoftArmReachEnvironment()
    
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 测试reset
    obs, action_space = env.reset()
    print(f"初始观测形状: {obs.shape}")
    print(f"观测格式: [关节({env.dof}) + achieved({env.spatial_dim}) + desired({env.spatial_dim})]")
    
    # 测试几步
    for step in range(3):
        action = action_space.sample()
        result = env.step(action)
        
        current_pos = result.observation[6:9]  # achieved_goal
        goal_pos = result.observation[9:12]    # desired_goal
        distance = torch.norm(current_pos - goal_pos)
        
        print(f"Step {step+1}: reward={result.reward:.1f}, distance={distance:.3f}, terminated={result.terminated}")
        
        if result.terminated:
            print("🎉 成功达到目标!")
            break
    
    print("✅ 软体臂Pearl环境测试完成!")


if __name__ == "__main__":
    test_soft_arm_pearl_env()