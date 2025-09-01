# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
变长软体机械臂Pearl环境实现
支持每个episode随机化segment长度，增强泛化能力
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


class VariableSoftArmReachEnvironment(Environment):
    """
    变长软体机械臂reaching环境，支持episode级segment长度随机化
    
    State: [joint_angles(n_segments*2), achieved_goal(3), desired_goal(3), segment_lengths(n_segments)]
    Action: joint_velocity_deltas (continuous, n_segments*2 DOF)
    Reward: sparse (50.0 if within threshold, -1.0 otherwise) - 与3DOF保持一致
    
    Args:
        n_segments: 软体臂节数，每节2个自由度（弯曲角度+方向角）
        base_segment_length: 基础段长度
        segment_length_range: 每episode随机长度范围 (min_length, max_length)
        workspace_bounds: [min_x, max_x, min_y, max_y, min_z, max_z] for 3D workspace (动态计算)
        goal_threshold: distance threshold to consider goal reached
        max_steps: maximum steps per episode
        include_lengths_in_obs: 是否在观测中包含当前segment长度信息
    """

    def __init__(
        self,
        n_segments: int = 3,  # 软体臂节数，默认3节
        base_segment_length: float = 0.21,  # 基础segment长度
        segment_length_range: Optional[Tuple[float, float]] = None,  # 每episode随机范围 (min, max)
        workspace_bounds: Optional[list[float]] = None,
        goal_threshold: float = 0.15,  # 相对3DOF的0.30，软体臂更精确
        max_steps: int = 200,
        include_lengths_in_obs: bool = True,  # 是否在观测中包含长度信息
        # 向后兼容参数
        dof: Optional[int] = None,  # 如果提供，会被忽略，用n_segments*2计算
    ) -> None:
        self.n_segments = n_segments
        self.dof = n_segments * 2  # 每节2个自由度：弯曲角度+方向角
        self.spatial_dim = 3  # 3D空间
        self.base_segment_length = base_segment_length  # 基础长度
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.include_lengths_in_obs = include_lengths_in_obs
        
        # Episode级随机segment长度配置
        if segment_length_range is None:
            # 默认±20%变化范围
            self.segment_length_range = (base_segment_length * 0.8, base_segment_length * 1.2)
        else:
            self.segment_length_range = segment_length_range
        
        # 当前episode的实际segment长度（每节可不同）
        self.current_segment_lengths = np.full(n_segments, base_segment_length, dtype=np.float32)
        
        # 向后兼容性检查
        if dof is not None and dof != self.dof:
            print(f"⚠️ dof参数({dof})被忽略，使用n_segments*2={self.dof}")
        
        # 工作空间边界将在每个episode动态计算
        self.fixed_workspace_bounds = workspace_bounds
        self.current_workspace_bounds = None
        
        # 软体机械臂实例（将在reset中重新配置）
        self.robot_arm = None
        
        # 初始化状态
        self.joint_angles = np.zeros(self.dof, dtype=np.float32)
        self.goal_position = np.zeros(self.spatial_dim, dtype=np.float32)  
        self.step_count = 0
        
        # Action space: joint velocity deltas
        self._action_space = BoxActionSpace(
            low=torch.full((self.dof,), -1.0),
            high=torch.full((self.dof,), 1.0)
        )
        
        # 观测空间: [joint_angles(n_segments*2), segment_lengths(n_segments), achieved_goal(3), desired_goal(3)]
        if include_lengths_in_obs:
            state_dim = self.dof + n_segments + self.spatial_dim + self.spatial_dim  # joints + lengths + achieved + desired
        else:
            state_dim = self.dof + self.spatial_dim + self.spatial_dim  # 标准格式
            
        self._observation_space = BoxSpace(
            low=np.full(state_dim, -np.inf),
            high=np.full(state_dim, np.inf)
        )
        
        print(f"✅ 变长软体臂Pearl环境初始化: {n_segments}节 {self.dof}DOF")
        print(f"   长度范围: {self.segment_length_range[0]:.3f}m - {self.segment_length_range[1]:.3f}m")
        print(f"   观测维度: {self._observation_space.shape} (包含长度: {include_lengths_in_obs})")
        print(f"   阈值: {goal_threshold}")

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property  
    def observation_space(self) -> Space:
        return self._observation_space

    def _compute_dynamic_workspace(self) -> list[float]:
        """基于当前segment长度动态计算工作空间"""
        if self.fixed_workspace_bounds is not None:
            return self.fixed_workspace_bounds
        
        # 计算当前配置的最大可达范围
        total_length = np.sum(self.current_segment_lengths)
        actual_safe_reach = total_length * 0.8  # 80%安全边界
        
        return [
            -actual_safe_reach, actual_safe_reach,  # x: 对称
            -actual_safe_reach, actual_safe_reach,  # y: 对称  
            0.05, actual_safe_reach  # z: 避免地面，但可达高处
        ]

    def _forward_kinematics(self) -> np.ndarray:
        """获取当前末端执行器位置 - 使用变长segment"""
        if self.robot_arm is not None:
            return np.array(self.robot_arm.get_ee_position(), dtype=np.float32)
        else:
            # 简化版正向运动学作为fallback
            current_pos = np.array([0, 0, 0], dtype=float)
            
            for i in range(self.n_segments):
                alpha = self.joint_angles[i*2]      # 弯曲角度
                beta = self.joint_angles[i*2+1]     # 方向角
                segment_len = self.current_segment_lengths[i]
                
                # 每段向前延伸
                dx = segment_len * np.cos(alpha) * np.cos(beta)
                dy = segment_len * np.cos(alpha) * np.sin(beta)
                dz = segment_len * np.sin(alpha)
                
                current_pos += [dx, dy, dz]
            
            return current_pos.astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """在动态工作空间内采样随机目标"""
        bounds = self.current_workspace_bounds
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[2], bounds[3])
        z = np.random.uniform(bounds[4], bounds[5])
        return np.array([x, y, z], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """完整状态: [joint_angles, segment_lengths, achieved_goal, desired_goal]"""
        achieved_goal = self._forward_kinematics()
        
        if self.include_lengths_in_obs:
            # 正确格式：配置信息(joint+length) + 目标信息(achieved+desired)
            return np.concatenate([
                self.joint_angles,              # joint_angles: 当前关节角度 (n_segments*2)
                self.current_segment_lengths,   # segment_lengths: 当前段长度 (n_segments) 
                achieved_goal,                  # achieved_goal: 当前end_effector位置 (3D)
                self.goal_position             # desired_goal: 目标end_effector位置 (3D)
            ], dtype=np.float32)
        else:
            # 标准格式：与固定长度版本兼容
            return np.concatenate([
                self.joint_angles,          # joint_angles: 当前关节角度 (n_segments*2)
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
        
        # 1. 随机化segment长度 (每节独立随机)
        min_len, max_len = self.segment_length_range
        self.current_segment_lengths = np.random.uniform(
            min_len, max_len, size=self.n_segments
        ).astype(np.float32)
        
        # 2. 基于新长度动态计算工作空间
        self.current_workspace_bounds = self._compute_dynamic_workspace()
        
        # 3. 重新配置软体机械臂（使用新的长度配置）
        # 注意：这里假设robot_arm可以动态重配置，或者我们使用简化的forward kinematics
        try:
            # 尝试创建新的robot_arm实例（如果支持变长）
            self.robot_arm = RobotArm(n_segments=self.n_segments, 
                                    segment_length=np.mean(self.current_segment_lengths))
            self.robot_arm.reset()
        except Exception as e:
            print(f"⚠️ Robot arm重配置失败，使用简化kinematics: {e}")
            self.robot_arm = None
        
        self.step_count = 0
        
        # 4. 随机初始关节配置 (小范围，避免极端姿态)
        self.joint_angles = np.random.uniform(-math.pi/8, math.pi/8, size=self.dof).astype(np.float32)
        
        # 5. 如果有robot_arm，同步关节状态
        if self.robot_arm is not None:
            for i, angle in enumerate(self.joint_angles):
                if hasattr(self.robot_arm, 'config_state'):
                    self.robot_arm.config_state[i] = angle
        
        # 6. 在新工作空间内生成随机目标
        self.goal_position = self._sample_goal()
        
        observation = self._get_observation()
        
        # Debug信息
        total_length = np.sum(self.current_segment_lengths)
        workspace_size = self.current_workspace_bounds[1] - self.current_workspace_bounds[0]
        print(f"🔄 Episode Reset - 总长度: {total_length:.3f}m, 工作空间: {workspace_size:.3f}m")
        
        return torch.tensor(observation), self.action_space

    def step(self, action: Action) -> ActionResult:
        # 确保action是正确格式
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.array(action)
        
        # 限制action范围
        action_np = np.clip(action_np, -1.0, 1.0)
        
        # 执行动作
        if self.robot_arm is not None:
            # 使用robot_arm执行 (如果可用)
            self.robot_arm.step(action_np, dt=0.02)
            self.step_count += 1
            
            # 更新关节状态
            if hasattr(self.robot_arm, 'config_state'):
                self.joint_angles = self.robot_arm.config_state.copy()
        else:
            # 简化版动作执行：直接更新关节角度
            self.joint_angles += action_np * 0.05  # 速度缩放
            self.joint_angles = np.clip(self.joint_angles, -math.pi/2, math.pi/2)  # 关节限制
            self.step_count += 1
        
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


# 兼容性别名
VariableSoftArmReachEnv = VariableSoftArmReachEnvironment


def test_variable_soft_arm_env():
    """测试变长软体臂环境"""
    print("🧪 测试变长软体臂Pearl环境...")
    
    env = VariableSoftArmReachEnvironment(
        n_segments=3,
        segment_length_range=(0.15, 0.25),  # ±25%变化
        include_lengths_in_obs=True
    )
    
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 测试多个episode的长度变化
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs, action_space = env.reset()
        print(f"Segment长度: {env.current_segment_lengths}")
        print(f"工作空间大小: {env.current_workspace_bounds[1] - env.current_workspace_bounds[0]:.3f}m")
        print(f"初始观测形状: {obs.shape}")
        
        # 测试几步
        for step in range(3):
            action = action_space.sample() * 0.1  # 小动作
            result = env.step(action)
            
            if env.include_lengths_in_obs:
                current_pos = result.observation[env.dof:env.dof + 3]  # achieved_goal
                goal_pos = result.observation[env.dof + 3:env.dof + 6]    # desired_goal
                lengths = result.observation[env.dof + 6:]  # segment_lengths
                print(f"  Step {step+1}: 长度{lengths.numpy()}, 距离={torch.norm(current_pos - goal_pos):.3f}")
            
            if result.terminated:
                print("🎉 成功达到目标!")
                break
    
    print("✅ 变长软体臂环境测试完成!")


if __name__ == "__main__":
    test_variable_soft_arm_env()