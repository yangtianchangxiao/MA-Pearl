# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
多DOF动态软体机械臂Pearl环境实现
支持训练过程中动态切换2-4节(4-8DOF)，测试Graph网络泛化能力
每个episode随机选择segment数量和长度，挑战Graph适应性
"""

import math
from typing import Optional, Tuple, List
import numpy as np
import torch
import os
import sys
import random

from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.api.reward import Reward
from pearl.api.space import Space
from pearl.utils.instantiations.spaces.box import BoxSpace
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace

# 导入软体臂实现
soft_arm_path = os.path.join(os.path.dirname(__file__), 'soft_arm', 'robot_catch', 'env')
sys.path.append(soft_arm_path)
from robot_arm_python import RobotArm


class MultiDOFVariableSoftArmReachEnvironment(Environment):
    """
    多DOF动态软体机械臂reaching环境
    
    🚀 核心特性:
    - Episode级随机DOF: 2节(4DOF), 3节(6DOF), 4节(8DOF)
    - 每节长度独立随机化
    - 统一观测空间: 固定最大DOF + padding/mask机制
    - Graph网络泛化能力终极测试
    
    State: [joint_angles(current_dof), achieved_goal(3), desired_goal(3), segment_lengths(current_n_segments)]
    Action: joint_velocity_deltas (continuous, 当前episode实际DOF)
    Reward: sparse (50.0 if within threshold, -1.0 otherwise)
    
    Args:
        dof_range: (min_segments, max_segments) 例如 (2, 4) 表示2-4节
        base_segment_length: 基础段长度
        segment_length_range: 长度随机范围 (min_length, max_length)
        goal_threshold: distance threshold to consider goal reached
        max_steps: maximum steps per episode
        dof_distribution: DOF采样分布 'uniform'|'weighted' 可以偏向某些DOF测试
    """

    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 4),  # 2-4节 (4-8 DOF)
        base_segment_length: float = 0.21,
        segment_length_range: Optional[Tuple[float, float]] = None,
        goal_threshold: float = 0.15,
        max_steps: int = 200,
        dof_distribution: str = 'uniform',  # 'uniform', 'weighted'
        dof_weights: Optional[List[float]] = None,  # 如果weighted，各DOF权重
    ) -> None:
        self.min_segments, self.max_segments = dof_range
        self.max_dof = self.max_segments * 2  # 最大DOF数
        self.spatial_dim = 3
        self.base_segment_length = base_segment_length
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.dof_distribution = dof_distribution
        
        # DOF采样权重
        if dof_weights is None:
            # 默认均匀分布
            self.dof_weights = [1.0] * (self.max_segments - self.min_segments + 1)
        else:
            self.dof_weights = dof_weights
        
        # Segment长度随机范围
        if segment_length_range is None:
            self.segment_length_range = (base_segment_length * 0.8, base_segment_length * 1.2)
        else:
            self.segment_length_range = segment_length_range
        
        # 当前episode配置（每次reset时随机）
        self.current_n_segments = None  # 当前节数
        self.current_dof = None  # 当前DOF
        self.current_segment_lengths = None  # 当前segment长度
        self.valid_joint_mask = None  # 有效关节mask
        
        # 软体机械臂实例
        self.robot_arm = None
        
        # 状态
        self.joint_angles = np.zeros(self.max_dof, dtype=np.float32)  # 使用最大DOF
        self.goal_position = np.zeros(self.spatial_dim, dtype=np.float32)
        self.step_count = 0
        
        # Action space: 使用最大DOF，但实际使用当前DOF
        # Graph网络会处理variable action dimensions
        self._action_space = BoxActionSpace(
            low=torch.full((self.max_dof,), -1.0),
            high=torch.full((self.max_dof,), 1.0)
        )
        
        # 观测空间: [joint_angles(max_dof), segment_lengths(max_segments), achieved_goal(3), desired_goal(3), valid_mask(max_segments)]
        state_dim = self.max_dof + self.max_segments + self.spatial_dim + self.spatial_dim + self.max_segments
        self._observation_space = BoxSpace(
            low=np.full(state_dim, -np.inf),
            high=np.full(state_dim, np.inf)
        )
        
        print(f"🚀 多DOF动态软体臂环境初始化")
        print(f"   DOF范围: {self.min_segments}-{self.max_segments}节 ({self.min_segments*2}-{self.max_dof}DOF)")
        print(f"   长度范围: {self.segment_length_range[0]:.3f}m - {self.segment_length_range[1]:.3f}m")
        print(f"   观测维度: {self._observation_space.shape}")
        print(f"   阈值: {goal_threshold}")
        print(f"   分布策略: {dof_distribution}")

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property  
    def observation_space(self) -> Space:
        return self._observation_space

    def _sample_episode_config(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """采样当前episode的DOF配置"""
        
        # 采样节数
        if self.dof_distribution == 'uniform':
            n_segments = random.randint(self.min_segments, self.max_segments)
        else:  # weighted
            segments_options = list(range(self.min_segments, self.max_segments + 1))
            n_segments = random.choices(segments_options, weights=self.dof_weights, k=1)[0]
        
        # 采样每节长度
        segment_lengths = np.array([
            random.uniform(*self.segment_length_range) 
            for _ in range(n_segments)
        ], dtype=np.float32)
        
        # 生成有效关节mask
        valid_mask = np.zeros(self.max_segments, dtype=np.float32)
        valid_mask[:n_segments] = 1.0
        
        return n_segments, segment_lengths, valid_mask

    def _forward_kinematics(self) -> np.ndarray:
        """使用当前配置的正向运动学"""
        if self.robot_arm is not None:
            return np.array(self.robot_arm.get_ee_position(), dtype=np.float32)
        else:
            # 简化正向运动学 - 只使用有效关节
            position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            current_angle = 0.0
            
            for i in range(self.current_n_segments):
                # 只处理当前有效的segment
                bend_angle = self.joint_angles[i*2]
                direction_angle = self.joint_angles[i*2 + 1]
                segment_length = self.current_segment_lengths[i]
                
                # 3D运动学计算
                current_angle += direction_angle
                position[0] += segment_length * np.cos(current_angle) * np.cos(bend_angle)
                position[1] += segment_length * np.sin(current_angle) * np.cos(bend_angle)
                position[2] += segment_length * np.sin(bend_angle)
            
            return position

    def _create_robot_arm(self):
        """使用当前segment配置创建机械臂"""
        try:
            # 将当前segment长度转换为列表传给C++实现
            lengths_list = self.current_segment_lengths.tolist()
            self.robot_arm = RobotArm(lengths_list)
        except Exception as e:
            print(f"⚠️ C++机械臂创建失败，使用Python fallback: {e}")
            self.robot_arm = None

    def _sample_goal(self) -> np.ndarray:
        """在当前配置的工作空间内采样目标"""
        # 基于当前总长度计算可达空间
        total_length = np.sum(self.current_segment_lengths)
        safe_reach = total_length * 0.7  # 70%安全范围
        
        # 在球形空间内均匀采样
        phi = random.uniform(0, 2 * np.pi)  # 方位角
        costheta = random.uniform(-1, 1)  # cos(极角)，保证球面均匀
        u = random.uniform(0, 1)
        theta = np.arccos(costheta)
        
        # 球坐标到直角坐标
        r = safe_reach * (u ** (1/3))  # 球体内均匀分布
        goal = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            max(0.05, r * np.cos(theta))  # 避免地面
        ], dtype=np.float32)
        
        return goal

    def _get_observation(self) -> torch.Tensor:
        """生成统一格式的观测"""
        # 当前末端位置
        achieved_goal = self._forward_kinematics()
        
        # 构建完整观测: [joints(max_dof), lengths(max_segments), achieved(3), desired(3), mask(max_segments)]
        obs_parts = []
        
        # 1. Joint angles (padded to max_dof)
        obs_parts.append(self.joint_angles)
        
        # 2. Segment lengths (padded to max_segments)
        padded_lengths = np.zeros(self.max_segments, dtype=np.float32)
        padded_lengths[:self.current_n_segments] = self.current_segment_lengths
        obs_parts.append(padded_lengths)
        
        # 3. Achieved goal
        obs_parts.append(achieved_goal)
        
        # 4. Desired goal  
        obs_parts.append(self.goal_position)
        
        # 5. Valid mask
        obs_parts.append(self.valid_joint_mask)
        
        observation = np.concatenate(obs_parts).astype(np.float32)
        return torch.tensor(observation)  # Pearl框架约定：环境tensor在CPU

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, ActionSpace]:
        """Reset environment with new random DOF configuration"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 采样新的episode配置
        self.current_n_segments, self.current_segment_lengths, self.valid_joint_mask = self._sample_episode_config()
        self.current_dof = self.current_n_segments * 2
        
        # 重置关节角度 (清零非活跃关节)
        self.joint_angles = np.zeros(self.max_dof, dtype=np.float32)
        
        # 创建对应配置的机械臂
        self._create_robot_arm()
        
        # 采样目标位置
        self.goal_position = self._sample_goal()
        
        # 重置步数
        self.step_count = 0
        
        # 计算总长度用于显示
        total_length = np.sum(self.current_segment_lengths)
        max_reach = total_length * 0.95
        
        print(f"🔄 Episode Reset - DOF: {self.current_n_segments}节({self.current_dof}DOF), 总长度: {total_length:.3f}m, 工作空间: {max_reach:.3f}m")
        
        observation = self._get_observation()
        return observation, self.action_space

    def step(self, action: Action) -> ActionResult:
        """执行动作，只使用当前有效的DOF"""
        self.step_count += 1
        
        # 只应用到当前有效的关节
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = action
        
        # 确保action_np是1D数组
        if action_np.ndim > 1:
            action_np = action_np.flatten()
        
        # 限制到当前DOF范围，忽略padding部分
        valid_action = action_np[:self.current_dof]
        
        # 更新关节角度 (0.01是速度比例因子)
        self.joint_angles[:self.current_dof] += valid_action * 0.01
        
        # 角度限制
        self.joint_angles[:self.current_dof] = np.clip(
            self.joint_angles[:self.current_dof], -np.pi/2, np.pi/2
        )
        
        # 如果使用C++实现，更新机械臂状态
        if self.robot_arm is not None:
            try:
                self.robot_arm.set_joint_angles(self.joint_angles[:self.current_dof].tolist())
            except Exception as e:
                print(f"⚠️ C++更新失败: {e}")
        
        # 计算奖励
        current_position = self._forward_kinematics()
        distance = np.linalg.norm(current_position - self.goal_position)
        
        # 稀疏奖励 - 与基线保持一致  
        if distance <= self.goal_threshold:
            reward = torch.tensor(50.0)  # 成功
            terminated = torch.tensor(True)
        else:
            reward = torch.tensor(-1.0)  # 继续尝试
            terminated = torch.tensor(False)
        
        # 截断条件
        truncated = torch.tensor(self.step_count >= self.max_steps)
        
        # 获取新观测
        observation = self._get_observation()
        
        return ActionResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def get_current_config(self) -> dict:
        """获取当前episode配置信息，用于调试"""
        return {
            'n_segments': self.current_n_segments,
            'current_dof': self.current_dof,
            'segment_lengths': self.current_segment_lengths.tolist(),
            'total_length': np.sum(self.current_segment_lengths),
            'valid_mask': self.valid_joint_mask.tolist(),
        }