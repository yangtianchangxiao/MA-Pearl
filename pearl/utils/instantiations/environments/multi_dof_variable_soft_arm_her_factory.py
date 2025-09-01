# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
多DOF动态软体机械臂HER Buffer工厂
支持2-4节动态DOF的HER训练，处理可变维度状态空间
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Callable

from pearl.replay_buffers import BasicReplayBuffer
from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.api.reward import Reward
from pearl.utils.tensor_like import assert_is_tensor_like


class MultiDOFVariableSoftArmHERBuffer(BasicReplayBuffer):
    """
    多DOF动态软体机械臂HER Buffer
    处理可变DOF状态空间的goal substitution和reward recomputation
    
    状态格式: [joint_angles(max_dof), segment_lengths(max_segments), achieved_goal(3), desired_goal(3), valid_mask(max_segments)]
    """
    
    def __init__(
        self,
        capacity: int,
        max_dof: int = 8,
        max_segments: int = 4,
        spatial_dim: int = 3,
        goal_threshold: float = 0.15,
        reward_fn: Callable[[SubjectiveState, Action], float] = None,
        terminated_fn: Callable[[SubjectiveState, Action], bool] = None,
        n_sampled_goals: int = 4,
        her_strategy: str = "future",
    ):
        super().__init__(capacity)
        
        self.max_dof = max_dof
        self.max_segments = max_segments  
        self.spatial_dim = spatial_dim
        self.goal_threshold = goal_threshold
        self.n_sampled_goals = n_sampled_goals
        self.her_strategy = her_strategy
        
        # 计算各部分在状态中的索引位置
        self.joint_start = 0
        self.joint_end = max_dof
        self.length_start = self.joint_end
        self.length_end = self.length_start + max_segments
        self.achieved_start = self.length_end
        self.achieved_end = self.achieved_start + spatial_dim
        self.desired_start = self.achieved_end
        self.desired_end = self.desired_start + spatial_dim
        self.mask_start = self.desired_end
        self.mask_end = self.mask_start + max_segments
        
        # Episode buffer for HER processing
        self._episode_buffer: List[dict] = []
        
        # Default reward and termination functions
        if reward_fn is None:
            self._reward_fn = self._compute_sparse_reward
        else:
            self._reward_fn = reward_fn
            
        if terminated_fn is None:
            self._terminated_fn = self._compute_termination
        else:
            self._terminated_fn = terminated_fn
        
        print(f"✅ 多DOF HER Buffer: capacity={capacity:,}, max_dof={max_dof}, max_segments={max_segments}")
        print(f"   HER策略: {her_strategy}, goals: {n_sampled_goals}")
        print(f"   阈值: {goal_threshold}")

    def _extract_achieved_goal(self, state: torch.Tensor) -> torch.Tensor:
        """从状态中提取achieved goal"""
        if state.dim() == 1:
            return state[self.achieved_start:self.achieved_end]
        else:  # batch
            return state[:, self.achieved_start:self.achieved_end]

    def _extract_desired_goal(self, state: torch.Tensor) -> torch.Tensor:
        """从状态中提取desired goal"""
        if state.dim() == 1:
            return state[self.desired_start:self.desired_end]
        else:  # batch
            return state[:, self.desired_start:self.desired_end]

    def _substitute_goal(self, state: torch.Tensor, new_goal: torch.Tensor) -> torch.Tensor:
        """将新目标替换到状态中"""
        new_state = state.clone()
        if state.dim() == 1:
            new_state[self.desired_start:self.desired_end] = new_goal
        else:  # batch
            new_state[:, self.desired_start:self.desired_end] = new_goal
        return new_state

    def _compute_sparse_reward(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        """计算稀疏奖励：成功=50.0，失败=-1.0"""
        distance = torch.norm(achieved_goal - desired_goal, dim=-1)
        
        if distance <= self.goal_threshold:
            return torch.tensor(50.0)  # 成功
        else:
            return torch.tensor(-1.0)  # 继续尝试

    def _compute_termination(self, achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        """计算终止条件"""
        distance = torch.norm(achieved_goal - desired_goal, dim=-1)
        return torch.tensor(distance <= self.goal_threshold)  # 返回tensor确保类型一致性
    
    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool = False,
        truncated: bool = False,
        curr_available_actions=None,
        next_state: SubjectiveState = None,
        next_available_actions=None,
        max_number_actions: int = None,
        cost: Reward = None,
    ) -> None:
        # Early return if no next_state
        if next_state is None:
            return
        
        # Store original experience using parent class
        # For continuous action spaces, we don't need curr_available_actions
        super().push(
            state, action, reward, terminated, truncated,
            None, next_state, None, max_number_actions, cost
        )
        
        # Store experience in episode buffer for HER processing
        # No tensor conversion - Pearl framework handles it
        experience = {
            'state': state,
            'action': action, 
            'reward': reward,
            'next_state': next_state,
            'curr_available_actions': curr_available_actions,
            'next_available_actions': next_available_actions,
            'terminated': terminated,
        }
        self._episode_buffer.append(experience)
        
        # Process HER experiences when episode ends
        # Extract boolean values for episode termination check
        terminated_bool = terminated.item() if isinstance(terminated, torch.Tensor) else terminated
        truncated_bool = truncated.item() if isinstance(truncated, torch.Tensor) else truncated
        
        if terminated_bool or truncated_bool:
            self._process_her_episode()
            self._episode_buffer.clear()
    
    def _process_her_episode(self):
        """处理episode的HER goal替换"""
        if len(self._episode_buffer) == 0:
            return
        
        episode_length = len(self._episode_buffer)
        
        # 为每个transition生成HER goals
        for i in range(episode_length):
            # 根据策略采样goals
            if self.her_strategy == "future":
                # Future策略：从当前step之后的states中采样achieved goals
                future_indices = list(range(i + 1, episode_length))
                if len(future_indices) == 0:
                    continue
                    
                sampled_indices = np.random.choice(
                    future_indices, 
                    size=min(self.n_sampled_goals, len(future_indices)), 
                    replace=False
                )
            else:
                # Random策略：从整个episode中随机采样
                sampled_indices = np.random.choice(
                    episode_length, 
                    size=min(self.n_sampled_goals, episode_length), 
                    replace=False
                )
            
            # 为每个采样的goal创建HER transition
            for idx in sampled_indices:
                her_goal = self._extract_achieved_goal(self._episode_buffer[idx]['next_state'])
                
                # 创建HER状态
                her_state = self._substitute_goal(self._episode_buffer[i]['state'], her_goal)
                her_next_state = self._substitute_goal(self._episode_buffer[i]['next_state'], her_goal)
                
                # 计算HER奖励
                achieved_goal = self._extract_achieved_goal(her_next_state)
                her_reward = self._compute_sparse_reward(achieved_goal, her_goal)
                her_terminated = self._compute_termination(achieved_goal, her_goal)
                
                # Push HER transition
                super().push(
                    her_state,
                    self._episode_buffer[i]['action'],
                    torch.tensor(her_reward),
                    her_terminated,
                    False,  # HER transitions are not truncated
                    None,  # No curr_available_actions for continuous spaces
                    her_next_state,
                    None,  # No next_available_actions for continuous spaces
                )


def create_multi_dof_variable_soft_arm_her_buffer(
    capacity: int,
    max_dof: int = 8,
    max_segments: int = 4,
    spatial_dim: int = 3,
    goal_threshold: float = 0.15,
    her_strategy: str = "future",
    her_goals: int = 4,
) -> MultiDOFVariableSoftArmHERBuffer:
    """
    创建多DOF动态软体机械臂HER Buffer
    
    Args:
        capacity: Buffer容量
        max_dof: 最大DOF数量
        max_segments: 最大segment数量
        spatial_dim: 空间维度 (3D)
        goal_threshold: 成功阈值
        her_strategy: HER策略 ("future", "episode", "random")
        her_goals: 每个transition生成的HER goal数量
    """
    
    her_buffer = MultiDOFVariableSoftArmHERBuffer(
        capacity=capacity,
        max_dof=max_dof,
        max_segments=max_segments,
        spatial_dim=spatial_dim,
        goal_threshold=goal_threshold,
        n_sampled_goals=her_goals,
        her_strategy=her_strategy,
    )
    
    return her_buffer