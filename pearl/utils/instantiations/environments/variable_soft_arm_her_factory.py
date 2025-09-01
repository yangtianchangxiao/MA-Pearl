# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.utils.instantiations.environments.variable_her_buffer import create_variable_arm_her_buffer as _create_buffer
from pearl.utils.tensor_like import assert_is_tensor_like


def create_variable_soft_arm_reward_fn(joint_dim: int, spatial_dim: int, n_segments: int, threshold: float = 0.1, include_lengths_in_obs: bool = True):
    """
    Create reward function for variable soft arm reaching task.
    
    State format (include_lengths_in_obs=True): 
    [joint_angles(joint_dim), segment_lengths(n_segments), achieved_goal(3), desired_goal(3)]
    
    State format (include_lengths_in_obs=False):
    [joint_angles(joint_dim), achieved_goal(3), desired_goal(3)]
    """
    def reward_fn(state: SubjectiveState, action: Action) -> float:
        if isinstance(state, torch.Tensor):
            s = state
        else:
            s = torch.tensor(state)
        
        # Extract goals from state based on format
        if include_lengths_in_obs:
            # [joint_angles(joint_dim), segment_lengths(n_segments), achieved_goal(3), desired_goal(3)]
            config_dim = joint_dim + n_segments  # 配置部分总维度：joints + lengths
            achieved_goal = s[config_dim:config_dim + spatial_dim]        # achieved_goal位置
            desired_goal = s[config_dim + spatial_dim:]                  # desired_goal位置
        else:
            # Standard format: [joint_angles(joint_dim), achieved_goal(3), desired_goal(3)]
            achieved_goal = s[joint_dim:joint_dim + spatial_dim]        # Current end position
            desired_goal = s[joint_dim + spatial_dim:]                 # Target end position
        
        # Direct goal comparison - this is what HER is all about  
        goal_distance = torch.norm(achieved_goal - desired_goal).item()
        
        # HER标准奖励：{0, -1} - 返回torch tensor确保一致性
        return torch.tensor(0.0 if goal_distance <= threshold else -1.0)
    
    return reward_fn


def create_variable_soft_arm_terminated_fn(joint_dim: int, spatial_dim: int, n_segments: int, threshold: float = 0.1, include_lengths_in_obs: bool = True):
    """Create termination function for variable soft arm reaching task."""
    reward_fn = create_variable_soft_arm_reward_fn(joint_dim, spatial_dim, n_segments, threshold, include_lengths_in_obs)
    return lambda state, action: torch.tensor(reward_fn(state, action) >= 0.0)  # 确保返回torch tensor


def create_variable_soft_arm_her_buffer(
    joint_dim: int = 6,
    spatial_dim: int = 3, 
    n_segments: int = 3,
    capacity: int = 100000, 
    threshold: float = 0.1,
    include_lengths_in_obs: bool = True
):
    """
    Create HER buffer for variable soft arm environment.
    
    State format支持两种:
    1. include_lengths_in_obs=True: [joint_angles(joint_dim), achieved_goal(3), desired_goal(3), segment_lengths(n_segments)]
    2. include_lengths_in_obs=False: [joint_angles(joint_dim), achieved_goal(3), desired_goal(3)]
    
    HER机制: Future策略 + 多采样 (对标SB3)
    """
    return _create_buffer(
        capacity=capacity,
        dof=joint_dim,
        spatial_dim=spatial_dim,
        n_segments=n_segments,
        reward_fn=create_variable_soft_arm_reward_fn(joint_dim, spatial_dim, n_segments, threshold, include_lengths_in_obs),
        terminated_fn=create_variable_soft_arm_terminated_fn(joint_dim, spatial_dim, n_segments, threshold, include_lengths_in_obs),
        n_sampled_goals=4,  # 对标SB3
        include_lengths_in_obs=include_lengths_in_obs,
    )