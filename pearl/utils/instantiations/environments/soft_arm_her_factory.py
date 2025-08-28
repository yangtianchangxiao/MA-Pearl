# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.utils.instantiations.environments.arm_her_buffer_fixed import FixedArmHERBuffer
from pearl.utils.tensor_like import assert_is_tensor_like


def create_soft_arm_reward_fn(joint_dim: int, spatial_dim: int, threshold: float = 0.1):
    """
    Create reward function for soft arm reaching task.
    
    State format: [joint_angles(6), achieved_goal(3), desired_goal(3)] (12D)
    """
    def reward_fn(state: SubjectiveState, action: Action) -> float:
        if isinstance(state, torch.Tensor):
            s = state
        else:
            s = torch.tensor(state)
        
        # Extract goals from: [joint_angles(6), achieved_goal(3), desired_goal(3)]
        achieved_goal = s[joint_dim:joint_dim + spatial_dim]        # Current end position
        desired_goal = s[joint_dim + spatial_dim:]                 # Target end position
        
        # Direct goal comparison - this is what HER is all about  
        goal_distance = torch.norm(achieved_goal - desired_goal).item()
        
        # HER标准奖励：{0, -1}
        return 0.0 if goal_distance <= threshold else -1.0
    
    return reward_fn


def create_soft_arm_terminated_fn(joint_dim: int, spatial_dim: int, threshold: float = 0.1):
    """Create termination function for soft arm reaching task."""
    reward_fn = create_soft_arm_reward_fn(joint_dim, spatial_dim, threshold)
    return lambda state, action: reward_fn(state, action) >= 0.0


def create_soft_arm_her_buffer(
    joint_dim: int = 6,
    spatial_dim: int = 3, 
    capacity: int = 100000, 
    threshold: float = 0.1
) -> FixedArmHERBuffer:
    """
    Create HER buffer for soft arm environment.
    
    State format: [joint_angles(6), achieved_goal(3), desired_goal(3)] (12D)
    HER机制: Future策略 + 多采样 (对标SB3)
    """
    return FixedArmHERBuffer(
        capacity=capacity,
        dof=joint_dim,
        spatial_dim=spatial_dim,
        reward_fn=create_soft_arm_reward_fn(joint_dim, spatial_dim, threshold),
        terminated_fn=create_soft_arm_terminated_fn(joint_dim, spatial_dim, threshold),
        n_sampled_goals=4,  # 对标SB3
    )