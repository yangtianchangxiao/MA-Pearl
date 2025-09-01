# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-DOF Dynamic Variable Soft Arm HER buffer implementation (WORKING VERSION)
Based on working variable_her_buffer.py with modifications for dynamic DOF
Supports observation format: [joint_angles(max_dof), segment_lengths(max_segments), achieved_goal(3), desired_goal(3), valid_mask(max_segments)]
"""

from typing import Callable, Optional, List
import torch
import numpy as np
from pearl.replay_buffers import BasicReplayBuffer
from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.api.reward import Reward
from pearl.utils.tensor_like import assert_is_tensor_like


class MultiDOFVariableArmHERBuffer(BasicReplayBuffer):
    """
    HER replay buffer for multi-DOF dynamic variable soft arm environment.
    
    Observation format: [joint_angles(max_dof), segment_lengths(max_segments), achieved_goal(3), desired_goal(3), valid_mask(max_segments)]
    
    Supports dynamic DOF (2-4 segments) with unified state space and HER processing.
    """
    
    def __init__(
        self,
        capacity: int,
        max_dof: int = 8,  # maximum joint angles dimension
        spatial_dim: int = 3,  # achieved/desired goal dimension
        max_segments: int = 4,  # maximum number of segments
        goal_threshold: float = 0.15,  # goal achievement threshold
        n_sampled_goals: int = 4,  # HER goal sampling count
        her_strategy: str = "future",  # HER strategy
    ):
        super().__init__(capacity)
        
        self.max_dof = max_dof
        self._spatial_dim = spatial_dim
        self.max_segments = max_segments
        self.goal_threshold = goal_threshold
        self._n_sampled_goals = n_sampled_goals
        self._her_strategy = her_strategy
        
        # Multi-DOF state indexing
        # State: [joint_angles(max_dof), segment_lengths(max_segments), achieved_goal(3), desired_goal(3), valid_mask(max_segments)]
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
        
        print(f"✅ Multi-DOF Variable Arm HER Buffer: capacity={capacity:,}, max_dof={max_dof}, max_segments={max_segments}")
        print(f"   HER strategy: {her_strategy}, goals: {n_sampled_goals}")
        print(f"   Goal threshold: {goal_threshold}")
    
    def _extract_goals_from_state(self, state: torch.Tensor):
        """Extract achieved and desired goals from multi-DOF state"""
        # Multi-DOF format: [joint_angles(max_dof), segment_lengths(max_segments), achieved_goal(3), desired_goal(3), valid_mask(max_segments)]
        achieved_goal = state[self.achieved_start:self.achieved_end]
        desired_goal = state[self.desired_start:self.desired_end]
        segment_lengths = state[self.length_start:self.length_end]
        valid_mask = state[self.mask_start:self.mask_end]
        return achieved_goal, desired_goal, segment_lengths, valid_mask
    
    def _create_her_state(self, original_state: torch.Tensor, new_desired_goal: torch.Tensor) -> torch.Tensor:
        """Create HER state by replacing desired goal in multi-DOF format"""
        her_state = original_state.clone()
        # Replace only the desired_goal part, keeping all other components
        her_state[self.desired_start:self.desired_end] = new_desired_goal
        return her_state
    
    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        terminated: bool = False,
        truncated: bool = False,
        curr_available_actions=None,
        next_state: SubjectiveState | None = None,
        next_available_actions=None,
        max_number_actions: int | None = None,
        cost: Reward | None = None,
    ) -> None:
        
        # Early return if no next_state
        if next_state is None:
            return
        
        # Store experience in episode buffer for HER processing
        # No need for manual tensor conversion - Pearl framework handles it now
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
        
        # Store original experience using fixed framework (now supports 0-dim tensors!)
        super().push(
            state, action, reward, terminated, truncated,
            curr_available_actions, next_state, next_available_actions,
            max_number_actions, cost
        )
        
        # Process HER experiences when episode ends
        # Extract boolean values for episode termination check
        terminated_bool = terminated.item() if isinstance(terminated, torch.Tensor) else terminated
        truncated_bool = truncated.item() if isinstance(truncated, torch.Tensor) else truncated
        
        if terminated_bool or truncated_bool:
            self._process_her_episode()
            self._episode_buffer.clear()
    
    def _process_her_episode(self):
        """Process episode buffer to generate HER experiences"""
        if len(self._episode_buffer) < 2:
            return
        
        episode_length = len(self._episode_buffer)
        
        # Extract all achieved goals from the episode
        achieved_goals = []
        for exp in self._episode_buffer:
            achieved_goal, _, _, _ = self._extract_goals_from_state(exp['next_state'])  # Use next_state for achieved goal
            achieved_goals.append(achieved_goal)
        
        # Generate HER experiences
        for t in range(episode_length):
            exp = self._episode_buffer[t]
            
            # Sample future achieved goals based on strategy
            future_indices = self._sample_future_goals(t, episode_length)
            
            for future_idx in future_indices:
                # Use future achieved goal as new desired goal
                future_achieved_goal = achieved_goals[future_idx]
                
                # Create HER states
                her_state = self._create_her_state(exp['state'], future_achieved_goal)
                her_next_state = self._create_her_state(exp['next_state'], future_achieved_goal)
                
                # Compute HER reward and termination using goal-based logic
                her_achieved_goal, _, _, _ = self._extract_goals_from_state(her_next_state)
                distance = torch.norm(her_achieved_goal - future_achieved_goal)
                
                if distance <= self.goal_threshold:
                    her_reward = 50.0  # Success reward
                    her_terminated = True
                else:
                    her_reward = -1.0  # Continue reward
                    her_terminated = False
                
                # Store HER experience using simplified approach - Pearl framework handles tensor conversion
                self._store_transition(
                    state=her_state,
                    action=exp['action'],
                    reward=her_reward,  # No manual tensor conversion needed
                    terminated=her_terminated,
                    truncated=False,
                    curr_available_actions_tensor_with_padding=None,
                    curr_unavailable_actions_mask=None,
                    next_state=her_next_state,
                    next_available_actions_tensor_with_padding=None,
                    next_unavailable_actions_mask=None,
                    cost=None,
                )
    
    def _sample_future_goals(self, current_timestep: int, episode_length: int) -> List[int]:
        """Sample future timesteps for HER goal replacement"""
        if self._her_strategy == "future":
            # Sample from future timesteps
            future_timesteps = list(range(current_timestep + 1, episode_length))
            if not future_timesteps:
                return []
            
            n_goals = min(self._n_sampled_goals, len(future_timesteps))
            return np.random.choice(future_timesteps, size=n_goals, replace=False).tolist()
        
        elif self._her_strategy == "episode":
            # Sample from entire episode
            all_timesteps = list(range(episode_length))
            n_goals = min(self._n_sampled_goals, len(all_timesteps))
            return np.random.choice(all_timesteps, size=n_goals, replace=False).tolist()
        
        else:
            raise ValueError(f"Unknown HER strategy: {self._her_strategy}")


def create_variable_arm_her_buffer(
    capacity: int,
    dof: int,
    spatial_dim: int = 3,
    n_segments: int = 3,
    reward_fn: Callable[[SubjectiveState, Action], float] = lambda x, y: 0.0,
    terminated_fn: Callable[[SubjectiveState, Action], bool] = lambda x, y: False,
    n_sampled_goals: int = 4,
    include_lengths_in_obs: bool = True,
) -> MultiDOFVariableArmHERBuffer:
    """Factory function to create variable arm HER buffer"""
    return MultiDOFVariableArmHERBuffer(
        capacity=capacity,
        max_dof=dof,
        spatial_dim=spatial_dim,
        max_segments=n_segments,
        goal_threshold=0.15,
        n_sampled_goals=n_sampled_goals,
        her_strategy="future",
    )


def create_multi_dof_variable_arm_her_buffer_working(
    capacity: int,
    max_dof: int = 8,
    spatial_dim: int = 3,
    max_segments: int = 4,
    goal_threshold: float = 0.15,
    n_sampled_goals: int = 4,
    her_strategy: str = "future",
) -> MultiDOFVariableArmHERBuffer:
    """Factory function to create multi-DOF variable arm HER buffer (WORKING VERSION)"""
    return MultiDOFVariableArmHERBuffer(
        capacity=capacity,
        max_dof=max_dof,
        spatial_dim=spatial_dim,
        max_segments=max_segments,
        goal_threshold=goal_threshold,
        n_sampled_goals=n_sampled_goals,
        her_strategy=her_strategy,
    )