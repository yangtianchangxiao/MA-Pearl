# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Variable-length soft arm HER buffer implementation
Supports observation format: [joint_angles, achieved_goal, desired_goal, segment_lengths]
"""

from typing import Callable, Optional, List
import torch
import numpy as np
from pearl.replay_buffers import BasicReplayBuffer
from pearl.api.action import Action
from pearl.api.state import SubjectiveState
from pearl.api.reward import Reward
from pearl.utils.tensor_like import assert_is_tensor_like


class VariableArmHERBuffer(BasicReplayBuffer):
    """
    HER replay buffer for variable soft arm environment.
    
    Observation format: [joint_angles(dof), achieved_goal(3), desired_goal(3), segment_lengths(n_segments)]
    
    The key difference from standard HER is that we need to handle the segment_lengths
    part of the observation when doing HER goal replacement.
    """
    
    def __init__(
        self,
        capacity: int,
        dof: int,  # joint angles dimension
        spatial_dim: int = 3,  # achieved/desired goal dimension
        n_segments: int = 3,  # number of segments (for segment_lengths)
        reward_fn: Callable[[SubjectiveState, Action], float] = lambda x, y: 0.0,
        terminated_fn: Callable[[SubjectiveState, Action], bool] = lambda x, y: False,
        n_sampled_goals: int = 4,  # HER goal sampling count
        her_strategy: str = "future",  # HER strategy
        include_lengths_in_obs: bool = True,  # whether obs includes segment lengths
    ):
        super().__init__(capacity)
        
        self._dof = dof
        self._spatial_dim = spatial_dim
        self._n_segments = n_segments
        self._reward_fn = reward_fn
        self._terminated_fn = terminated_fn
        self._n_sampled_goals = n_sampled_goals
        self._her_strategy = her_strategy
        self._include_lengths_in_obs = include_lengths_in_obs
        
        # Episode buffer for HER processing
        self._episode_buffer: List[dict] = []
        
        print(f"✅ Variable Arm HER Buffer: capacity={capacity:,}, dof={dof}, segments={n_segments}")
        print(f"   HER strategy: {her_strategy}, goals: {n_sampled_goals}")
        print(f"   Include lengths: {include_lengths_in_obs}")
    
    def _extract_goals_from_state(self, state: torch.Tensor):
        """Extract achieved and desired goals from state"""
        if self._include_lengths_in_obs:
            # Format: [joint_angles(dof), achieved_goal(3), desired_goal(3), segment_lengths(n_segments)]
            achieved_goal = state[self._dof:self._dof + self._spatial_dim]
            desired_goal = state[self._dof + self._spatial_dim:self._dof + 2*self._spatial_dim]
            segment_lengths = state[self._dof + 2*self._spatial_dim:]
            return achieved_goal, desired_goal, segment_lengths
        else:
            # Standard format: [joint_angles(dof), achieved_goal(3), desired_goal(3)]
            achieved_goal = state[self._dof:self._dof + self._spatial_dim]
            desired_goal = state[self._dof + self._spatial_dim:]
            return achieved_goal, desired_goal, None
    
    def _create_her_state(self, original_state: torch.Tensor, new_desired_goal: torch.Tensor) -> torch.Tensor:
        """Create HER state by replacing desired goal"""
        her_state = original_state.clone()
        
        if self._include_lengths_in_obs:
            # Replace desired_goal while keeping joint_angles, achieved_goal, and segment_lengths
            her_state[self._dof + self._spatial_dim:self._dof + 2*self._spatial_dim] = new_desired_goal
        else:
            # Standard replacement
            her_state[self._dof + self._spatial_dim:] = new_desired_goal
            
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
            achieved_goal, _, _ = self._extract_goals_from_state(exp['state'])
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
                
                # Compute HER reward and termination
                her_reward = self._reward_fn(her_state, exp['action'])
                her_terminated = self._terminated_fn(her_state, exp['action'])
                
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
) -> VariableArmHERBuffer:
    """Factory function to create variable arm HER buffer"""
    return VariableArmHERBuffer(
        capacity=capacity,
        dof=dof,
        spatial_dim=spatial_dim,
        n_segments=n_segments,
        reward_fn=reward_fn,
        terminated_fn=terminated_fn,
        n_sampled_goals=n_sampled_goals,
        include_lengths_in_obs=include_lengths_in_obs,
    )