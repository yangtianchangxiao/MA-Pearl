# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Optional, Tuple

import numpy as np
import torch
from pearl.api.action import Action
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace
from pearl.api.environment import Environment
from pearl.api.observation import Observation
from pearl.api.reward import Reward
from pearl.api.space import Space
from pearl.utils.instantiations.spaces.box import BoxSpace
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace


class NDOFArmEnvironment(Environment):
    """
    N-DOF robotic arm reaching environment with goal-conditioned rewards.
    
    State: [joint_angles, end_effector_pos, goal_pos]
    Action: joint_velocity_deltas (continuous)
    Reward: sparse (1.0 if within threshold, 0.0 otherwise)
    
    Args:
        dof: degrees of freedom (3 for 2D, 4-5 for 3D)
        link_lengths: length of each link
        workspace_bounds: [min_x, max_x, min_y, max_y] for 2D or [..., min_z, max_z] for 3D
        goal_threshold: distance threshold to consider goal reached
        max_steps: maximum steps per episode
    """

    def __init__(
        self,
        dof: int = 3,
        link_lengths: Optional[list[float]] = None,
        workspace_bounds: Optional[list[float]] = None,
        goal_threshold: float = 0.30,
        max_steps: int = 200,
    ) -> None:
        self.dof = dof
        self.spatial_dim = 2 if dof == 3 else 3
        
        self.link_lengths = link_lengths or [1.0] * dof
        assert len(self.link_lengths) == dof
        
        if workspace_bounds is None:
            total_reach = sum(self.link_lengths)
            if self.spatial_dim == 2:
                self.workspace_bounds = [-total_reach, total_reach, -total_reach, total_reach]
            else:
                self.workspace_bounds = [-total_reach, total_reach, -total_reach, total_reach, 0.0, total_reach]
        else:
            self.workspace_bounds = workspace_bounds
            
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        
        self.joint_angles = np.zeros(dof, dtype=np.float32)
        self.goal_end_pos = np.zeros(self.spatial_dim, dtype=np.float32)
        self.step_count = 0
        
        # Action space: joint velocity deltas - 足够大的范围确保有效探索
        self._action_space = BoxActionSpace(
            low=torch.full((dof,), -1.0),
            high=torch.full((dof,), 1.0)
        )
        
        # 恢复完整状态表示: [joint_angles, achieved_goal, desired_goal]
        # joint_angles: 控制所需的关节状态 (3D)
        # achieved_goal: 当前end_effector位置 (2D) 
        # desired_goal: 目标end_effector位置 (2D)
        state_dim = dof + self.spatial_dim + self.spatial_dim  # 3 + 2 + 2 = 7
        self._observation_space = BoxSpace(
            low=np.full(state_dim, -np.inf),
            high=np.full(state_dim, np.inf)
        )

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def _forward_kinematics(self, angles: np.ndarray) -> np.ndarray:
        """Compute end-effector position from joint angles."""
        if self.spatial_dim == 2:
            # 2D forward kinematics
            x = 0.0
            y = 0.0
            cumulative_angle = 0.0
            
            for i in range(self.dof):
                cumulative_angle += angles[i]
                x += self.link_lengths[i] * math.cos(cumulative_angle)
                y += self.link_lengths[i] * math.sin(cumulative_angle)
                
            return np.array([x, y], dtype=np.float32)
        else:
            # 3D forward kinematics (simplified - assumes first joint is base rotation)
            base_rotation = angles[0]
            # Project remaining joints onto 2D plane, then rotate by base
            planar_angles = angles[1:]
            
            x_planar = 0.0
            y_planar = 0.0
            cumulative_angle = 0.0
            
            for i in range(len(planar_angles)):
                cumulative_angle += planar_angles[i]
                x_planar += self.link_lengths[i + 1] * math.cos(cumulative_angle)
                y_planar += self.link_lengths[i + 1] * math.sin(cumulative_angle)
            
            # Add base link
            x_planar += self.link_lengths[0]
            
            # Rotate by base joint
            x = x_planar * math.cos(base_rotation)
            y = x_planar * math.sin(base_rotation)
            z = y_planar
            
            return np.array([x, y, z], dtype=np.float32)

    def _sample_goal(self) -> np.ndarray:
        """Sample a random goal within workspace bounds."""
        if self.spatial_dim == 2:
            x = np.random.uniform(self.workspace_bounds[0], self.workspace_bounds[1])
            y = np.random.uniform(self.workspace_bounds[2], self.workspace_bounds[3])
            return np.array([x, y], dtype=np.float32)
        else:
            x = np.random.uniform(self.workspace_bounds[0], self.workspace_bounds[1])
            y = np.random.uniform(self.workspace_bounds[2], self.workspace_bounds[3])
            z = np.random.uniform(self.workspace_bounds[4], self.workspace_bounds[5])
            return np.array([x, y, z], dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """完整状态: [joint_angles, achieved_goal, desired_goal]"""
        achieved_goal = self._forward_kinematics(self.joint_angles)
        return np.concatenate([
            self.joint_angles,          # joint_angles: 当前关节角度 (3D)
            achieved_goal,              # achieved_goal: 当前end_effector位置 (2D)
            self.goal_end_pos          # desired_goal: 目标end_effector位置 (2D)
        ], dtype=np.float32)

    def _compute_reward(self, joint_angles: np.ndarray) -> float:
        """Sparse reward: -1 per step + big success reward."""
        current_end_pos = self._forward_kinematics(joint_angles)
        end_distance = np.linalg.norm(current_end_pos - self.goal_end_pos)
        
        if end_distance <= self.goal_threshold:
            return 50.0  # Big success reward - terminate episode
        else:
            return -1.0  # Step penalty - encourages faster completion

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, ActionSpace]:
        if seed is not None:
            np.random.seed(seed)
            
        # 更中心化的初始关节配置，避免极端角度
        self.joint_angles = np.random.uniform(-math.pi/6, math.pi/6, size=self.dof).astype(np.float32)
        
        # Generate goal within workspace bounds - independent of current position
        if self.spatial_dim == 2:
            # 2D workspace: 3-link arm保守可达范围
            workspace_radius = 0.8  # 极度保守，确保所有目标都可达
            attempts = 0
            while attempts < 50:
                # Uniform sampling in workspace
                x = np.random.uniform(-workspace_radius, workspace_radius)
                y = np.random.uniform(-workspace_radius, workspace_radius)
                candidate_goal = np.array([x, y], dtype=np.float32)
                
                # Check if goal is reachable (within workspace radius)
                if np.linalg.norm(candidate_goal) <= workspace_radius:
                    self.goal_end_pos = candidate_goal
                    break
                attempts += 1
            
            # Fallback: random point on circle
            if attempts >= 50:
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(0.5, workspace_radius)
                self.goal_end_pos = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle)
                ], dtype=np.float32)
        else:
            # 3D workspace
            workspace_radius = 2.5
            attempts = 0
            while attempts < 50:
                x = np.random.uniform(-workspace_radius, workspace_radius)
                y = np.random.uniform(-workspace_radius, workspace_radius) 
                z = np.random.uniform(0, workspace_radius)  # Above ground
                candidate_goal = np.array([x, y, z], dtype=np.float32)
                
                if np.linalg.norm(candidate_goal) <= workspace_radius:
                    self.goal_end_pos = candidate_goal
                    break
                attempts += 1
            
            if attempts >= 50:
                angle1 = np.random.uniform(0, 2*np.pi)
                angle2 = np.random.uniform(0, np.pi)
                radius = np.random.uniform(0.5, workspace_radius)
                self.goal_end_pos = np.array([
                    radius * np.sin(angle2) * np.cos(angle1),
                    radius * np.sin(angle2) * np.sin(angle1),
                    radius * np.cos(angle2)
                ], dtype=np.float32)
        
        # No need for goal joint angles - only end position matters
        
        self.step_count = 0
        
        observation = torch.from_numpy(self._get_observation())
        return observation, self.action_space

    def step(self, action: Action) -> ActionResult:
        # Apply action (joint velocity deltas) with proper scaling
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)
        
        # Scale action for high-frequency control with full 360° reachability
        # 保证每关节可完整旋转: 550步 × 0.012 = 6.6 rad > 2π rad
        action_scale = 0.012
        self.joint_angles += action_np * action_scale
        
        # Clip joint angles to reasonable range
        self.joint_angles = np.clip(self.joint_angles, -math.pi, math.pi)
        
        # Compute reward
        reward = self._compute_reward(self.joint_angles)
        
        # Check termination
        self.step_count += 1
        terminated = reward > 0.0  # Goal reached (success reward > 0)
        truncated = self.step_count >= self.max_steps
        
        observation = torch.from_numpy(self._get_observation())
        
        return ActionResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def render(self) -> None:
        """Optional: basic text output for debugging."""
        current_end_pos = self._forward_kinematics(self.joint_angles)
        joint_distance = np.linalg.norm(self.joint_angles - self.goal_joint_angles)
        end_distance = np.linalg.norm(current_end_pos - self.goal_end_pos)
        print(f"Step {self.step_count}: "
              f"End pos {current_end_pos} -> Goal {self.goal_end_pos} (dist: {end_distance:.3f}), "
              f"Joints {self.joint_angles} -> Goal {self.goal_joint_angles} (dist: {joint_distance:.3f})")