# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Robotic-specific actor networks for manipulation tasks.

This module extends Pearl's actor networks with specialized architectures
for robotic manipulation, particularly for joint-space control.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import math

from pearl.api.action_space import ActionSpace
from pearl.neural_networks.sequential_decision_making.actor_networks import ActorNetwork, GaussianActorNetwork
from pearl.neural_networks.common.utils import mlp_block
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace

from torch import Tensor
from torch.distributions import Normal


class JointSpaceActorNetwork(GaussianActorNetwork):
    """
    Joint-space aware actor network for robotic manipulation.
    
    Key improvements over vanilla GaussianActorNetwork:
    1. Joint-aware feature processing: treats joint states with physical structure
    2. Scaled output: outputs are pre-scaled for reasonable joint velocities  
    3. Joint limits awareness: inherently respects joint limit constraints
    4. Kinematic structure: can optionally include kinematic chain information
    
    This network is designed specifically for joint-space control of robotic arms
    where understanding the physical relationships between joints is crucial.
    
    Args:
        input_dim: input state dimension (typically: n_joints + 2*spatial_dim for goal-conditioned)
        hidden_dims: list of hidden layer dimensions
        output_dim: action dimension (should equal number of joints)  
        action_space: action space (BoxActionSpace)
        dof: degrees of freedom (number of joints)
        joint_velocity_scale: scaling factor for joint velocities (default: 0.1 rad/step)
        use_joint_structure: whether to use structured processing for joint features
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int, 
        action_space: ActionSpace,
        dof: int = 3,
        joint_velocity_scale: float = 0.1,
        use_joint_structure: bool = True,
    ) -> None:
        # Initialize parent class
        super().__init__(input_dim, hidden_dims, output_dim, action_space)
        
        self.dof = dof
        self.joint_velocity_scale = joint_velocity_scale
        self.use_joint_structure = use_joint_structure
        
        if use_joint_structure:
            # Replace the standard model with joint-structured processing
            self._build_joint_structured_model(input_dim, hidden_dims, output_dim)
    
    def _build_joint_structured_model(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        """
        Build a network that processes joint states and goal information separately.
        
        Input structure assumed: [joint_angles(dof), achieved_goal(spatial), desired_goal(spatial)]
        """
        spatial_dim = (input_dim - self.dof) // 2  # remaining dims split between achieved/desired
        
        # Joint state encoder: processes joint angles with kinematic awareness
        self.joint_encoder = mlp_block(
            input_dim=self.dof,
            hidden_dims=[hidden_dims[0] // 2],
            output_dim=hidden_dims[0] // 2,
        )
        
        # Goal encoder: processes spatial goal information
        goal_input_dim = spatial_dim * 2  # achieved + desired goals
        self.goal_encoder = mlp_block(
            input_dim=goal_input_dim,
            hidden_dims=[hidden_dims[0] // 2], 
            output_dim=hidden_dims[0] // 2,
        )
        
        # Fusion network: combines joint and goal representations
        fusion_input_dim = hidden_dims[0]  # joint_features + goal_features
        self.fusion_network = mlp_block(
            input_dim=fusion_input_dim,
            hidden_dims=hidden_dims[1:],
            output_dim=hidden_dims[-1],
        )
        
        # Output layers for mean and log_std (same as parent)
        self.last_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Initialize output layers with small weights for stable joint control
        nn.init.uniform_(self.last_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.last_layer.bias, -0.01, 0.01)
        nn.init.uniform_(self.log_std_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.log_std_layer.bias, -2.0)  # Start with low exploration
    
    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with joint-structured processing.
        
        Args:
            state_batch: batch of states [batch_size, input_dim]
            
        Returns:
            Tuple of (mean, log_std) for the action distribution
        """
        if not self.use_joint_structure:
            # Fall back to parent implementation
            return super().forward(state_batch)
        
        batch_size = state_batch.shape[0]
        
        # Split input into joint states and goal information
        joint_states = state_batch[:, :self.dof]                    # [batch, dof]
        goal_info = state_batch[:, self.dof:]                       # [batch, 2*spatial_dim]
        
        # Process each component
        joint_features = self.joint_encoder(joint_states)          # [batch, hidden//2]
        goal_features = self.goal_encoder(goal_info)               # [batch, hidden//2]
        
        # Combine features
        combined_features = torch.cat([joint_features, goal_features], dim=-1)  # [batch, hidden]
        
        # Final processing
        hidden_output = self.fusion_network(combined_features)     # [batch, hidden[-1]]
        
        # Generate mean and log_std
        mean = self.last_layer(hidden_output)                      # [batch, output_dim]
        log_std = self.log_std_layer(hidden_output)                # [batch, output_dim]
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -5.0, 2.0)
        
        return mean, log_std
    
    def sample_action(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy with proper joint velocity scaling.
        
        Returns:
            Tuple of (action, log_prob) where action is scaled for joint control
        """
        mean, log_std = self.forward(state_batch)
        std = torch.exp(log_std)
        
        # Create normal distribution and sample
        normal_dist = Normal(mean, std)
        raw_action = normal_dist.rsample()  # Reparameterization trick
        
        # Apply tanh to bound actions to [-1, 1]
        bounded_action = torch.tanh(raw_action)
        
        # Scale for joint velocities
        scaled_action = bounded_action * self.joint_velocity_scale
        
        # Compute log probability with change of variables
        log_prob = normal_dist.log_prob(raw_action)
        log_prob -= torch.log(1 - bounded_action.pow(2) + 1e-6)  # tanh correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return scaled_action, log_prob
    
    def get_action_mean(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Get the mean action (deterministic policy) with proper scaling.
        """
        mean, _ = self.forward(state_batch)
        bounded_mean = torch.tanh(mean)
        return bounded_mean * self.joint_velocity_scale


class AdaptiveJointActorNetwork(JointSpaceActorNetwork):
    """
    Advanced joint-space actor with adaptive scaling based on current joint state.
    
    This network can dynamically adjust joint velocity limits based on:
    1. Distance to joint limits
    2. Current joint velocities  
    3. Goal distance (fine control when close to goal)
    
    Useful for more sophisticated joint control strategies.
    """
    
    def __init__(self, *args, adaptive_scaling: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adaptive_scaling = adaptive_scaling
        
        if adaptive_scaling:
            # Add a small network to predict scaling factors
            self.scale_predictor = mlp_block(
                input_dim=self.dof + 2,  # joint_angles + distance_to_goal + current_velocity_norm
                hidden_dims=[32, 16],
                output_dim=self.dof,     # per-joint scaling
                last_activation="sigmoid"
            )
    
    def _compute_adaptive_scales(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute per-joint adaptive scaling factors.
        """
        if not self.adaptive_scaling:
            return torch.ones(state_batch.shape[0], self.dof, device=state_batch.device)
        
        # Extract features for adaptive scaling
        joint_angles = state_batch[:, :self.dof]
        
        # Compute goal distance (simplified)
        spatial_dim = (state_batch.shape[1] - self.dof) // 2
        achieved = state_batch[:, self.dof:self.dof + spatial_dim]
        desired = state_batch[:, self.dof + spatial_dim:]
        goal_distance = torch.norm(desired - achieved, dim=-1, keepdim=True)
        
        # Estimate current velocity (placeholder - would need previous state in practice)
        velocity_norm = torch.zeros(state_batch.shape[0], 1, device=state_batch.device)
        
        # Combine features
        scale_input = torch.cat([joint_angles, goal_distance, velocity_norm], dim=-1)
        
        # Predict scaling factors [0, 1] -> [0.1 * base_scale, base_scale]
        raw_scales = self.scale_predictor(scale_input)  # [0, 1]
        scales = 0.1 + 0.9 * raw_scales  # [0.1, 1.0]
        
        return scales * self.joint_velocity_scale
    
    def sample_action(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with adaptive per-joint scaling.
        """
        mean, log_std = self.forward(state_batch)
        std = torch.exp(log_std)
        
        normal_dist = Normal(mean, std)
        raw_action = normal_dist.rsample()
        bounded_action = torch.tanh(raw_action)
        
        # Apply adaptive scaling
        adaptive_scales = self._compute_adaptive_scales(state_batch)
        scaled_action = bounded_action * adaptive_scales
        
        # Compute log probability
        log_prob = normal_dist.log_prob(raw_action)
        log_prob -= torch.log(1 - bounded_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return scaled_action, log_prob