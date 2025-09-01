# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Graph neural network components for robotic applications.
Provides reusable graph blocks for kinematic chains and articulated structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import math


class StructuralEmbedding(nn.Module):
    """
    Encodes structural properties of robotic joints into embeddings.
    Handles segment lengths, joint types, and positional information.
    """
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Joint type embedding: revolute=0, prismatic=1, fixed=2, end_effector=3
        self.joint_type_embedding = nn.Embedding(4, embedding_dim // 4)
        
        # Physical property encoders
        self.length_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # DOF encoder (degrees of freedom per joint)
        self.dof_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4)
        )
        
        # Positional encoding along kinematic chain
        self.position_embedding = nn.Embedding(20, embedding_dim // 4)
        
    def forward(self, segment_lengths: Tensor, joint_types: Tensor, positions: Tensor) -> Tensor:
        """
        Args:
            segment_lengths: [batch_size, max_segments] Joint segment lengths
            joint_types: [batch_size, max_segments] Joint type IDs
            positions: [batch_size, max_segments] Position indices in kinematic chain
        Returns:
            structural_embedding: [batch_size, max_segments, embedding_dim] Combined structural features
        """
        # Generate individual embeddings
        type_emb = self.joint_type_embedding(joint_types)
        length_emb = self.length_encoder(segment_lengths.unsqueeze(-1))
        dof_emb = self.dof_encoder(torch.ones_like(segment_lengths.unsqueeze(-1)) * 2)  # Assume 2-DOF joints
        pos_emb = self.position_embedding(positions)
        
        # Concatenate all structural features
        structural_embedding = torch.cat([type_emb, length_emb, dof_emb, pos_emb], dim=-1)
        
        return structural_embedding


class GraphAttentionLayer(nn.Module):
    """
    Multi-head attention layer for graph neural networks.
    Processes relationships between joints in kinematic chains.
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # Multi-head attention with batch_first=True for efficiency
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection if dimensions differ
        if in_dim != out_dim:
            self.output_proj = nn.Linear(in_dim, out_dim)
        else:
            self.output_proj = nn.Identity()
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features: Tensor, node_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            node_features: [batch_size, max_nodes, in_dim] Node feature tensor
            node_mask: [batch_size, max_nodes] Boolean mask for valid nodes
        Returns:
            output: [batch_size, max_nodes, out_dim] Updated node features
        """
        # Self-attention over nodes
        attended_features, _ = self.attention(
            node_features, node_features, node_features,
            key_padding_mask=~node_mask if node_mask is not None else None
        )
        
        # Project to output dimension
        attended_features = self.output_proj(attended_features)
        
        # Residual connection and normalization
        if self.in_dim == self.out_dim:
            output = self.layer_norm(attended_features + node_features)
        else:
            output = self.layer_norm(attended_features)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Apply node mask
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)
        
        return output


class KinematicChainProcessor(nn.Module):
    """
    Processes kinematic chain structure of robotic manipulators.
    Handles sequential dependencies between joints.
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM to capture sequential dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Output projection to maintain dimension
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features: Tensor, node_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            node_features: [batch_size, max_nodes, hidden_dim] Sequential joint features
            node_mask: [batch_size, max_nodes] Boolean mask for valid joints
        Returns:
            output: [batch_size, max_nodes, hidden_dim] Sequence-aware features
        """
        # Process through LSTM
        lstm_out, _ = self.lstm(node_features)
        
        # Project back to original dimension
        output = self.output_proj(lstm_out)
        
        # Apply mask if provided
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)
        
        return output


class RobotGraphTransformer(nn.Module):
    """
    Complete Graph Transformer for robotic manipulators.
    Combines structural embeddings, attention, and sequential processing.
    """
    def __init__(
        self,
        node_feature_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        max_nodes: int = 10,
        num_heads: int = 4,
        structural_embedding_dim: int = 32,
        use_kinematic_chain: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.use_kinematic_chain = use_kinematic_chain
        
        # Structural embedding module
        self.structural_embedding = StructuralEmbedding(structural_embedding_dim)
        
        # Input feature projection
        input_dim = node_feature_dim + structural_embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Kinematic chain processor (optional)
        if use_kinematic_chain:
            self.kinematic_processor = KinematicChainProcessor(hidden_dim)
        
        # Global feature aggregation
        self.global_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def parse_robot_observation(self, observations: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        Parses robot observations into components.
        
        Args:
            observations: [batch_size, obs_dim] Raw observations
        Returns:
            joint_angles: [batch_size, n_joints] Joint angle values
            segment_lengths: [batch_size, n_segments] Physical segment lengths  
            achieved_goal: [batch_size, goal_dim] Current end-effector position
            desired_goal: [batch_size, goal_dim] Target end-effector position
            n_segments: Number of segments in the manipulator
        """
        obs_dim = observations.shape[1]
        
        if obs_dim == 15:  # Variable-length soft arm: 6 joints + 3 lengths + 3 achieved + 3 desired
            joint_angles = observations[:, :6]
            segment_lengths = observations[:, 6:9]
            achieved_goal = observations[:, 9:12]
            desired_goal = observations[:, 12:15]
            n_segments = 3
        elif obs_dim == 7:  # NDOF arm: 3 joints + 2 achieved + 2 desired
            joint_angles = observations[:, :3]
            segment_lengths = torch.ones(observations.shape[0], 3, device=observations.device)  # Default lengths
            achieved_goal = observations[:, 3:5]
            desired_goal = observations[:, 5:7]
            n_segments = 3
        else:
            raise ValueError(f"Unsupported observation dimension: {obs_dim}")
        
        return joint_angles, segment_lengths, achieved_goal, desired_goal, n_segments
    
    def create_node_features(self, joint_angles: Tensor, achieved_goal: Tensor, 
                           desired_goal: Tensor, n_segments: int) -> Tensor:
        """
        Creates node features for each joint in the manipulator.
        
        Args:
            joint_angles: [batch_size, n_joints] Joint configurations
            achieved_goal: [batch_size, goal_dim] Current end-effector state
            desired_goal: [batch_size, goal_dim] Target end-effector state
            n_segments: Number of manipulator segments
        Returns:
            node_features: [batch_size, n_segments, node_feature_dim] Per-node features
        """
        batch_size = joint_angles.shape[0]
        device = joint_angles.device
        
        # Calculate degrees of freedom per segment
        total_dof = joint_angles.shape[1]
        dof_per_segment = total_dof // n_segments
        
        node_features = []
        for i in range(n_segments):
            # Extract joint angles for this segment
            start_idx = i * dof_per_segment
            end_idx = (i + 1) * dof_per_segment
            segment_joints = joint_angles[:, start_idx:end_idx]
            
            # Combine with global goal information
            node_feat = torch.cat([
                segment_joints,    # Local joint state
                achieved_goal,     # Global current state
                desired_goal       # Global target state
            ], dim=1)
            node_features.append(node_feat)
        
        return torch.stack(node_features, dim=1)
    
    def forward(self, observations: Tensor, segment_lengths: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the robot graph transformer.
        
        Args:
            observations: [batch_size, obs_dim] Raw robot observations
            segment_lengths: [batch_size, n_segments] Optional explicit segment lengths
        Returns:
            node_features: [batch_size, max_nodes, hidden_dim] Per-node representations
            graph_features: [batch_size, hidden_dim] Global graph representation
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        # Parse observations
        joint_angles, seg_lengths, achieved_goal, desired_goal, n_segments = self.parse_robot_observation(observations)
        
        # Use provided lengths if available
        if segment_lengths is not None:
            seg_lengths = segment_lengths
        
        # Create node features
        node_features = self.create_node_features(joint_angles, achieved_goal, desired_goal, n_segments)
        
        # Create structural embeddings
        joint_types = torch.zeros(batch_size, n_segments, dtype=torch.long, device=device)  # Default: revolute joints
        positions = torch.arange(n_segments, device=device).unsqueeze(0).expand(batch_size, -1)
        
        structural_emb = self.structural_embedding(seg_lengths, joint_types, positions)
        
        # Combine state and structural features
        combined_features = torch.cat([node_features, structural_emb], dim=-1)
        
        # Project to hidden dimension
        x = self.input_projection(combined_features)
        
        # Create node validity mask
        node_mask = torch.ones(batch_size, n_segments, dtype=torch.bool, device=device)
        
        # Apply graph attention layers
        for attention_layer in self.attention_layers:
            x = attention_layer(x, node_mask)
        
        # Apply kinematic chain processing
        if self.use_kinematic_chain:
            x = self.kinematic_processor(x, node_mask)
        
        # Global feature aggregation (mean pooling)
        graph_features = torch.mean(x * node_mask.unsqueeze(-1), dim=1)
        graph_features = self.global_pooling(graph_features)
        
        return x, graph_features