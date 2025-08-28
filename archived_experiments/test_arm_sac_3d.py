#!/usr/bin/env python3

"""
Test script for 4-5DOF arm environment (3D) with SAC.
"""

import torch
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.replay_buffers.basic_replay_buffer import BasicReplayBuffer
from pearl.utils.instantiations.environments import NDOFArmEnvironment


def create_ndof_arm_agent(action_space, state_dim: int, name: str) -> PearlAgent:
    """Create SAC agent for N-DOF arm."""
    return PearlAgent(
        policy_learner=ContinuousSoftActorCritic(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            batch_size=64,
            training_rounds=1,
        ),
        replay_buffer=BasicReplayBuffer(capacity=10000),
    )


def test_4dof_arm():
    """Test 4DOF (3D) arm environment."""
    print("\n=== Testing 4DOF (3D) Arm Environment ===")
    
    env = NDOFArmEnvironment(
        dof=4,
        link_lengths=[1.0, 0.8, 0.6, 0.4],
        goal_threshold=0.15,
        max_steps=50
    )
    
    state_dim = env.observation_space.shape[0]  # 4 joint + 3 end_pos + 3 goal_pos = 10
    action_dim = env.action_space.shape[0]  # 4 joint velocities
    print(f"4DOF - State dim: {state_dim}, Action dim: {action_dim}")
    
    agent = create_ndof_arm_agent(env.action_space, state_dim, "4DOF")
    
    # Quick test
    observation, action_space = env.reset()
    agent.reset(observation, action_space)
    
    for step in range(10):
        action = agent.act(exploit=False)
        action_result = env.step(action)
        agent.observe(action_result)
        if step > 3:
            agent.learn()
    
    env.render()
    print("✅ 4DOF test passed!")


def test_5dof_arm():
    """Test 5DOF (3D) arm environment."""
    print("\n=== Testing 5DOF (3D) Arm Environment ===")
    
    env = NDOFArmEnvironment(
        dof=5,
        link_lengths=[1.0, 0.8, 0.6, 0.4, 0.3],
        goal_threshold=0.15,
        max_steps=50
    )
    
    state_dim = env.observation_space.shape[0]  # 5 joint + 3 end_pos + 3 goal_pos = 11
    action_dim = env.action_space.shape[0]  # 5 joint velocities
    print(f"5DOF - State dim: {state_dim}, Action dim: {action_dim}")
    
    agent = create_ndof_arm_agent(env.action_space, state_dim, "5DOF")
    
    # Quick test
    observation, action_space = env.reset()
    agent.reset(observation, action_space)
    
    for step in range(10):
        action = agent.act(exploit=False)
        action_result = env.step(action)
        agent.observe(action_result)
        if step > 3:
            agent.learn()
    
    env.render()
    print("✅ 5DOF test passed!")


def test_workspace_bounds():
    """Test different workspace configurations."""
    print("\n=== Testing Workspace Bounds ===")
    
    # Smaller workspace for 4DOF
    env_small = NDOFArmEnvironment(
        dof=4,
        workspace_bounds=[-1.5, 1.5, -1.5, 1.5, 0.0, 1.5],
        goal_threshold=0.1
    )
    
    observation, _ = env_small.reset(seed=42)
    state = observation.cpu().numpy()
    
    # Extract goal position (last 3 elements)
    goal_pos = state[-3:]
    print(f"Goal in small workspace: {goal_pos}")
    
    # Check bounds
    assert -1.5 <= goal_pos[0] <= 1.5
    assert -1.5 <= goal_pos[1] <= 1.5
    assert 0.0 <= goal_pos[2] <= 1.5
    
    print("✅ Workspace bounds test passed!")


if __name__ == "__main__":
    test_4dof_arm()
    test_5dof_arm()
    test_workspace_bounds()
    print("\n🎯 All 3D arm tests completed successfully!")