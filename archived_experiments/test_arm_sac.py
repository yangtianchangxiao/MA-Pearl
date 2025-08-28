#!/usr/bin/env python3

"""
Test script for 3DOF arm environment with SAC.
Simple validation that environment + SAC can run without HER first.
"""

import torch
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.replay_buffers.basic_replay_buffer import BasicReplayBuffer
from pearl.utils.instantiations.environments import NDOFArmEnvironment


def create_3dof_arm_agent(action_space, state_dim: int) -> PearlAgent:
    """Create SAC agent for 3DOF arm."""
    return PearlAgent(
        policy_learner=ContinuousSoftActorCritic(
            state_dim=state_dim,
            action_space=action_space,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            batch_size=64,  # Small batch for testing
            training_rounds=1,
        ),
        replay_buffer=BasicReplayBuffer(capacity=10000),
    )


def test_basic_functionality():
    """Test basic environment + SAC functionality."""
    print("Testing 3DOF Arm Environment with SAC...")
    
    # Create environment
    env = NDOFArmEnvironment(
        dof=3,
        link_lengths=[1.0, 0.8, 0.6],
        goal_threshold=0.1,
        max_steps=50  # Short episodes for testing
    )
    
    # Calculate state dimension
    state_dim = env.observation_space.shape[0]  # 3 joint + 2 end_pos + 2 goal_pos = 7
    action_dim = env.action_space.shape[0]  # 3 joint velocities
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create agent
    agent = create_3dof_arm_agent(env.action_space, state_dim)
    
    # Test episodes
    num_episodes = 5
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        observation, action_space = env.reset()
        agent.reset(observation, action_space)
        
        episode_reward = 0.0
        step = 0
        done = False
        
        while not done and step < 50:
            # Agent acts
            action = agent.act(exploit=False)  # Exploration mode
            
            # Environment step
            action_result = env.step(action)
            
            # Agent observes and learns
            agent.observe(action_result)
            if step > 10:  # Start learning after some exploration
                agent.learn()
            
            episode_reward += action_result.reward
            done = action_result.terminated or action_result.truncated
            step += 1
            
            if step % 10 == 0:
                env.render()
        
        print(f"Episode {episode + 1}: Steps = {step}, Reward = {episode_reward}")
        if episode_reward > 0:
            print("  🎯 Goal reached!")
    
    print("\n✅ Basic functionality test completed!")


if __name__ == "__main__":
    test_basic_functionality()