#!/usr/bin/env python3
"""
Fixed complex kinematics training - using working debug logic
"""

import torch
import time
import numpy as np
from pathlib import Path

from complex_kinematics_her_wrapper import ComplexKinematicsHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer

def run_single_episode(agent, env, config, total_steps_ref):
    """Run single episode with step-level learning like original"""
    try:
        obs, action_space = env.reset()
        agent.reset(obs, action_space)
        
        episode_reward = 0
        step_count = 0
        episode_success = False  # 初始化为False，像原版一样
        
        for step in range(config['max_episode_steps']):
            # Agent action
            action = agent.act(exploit=False)
            
            # Environment step
            action_result = env.step(action)
            
            episode_reward += action_result.reward.item()
            total_steps_ref[0] += 1  # Update global step counter
            step_count += 1
            
            # Agent observe
            agent.observe(action_result)
            
            # Learning - 每50步学习一次 (像原来一样)
            if total_steps_ref[0] >= config['learning_starts'] and total_steps_ref[0] % config['learn_every'] == 0:
                agent.learn()
            
            if action_result.terminated or action_result.truncated:
                episode_success = action_result.terminated.item()
                break
        # 如果没有提前终止，episode_success保持初始值False
        
        final_distance = action_result.info.get('distance', float('inf'))
        
        return episode_reward, episode_success, final_distance, step_count
        
    except Exception as e:
        print(f"❌ Episode error: {e}")
        return -200, 0, float('inf'), 1  # Return failure values

def main():
    """Main training function"""
    print("🚀 Fixed Complex Kinematics Training")
    
    # Same config as before
    config = {
        'dof_range': (2, 5),
        'segment_length_range': (0.1, 0.35),
        'goal_threshold': 0.10,  # 中等难度阈值
        'max_episode_steps': 200,
        'hidden_dim': 128,
        'num_gnn_layers': 2,
        'critic_hidden_dims': [512, 512],
        'episodes': 5000,  # 与原版一致
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_capacity': 200000,
        'batch_size': 256,
        'training_rounds': 25,
        'learning_starts': 10000,
        'learn_every': 50,
        'save_dir': 'complex_kinematics_fixed_results'
    }
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Create environment (same as debug)
    env = ComplexKinematicsHERWrapper(
        dof_range=config['dof_range'],
        segment_length_range=config['segment_length_range'],
        goal_threshold=config['goal_threshold'],
        max_steps=config['max_episode_steps']
    )
    
    # Create actor (same as debug)
    actor_network = UltraLightGNNActor(
        action_dim=10,
        dof_range=config['dof_range'],
        hidden_dim=config['hidden_dim'],
        num_gnn_layers=config['num_gnn_layers']
    ).to(config['device'])
    
    # Action representation
    action_rep_module = IdentityActionRepresentationModule(
        max_number_actions=10,
        representation_dim=10
    )
    
    # Create SAC (same as debug)
    sac = ContinuousSoftActorCritic(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        actor_network_instance=actor_network,
        critic_hidden_dims=config['critic_hidden_dims'],
        action_representation_module=action_rep_module,
        training_rounds=config['training_rounds'],
        batch_size=config['batch_size']
    )
    
    # Create HER buffer
    replay_buffer = create_variable_soft_arm_her_buffer(
        capacity=config['buffer_capacity'],
        joint_dim=10,
        spatial_dim=3,
        n_segments=5,
        threshold=config['goal_threshold'],
        include_lengths_in_obs=False
    )
    
    # Create Agent (same as debug)
    agent = PearlAgent(
        policy_learner=sac,
        replay_buffer=replay_buffer
    )
    agent._action_space = env.action_space
    
    print("✅ Setup complete - starting training")
    
    # Training loop
    episode_rewards = []
    episode_successes = []
    total_steps_ref = [0]  # Use list to pass by reference
    start_time = time.time()
    
    for episode in range(1, config['episodes'] + 1):
        # Run episode with step-level learning (like original)
        episode_reward, episode_success, final_distance, step_count = run_single_episode(
            agent, env, config, total_steps_ref
        )
        
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)
        
        # Calculate statistics
        recent_success_rate = np.mean(episode_successes[-100:]) * 100
        avg_reward = np.mean(episode_rewards[-100:])
        
        # Print progress (every 10 episodes)
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            eps_per_hour = episode / (elapsed_time / 3600)
            
            print(f"Ep {episode:4d} | Success: {recent_success_rate:5.1f}% | "
                  f"Reward: {avg_reward:7.1f} | Steps: {step_count:3d} | "
                  f"Total Steps: {total_steps_ref[0]} | Speed: {eps_per_hour:.1f} eps/h")
    
    print(f"\\n🎉 Training completed!")
    print(f"Final success rate: {np.mean(episode_successes[-100:]) * 100:.1f}%")

if __name__ == "__main__":
    main()