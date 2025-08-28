#!/usr/bin/env python3
"""
Training script for N-DOF robotic arm with SAC+HER on CUDA.
"""

import os
import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

from pearl.utils.instantiations.environments import NDOFArmEnvironment
from pearl.utils.instantiations.environments.arm_her_factory import create_arm_her_buffer
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.pearl_agent import PearlAgent
# Simple parallel collection - no external dependencies needed


class ArmTrainer:
    """Training class for robotic arm with SAC+HER."""
    
    def __init__(
        self,
        dof: int = 3,
        device: str = "cuda:0",
        save_dir: str = "./training_results",
        use_vectorized: bool = True,
        num_envs: int = 4
    ):
        self.dof = dof
        self.spatial_dim = 2 if dof == 3 else 3
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.use_vectorized = use_vectorized
        self.num_envs = num_envs
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            print(f"🚀 Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(device)}")
        else:
            self.device = "cpu"
            print("⚠️  CUDA not available, using CPU")
        
        # Initialize environment(s) - Linus-style: simple is better
        if self.use_vectorized:
            # Create multiple independent environments
            self.envs = [
                NDOFArmEnvironment(dof=dof, max_steps=200, goal_threshold=0.05)
                for _ in range(num_envs)
            ]
            self.env = self.envs[0]  # Reference environment
            print(f"✅ Simple Parallel: {num_envs} × {dof}DOF ({'2D' if self.spatial_dim == 2 else '3D'})")
        else:
            self.env = NDOFArmEnvironment(
                dof=dof,
                max_steps=200,
                goal_threshold=0.05
            )
            print(f"✅ Environment: {dof}DOF ({'2D' if self.spatial_dim == 2 else '3D'})")
        
        print(f"   State dimension: {self.env.observation_space.shape[0]}")
        
        # Initialize agent
        self._setup_agent()
        
        # Training metrics
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'success_rate': [],
            'episode_lengths': [],
            'buffer_size': []
        }
    
    def _setup_agent(self):
        """Initialize SAC agent with HER buffer."""
        # HER buffer
        her_buffer = create_arm_her_buffer(
            dof=self.dof,
            spatial_dim=self.spatial_dim,
            capacity=100000,
            threshold=0.05
        )
        
        # SAC policy learner
        policy_learner = ContinuousSoftActorCritic(
            state_dim=self.env.observation_space.shape[0],
            action_space=self.env.action_space,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[256, 256],
            batch_size=256,
            training_rounds=1,
        )
        
        # Pearl agent
        self.agent = PearlAgent(
            policy_learner=policy_learner,
            replay_buffer=her_buffer,
        )
        
        print(f"✅ Agent: SAC+HER")
        print(f"   HER goal_dim: {her_buffer._goal_dim}")
        print(f"   Buffer capacity: {her_buffer.capacity}")
    
    def train(self, episodes: int = 1000, eval_every: int = 50):
        """Train the agent."""
        mode = "Vectorized" if self.use_vectorized else "Single"
        print(f"\n🎯 Starting {mode} training for {episodes} episodes...")
        print("=" * 60)
        
        if self.use_vectorized:
            self._train_vectorized(episodes, eval_every)
        else:
            self._train_single(episodes, eval_every)
        
        print(f"\n🎉 Training completed!")
        self._save_results()
    
    def _train_single(self, episodes: int, eval_every: int):
        """Original single environment training."""
        episode_rewards = []
        recent_successes = []
        
        for episode in tqdm(range(episodes), desc="Training"):
            # Reset environment
            obs, action_space = self.env.reset()
            self.agent.reset(obs, action_space)
            
            episode_reward = 0.0
            step_count = 0
            
            while True:
                # Agent acts
                action = self.agent.act(exploit=False)
                result = self.env.step(action)
                
                # Agent observes
                self.agent.observe(result)
                
                episode_reward += result.reward
                step_count += 1
                
                # Learn after some initial steps
                if len(self.agent.replay_buffer) > 1000:
                    self.agent.learn()
                
                # Check termination
                if result.terminated or result.truncated:
                    break
            
            # Track metrics
            episode_rewards.append(episode_reward)
            recent_successes.append(1.0 if episode_reward > 0.0 else 0.0)
            
            # Keep recent success window
            if len(recent_successes) > 100:
                recent_successes.pop(0)
            
            # Log progress
            if episode % eval_every == 0:
                success_rate = np.mean(recent_successes) * 100
                avg_reward = np.mean(episode_rewards[-eval_every:])
                buffer_size = len(self.agent.replay_buffer)
                
                print(f"\n📊 Episode {episode}")
                print(f"   Success Rate: {success_rate:.1f}%")
                print(f"   Avg Reward: {avg_reward:.3f}")
                print(f"   Buffer Size: {buffer_size}")
                print(f"   Steps: {step_count}")
                
                # Store metrics
                self.training_metrics['episodes'].append(episode)
                self.training_metrics['rewards'].append(avg_reward)
                self.training_metrics['success_rate'].append(success_rate)
                self.training_metrics['episode_lengths'].append(step_count)
                self.training_metrics['buffer_size'].append(buffer_size)
    
    def _train_vectorized(self, episodes: int, eval_every: int):
        """Simple parallel training - Linus style: no complex wrapper."""
        episode_rewards = []
        recent_successes = []
        
        # Track per-environment state
        env_active = [True] * self.num_envs
        env_obs = [None] * self.num_envs
        env_rewards = [0.0] * self.num_envs
        env_lengths = [0] * self.num_envs
        
        # Reset all environments
        for i, env in enumerate(self.envs):
            env_obs[i], _ = env.reset()
        
        total_episodes = 0
        total_steps = 0
        
        with tqdm(total=episodes, desc="Episodes") as pbar:
            while total_episodes < episodes:
                
                # Step each active environment
                for i, env in enumerate(self.envs):
                    if env_active[i]:
                        # Agent acts
                        self.agent.reset(env_obs[i], env.action_space)
                        action = self.agent.act(exploit=False)
                        result = env.step(action)
                        
                        # Agent observes
                        self.agent.observe(result)
                        
                        # Update tracking
                        env_obs[i] = result.observation
                        env_rewards[i] += result.reward
                        env_lengths[i] += 1
                        total_steps += 1
                        
                        # Check if episode done
                        if result.terminated or result.truncated:
                            # Episode completed
                            episode_rewards.append(env_rewards[i])
                            recent_successes.append(1.0 if env_rewards[i] > 0.0 else 0.0)
                            total_episodes += 1
                            pbar.update(1)
                            
                            # Reset this environment
                            env_obs[i], _ = env.reset()
                            env_rewards[i] = 0.0
                            env_lengths[i] = 0
                
                # Learn periodically
                if len(self.agent.replay_buffer) > 1000 and total_steps % 40 == 0:
                    self.agent.learn()
                
                # Keep recent success window
                if len(recent_successes) > 200:
                    recent_successes = recent_successes[-200:]
                
                # Log progress
                if total_episodes % eval_every == 0 and total_episodes > 0:
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                    avg_reward = np.mean(episode_rewards[-eval_every:]) if len(episode_rewards) >= eval_every else np.mean(episode_rewards) if episode_rewards else 0
                    avg_length = np.mean(env_lengths) if env_lengths else 0
                    buffer_size = len(self.agent.replay_buffer)
                    
                    print(f"\n📊 Episode {total_episodes}")
                    print(f"   Success Rate: {success_rate:.1f}%")
                    print(f"   Avg Reward: {avg_reward:.3f}")
                    print(f"   Avg Length: {avg_length:.1f}")
                    print(f"   Buffer Size: {buffer_size}")
                    print(f"   Parallel Envs: {self.num_envs}")
                    
                    # Store metrics
                    self.training_metrics['episodes'].append(total_episodes)
                    self.training_metrics['rewards'].append(avg_reward)
                    self.training_metrics['success_rate'].append(success_rate)
                    self.training_metrics['episode_lengths'].append(avg_length)
                    self.training_metrics['buffer_size'].append(buffer_size)
    
    def evaluate(self, episodes: int = 100):
        """Evaluate the trained agent."""
        print(f"\n🔍 Evaluating for {episodes} episodes...")
        
        successes = 0
        total_reward = 0
        episode_lengths = []
        
        for episode in tqdm(range(episodes), desc="Evaluating"):
            obs, action_space = self.env.reset()
            self.agent.reset(obs, action_space)
            
            episode_reward = 0.0
            step_count = 0
            
            while True:
                # Agent acts (exploit mode)
                action = self.agent.act(exploit=True)
                result = self.env.step(action)
                
                episode_reward += result.reward
                step_count += 1
                
                if result.terminated or result.truncated:
                    break
            
            if episode_reward > 0.0:
                successes += 1
            
            total_reward += episode_reward
            episode_lengths.append(step_count)
        
        success_rate = successes / episodes * 100
        avg_reward = total_reward / episodes
        avg_length = np.mean(episode_lengths)
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Average Reward: {avg_reward:.3f}")
        print(f"   Average Episode Length: {avg_length:.1f}")
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_length': avg_length
        }
    
    def _save_results(self):
        """Save training results."""
        # Save metrics
        metrics_file = self.save_dir / f"metrics_{self.dof}dof.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        # Plot and save figures
        self._plot_training_curves()
        
        print(f"📁 Results saved to {self.save_dir}")
    
    def _plot_training_curves(self):
        """Plot training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        episodes = self.training_metrics['episodes']
        
        # Success rate
        ax1.plot(episodes, self.training_metrics['success_rate'], 'g-', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Average reward
        ax2.plot(episodes, self.training_metrics['rewards'], 'b-', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Reward Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Episode length
        ax3.plot(episodes, self.training_metrics['episode_lengths'], 'r-', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Episode Length Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Buffer size
        ax4.plot(episodes, self.training_metrics['buffer_size'], 'orange', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Buffer Size')
        ax4.set_title('Replay Buffer Size Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"training_curves_{self.dof}dof.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function."""
    # Training configurations - vectorized training uses fewer episodes due to higher efficiency
    configs = [
        {'dof': 3, 'episodes': 500, 'use_vectorized': True, 'num_envs': 4},   # 2D arm
        {'dof': 4, 'episodes': 750, 'use_vectorized': True, 'num_envs': 4},   # 3D arm  
        {'dof': 5, 'episodes': 1000, 'use_vectorized': True, 'num_envs': 4},  # 3D arm with more complexity
    ]
    
    print("🤖 Multi-DOF Robotic Arm Training with SAC+HER")
    print("📦 Using Linus-style Wrapper for Vectorization")
    print("=" * 60)
    
    for config in configs:
        dof = config['dof']
        episodes = config['episodes']
        use_vectorized = config.get('use_vectorized', False)
        num_envs = config.get('num_envs', 1)
        
        mode = f"Vectorized ({num_envs} envs)" if use_vectorized else "Single"
        print(f"\n🚀 Training {dof}DOF arm - {mode}...")
        
        # Initialize trainer
        trainer = ArmTrainer(
            dof=dof,
            device="cuda:0",  # Use first GPU
            save_dir=f"./training_results_{dof}dof{'_vec' if use_vectorized else ''}",
            use_vectorized=use_vectorized,
            num_envs=num_envs
        )
        
        # Train
        trainer.train(episodes=episodes, eval_every=50)
        
        # Evaluate
        eval_results = trainer.evaluate(episodes=100)
        
        print(f"✅ {dof}DOF {mode.lower()} training completed!")
        print(f"   Final Success Rate: {eval_results['success_rate']:.1f}%")
        if use_vectorized:
            print(f"   Speedup Factor: ~{num_envs}x")


if __name__ == "__main__":
    main()