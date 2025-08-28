#!/usr/bin/env python3
"""
Official Pearl training script for N-DOF robotic arms with SAC+HER and multiprocessing.

This script is integrated into Pearl's standard structure and follows Pearl conventions.
Supports true multiprocessing with subprocess vector environments.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# Pearl imports
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.utils.instantiations.environments import NDOFArmEnvironment
from pearl.utils.instantiations.environments.arm_her_factory import create_arm_her_buffer
from pearl.user_envs.wrappers.subprocess_vector_env import SubprocVectorEnv


class PearlArmTrainer:
    """
    Official Pearl trainer for robotic arms with SAC+HER multiprocessing.
    
    Integrated into Pearl's utils.scripts structure following Pearl conventions.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        save_dir: str = "./pearl_arm_results"
    ):
        self.config = config
        self.dof = config['dof']
        self.spatial_dim = 2 if self.dof == 3 else 3
        self.num_processes = config['num_processes']
        self.device = config['device']
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup device
        if torch.cuda.is_available() and 'cuda' in self.device:
            torch.cuda.set_device(self.device)
            print(f"🚀 Pearl ARM Training - Device: {self.device}")
            print(f"   GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = "cpu"
            print("⚠️ Pearl ARM Training - Using CPU")
        
        # Initialize components following Pearl patterns
        self._setup_vector_environment()
        self._setup_pearl_agent()
        
        # Training metrics
        self.metrics = {
            'episodes': [],
            'success_rate': [],
            'avg_reward': [],
            'buffer_size': [],
            'throughput': [],
            'config': config
        }
        
        print(f"✅ Pearl ARM Trainer initialized")
        print(f"   DOF: {self.dof} ({'2D' if self.spatial_dim == 2 else '3D'})")
        print(f"   Processes: {self.num_processes}")
        print(f"   Algorithm: SAC + HER")
    
    def _setup_vector_environment(self):
        """Setup multiprocess vector environment following Pearl patterns."""
        def make_env():
            return NDOFArmEnvironment(
                dof=self.dof,
                max_steps=self.config['max_episode_steps'],
                goal_threshold=self.config['goal_threshold']
            )
        
        env_fns = [make_env for _ in range(self.num_processes)]
        self.vec_env = SubprocVectorEnv(env_fns)
        
        print(f"✅ Vector Environment: {self.num_processes} processes")
        print(f"   State dimension: {self.vec_env.observation_space.shape[0]}")
    
    def _setup_pearl_agent(self):
        """Setup Pearl agent with SAC + HER following Pearl conventions."""
        # HER replay buffer
        her_buffer = create_arm_her_buffer(
            dof=self.dof,
            spatial_dim=self.spatial_dim,
            capacity=self.config['buffer_capacity'],
            threshold=self.config['goal_threshold']
        )
        
        # SAC policy learner - 标准版本，稍后添加梯度裁剪
        sac_learner = ContinuousSoftActorCritic(
            state_dim=self.vec_env.observation_space.shape[0],
            action_space=self.vec_env.action_space,
            actor_hidden_dims=self.config['actor_hidden_dims'],
            critic_hidden_dims=self.config['critic_hidden_dims'],
            batch_size=self.config['batch_size'],
            training_rounds=self.config['training_rounds'],
            entropy_coef=0.2,           
            entropy_autotune=True,      
            actor_learning_rate=0.0003,
            critic_learning_rate=0.0003,  # 回到标准学习率
        )
        
        # 添加梯度裁剪的猴子补丁
        original_learn = sac_learner.learn
        def learn_with_grad_clip(replay_buffer):
            result = original_learn(replay_buffer)
            
            # 在每次学习后应用梯度裁剪
            import torch.nn.utils as utils
            utils.clip_grad_norm_(sac_learner._critic.parameters(), 1.0)
            utils.clip_grad_norm_(sac_learner._actor.parameters(), 1.0)
            
            return result
        
        sac_learner.learn = learn_with_grad_clip
        print("✅ 添加梯度裁剪 (clip_norm=1.0) 到SAC")
        
        # Pearl agent
        self.agent = PearlAgent(
            policy_learner=sac_learner,
            replay_buffer=her_buffer,
        )
        
        print(f"✅ Pearl Agent: SAC + HER")
        if hasattr(her_buffer, '_goal_dim'):
            print(f"   HER goal_dim: {her_buffer._goal_dim}")
        elif hasattr(her_buffer, '_spatial_dim'):
            print(f"   HER spatial_dim: {her_buffer._spatial_dim}")
        print(f"   Buffer capacity: {her_buffer.capacity}")
        print(f"   Batch size: {self.config['batch_size']}")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the agent using Pearl's multiprocessing approach.
        
        Returns:
            Training results dictionary
        """
        episodes = self.config['episodes']
        eval_every = self.config['eval_every']
        
        print(f"\n🎯 Pearl ARM Training Started")
        print(f"📊 Target episodes: {episodes}")
        print(f"🔄 Evaluation every: {eval_every}")
        print("=" * 60)
        
        try:
            # Initialize environments
            observations, action_space = self.vec_env.reset()
            
            episode_rewards = []
            recent_successes = []
            completed_episodes = 0
            total_steps = 0
            start_time = time.time()
            
            # Per-environment tracking
            env_rewards = [0.0] * self.num_processes
            env_lengths = [0] * self.num_processes
            
            with tqdm(total=episodes, desc="Episodes", unit="eps") as pbar:
                while completed_episodes < episodes:
                    
                    # Get actions for all environments
                    actions = []
                    for obs in observations:
                        self.agent.reset(obs, action_space)
                        action = self.agent.act(exploit=False)
                        actions.append(action)
                    
                    # Step all environments in parallel
                    self.vec_env.step_async(actions)
                    obs_list, rewards, terminated, truncated, infos = self.vec_env.step_wait()
                    
                    # Process results from all environments
                    for i, (obs, reward, term, trunc) in enumerate(
                        zip(obs_list, rewards, terminated, truncated)
                    ):
                        # Agent observes transition
                        from pearl.api.action_result import ActionResult
                        result = ActionResult(
                            observation=obs,
                            reward=reward,
                            terminated=term,
                            truncated=trunc
                        )
                        self.agent.observe(result)
                        
                        # Update episode tracking
                        env_rewards[i] += reward
                        env_lengths[i] += 1
                        total_steps += 1
                        
                        # Check episode completion
                        if term or trunc:
                            episode_rewards.append(env_rewards[i])
                            # 正确的成功判定：episode以terminated=True结束
                            recent_successes.append(1.0 if term else 0.0)
                            completed_episodes += 1
                            pbar.update(1)
                            
                            # 立即检查是否需要评估 - 放在episode完成时
                            if completed_episodes % eval_every == 0:
                                success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                                avg_reward = np.mean(episode_rewards[-eval_every:]) if len(episode_rewards) >= eval_every else np.mean(episode_rewards) if episode_rewards else 0
                                buffer_size = len(self.agent.replay_buffer)
                                elapsed = time.time() - start_time
                                throughput = completed_episodes / elapsed if elapsed > 0 else 0
                                
                                # 使用pbar.write确保输出不被进度条覆盖
                                pbar.write(f"\n📊 Pearl ARM Training Progress")
                                pbar.write(f"   Episode: {completed_episodes}")
                                pbar.write(f"   Success Rate: {success_rate:.1f}%")
                                pbar.write(f"   Avg Reward: {avg_reward:.3f}")
                                pbar.write(f"   Buffer Size: {buffer_size:,}")
                                pbar.write(f"   Throughput: {throughput:.1f} eps/sec")
                                pbar.write(f"   Total Steps: {total_steps:,}")
                                pbar.write("=" * 60)
                            
                            # Reset tracking
                            env_rewards[i] = 0.0
                            env_lengths[i] = 0
                    
                    # Update observations for next iteration
                    observations = obs_list
                    
                    # Pearl agent learning
                    if len(self.agent.replay_buffer) > self.config['learning_starts'] and \
                       total_steps % self.config['learn_every'] == 0:
                        self.agent.learn()
                    
                    # 评估已经在episode完成时进行，这里不需要重复
        
        finally:
            self.vec_env.close()
            print("✅ Vector environment closed")
        
        # Final results
        final_success_rate = np.mean(recent_successes[-100:]) * 100 if recent_successes else 0.0
        total_time = time.time() - start_time
        
        results = {
            'final_success_rate': final_success_rate,
            'total_episodes': completed_episodes,
            'total_time': total_time,
            'avg_throughput': completed_episodes / total_time,
            'config': self.config
        }
        
        self._save_results(results)
        return results
    
    def _log_progress(
        self, 
        completed_episodes: int, 
        episode_rewards: list, 
        recent_successes: list,
        start_time: float,
        total_steps: int
    ):
        """Log training progress following Pearl conventions."""
        # Keep recent success window
        if len(recent_successes) > 200:
            recent_successes = recent_successes[-200:]
        
        # Calculate metrics
        current_time = time.time()
        elapsed = current_time - start_time
        throughput = completed_episodes / elapsed if elapsed > 0 else 0
        
        eval_every = self.config['eval_every']
        success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
        avg_reward = np.mean(episode_rewards[-eval_every:]) if len(episode_rewards) >= eval_every else np.mean(episode_rewards) if episode_rewards else 0
        buffer_size = len(self.agent.replay_buffer)
        
        print(f"\n📊 Pearl ARM Training Progress")
        print(f"   Episode: {completed_episodes}")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Avg Reward: {avg_reward:.3f}")
        print(f"   Buffer Size: {buffer_size:,}")
        print(f"   Throughput: {throughput:.1f} eps/sec")
        print(f"   Total Steps: {total_steps:,}")

    def _log_progress_tqdm(self, completed_episodes: int, episode_rewards: list, 
                          recent_successes: list, start_time: float, total_steps: int, pbar) -> None:
        """Log training progress using tqdm pbar.write() to avoid being overwritten"""
        current_time = time.time()
        elapsed = current_time - start_time
        throughput = completed_episodes / elapsed if elapsed > 0 else 0
        
        eval_every = self.config['eval_every']
        success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
        avg_reward = np.mean(episode_rewards[-eval_every:]) if len(episode_rewards) >= eval_every else np.mean(episode_rewards) if episode_rewards else 0
        buffer_size = len(self.agent.replay_buffer)
        
        pbar.write(f"\n📊 Pearl ARM Training Progress")
        pbar.write(f"   Episode: {completed_episodes}")
        pbar.write(f"   Success Rate: {success_rate:.1f}%")
        pbar.write(f"   Avg Reward: {avg_reward:.3f}")
        pbar.write(f"   Buffer Size: {buffer_size:,}")
        pbar.write(f"   Throughput: {throughput:.1f} eps/sec")
        pbar.write(f"   Total Steps: {total_steps:,}")
        
        # Store metrics
        self.metrics['episodes'].append(completed_episodes)
        self.metrics['success_rate'].append(success_rate)
        self.metrics['avg_reward'].append(avg_reward)
        self.metrics['buffer_size'].append(buffer_size)
        self.metrics['throughput'].append(throughput)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results and model weights following Pearl conventions."""
        # Save detailed metrics
        metrics_file = self.save_dir / f"pearl_arm_{self.dof}dof_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save final results
        results_file = self.save_dir / f"pearl_arm_{self.dof}dof_results.json"  
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save trained model weights
        model_file = self.save_dir / f"pearl_arm_{self.dof}dof_model.pt"
        torch.save({
            'policy_learner_state_dict': self.agent.policy_learner.state_dict(),
            'config': self.config,
            'final_success_rate': results['final_success_rate'],
            'total_episodes': results['total_episodes']
        }, model_file)
        
        print(f"\n📁 Results saved to {self.save_dir}")
        print(f"   Metrics: {metrics_file}")
        print(f"   Results: {results_file}")
        print(f"   Model: {model_file}")


def get_default_config() -> Dict[str, Any]:
    """Get default training configuration following Pearl conventions."""
    return {
        # Environment - 高频率长时间配置 (100Hz × 5.5s)
        'dof': 3,
        'max_episode_steps': 550,   # 100Hz × 5.5s = 550 steps，保证360°可达性
        'goal_threshold': 0.30,     # 可行的目标阈值
        
        # Training - 真正的长时间训练
        'episodes': 10000,          # 大幅增加训练量
        'num_processes': 4,         # 稳定的进程数
        'device': 'cuda:0',
        
        # Agent - 生产级配置
        'buffer_capacity': 500000,  # 更大buffer容纳更多经验
        'batch_size': 512,          # 大batch size配合大warmup提高学习稳定性
        'training_rounds': 1,       # 标准SAC: 每次学习1轮梯度更新
        'actor_hidden_dims': [512, 512],  # 更大网络容量
        'critic_hidden_dims': [512, 512],
        
        # Learning - 大规模warmup + 频繁评估配置  
        'learning_starts': 50000,   # 10% of buffer: 充分随机探索HER经验
        'learn_every': 1,           # 每步都学(适合高频控制)
        'eval_every': 10,           # 每10个episodes评估一次
    }


def main():
    """Main training function following Pearl script conventions."""
    parser = argparse.ArgumentParser(
        description="Pearl ARM Training with SAC+HER Multiprocessing"
    )
    parser.add_argument('--dof', type=int, default=3, choices=[3, 4, 5],
                       help='Degrees of freedom (3=2D, 4-5=3D)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--processes', type=int, default=6,
                       help='Number of parallel processes')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Training device')
    parser.add_argument('--save-dir', type=str, default='./pearl_arm_results',
                       help='Results save directory')
    
    args = parser.parse_args()
    
    # Setup configuration
    config = get_default_config()
    config.update({
        'dof': args.dof,
        'episodes': args.episodes,
        'num_processes': args.processes,
        'device': args.device,
    })
    
    print("🤖 Pearl ARM Training with SAC+HER Multiprocessing")
    print("📦 Integrated into Pearl's official structure")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = PearlArmTrainer(config, args.save_dir)
        
        # Train
        results = trainer.train()
        
        # Final report
        print(f"\n🎉 Pearl ARM Training Completed!")
        print(f"   DOF: {config['dof']}")
        print(f"   Final Success Rate: {results['final_success_rate']:.1f}%")
        print(f"   Episodes: {results['total_episodes']}")
        print(f"   Throughput: {results['avg_throughput']:.1f} eps/sec")
        print(f"   Processes: {config['num_processes']}")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()