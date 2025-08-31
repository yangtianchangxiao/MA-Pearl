#!/usr/bin/env python3
"""
软体机械臂Pearl训练脚本
使用单进程SAC+HER，基于成功的3DOF配置
"""
import argparse
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
from pearl.utils.instantiations.environments import SoftArmReachEnvironment
from pearl.utils.instantiations.environments.soft_arm_her_factory import create_soft_arm_her_buffer


class SoftArmPearlTrainer:
    """
    软体机械臂Pearl训练器 - 单进程版本
    基于成功的3DOF配置，避免多进程HER问题
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        save_dir: str = "./soft_arm_pearl_results"
    ):
        self.config = config
        self.device = config['device']
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup device
        if torch.cuda.is_available() and 'cuda' in self.device:
            torch.cuda.set_device(self.device)
            print(f"🚀 软体臂Pearl训练 - Device: {self.device}")
            print(f"   GPU: {torch.cuda.get_device_name(self.device)}")
            
            # Set process title for nvidia-smi identification
            try:
                import setproctitle
                setproctitle.setproctitle("SoftArm_50step_25x")
            except ImportError:
                print("   (setproctitle not available for process naming)")
        else:
            self.device = "cpu"
            print("⚠️ 软体臂Pearl训练 - Using CPU")
        
        # Initialize components
        self._setup_environment()
        self._setup_pearl_agent()
        
        # Training metrics
        self.metrics = {
            'episodes': [],
            'success_rate': [],
            'avg_reward': [],
            'buffer_size': [],
            'config': config
        }
        
        # Checkpoint tracking
        self.best_success_rate = -1.0
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"✅ 软体臂Pearl训练器初始化完成")
        print(f"   节数: {self.config.get('n_segments', 3)} ({self.config.get('n_segments', 3)*2}DOF)")
        print(f"   算法: SAC + HER (单进程)")
        print(f"   阈值: {config['goal_threshold']}")
    
    def _setup_environment(self):
        """设置软体臂环境"""
        n_segments = self.config.get('n_segments', 3)  # 默认3节
        self.env = SoftArmReachEnvironment(
            n_segments=n_segments,
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps']
        )
        
        print(f"✅ 软体臂环境: {n_segments}节 {self.env.dof}DOF, 3D工作空间")
        print(f"   观测维度: {self.env.observation_space.shape}")
        print(f"   动作维度: {self.env.action_space.shape}")
    
    def _setup_pearl_agent(self):
        """设置Pearl agent with SAC + HER"""
        # HER replay buffer - 使用软体臂专用版本，支持动态DOF
        her_buffer = create_soft_arm_her_buffer(
            joint_dim=self.env.dof,  # 动态DOF，基于节数
            spatial_dim=3,
            capacity=self.config['buffer_capacity'],
            threshold=self.config['goal_threshold']
        )
        
        # SAC policy learner
        sac_learner = ContinuousSoftActorCritic(
            state_dim=self.env.observation_space.shape[0],
            action_space=self.env.action_space,
            actor_hidden_dims=self.config['actor_hidden_dims'],
            critic_hidden_dims=self.config['critic_hidden_dims'],
            batch_size=self.config['batch_size'],
            training_rounds=self.config['training_rounds'],
            entropy_coef=0.2,
            entropy_autotune=True,
            actor_learning_rate=0.0003,
            critic_learning_rate=0.0003,
        )
        
        # Pearl agent
        self.agent = PearlAgent(
            policy_learner=sac_learner,
            replay_buffer=her_buffer,
        )
        
        print(f"✅ Pearl Agent: SAC + HER")
        print(f"   Buffer容量: {her_buffer.capacity:,}")
        print(f"   批量大小: {self.config['batch_size']}")
        print(f"   HER策略: future + 4目标采样")
    
    def save_checkpoint(self, success_rate, episode, is_best=False):
        """保存训练checkpoint"""
        checkpoint = {
            'episode': episode,
            'success_rate': success_rate,
            'agent_state': self.agent.get_state() if hasattr(self.agent, 'get_state') else None,
            'metrics': self.metrics,
            'config': self.config
        }
        
        # 保存最新checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳性能，保存best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f"🎉 新的最佳模型保存! 成功率: {success_rate:.1f}% -> {best_path}")
        
        # 定期保存编号checkpoint
        if episode % 1000 == 0:
            episode_path = self.checkpoint_dir / f'checkpoint_episode_{episode}.pt'
            torch.save(checkpoint, episode_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载训练checkpoint"""
        if not checkpoint_path.exists():
            print(f"⚠️ Checkpoint不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if checkpoint.get('agent_state') and hasattr(self.agent, 'load_state'):
                self.agent.load_state(checkpoint['agent_state'])
            
            self.metrics = checkpoint.get('metrics', self.metrics)
            self.best_success_rate = checkpoint.get('success_rate', -1.0)
            
            print(f"✅ Checkpoint加载成功: Episode {checkpoint.get('episode', 0)}, 成功率: {self.best_success_rate:.1f}%")
            return True
        except Exception as e:
            print(f"❌ Checkpoint加载失败: {e}")
            return False
    
    def train(self) -> Dict[str, Any]:
        """训练agent - 单进程版本"""
        episodes = self.config['episodes']
        eval_every = self.config.get('eval_every', 500)
        learning_starts = self.config['learning_starts']
        learn_every = self.config.get('learn_every', 1)
        
        print(f"\n🚀 开始软体臂训练...")
        print(f"📝 配置: {episodes} episodes, 单进程")
        print(f"💡 学习开始: {learning_starts}, 学习频率: 每{learn_every}步")
        print("=" * 80)
        
        episode_rewards = []
        recent_successes = []
        total_steps = 0
        start_time = time.time()
        
        with tqdm(total=episodes, desc="Episodes", unit="eps") as pbar:
            for episode in range(episodes):
                # Reset环境
                obs, action_space = self.env.reset()
                self.agent.reset(obs, action_space)
                
                episode_reward = 0
                episode_steps = 0
                
                for step in range(self.config['max_episode_steps']):
                    # 获取action
                    action = self.agent.act(exploit=False)
                    
                    # 执行action
                    result = self.env.step(action)
                    episode_reward += result.reward.item()
                    episode_steps += 1
                    total_steps += 1
                    
                    # Agent观察结果
                    self.agent.observe(result)
                    
                    # 学习 - 每50步训练25次
                    if total_steps >= learning_starts and total_steps % learn_every == 0:
                        print(f"🧠 开始学习: Step {total_steps}, Episode {episode+1} (每50步训练25次)")
                        start_learn_time = time.time()
                        self.agent.learn()
                        learn_time = time.time() - start_learn_time
                        print(f"✅ 学习完成: 耗时 {learn_time:.3f}s")
                    
                    # 检查终止
                    if result.terminated or result.truncated:
                        # 记录成功状态
                        success = result.terminated.item()
                        recent_successes.append(1.0 if success else 0.0)
                        break
                
                episode_rewards.append(episode_reward)
                pbar.update(1)
                
                # 每个episode都统计 (eval_every=1)
                if (episode + 1) % eval_every == 0:
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                    avg_reward = np.mean(episode_rewards[-eval_every:]) if len(episode_rewards) >= eval_every else np.mean(episode_rewards)
                    buffer_size = len(self.agent.replay_buffer) if hasattr(self.agent, 'replay_buffer') else 0
                    elapsed = time.time() - start_time
                    throughput = (episode + 1) / elapsed if elapsed > 0 else 0
                    
                    pbar.write(f"\n📊 软体臂训练进度")
                    pbar.write(f"   Episode: {episode + 1}")
                    pbar.write(f"   成功率: {success_rate:.1f}%")
                    pbar.write(f"   平均奖励: {avg_reward:.3f}")
                    pbar.write(f"   Buffer大小: {buffer_size:,}")
                    pbar.write(f"   吞吐量: {throughput:.1f} eps/sec")
                    pbar.write(f"   总步数: {total_steps:,}")
                    pbar.write("=" * 60)
                    
                    # 保存metrics
                    self.metrics['episodes'].append(episode + 1)
                    self.metrics['success_rate'].append(success_rate)
                    self.metrics['avg_reward'].append(avg_reward)
                    self.metrics['buffer_size'].append(buffer_size)
                    
                    # 检查是否需要保存checkpoint
                    is_best = success_rate > self.best_success_rate
                    if is_best:
                        self.best_success_rate = success_rate
                    
                    # 保存checkpoint
                    self.save_checkpoint(success_rate, episode + 1, is_best=is_best)
        
        # 最终结果
        final_success_rate = np.mean(recent_successes[-200:]) * 100 if len(recent_successes) >= 200 else np.mean(recent_successes) * 100
        total_time = time.time() - start_time
        
        results = {
            'final_success_rate': final_success_rate,
            'total_episodes': episodes,
            'total_time': total_time,
            'avg_throughput': episodes / total_time if total_time > 0 else 0,
            'metrics': self.metrics
        }
        
        # 保存结果
        results_file = self.save_dir / 'training_results.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n🎉 软体臂训练完成!")
        print(f"📈 最终成功率: {final_success_rate:.1f}%")
        print(f"⏱️  总训练时间: {total_time:.1f}s")
        print(f"🔄 平均速度: {results['avg_throughput']:.1f} eps/sec")
        print(f"💾 结果保存至: {results_file}")
        
        return results


def get_default_config():
    """获取默认配置 - 匹配3DOF大训练量配置"""
    return {
        'device': 'cuda:0',
        'episodes': 10000,  # 匹配3DOF大训练量
        'max_episode_steps': 200,  # 软体臂步数
        'goal_threshold': 0.15,  # 软体臂阈值
        'n_segments': 3,  # 默认3节软体臂 (6DOF)
        
        # SAC配置 - 匹配3DOF
        'actor_hidden_dims': [512, 512],
        'critic_hidden_dims': [512, 512], 
        'batch_size': 512,  # 匹配3DOF batch size
        'training_rounds': 25,  # 每50步学习25次
        
        # HER配置 - 匹配3DOF
        'buffer_capacity': 500000,
        
        # 训练配置 - 每50步训练25次
        'learning_starts': 50000,  # 匹配3DOF大warmup
        'learn_every': 50,  # 每50步学习一次
        'eval_every': 1,  # 每1个episode评估一次
    }


def main():
    parser = argparse.ArgumentParser(description='软体机械臂Pearl训练')
    parser.add_argument('--episodes', type=int, default=5000, help='训练episodes数')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--threshold', type=float, default=0.15, help='目标阈值')
    parser.add_argument('--segments', type=int, default=3, help='软体臂节数')
    
    args = parser.parse_args()
    
    # 配置
    config = get_default_config()
    config['episodes'] = args.episodes
    config['device'] = args.device  
    config['goal_threshold'] = args.threshold
    config['n_segments'] = args.segments
    
    print(f"🤖 软体机械臂Pearl训练启动")
    print(f"🔧 配置: {args.episodes} episodes, {args.segments}节({args.segments*2}DOF), 阈值={args.threshold}")
    
    # 创建训练器并开始训练
    trainer = SoftArmPearlTrainer(config)
    results = trainer.train()
    
    print(f"✨ 训练完成! 最终成功率: {results['final_success_rate']:.1f}%")


if __name__ == "__main__":
    main()