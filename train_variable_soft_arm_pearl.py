#!/usr/bin/env python3
"""
变长软体机械臂Pearl训练脚本
支持episode级segment长度随机化，增强泛化能力
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
from pearl.utils.instantiations.environments import VariableSoftArmReachEnvironment
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


class VariableSoftArmPearlTrainer:
    """
    变长软体机械臂Pearl训练器 - 单进程版本
    支持episode级segment长度随机化，增强泛化能力
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        save_dir: str = "./variable_soft_arm_results"
    ):
        self.config = config
        self.device = config['device']
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup device
        if torch.cuda.is_available() and 'cuda' in self.device:
            torch.cuda.set_device(self.device)
            print(f"🚀 变长软体臂Pearl训练 - Device: {self.device}")
            print(f"   GPU: {torch.cuda.get_device_name(self.device)}")
            
            # Set process title for nvidia-smi identification
            try:
                import setproctitle
                setproctitle.setproctitle("VarSoftArm_Pearl_Training")
            except ImportError:
                print("   (setproctitle not available for process naming)")
        else:
            self.device = "cpu"
            print("⚠️ 变长软体臂Pearl训练 - Using CPU")
        
        # Initialize components
        self._setup_environment()
        self._setup_pearl_agent()
        
        # Training metrics
        self.metrics = {
            'episodes': [],
            'success_rate': [],
            'avg_reward': [],
            'buffer_size': [],
            'avg_segment_length': [],  # 新增：平均segment长度追踪
            'config': config
        }
        
        # Checkpoint tracking
        self.best_success_rate = -1.0
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"✅ 变长软体臂Pearl训练器初始化完成")
        print(f"   节数: {self.config.get('n_segments', 3)} ({self.config.get('n_segments', 3)*2}DOF)")
        print(f"   长度范围: {config.get('segment_length_range', (0.15, 0.30))}")
        print(f"   算法: SAC + HER (单进程)")
        print(f"   阈值: {config['goal_threshold']}")
    
    def _setup_environment(self):
        """设置变长软体臂环境"""
        n_segments = self.config.get('n_segments', 3)
        segment_length_range = self.config.get('segment_length_range', (0.15, 0.30))
        include_lengths_in_obs = self.config.get('include_lengths_in_obs', True)
        
        self.env = VariableSoftArmReachEnvironment(
            n_segments=n_segments,
            base_segment_length=0.21,
            segment_length_range=segment_length_range,
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps'],
            include_lengths_in_obs=include_lengths_in_obs
        )
        
        print(f"✅ 变长软体臂环境: {n_segments}节 {self.env.dof}DOF")
        print(f"   长度变化: {segment_length_range[0]:.3f}m - {segment_length_range[1]:.3f}m")
        print(f"   观测维度: {self.env.observation_space.shape}")
        print(f"   动作维度: {self.env.action_space.shape}")
    
    def _setup_pearl_agent(self):
        """设置Pearl agent with SAC + HER"""
        # HER replay buffer - 使用变长专用HER buffer（包含segment长度）
        include_lengths_in_obs = self.config.get('include_lengths_in_obs', True)
        her_buffer = create_variable_soft_arm_her_buffer(
            joint_dim=self.env.dof,  # 动态DOF，基于节数  
            spatial_dim=3,
            n_segments=self.env.n_segments,
            capacity=self.config['buffer_capacity'],
            threshold=self.config['goal_threshold'],
            include_lengths_in_obs=include_lengths_in_obs
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
        
        print(f"✅ Pearl Agent: SAC + HER (标准兼容)")
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
        """训练agent - 单进程版本，支持变长segment"""
        episodes = self.config['episodes']
        eval_every = self.config.get('eval_every', 500)
        learning_starts = self.config['learning_starts']
        learn_every = self.config.get('learn_every', 50)
        
        print(f"\\n🚀 开始变长软体臂训练...")
        print(f"📝 配置: {episodes} episodes, 单进程, 变长segment")
        print(f"💡 学习开始: {learning_starts}, 学习频率: 每{learn_every}步")
        print("=" * 80)
        
        episode_rewards = []
        recent_successes = []
        segment_length_history = []  # 追踪长度变化
        total_steps = 0
        start_time = time.time()
        
        with tqdm(total=episodes, desc="Episodes", unit="eps") as pbar:
            for episode in range(episodes):
                # Reset环境 (会随机化segment长度)
                obs, action_space = self.env.reset()
                self.agent.reset(obs, action_space)
                
                # 记录当前episode的segment长度
                current_avg_length = np.mean(self.env.current_segment_lengths)
                segment_length_history.append(current_avg_length)
                
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
                    
                    # 学习 - 按配置频率训练
                    if total_steps >= learning_starts and total_steps % learn_every == 0:
                        self.agent.learn()
                    
                    # 检查终止
                    if result.terminated or result.truncated:
                        # 记录成功状态
                        success = result.terminated.item()
                        recent_successes.append(1.0 if success else 0.0)
                        break
                
                episode_rewards.append(episode_reward)
                pbar.update(1)
                
                # 评估和checkpoint保存
                if (episode + 1) % eval_every == 0:
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                    avg_reward = np.mean(episode_rewards[-eval_every:]) if len(episode_rewards) >= eval_every else np.mean(episode_rewards)
                    buffer_size = len(self.agent.replay_buffer) if hasattr(self.agent, 'replay_buffer') else 0
                    avg_seg_length = np.mean(segment_length_history[-eval_every:]) if len(segment_length_history) >= eval_every else np.mean(segment_length_history)
                    elapsed = time.time() - start_time
                    throughput = (episode + 1) / elapsed if elapsed > 0 else 0
                    
                    pbar.write(f"\\n📊 变长软体臂训练进度")
                    pbar.write(f"   Episode: {episode + 1}")
                    pbar.write(f"   成功率: {success_rate:.1f}%")
                    pbar.write(f"   平均奖励: {avg_reward:.3f}")
                    pbar.write(f"   平均长度: {avg_seg_length:.3f}m")
                    pbar.write(f"   Buffer大小: {buffer_size:,}")
                    pbar.write(f"   吞吐量: {throughput:.1f} eps/sec")
                    pbar.write(f"   总步数: {total_steps:,}")
                    pbar.write("=" * 60)
                    
                    # 保存metrics
                    self.metrics['episodes'].append(episode + 1)
                    self.metrics['success_rate'].append(success_rate)
                    self.metrics['avg_reward'].append(avg_reward)
                    self.metrics['buffer_size'].append(buffer_size)
                    self.metrics['avg_segment_length'].append(avg_seg_length)
                    
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
            'avg_segment_length': np.mean(segment_length_history),
            'segment_length_std': np.std(segment_length_history),
            'metrics': self.metrics
        }
        
        # 保存结果
        results_file = self.save_dir / 'training_results.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n🎉 变长软体臂训练完成!")
        print(f"📈 最终成功率: {final_success_rate:.1f}%")
        print(f"📏 平均segment长度: {results['avg_segment_length']:.3f}±{results['segment_length_std']:.3f}m")
        print(f"⏱️  总训练时间: {total_time:.1f}s")
        print(f"🔄 平均速度: {results['avg_throughput']:.1f} eps/sec")
        print(f"💾 结果保存至: {results_file}")
        
        return results


def get_default_config():
    """获取默认配置 - 变长软体臂版本"""
    return {
        'device': 'cuda:0',
        'episodes': 5000,  # 较少episodes用于快速验证
        'max_episode_steps': 200,  # 软体臂步数
        'goal_threshold': 0.15,  # 软体臂阈值
        'n_segments': 3,  # 默认3节软体臂
        'segment_length_range': (0.15, 0.30),  # ±30%长度变化
        'include_lengths_in_obs': True,   # 包含segment长度信息
        
        # SAC配置 - 匹配成功配置
        'actor_hidden_dims': [512, 512],
        'critic_hidden_dims': [512, 512], 
        'batch_size': 256,  # 中等batch size
        'training_rounds': 25,  # 每50步学习25次
        
        # HER配置
        'buffer_capacity': 200000,  # 适中的buffer
        
        # 训练配置
        'learning_starts': 10000,  # 较少的warmup
        'learn_every': 50,  # 每50步学习一次
        'eval_every': 500,  # 每500个episode评估一次
    }


def main():
    parser = argparse.ArgumentParser(description='变长软体机械臂Pearl训练')
    parser.add_argument('--episodes', type=int, default=5000, help='训练episodes数')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--threshold', type=float, default=0.15, help='目标阈值')
    parser.add_argument('--segments', type=int, default=3, help='软体臂节数')
    parser.add_argument('--length-min', type=float, default=0.15, help='最小segment长度')
    parser.add_argument('--length-max', type=float, default=0.30, help='最大segment长度')
    
    args = parser.parse_args()
    
    # 配置
    config = get_default_config()
    config['episodes'] = args.episodes
    config['device'] = args.device  
    config['goal_threshold'] = args.threshold
    config['n_segments'] = args.segments
    config['segment_length_range'] = (args.length_min, args.length_max)
    
    print(f"🤖 变长软体机械臂Pearl训练启动")
    print(f"🔧 配置: {args.episodes} episodes, {args.segments}节({args.segments*2}DOF)")
    print(f"📏 长度范围: {args.length_min:.3f}m - {args.length_max:.3f}m")
    print(f"🎯 阈值: {args.threshold}")
    
    # 创建训练器并开始训练
    trainer = VariableSoftArmPearlTrainer(config)
    results = trainer.train()
    
    print(f"✨ 训练完成! 最终成功率: {results['final_success_rate']:.1f}%")
    print(f"📊 segment长度统计: {results['avg_segment_length']:.3f}±{results['segment_length_std']:.3f}m")


if __name__ == "__main__":
    main()