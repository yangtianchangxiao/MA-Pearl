#!/usr/bin/env python3
"""
正式训练脚本：变长软体机械臂 Pearl SAC+HER训练
配置：6DOF 3节机械臂，不定长度，动态workspace计算

训练配置与原arm训练保持一致：
- 50 training steps，25 learn_every
- batch_size=512, episodes=100000  
- 动态segment长度变化：±20% (0.168m-0.252m)
- 动态workspace：基于总长度80%计算
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from tqdm import tqdm

# 设置进程名称，让nvidia-smi显示有意义的名字
import setproctitle
setproctitle.setproctitle("Variable-Arm-Pearl-SAC")

# Pearl imports
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.utils.instantiations.environments import VariableSoftArmReachEnvironment
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


class VariableArmTrainer:
    """变长软体机械臂Pearl训练器"""
    
    def __init__(self, config: Dict[str, Any], save_dir: str = "./variable_arm_results"):
        self.config = config
        self.device = config['device']
        # 创建任务专用子目录
        task_name = "variable_soft_arm_6dof"
        self.save_dir = Path(save_dir) / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # 跟踪最佳成功率
        self.best_success_rate = -1.0
        
        print(f"🚀 变长软体机械臂训练器初始化")
        print(f"   配置: {config['n_segments']}节 {config['n_segments']*2}DOF")
        print(f"   Segment长度范围: {config['segment_length_range']} (动态±20%)")
        print(f"   目标阈值: {config['goal_threshold']}")
        print(f"   Episodes: {config['episodes']:,}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Learning: {config['training_rounds']}rounds every {config['learn_every']}steps")
        
    def create_environment(self):
        """创建变长软体臂环境"""
        return VariableSoftArmReachEnvironment(
            n_segments=self.config['n_segments'],
            segment_length_range=self.config['segment_length_range'],  
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps'],
            include_lengths_in_obs=True,  # 在观测中包含当前长度信息
        )
        
    def create_agent(self, env):
        """创建Pearl Agent with SAC+HER"""
        # HER buffer
        her_buffer = create_variable_soft_arm_her_buffer(
            joint_dim=env.dof,
            spatial_dim=3,
            n_segments=env.n_segments,
            capacity=self.config['buffer_capacity'],
            threshold=self.config['goal_threshold'],
            include_lengths_in_obs=True,
        )
        
        # Action representation module
        action_rep_module = IdentityActionRepresentationModule(
            max_number_actions=env.action_space.shape[0],
            representation_dim=env.action_space.shape[0]
        )
        
        # Continuous SAC policy learner
        sac = ContinuousSoftActorCritic(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            actor_hidden_dims=self.config['actor_hidden_dims'],
            critic_hidden_dims=self.config['critic_hidden_dims'],
            action_representation_module=action_rep_module,
            training_rounds=self.config['training_rounds'],
            batch_size=self.config['batch_size'],
        )
        
        # Pearl Agent
        agent = PearlAgent(
            policy_learner=sac,
            replay_buffer=her_buffer,
        )
        
        return agent
        
    def train(self):
        """执行训练"""
        env = self.create_environment()
        agent = self.create_agent(env)
        
        # 训练统计 - 按照固定长度的模式
        episode_rewards = []
        recent_successes = []  # 跟踪成功状态，像固定长度那样
        success_count = 0
        total_steps = 0
        
        # 开始训练
        start_time = time.time()
        print(f"\n🎯 开始变长软体臂训练 {self.config['episodes']:,} episodes...")
        print(f"💡 学习开始: {self.config['learning_starts']}, 学习频率: 每{self.config['learn_every']}步")
        print("="*80)
        
        with tqdm(total=self.config['episodes'], desc="Episodes", unit="eps") as pbar:
            for episode in range(self.config['episodes']):
                obs, action_space = env.reset()
                agent.reset(obs, action_space)
                
                episode_reward = 0
                episode_success = False
                
                for step in range(self.config['max_episode_steps']):
                    # 选择动作
                    action = agent.act(exploit=False)
                    
                    # 执行动作
                    result = env.step(action)
                    episode_reward += result.reward.item()
                    total_steps += 1
                    
                    # Agent观察结果
                    agent.observe(result)
                    
                    # 学习 - 按照固定长度的模式
                    if total_steps >= self.config['learning_starts'] and total_steps % self.config['learn_every'] == 0:
                        agent.learn()
                    
                    # 检查终止
                    if result.terminated or result.truncated:
                        # 记录成功状态 - 像固定长度那样
                        success = result.terminated.item()
                        recent_successes.append(1.0 if success else 0.0)
                        if success:
                            episode_success = True
                            success_count += 1
                        break
                        
                    obs = result.observation
                
                episode_rewards.append(episode_reward)
                pbar.update(1)
                
                # 每个episode都统计 - 完全按照固定长度的模式
                eval_every = self.config.get('eval_every', 500)
                if (episode + 1) % eval_every == 0:
                    success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                    avg_reward = np.mean(episode_rewards[-eval_every:]) if len(episode_rewards) >= eval_every else np.mean(episode_rewards)
                    buffer_size = len(agent.replay_buffer) if hasattr(agent, 'replay_buffer') else 0
                    elapsed = time.time() - start_time
                    throughput = (episode + 1) / elapsed if elapsed > 0 else 0
                    
                    pbar.write(f"\n📊 变长软体臂训练进度")
                    pbar.write(f"   Episode: {episode + 1}")
                    pbar.write(f"   成功率: {success_rate:.1f}%")
                    pbar.write(f"   平均奖励: {avg_reward:.3f}")
                    pbar.write(f"   Buffer大小: {buffer_size:,}")
                    pbar.write(f"   吞吐量: {throughput:.1f} eps/sec")
                    pbar.write(f"   总步数: {total_steps:,}")
                    pbar.write(f"   当前segments: {env.current_segment_lengths}, 总长: {np.sum(env.current_segment_lengths):.3f}m")
                    pbar.write("=" * 60)
                    
                    # 保存检查点 - 如果成功率创新高
                    if not hasattr(self, 'best_success_rate'):
                        self.best_success_rate = -1.0
                    is_best = success_rate > self.best_success_rate
                    if is_best:
                        self.best_success_rate = success_rate
                        self._save_checkpoint(agent, episode+1, success_rate)
        
        # 训练完成 - 按照固定长度的模式计算最终成功率
        total_time = time.time() - start_time
        final_success_rate = np.mean(recent_successes[-200:]) * 100 if len(recent_successes) >= 200 else np.mean(recent_successes) * 100
        avg_throughput = self.config['episodes'] / total_time
        
        results = {
            "episodes": self.config['episodes'],
            "final_success_rate": final_success_rate,
            "total_successes": success_count, 
            "avg_reward": np.mean(episode_rewards),
            "total_time": total_time,
            "avg_throughput": avg_throughput,
            "buffer_size": len(agent.replay_buffer),
            "config": self.config
        }
        
        # 保存最终结果
        self._save_results(results)
        
        print(f"\n🎉 训练完成!")
        print(f"   总Episodes: {self.config['episodes']:,}")
        print(f"   最终成功率: {final_success_rate:.1f}%")
        print(f"   总用时: {total_time:.1f}s")
        print(f"   平均吞吐量: {avg_throughput:.1f} eps/sec")
        print(f"   最终Buffer大小: {len(agent.replay_buffer):,}")
        
        return results
        
    def _save_checkpoint(self, agent, episode, success_rate):
        """只保存成功率最高的检查点"""
        if success_rate > self.best_success_rate:
            # 更新最佳成功率
            self.best_success_rate = success_rate
            
            # 固定的最佳checkpoint文件名
            checkpoint_path = self.save_dir / "best_checkpoint.pt"
            torch.save({
                'episode': episode,
                'success_rate': success_rate,
                'agent_state': agent.policy_learner.state_dict(),
                'config': self.config
            }, checkpoint_path)
            print(f"   🏆 新最佳成功率 {success_rate:.1f}%! 保存至: best_checkpoint.pt")
        else:
            print(f"   📊 当前成功率 {success_rate:.1f}% (最佳: {self.best_success_rate:.1f}%)")
        
    def _save_results(self, results):
        """保存训练结果"""
        results_path = self.save_dir / f"training_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 结果已保存: {results_path}")


def main():
    # 训练配置 - 与原arm训练保持一致
    config = {
        # 环境配置
        'n_segments': 3,                              # 3节机械臂
        'segment_length_range': (0.168, 0.252),       # ±20% 变化范围 (0.21*0.8, 0.21*1.2)
        'goal_threshold': 0.15,                       # 目标阈值
        'max_episode_steps': 200,                     # 每episode最大步数 (匹配固定长度软体臂)
        
        # 训练配置
        'episodes': 100000,                           # 总episodes数量
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # SAC+HER配置
        'buffer_capacity': 1000000,                   # HER buffer容量
        'batch_size': 512,                           # 与原arm保持一致
        'training_rounds': 25,                       # 每次学习的训练轮数 (匹配3DOF)
        'learning_starts': 50000,                    # 开始学习的buffer大小 (匹配3DOF大warmup)
        'learn_every': 50,                           # 每50步学习一次 (匹配3DOF)
        
        # 网络配置
        'actor_hidden_dims': [512, 512],             # Actor网络隐藏层
        'critic_hidden_dims': [512, 512],            # Critic网络隐藏层
        
        # 评估配置  
        'eval_every': 1,                             # 每1个episode评估一次 (按用户要求)
    }
    
    print("🚀 变长软体机械臂Pearl SAC+HER训练")
    print("="*60)
    print("📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("="*60)
    
    # 创建并运行训练器
    trainer = VariableArmTrainer(config)
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    results = main()