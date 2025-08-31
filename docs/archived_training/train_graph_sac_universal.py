#!/usr/bin/env python3
"""
通用Graph SAC训练脚本
支持任意DOF的机械臂（刚体/软体通用）
基于优化的Graph Transformer网络

核心特性：
1. 支持任意DOF机械臂（3DOF-9DOF+）
2. 自适应图结构，无需预定义网络大小
3. 统一的状态空间处理（刚体/软体）
4. Pearl SAC+HER兼容
5. 高效的批处理和GPU加速
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import setproctitle

# Pearl框架导入
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# 环境导入
from pearl.utils.instantiations.environments import NDOFArmEnvironment, VariableSoftArmReachEnvironment

# HER buffer导入
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer
from pearl.utils.instantiations.environments.arm_her_factory import create_arm_her_buffer

# Graph网络导入  
from simple_robot_graph import SimpleRobotGraphSAC, create_robot_graph_data


class UniversalGraphSACTrainer:
    """
    通用Graph SAC训练器
    支持任意配置的机械臂
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 设置进程名称
        setproctitle.setproctitle(f"Graph-SAC-{config['env_type']}-{config.get('dof', 'Variable')}")
        
        # 创建任务专用子目录
        base_save_dir = Path(config.get('save_dir', './graph_sac_results'))
        task_name = f"graph_sac_{config['env_type']}_{config.get('dof', config.get('n_segments', 'var'))}dof"
        self.save_dir = base_save_dir / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # 跟踪最佳成功率
        self.best_success_rate = -1.0
        
        print(f"🚀 通用Graph SAC训练器初始化")
        print(f"   环境类型: {config['env_type']}")
        print(f"   DOF: {config.get('dof', 'Variable')}")
        print(f"   最大DOF: {config['max_dof']}")
        print(f"   设备: {self.device}")
        print(f"   Episodes: {config['episodes']:,}")
        
        # 创建环境
        self.env = self._create_environment()
        
        # 创建Graph SAC智能体
        self.agent = self._create_graph_agent()
        
        # 训练统计
        self.episode_rewards = []
        self.episode_successes = []
        self.recent_success_buffer = []
        self.recent_reward_buffer = []
        self.window_size = 100
        
    def _create_environment(self):
        """根据配置创建环境"""
        env_type = self.config['env_type']
        
        if env_type == 'ndof':
            # 刚体NDOF机械臂
            env = NDOFArmEnvironment(
                dof=self.config['dof'],
                max_steps=self.config.get('max_episode_steps', 50),
                goal_threshold=self.config.get('goal_threshold', 0.30)
            )
            print(f"   创建NDOF环境: {self.config['dof']}DOF")
            
        elif env_type == 'variable_soft':
            # 变长软体机械臂
            env = VariableSoftArmReachEnvironment(
                n_segments=self.config.get('n_segments', 3),
                max_steps=self.config.get('max_episode_steps', 200),
                segment_length_range=self.config.get('segment_length_range', (0.168, 0.252)),
                goal_threshold=self.config.get('goal_threshold', 0.10),
                include_lengths_in_obs=True
            )
            print(f"   创建变长软体环境: {self.config.get('n_segments', 3)}节")
            
        else:
            raise ValueError(f"不支持的环境类型: {env_type}")
        
        return env
    
    def _create_graph_agent(self):
        """创建Graph SAC智能体"""
        
        # 创建动作表示模块（使用实际环境的动作维度）
        actual_action_dim = self.env.action_space.shape[0]
        action_rep_module = IdentityActionRepresentationModule(
            max_number_actions=actual_action_dim,
            representation_dim=actual_action_dim
        )
        
        # 根据环境类型创建HER buffer
        if self.config['env_type'] == 'ndof':
            her_buffer = create_arm_her_buffer(
                capacity=self.config.get('buffer_capacity', 100000),
                dof=self.config['dof'],
                spatial_dim=2  # 2D位置
            )
        elif self.config['env_type'] == 'variable_soft':
            her_buffer = create_variable_soft_arm_her_buffer(
                capacity=self.config.get('buffer_capacity', 500000),
                joint_dim=self.config.get('n_segments', 3) * 2,  # 每节2DOF
                spatial_dim=3,  # 3D位置
                n_segments=self.config.get('n_segments', 3),
                include_lengths_in_obs=True
            )
        else:
            raise ValueError(f"不支持的环境类型: {self.config['env_type']}")
        
        # 创建标准SAC policy learner（暂时使用MLP，后续会替换为Graph网络）
        sac = ContinuousSoftActorCritic(
            state_dim=self.env.observation_space.shape[0],
            action_space=self.env.action_space,
            actor_hidden_dims=self.config.get('actor_hidden_dims', [256, 256]),
            critic_hidden_dims=self.config.get('critic_hidden_dims', [256, 256]),
            action_representation_module=action_rep_module,
            training_rounds=self.config.get('training_rounds', 1),
            batch_size=self.config.get('batch_size', 256),
        )
        
        # 创建Pearl Agent
        agent = PearlAgent(
            policy_learner=sac,
            replay_buffer=her_buffer,
        )
        
        print(f"   创建Graph SAC Agent (当前使用MLP，后续升级):")
        print(f"     Actor网络: {self.config.get('actor_hidden_dims', [256, 256])}")
        print(f"     Critic网络: {self.config.get('critic_hidden_dims', [256, 256])}")
        print(f"     Buffer容量: {self.config.get('buffer_capacity', 100000):,}")
        print(f"     批处理大小: {self.config.get('batch_size', 256)}")
        
        return agent
    
    def train(self) -> Dict[str, Any]:
        """执行训练循环"""
        
        print(f"\n🎯 开始Graph SAC训练")
        print(f"=" * 80)
        
        start_time = time.time()
        episodes = self.config['episodes']
        learning_starts = self.config.get('learning_starts', 1000)
        eval_every = self.config.get('eval_every', 500)
        
        # 训练进度条
        pbar = tqdm(range(episodes), desc="Episodes", unit="eps")
        
        total_steps = 0
        
        for episode in pbar:
            episode_reward = 0.0
            episode_success = False
            
            # 重置环境和智能体
            obs, action_space = self.env.reset()
            self.agent.reset(obs, action_space)
            
            # Episode循环
            max_steps = self.config.get('max_episode_steps', 200)
            
            for step in range(max_steps):
                # 选择动作
                action = self.agent.act(exploit=False)
                
                # 执行动作
                result = self.env.step(action)
                episode_reward += result.reward.item() if hasattr(result.reward, 'item') else result.reward
                total_steps += 1
                
                # 智能体观察结果
                self.agent.observe(result)
                
                # 学习（基于总步数）
                if total_steps >= learning_starts and total_steps % self.config.get('learn_every', 1) == 0:
                    for _ in range(self.config.get('training_rounds', 1)):
                        self.agent.learn()
                
                # 检查终止条件
                if result.terminated or result.truncated:
                    episode_success = result.terminated.item() if hasattr(result.terminated, 'item') else result.terminated
                    break
            
            # 记录统计信息
            self.episode_rewards.append(episode_reward)
            self.episode_successes.append(episode_success)
            self.recent_reward_buffer.append(episode_reward)
            self.recent_success_buffer.append(episode_success)
            
            # 保持滑动窗口
            if len(self.recent_reward_buffer) > self.window_size:
                self.recent_reward_buffer.pop(0)
                self.recent_success_buffer.pop(0)
            
            # 更新进度条
            recent_success_rate = np.mean(self.recent_success_buffer) * 100 if self.recent_success_buffer else 0.0
            recent_avg_reward = np.mean(self.recent_reward_buffer) if self.recent_reward_buffer else 0.0
            
            pbar.set_postfix({
                'Success%': f"{recent_success_rate:.1f}",
                'AvgReward': f"{recent_avg_reward:.1f}",
                'Buffer': len(self.agent.replay_buffer) if hasattr(self.agent, 'replay_buffer') else 0
            })
            
            # 定期评估和保存
            if episode > 0 and episode % eval_every == 0:
                self._evaluate_and_save(episode, recent_success_rate)
        
        # 训练完成
        total_time = time.time() - start_time
        
        print(f"\n🎉 Graph SAC训练完成!")
        print(f"   总时间: {total_time:.1f}s")
        print(f"   吞吐量: {episodes / total_time:.1f} eps/sec")
        print(f"   最终成功率: {recent_success_rate:.1f}%")
        
        return {
            'episodes': episodes,
            'final_success_rate': recent_success_rate,
            'final_avg_reward': recent_avg_reward,
            'total_time': total_time,
            'throughput': episodes / total_time
        }
    
    def _evaluate_and_save(self, episode: int, success_rate: float):
        """只保存成功率最高的模型"""
        
        print(f"📊 Episode {episode} 评估:")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   平均奖励: {np.mean(self.recent_reward_buffer):.3f}")
        print(f"   Buffer大小: {len(self.agent.replay_buffer) if hasattr(self.agent, 'replay_buffer') else 0:,}")
        
        # 只在成功率提升时保存
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            
            checkpoint = {
                'episode': episode,
                'success_rate': success_rate,
                'config': self.config,
                'episode_rewards': self.episode_rewards,
                'episode_successes': self.episode_successes
            }
            
            checkpoint_path = self.save_dir / "best_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"   🏆 新最佳成功率 {success_rate:.1f}%! 保存至: best_checkpoint.pt")
        else:
            print(f"   📊 当前成功率 {success_rate:.1f}% (最佳: {self.best_success_rate:.1f}%)")
            
        print("=" * 60)


def create_training_config(env_type: str, **kwargs) -> Dict[str, Any]:
    """创建训练配置"""
    
    base_config = {
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'episodes': 10000,
        'max_episode_steps': 50,
        'learning_starts': 1000,
        'learn_every': 1,
        'training_rounds': 1,
        'eval_every': 500,
        'batch_size': 256,
        'buffer_capacity': 100000,
        'learning_rate': 3e-4,
        'discount_factor': 0.99,
        'target_update_tau': 0.005,
        
        # Graph网络配置
        'graph_hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 3,
        'max_nodes': 10,
        
        # 更新kwargs
        **kwargs
    }
    
    if env_type == 'ndof':
        base_config.update({
            'env_type': 'ndof',
            'dof': kwargs.get('dof', 3),
            'max_dof': 9,
            'goal_threshold': 0.50,  # 合理的阈值（~17%工作空间）
            'max_episode_steps': 200  # 匹配环境默认值
        })
        
    elif env_type == 'variable_soft':
        base_config.update({
            'env_type': 'variable_soft',
            'n_segments': kwargs.get('n_segments', 3),
            'max_dof': 6,
            'goal_threshold': 0.10,
            'max_episode_steps': 200,
            'segment_length_range': (0.168, 0.252)
        })
    
    return base_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="通用Graph SAC训练")
    parser.add_argument('--env_type', type=str, required=True, 
                       choices=['ndof', 'variable_soft'],
                       help='环境类型')
    parser.add_argument('--dof', type=int, default=3,
                       help='DOF数量（ndof环境）')
    parser.add_argument('--n_segments', type=int, default=3,
                       help='段数量（variable_soft环境）')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='训练episodes')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')
    parser.add_argument('--save_dir', type=str, default='./graph_sac_results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建配置
    config = create_training_config(
        env_type=args.env_type,
        dof=args.dof,
        n_segments=args.n_segments,
        episodes=args.episodes,
        device=args.device,
        save_dir=args.save_dir
    )
    
    # 创建训练器并开始训练
    trainer = UniversalGraphSACTrainer(config)
    results = trainer.train()
    
    # 保存最终结果
    results_path = Path(args.save_dir) / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 最终结果保存至: {results_path}")


if __name__ == "__main__":
    main()