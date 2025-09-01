#!/usr/bin/env python3
"""
多DOF动态软体机械臂Graph SAC+HER训练
终极Graph网络泛化能力测试：2-4节动态DOF训练

核心实验目标: 验证Graph Transformer在动态DOF场景下的泛化能力
- Episode级DOF随机：2节(4DOF) → 3节(6DOF) → 4节(8DOF)
- 统一网络处理所有配置
- 与固定DOF性能对比
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

# 设置进程名称
import setproctitle
setproctitle.setproctitle("MultiDOF-Graph-SAC")

# Pearl imports
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# Multi-DOF Graph network imports
from pearl.neural_networks.sequential_decision_making.actor_networks import MultiDOFGraphActorNetwork

# Multi-DOF Environment and HER imports
from pearl.utils.instantiations.environments.multi_dof_variable_soft_arm_environment import MultiDOFVariableSoftArmReachEnvironment
from pearl.utils.instantiations.environments.multi_dof_variable_soft_arm_her_factory import create_multi_dof_variable_soft_arm_her_buffer


class MultiDOFGraphArmTrainer:
    """多DOF动态Graph软体机械臂Pearl训练器"""
    
    def __init__(self, config: Dict[str, Any], save_dir: str = "./multi_dof_graph_results"):
        self.config = config
        self.device = config['device']
        
        # 创建任务专用子目录
        task_name = "multi_dof_graph_soft_arm_2to4"
        self.save_dir = Path(save_dir) / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 跟踪最佳成功率
        self.best_success_rate = -1.0
        
    def create_environment(self):
        """创建多DOF动态软体机械臂环境"""
        env = MultiDOFVariableSoftArmReachEnvironment(
            dof_range=self.config['dof_range'],
            base_segment_length=0.21,
            segment_length_range=self.config['segment_length_range'],
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps'],
            dof_distribution=self.config.get('dof_distribution', 'uniform')
        )
        
        return env
    
    def create_agent(self, env):
        """创建多DOF Graph Agent"""
        
        # 动作表示模块
        action_rep_module = IdentityActionRepresentationModule(
            max_number_actions=self.config['max_dof'],
            representation_dim=self.config['max_dof']
        )
        
        # Multi-DOF HER Buffer
        her_buffer = create_multi_dof_variable_soft_arm_her_buffer(
            capacity=self.config['buffer_capacity'],
            max_dof=self.config['max_dof'],
            max_segments=self.config['max_segments'],
            spatial_dim=3,
            goal_threshold=self.config['goal_threshold']
        )
        
        # Multi-DOF Graph Actor网络
        actor = MultiDOFGraphActorNetwork(
            action_space=env.action_space,
            max_dof=self.config['max_dof'],
            max_segments=self.config['max_segments'],
            hidden_dims=self.config['actor_hidden_dims'],
            node_feature_dim=8,
            num_graph_layers=self.config['num_graph_layers'],
            num_attention_heads=self.config['num_attention_heads'],
            use_kinematic_chain=True
        )
        
        # Multi-DOF Graph SAC
        sac = ContinuousSoftActorCritic(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            actor_network_instance=actor,
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
        """执行多DOF动态训练"""
        env = self.create_environment()
        agent = self.create_agent(env)
        
        # 训练统计
        episode_rewards = []
        recent_successes = []
        success_count = 0
        total_steps = 0
        dof_distribution_stats = {2: 0, 3: 0, 4: 0}
        
        print("🎯 开始多DOF动态Graph软体机械臂训练")
        print("🔥 实验核心: 单一Graph网络处理2-4节动态DOF")
        print("=" * 60)
        
        start_time = time.time()
        
        # 训练循环
        for episode in range(self.config['episodes']):
            # 重置环境和agent
            obs, action_space = env.reset()
            agent.reset(obs, action_space)
            
            episode_reward = 0
            episode_success = False
            
            # 记录当前episode的DOF配置
            current_config = env.get_current_config()
            current_n_segments = current_config['n_segments']
            dof_distribution_stats[current_n_segments] += 1
            
            for step in range(self.config['max_episode_steps']):
                # 选择动作
                action = agent.act(exploit=False)
                
                # 执行动作
                result = env.step(action)
                episode_reward += result.reward.item()
                total_steps += 1
                
                # Agent观察结果
                agent.observe(result)
                
                # 学习
                if total_steps >= self.config['learning_starts'] and total_steps % self.config['learn_every'] == 0:
                    agent.learn()
                
                # 检查终止
                if result.terminated or result.truncated:
                    success = result.terminated.item()
                    recent_successes.append(1.0 if success else 0.0)
                    if success:
                        episode_success = True
                        success_count += 1
                    break
                    
                obs = result.observation
            
            episode_rewards.append(episode_reward)
            
            # 评估和显示
            if (episode + 1) % 100 == 0:
                success_rate = np.mean(recent_successes[-500:]) * 100 if len(recent_successes) >= 500 else np.mean(recent_successes) * 100 if recent_successes else 0
                
                current_throughput = (episode + 1) / (time.time() - start_time)
                
                print(f"Episode: {episode + 1}, Success: {success_rate:.1f}%, Throughput: {current_throughput:.1f} eps/sec")
        
        # 训练完成统计
        final_success_rate = np.mean(recent_successes[-100:]) * 100 if len(recent_successes) >= 100 else 0
        total_time = time.time() - start_time
        
        print(f"🎉 多DOF动态训练完成! 最终成功率: {final_success_rate:.1f}%")
        
        return {
            'final_success_rate': final_success_rate,
            'total_episodes': self.config['episodes'],
            'total_time': total_time,
            'episode_rewards': episode_rewards,
            'recent_successes': recent_successes,
            'dof_distribution': dof_distribution_stats,
        }


def main():
    # 快速测试配置
    config = {
        'dof_range': (2, 4),
        'max_dof': 8,
        'max_segments': 4,
        'segment_length_range': (0.168, 0.252),
        'goal_threshold': 0.15,
        'max_episode_steps': 200,
        'dof_distribution': 'uniform',
        
        'episodes': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_capacity': 50000,
        'batch_size': 64,
        'training_rounds': 5,
        'learning_starts': 2000,
        'learn_every': 10,
        
        'actor_hidden_dims': [128, 128],
        'critic_hidden_dims': [128, 128],
        'num_graph_layers': 3,
        'num_attention_heads': 4,
    }
    
    print("🚀 多DOF动态Graph软体机械臂训练")
    print("🧪 实验: 单一Graph网络适应2-4节动态DOF配置")
    print("=" * 50)
    
    trainer = MultiDOFGraphArmTrainer(config)
    results = trainer.train()


if __name__ == "__main__":
    main()