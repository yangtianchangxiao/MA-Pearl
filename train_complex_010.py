#!/usr/bin/env python3
"""
复杂运动学随机DOF GNN训练脚本

基于train_ultra_light_gnn_random_dof.py，但使用复杂运动学
训练出的模型可以直接部署到C++硬件上
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import time
import json

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# 使用复杂运动学环境
from complex_kinematics_her_wrapper import ComplexKinematicsHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor  # 复用现有网络架构
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer
from pearl.utils.instantiations.spaces.box import BoxSpace


class ComplexKinematicsGNNSACTrainer:
    """复杂运动学随机DOF GNN SAC训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config['device']
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(exist_ok=True)
        
        # 最佳成功率追踪
        self.best_success_rate = 0.0
        
        # 全局step计数器 (用于学习频率控制)
        self.total_steps = 0
        
        print("🚀 复杂运动学随机DOF GNN SAC训练器初始化")
        print("=" * 70)
        print(f"🎯 关键改进: 使用C++硬件一致的复杂运动学")
        print(f"🎯 训练配置:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        self._setup_environment()
        self._setup_agent()
        
    def _setup_environment(self):
        """设置复杂运动学环境"""
        print(f"\n🌍 复杂运动学环境设置:")
        
        self.env = ComplexKinematicsHERWrapper(
            dof_range=self.config['dof_range'],
            segment_length_range=self.config.get('segment_length_range', (0.1, 0.35)),
            goal_threshold=self.config['goal_threshold'], 
            max_steps=self.config['max_episode_steps']
        )
        
        print(f"✅ 复杂运动学环境创建完成")
        
    def _setup_agent(self):
        """设置智能体 - 复用现有GNN架构"""
        print(f"\n🤖 智能体设置 (复用UltraLightGNNActor):")
        
        # 创建GNN Actor (复用现有架构)
        action_dim = self.env.action_space.shape[0]
        
        self.actor_network = UltraLightGNNActor(
            action_dim=action_dim,
            dof_range=self.config['dof_range'],
            hidden_dim=self.config.get('hidden_dim', 128),
            num_gnn_layers=self.config.get('num_gnn_layers', 2)
        ).to(self.device)
        
        # Action representation module
        max_dof = max(self.config['dof_range']) * 2
        action_rep_module = IdentityActionRepresentationModule(
            max_number_actions=max_dof,
            representation_dim=max_dof
        )
        
        # 创建SAC learner
        state_dim = self.env.observation_space.shape[0]
        
        sac = ContinuousSoftActorCritic(
            state_dim=state_dim,
            action_space=self.env.action_space,
            actor_network_instance=self.actor_network,
            critic_hidden_dims=self.config.get('critic_hidden_dims', [512, 512]),
            action_representation_module=action_rep_module,
            training_rounds=self.config.get('training_rounds', 25),
            batch_size=self.config.get('batch_size', 256)
        )
        
        # 创建HER buffer  
        replay_buffer = create_variable_soft_arm_her_buffer(
            capacity=self.config['buffer_capacity'],
            joint_dim=max_dof,
            spatial_dim=3,
            n_segments=max(self.config['dof_range']),
            threshold=self.config['goal_threshold'],
            include_lengths_in_obs=False
        )
        
        # 创建Agent
        self.agent = PearlAgent(
            policy_learner=sac,
            replay_buffer=replay_buffer
        )
        
        self.agent._action_space = self.env.action_space
        
        print(f"✅ 复杂运动学Agent创建完成")
        print(f"   Actor参数: ~{self._count_parameters():,}")
        print(f"   Buffer容量: {self.config['buffer_capacity']:,}")
        
    def _count_parameters(self):
        return sum(p.numel() for p in self.actor_network.parameters())
        
    def train(self):
        """主训练循环"""
        print(f"\n🎓 开始复杂运动学随机DOF训练")
        print("=" * 70)
        
        episode_rewards = []
        episode_successes = []
        episode_distances = []
        training_start_time = time.time()
        
        for episode in range(1, self.config['episodes'] + 1):
            episode_reward, episode_success, final_distance = self._run_episode()
            
            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)
            episode_distances.append(final_distance)
            
            # 定期评估和保存
            if episode % self.config['eval_every'] == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                success_rate = np.mean(episode_successes[-100:]) * 100
                avg_distance = np.mean(episode_distances[-100:])
                
                elapsed_time = time.time() - training_start_time
                episodes_per_hour = episode / (elapsed_time / 3600)
                
                learning_status = "🎓学习中" if self.total_steps >= self.config['learning_starts'] else f"🔍探索中({self.total_steps}/{self.config['learning_starts']})"
                print(f"Episode {episode:4d} | "
                      f"成功率: {success_rate:5.1f}% | "
                      f"平均奖励: {avg_reward:7.1f} | "
                      f"平均距离: {avg_distance:.3f}m | "
                      f"速度: {episodes_per_hour:.1f} eps/h | {learning_status}")
                
                # 保存最佳模型 - 只有开始学习后才保存
                if (success_rate > self.best_success_rate and 
                    self.total_steps >= self.config['learning_starts']):
                    self.best_success_rate = success_rate
                    self._save_checkpoint(episode, success_rate, avg_reward, avg_distance)
                    print(f"🏆 新纪录! 复杂运动学成功率: {success_rate:.1f}%")
            
            # 训练网络 - step-level学习已在_run_episode()中处理，这里不需要额外训练
        
        print(f"\n🎉 复杂运动学训练完成!")
        print(f"   最佳成功率: {self.best_success_rate:.1f}%")
        print(f"   总用时: {(time.time() - training_start_time)/3600:.1f}小时")
        print(f"🎯 关键: 训练的模型可直接部署到C++硬件!")
        
    def _run_episode(self):
        """运行单个episode"""
        obs, action_space = self.env.reset()
        self.agent.reset(obs, action_space)
        
        episode_reward = 0
        step_count = 0
        final_distance = float('inf')
        episode_success = False  # 初始化为False，像原版一样
        
        while step_count < self.config['max_episode_steps']:
            # Agent动作
            action = self.agent.act(exploit=False)
            
            # 环境step
            action_result = self.env.step(action)
            
            episode_reward += action_result.reward.item()
            final_distance = action_result.info.get('distance', float('inf'))
            self.total_steps += 1  # 全局step计数
            step_count += 1
            
            # Agent观察
            self.agent.observe(action_result)
            
            # 学习 - 每50步学习一次 (像原版一样)
            if self.total_steps >= self.config['learning_starts'] and self.total_steps % self.config['learn_every'] == 0:
                self.agent.learn()
            
            if action_result.terminated or action_result.truncated:
                episode_success = action_result.terminated.item()
                break
        
        return episode_reward, episode_success, final_distance
    
    def _save_checkpoint(self, episode, success_rate, avg_reward, avg_distance):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_distance': avg_distance,
            'agent_state': self.agent.state_dict(),
            'config': self.config,
            'kinematics_type': 'complex',  # 标记运动学类型
            'hardware_compatible': True    # 标记硬件兼容
        }
        
        checkpoint_path = self.save_dir / "best_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存配置文件
        config_path = self.save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"💾 保存复杂运动学模型: {checkpoint_path}")


def main():
    """主函数"""
    print("🚀 复杂运动学随机DOF GNN训练启动")
    print("训练与C++硬件兼容的模型")
    print()
    
    # 复杂运动学训练配置
    config = {
        # 环境配置
        'dof_range': (2, 5),  # 与原来一致
        'segment_length_range': (0.1, 0.35),  # 大长度变化
        'goal_threshold': 0.10,  # 中等阈值
        'max_episode_steps': 200,
        
        # 网络配置 (原来的轻量配置 - 备注)
        # 'hidden_dim': 128,           # 轻量版本
        # 'num_gnn_layers': 2,         # 轻量版本  
        # 'critic_hidden_dims': [512, 512],  # 轻量版本
        
        # 网络配置 (更大网络 - 解决困难任务)
        'hidden_dim': 256,           # 从128增加到256
        'num_gnn_layers': 3,         # 从2增加到3层
        'critic_hidden_dims': [512, 512, 256],  # 更深的critic
        
        # 训练配置 (原来的配置 - 备注)
        # 'batch_size': 256,           # 轻量版本
        # 'training_rounds': 25,       # 轻量版本
        # 'learn_every': 50,           # 轻量版本
        
        # 训练配置 (更大batch - 提高学习稳定性)
        'episodes': 5000,  # 复杂运动学可能需要更多训练
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_capacity': 200000,
        'batch_size': 512,           # 从256增加到512
        'training_rounds': 50,       # 从25增加到50
        'learning_starts': 10000,
        'learn_every': 25,           # 每25步学习一次
        'eval_every': 1,  # 更频繁评估
        
        # 保存路径 - 0.10阈值专用  
        'save_dir': 'complex_010_results'
    }
    
    print(f"🔄 配置对比:")
    print(f"   原来简化运动学: random_dof_gnn_results/")
    print(f"   新的复杂运动学: complex_kinematics_gnn_results/")
    print(f"   网络架构: 复用UltraLightGNNActor (已验证成功)")
    print(f"   关键差异: 使用C++硬件运动学训练")
    print()
    
    # 创建训练器
    trainer = ComplexKinematicsGNNSACTrainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()