#!/usr/bin/env python3
"""
Graph-to-Graph SAC训练脚本

与现有系统并行：
- 现有: lightweight_gnn_actor.py → random_dof_gnn_results/  
- 新系统: graph_to_graph_actor.py → graph_to_graph_results/
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time

from optimized_graph_her_wrapper import OptimizedGraphHERWrapper
from graph_to_graph_actor import GraphToGraphActor
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.pearl_agent import PearlAgent
from pearl.utils.instantiations.environments.variable_her_buffer import create_variable_soft_arm_her_buffer


class GraphToGraphTrainer:
    """Graph-to-Graph SAC训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(exist_ok=True)
        
        # 最佳成功率追踪
        self.best_success_rate = 0.0
        
        print("🚀 Graph-to-Graph SAC训练器初始化")
        print("=" * 60)
        print(f"🎯 训练配置:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        self._setup_environment()
        self._setup_agent()
        
    def _setup_environment(self):
        """设置环境"""
        print(f"\n🌍 环境设置:")
        
        self.env = OptimizedGraphHERWrapper(
            dof_range=self.config['dof_range'],
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps']
        )
        
        print(f"   DOF范围: {self.config['dof_range']} (真正动态!)")
        print(f"   最大steps: {self.config['max_episode_steps']}")
        print(f"   成功阈值: {self.config['goal_threshold']}")
        
    def _setup_agent(self):
        """设置智能体"""
        print(f"\n🤖 智能体设置:")
        
        # 创建Graph-to-Graph Actor
        self.actor_network = GraphToGraphActor(
            dof_range=self.config['dof_range'],
            hidden_dim=self.config['hidden_dim'],
            num_gnn_layers=self.config['num_gnn_layers']
        ).to(self.device)
        
        # 创建SAC learner
        action_space = self.env.action_space
        state_dim = self.env.observation_space.shape[0]
        
        learner = ContinuousSoftActorCritic(
            action_space=action_space,
            state_dim=state_dim,
            actor_network_instance=self.actor_network,
            critic_hidden_dims=self.config['critic_hidden_dims'],
            actor_learning_rate=self.config['actor_lr'],
            critic_learning_rate=self.config['critic_lr']
        )
        
        # 创建HER buffer
        replay_buffer = create_variable_soft_arm_her_buffer(
            capacity=self.config['buffer_capacity']
        )
        
        # 创建Agent
        self.agent = PearlAgent(
            policy_learner=learner,
            replay_buffer=replay_buffer,
            device=self.device
        )
        
        self.agent._action_space = action_space
        
        print(f"✅ Graph-to-Graph Agent创建完成")
        print(f"   Actor参数: {self.actor_network._count_parameters():,}")
        print(f"   Buffer容量: {self.config['buffer_capacity']:,}")
        
    def train(self):
        """主训练循环"""
        print(f"\n🎓 开始Graph-to-Graph训练")
        print("=" * 60)
        
        episode_rewards = []
        episode_successes = []
        training_start_time = time.time()
        
        for episode in range(1, self.config['episodes'] + 1):
            episode_reward, episode_success = self._run_episode()
            
            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)
            
            # 定期评估和保存
            if episode % self.config['eval_every'] == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                success_rate = np.mean(episode_successes[-100:]) * 100
                
                elapsed_time = time.time() - training_start_time
                episodes_per_hour = episode / (elapsed_time / 3600)
                
                print(f"Episode {episode:4d} | "
                      f"成功率: {success_rate:5.1f}% | "
                      f"平均奖励: {avg_reward:7.1f} | "
                      f"速度: {episodes_per_hour:.1f} eps/h")
                
                # 保存最佳模型
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self._save_checkpoint(episode, success_rate, avg_reward)
                    print(f"🏆 新纪录! 成功率: {success_rate:.1f}%")
            
            # 训练网络
            if (episode >= self.config['learning_starts'] and 
                episode % self.config['learn_every'] == 0):
                
                for _ in range(self.config['training_rounds']):
                    self.agent.learn()
        
        print(f"\n🎉 Graph-to-Graph训练完成!")
        print(f"   最佳成功率: {self.best_success_rate:.1f}%")
        print(f"   总用时: {(time.time() - training_start_time)/3600:.1f}小时")
        
    def _run_episode(self):
        """运行单个episode"""
        obs, action_space = self.env.reset()
        self.agent.reset(obs, action_space)
        
        episode_reward = 0
        step = 0
        
        while step < self.config['max_episode_steps']:
            # Agent动作
            action = self.agent.act(exploit=False)
            
            # 环境step
            action_result = self.env.step(action)
            self.agent.observe(action_result)
            
            episode_reward += action_result.reward.item()
            step += 1
            
            if action_result.terminated or action_result.truncated:
                episode_success = action_result.terminated.item()
                break
        else:
            episode_success = 0
        
        return episode_reward, episode_success
    
    def _save_checkpoint(self, episode, success_rate, avg_reward):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'agent_state': self.agent.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = self.save_dir / "best_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 同时保存配置文件
        config_path = self.save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)


def main():
    """主函数"""
    print("🚀 Graph-to-Graph SAC训练启动")
    print("保护现有系统，创建新的训练管道")
    print()
    
    # Graph-to-Graph训练配置
    config = {
        # 环境配置
        'dof_range': (2, 8),  # 扩展到8节！
        'goal_threshold': 0.15,
        'max_episode_steps': 200,
        
        # 网络配置
        'hidden_dim': 128,
        'num_gnn_layers': 2,
        'critic_hidden_dims': [512, 512],
        
        # 训练配置
        'episodes': 10000,  # 更多训练
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_capacity': 500000,  # 更大buffer
        'batch_size': 256,
        'training_rounds': 25,
        'learning_starts': 5000,
        'learn_every': 50,
        'eval_every': 100,
        
        # 学习率
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        
        # 保存路径 - 与现有系统分离
        'save_dir': 'graph_to_graph_results'
    }
    
    # 创建训练器
    trainer = GraphToGraphTrainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()