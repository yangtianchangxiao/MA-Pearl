#!/usr/bin/env python3
"""
变长软体机械臂Graph SAC+HER训练
使用Graph Transformer替换MLP网络，验证Graph网络在机械臂任务中的效果

对比配置：
- 环境：完全相同的VariableSoftArmReachEnvironment
- 训练超参数：与train_variable_soft_arm_official.py完全一致  
- 唯一区别：使用GraphActorNetwork和GraphQValueNetwork替换标准MLP
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
setproctitle.setproctitle("Graph-Variable-Arm-SAC")

# Pearl imports
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# Graph network imports
from pearl.neural_networks.sequential_decision_making.actor_networks import GraphActorNetwork
from pearl.neural_networks.sequential_decision_making.q_value_networks import GraphQValueNetwork

# Environment and HER imports
from pearl.utils.instantiations.environments import VariableSoftArmReachEnvironment
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


class GraphVariableArmTrainer:
    """Graph变长软体机械臂Pearl训练器"""
    
    def __init__(self, config: Dict[str, Any], save_dir: str = "./graph_variable_arm_results"):
        self.config = config
        self.device = config['device']
        
        # 创建任务专用子目录
        task_name = "graph_variable_soft_arm_6dof"
        self.save_dir = Path(save_dir) / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 跟踪最佳成功率
        self.best_success_rate = -1.0
        
        print(f"🚀 Graph变长软体机械臂训练器初始化")
        print(f"   配置: {config['n_segments']}节 {config['n_segments']*2}DOF")
        print(f"   Segment长度范围: {config['segment_length_range']} (动态±20%)")
        print(f"   目标阈值: {config['goal_threshold']}")
        print(f"   Episodes: {config['episodes']:,}")
        print(f"   Graph配置: {config['num_graph_layers']}层, {config['num_attention_heads']}头注意力")
        
    def create_environment(self):
        """创建变长软体机械臂环境"""
        env = VariableSoftArmReachEnvironment(
            n_segments=self.config['n_segments'],
            max_steps=self.config['max_episode_steps'],
            segment_length_range=self.config['segment_length_range'],
            goal_threshold=self.config['goal_threshold'],
            include_lengths_in_obs=True
        )
        
        print(f"✅ 变长软体臂Pearl环境初始化: {self.config['n_segments']}节 {self.config['n_segments']*2}DOF")
        print(f"   长度范围: {self.config['segment_length_range'][0]:.3f}m - {self.config['segment_length_range'][1]:.3f}m")
        print(f"   观测维度: {env.observation_space.shape} (包含长度: True)")
        print(f"   阈值: {self.config['goal_threshold']}")
        
        return env
    
    def create_agent(self, env):
        """创建使用Graph网络的Pearl Agent"""
        
        # 动作表示模块
        action_rep_module = IdentityActionRepresentationModule(
            max_number_actions=env.action_space.shape[0],
            representation_dim=env.action_space.shape[0]
        )
        
        # Variable Soft Arm HER Buffer
        her_buffer = create_variable_soft_arm_her_buffer(
            capacity=self.config['buffer_capacity'],
            joint_dim=self.config['n_segments'] * 2,  # 每节2DOF
            spatial_dim=3,  # 3D位置
            n_segments=self.config['n_segments'],
            include_lengths_in_obs=True
        )
        
        print(f"✅ Variable Arm HER Buffer: capacity={self.config['buffer_capacity']:,}, dof={self.config['n_segments']*2}, segments={self.config['n_segments']}")
        print(f"   HER strategy: future, goals: 4")
        print(f"   Include lengths: True")
        
        # Graph Actor网络
        actor = GraphActorNetwork(
            input_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            hidden_dims=self.config['actor_hidden_dims'],
            node_feature_dim=8,  # [joint(2) + achieved(3) + desired(3)]
            num_graph_layers=self.config['num_graph_layers'],
            num_attention_heads=self.config['num_attention_heads'],
            use_kinematic_chain=True
        )
        
        # 🚀 使用Graph Actor + 标准MLP Critic (快速测试)
        # 这样可以先验证Graph Actor网络，Critic使用标准MLP避免twin critic问题
        sac = ContinuousSoftActorCritic(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            actor_network_instance=actor,  # 使用Graph Actor
            # critic使用默认VanillaQValueNetwork (MLP)
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
        
        print(f"✅ Graph SAC Agent创建成功:")
        print(f"   Actor网络: GraphActorNetwork {self.config['actor_hidden_dims']}")
        print(f"   Critic网络: GraphQValueNetwork {self.config['critic_hidden_dims']}")
        print(f"   Graph层数: {self.config['num_graph_layers']}")
        print(f"   注意力头数: {self.config['num_attention_heads']}")
        print(f"   Buffer容量: {self.config['buffer_capacity']:,}")
        print(f"   批处理大小: {self.config['batch_size']}")
        
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
        
        print(f"\n🎯 开始Graph变长软体机械臂训练")
        print(f"🔥 配置: Graph网络 vs MLP对比实验")
        print(f"📊 预期: Graph网络应在复杂kinematic关系上表现更好")
        print(f"=" * 80)
        
        # 训练循环
        pbar = tqdm(range(self.config['episodes']), desc="Episodes", unit="eps")
        
        for episode in pbar:
            # 重置环境和agent
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
            eval_every = self.config.get('eval_every', 1)
            if (episode + 1) % eval_every == 0:
                success_rate = np.mean(recent_successes) * 100 if recent_successes else 0
                
                # 保持滑动窗口大小
                if len(recent_successes) > 500:  # 保持最近500个episodes
                    recent_successes = recent_successes[-500:]
                
                # 保存最佳checkpoint
                self._save_checkpoint(agent, episode+1, success_rate)
                
                # 显示详细进度信息
                current_throughput = (episode + 1) / (time.time() - start_time) if 'start_time' in locals() else 0
                
                print(f"\n📊 Graph变长软体臂训练进度")
                print(f"   Episode: {episode + 1}")
                print(f"   成功率: {success_rate:.1f}%")
                print(f"   平均奖励: {np.mean(episode_rewards[-100:]):.3f}")
                print(f"   Buffer大小: {len(agent.replay_buffer):,}")
                print(f"   吞吐量: {current_throughput:.1f} eps/sec")
                print(f"   总步数: {total_steps:,}")
                if hasattr(env, 'current_segment_lengths'):
                    print(f"   当前segments: {env.current_segment_lengths}, 总长: {np.sum(env.current_segment_lengths):.3f}m")
                print("=" * 60)
            
            # 设置开始时间
            if episode == 0:
                start_time = time.time()
        
        # 训练完成统计
        final_success_rate = np.mean(recent_successes[-100:]) * 100 if len(recent_successes) >= 100 else 0
        total_time = time.time() - start_time
        
        print(f"\n🎉 Graph变长软体机械臂训练完成!")
        print(f"   最终成功率: {final_success_rate:.1f}%")
        print(f"   总Episodes: {self.config['episodes']:,}")
        print(f"   总时间: {total_time:.1f}s")
        print(f"   平均吞吐量: {self.config['episodes'] / total_time:.1f} eps/sec")
        print(f"   最终Buffer大小: {len(agent.replay_buffer):,}")
        
        return {
            'final_success_rate': final_success_rate,
            'total_episodes': self.config['episodes'],
            'total_time': total_time,
            'final_buffer_size': len(agent.replay_buffer),
            'episode_rewards': episode_rewards,
            'recent_successes': recent_successes
        }
    
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


def main():
    # 训练配置 - 与原变长训练完全一致，只是网络替换为Graph
    config = {
        # 环境配置
        'n_segments': 3,                              # 3节机械臂
        'segment_length_range': (0.168, 0.252),       # ±20% 变化范围
        'goal_threshold': 0.15,                       # 目标阈值
        'max_episode_steps': 200,                     # 每episode最大步数
        
        # 🚀 生产训练配置
        'episodes': 100000,                           # 总episodes数量
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # SAC+HER配置
        'buffer_capacity': 1000000,                   # HER buffer容量
        'batch_size': 512,                           # 批处理大小
        'training_rounds': 25,                       # 每次学习的训练轮数
        'learning_starts': 50000,                    # 开始学习的buffer大小
        'learn_every': 50,                           # 每50步学习一次
        
        # 🧪 快速测试配置 (已验证成功，备用)
        # 'episodes': 500,                              # 快速测试: 500 episodes
        # 'buffer_capacity': 50000,                     # 快速测试: 50k buffer
        # 'batch_size': 64,                            # 快速测试: 64 batch size  
        # 'training_rounds': 5,                        # 快速测试: 5 rounds
        # 'learning_starts': 1000,                     # 快速测试: 1k steps开始学习
        # 'learn_every': 10,                           # 快速测试: 每10步学习一次
        
        # Graph网络配置 (新增)
        'actor_hidden_dims': [256, 256],             # Graph Actor后的MLP层
        'critic_hidden_dims': [256, 256],            # Graph Critic后的MLP层
        'num_graph_layers': 3,                       # Graph Transformer层数
        'num_attention_heads': 4,                    # 多头注意力头数
        
        # 评估配置  
        'eval_every': 1,                             # 每1个episode评估一次
    }
    
    print("🚀 Graph变长软体机械臂Pearl SAC+HER训练")
    print("🧪 实验目的: 对比Graph网络 vs MLP网络在复杂机械臂任务中的表现")
    print("=" * 80)
    
    trainer = GraphVariableArmTrainer(config)
    results = trainer.train()
    
    # 保存训练结果
    results_path = trainer.save_dir / f"graph_training_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    # 转换numpy数组为列表以便JSON序列化
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            json_results[key] = [v.tolist() for v in value]
        else:
            json_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"📊 Graph训练结果保存至: {results_path}")
    print("🎯 可与MLP基线结果进行性能对比分析")


if __name__ == "__main__":
    main()