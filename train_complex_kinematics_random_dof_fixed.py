#!/usr/bin/env python3
"""
超轻量GNN随机DOF训练脚本
基于成功的train_variable_soft_arm_pearl.py配置
随机DOF (2-5节) + 大长度变化 + 每个episode评估
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import time

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# 使用复杂运动学环境
from complex_kinematics_her_wrapper import ComplexKinematicsHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


class RandomDOFGNNSACTrainer:
    """随机DOF的超轻量GNN SAC训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config['device']
        
        self.save_dir = Path(config.get('save_dir', './random_dof_gnn_results'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_success_rate = -1.0
        
        print(f"🚀 随机DOF GNN训练器")
        print(f"   DOF范围: {config['dof_range'][0]}-{config['dof_range'][1]}节 ({config['dof_range'][0]*2}-{config['dof_range'][1]*2}DOF)")
        print(f"   长度变化: {config['segment_length_range'][0]:.1f}-{config['segment_length_range'][1]:.2f}m")
        print(f"   基于: train_variable_soft_arm_pearl.py 成功配置")
    
    def create_environment(self):
        """创建复杂运动学随机DOF环境"""
        env = ComplexKinematicsHERWrapper(
            dof_range=self.config['dof_range'],
            segment_length_range=self.config['segment_length_range'],
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps']
        )
        print(f"✅ 随机DOF环境: {self.config['dof_range'][0]}-{self.config['dof_range'][1]}节")
        print(f"   长度变化: {self.config['segment_length_range'][0]:.2f}m - {self.config['segment_length_range'][1]:.2f}m ({self.config['segment_length_range'][1]/self.config['segment_length_range'][0]:.1f}x)")
        print(f"   观测维度: {env.observation_space.shape}")
        return env
    
    def create_agent(self, env):
        """创建agent - 基于成功配置"""
        
        # Action representation
        max_dof = max(self.config['dof_range']) * 2  # 最大DOF
        action_rep_module = IdentityActionRepresentationModule(
            max_number_actions=max_dof,
            representation_dim=max_dof
        )
        
        # HER Buffer - 匹配成功配置
        her_buffer = create_variable_soft_arm_her_buffer(
            capacity=self.config['buffer_capacity'],
            joint_dim=max_dof,
            spatial_dim=3,
            n_segments=max(self.config['dof_range']),  # 最大节数
            threshold=self.config['goal_threshold'],
            include_lengths_in_obs=False
        )
        
        print(f"✅ HER Buffer: capacity={self.config['buffer_capacity']:,}")
        
        # 轻量GNN Actor网络
        actor = UltraLightGNNActor(
            action_dim=max_dof,
            dof_range=self.config['dof_range'],
            hidden_dim=self.config['hidden_dim'],
            num_gnn_layers=self.config['num_gnn_layers']
        ).to(self.device)
        
        print(f"✅ 轻量GNN Actor: {self.config['hidden_dim']}维, {self.config['num_gnn_layers']}层")
        
        # SAC Policy Learner - 完全匹配成功配置
        sac = ContinuousSoftActorCritic(
            state_dim=env.observation_space.shape[0],
            action_space=env.action_space,
            actor_network_instance=actor,
            critic_hidden_dims=self.config['critic_hidden_dims'],  # 成功的critic配置
            action_representation_module=action_rep_module,
            training_rounds=self.config['training_rounds'],  # 成功的training rounds
            batch_size=self.config['batch_size']  # 成功的batch size
        )
        
        # Pearl Agent
        agent = PearlAgent(
            policy_learner=sac,
            replay_buffer=her_buffer
        )
        
        print(f"✅ Pearl Agent: 随机DOF GNN SAC + HER")
        print(f"   成功配置: batch_size={self.config['batch_size']}, learn_every={self.config['learn_every']}")
        
        return agent
    
    def train(self):
        """训练 - 完全匹配成功的训练循环"""
        env = self.create_environment()
        agent = self.create_agent(env)
        
        print(f"\n🎯 开始随机DOF GNN训练")
        print(f"🧠 核心验证: GNN能否适应随机结构 (2-5节 + 大长度变化)")
        print(f"📊 每个episode都显示进度，快速反馈")
        print(f"=" * 60)
        
        # 训练统计
        success_count = 0
        total_episodes = 0
        recent_successes = []
        total_steps = 0
        dof_stats = {4: 0, 6: 0, 8: 0, 10: 0}  # 统计各DOF出现次数
        start_time = time.time()
        
        for episode in range(self.config['episodes']):
            # 重置环境
            obs, action_space = env.reset()
            agent.reset(obs, action_space)
            
            # 记录当前DOF
            current_dof = env.env.current_n_segments * 2
            if current_dof in dof_stats:
                dof_stats[current_dof] += 1
            
            episode_reward = 0
            episode_success = False
            
            for step in range(self.config['max_episode_steps']):
                # 选择动作
                action = agent.act(exploit=False)
                
                # 执行动作
                result = env.step(action)
                episode_reward += result.reward.item()
                total_steps += 1
                
                # Agent观察
                agent.observe(result)
                
                # 学习 - 完全匹配成功配置
                if total_steps >= self.config['learning_starts'] and total_steps % self.config['learn_every'] == 0:
                    agent.learn()
                
                # 检查终止
                if result.terminated or result.truncated:
                    episode_success = result.terminated.item()
                    break
            
            # 统计
            total_episodes += 1
            if episode_success:
                success_count += 1
            
            recent_successes.append(1.0 if episode_success else 0.0)
            if len(recent_successes) > 100:
                recent_successes.pop(0)
            
            # 每个episode评估
            if (episode + 1) % self.config['eval_every'] == 0:
                success_rate = sum(recent_successes) / len(recent_successes) * 100
                elapsed_time = time.time() - start_time
                
                print(f"Ep {episode + 1:4}: 成功率 {success_rate:.1f}% (近100ep), "
                      f"总成功率 {success_count/total_episodes*100:.1f}%, "
                      f"{current_dof}DOF, 时间 {elapsed_time/60:.1f}min")
                
                # 保存最佳模型
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self._save_checkpoint(agent, episode + 1, success_rate)
                    print(f"     🏆 新最佳成功率: {success_rate:.1f}%!")
        
        final_success_rate = success_count / total_episodes * 100
        total_time = time.time() - start_time
        
        print(f"\n🎉 随机DOF GNN训练完成!")
        print(f"   最终成功率: {final_success_rate:.1f}%")
        print(f"   最佳成功率: {self.best_success_rate:.1f}%")
        print(f"   总episodes: {total_episodes:,}")
        print(f"   总steps: {total_steps:,}")
        print(f"   总时间: {total_time/3600:.1f}小时")
        
        # DOF分布统计
        print(f"\n📊 DOF分布统计:")
        for dof, count in dof_stats.items():
            if count > 0:
                percentage = count / total_episodes * 100
                print(f"   {dof}DOF ({dof//2}节): {count}次 ({percentage:.1f}%)")
        
        return {
            'final_success_rate': final_success_rate,
            'best_success_rate': self.best_success_rate,
            'total_episodes': total_episodes,
            'total_steps': total_steps,
            'total_time': total_time,
            'dof_stats': dof_stats,
            'architecture': 'Random-DOF-GNN + Successful-Config'
        }
    
    def _save_checkpoint(self, agent, episode, success_rate):
        """保存checkpoint"""
        checkpoint_path = self.save_dir / "best_checkpoint.pt"
        torch.save({
            'episode': episode,
            'success_rate': success_rate,
            'agent_state': agent.policy_learner.state_dict(),
            'config': self.config
        }, checkpoint_path)


def main():
    """主函数 - 随机DOF + 成功配置"""
    
    # 🚀 随机DOF + 成功配置
    config = {
        # 🌟 随机结构多样性 (2-5节)
        'dof_range': (2, 5),  # 2-5节随机选择 (4-10DOF) 
        'segment_length_range': (0.1, 0.35),  # 大长度变化 (3.5x范围)
        'goal_threshold': 0.05,  # 严格阈值
        'max_episode_steps': 200,  # 成功步数
        
        # 📚 基于成功的train_variable_soft_arm_pearl.py配置
        'episodes': 5000,  # 成功的episodes数
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_capacity': 200000,  # 成功的buffer大小
        'batch_size': 256,  # 成功的batch size
        'training_rounds': 25,  # 成功的training rounds
        'learning_starts': 10000,  # 成功的warmup
        'learn_every': 50,  # 成功的学习频率
        'eval_every': 1,  # 每个episode评估，快速看进度
        
        # 🧠 轻量GNN配置 (适应随机DOF)
        'hidden_dim': 128,   # 适中隐藏层处理随机结构
        'num_gnn_layers': 2,  # 2层GNN足够处理结构变化
        'critic_hidden_dims': [512, 512],  # 成功的critic配置
    }
    
    print("🚀 随机DOF轻量GNN训练")
    print("🌟 核心挑战: 2-5节随机 (4-10DOF) + 大长度变化 (3.5x)")
    print("📚 基于: train_variable_soft_arm_pearl.py 成功配置")
    print("🎯 目标: 验证Graph方法在随机结构下的强大能力")
    print("=" * 60)
    
    trainer = RandomDOFGNNSACTrainer(config)
    results = trainer.train()
    
    print(f"\n📊 最终结果:")
    print(f"   架构: {results['architecture']}")
    print(f"   最终成功率: {results['final_success_rate']:.1f}%")
    print(f"   最佳成功率: {results['best_success_rate']:.1f}%")
    print(f"   训练时间: {results['total_time']/3600:.1f}小时")
    
    print(f"\n🧠 随机DOF挑战验证:")
    if results['best_success_rate'] > 70:
        print(f"   🏆 OUTSTANDING! Graph网络完美处理随机DOF!")
        print(f"   ✅ 2-5节随机 + 3.5x长度变化 = 完全掌控!")
        print(f"   🌟 证明了Graph方法的超强适应性!")
    elif results['best_success_rate'] > 50:
        print(f"   🎯 EXCELLENT! Graph网络很好适应随机结构!")
        print(f"   ✅ 随机DOF变化对Graph网络影响有限")
        print(f"   💪 成功配置 + Graph适应性 = 成功!")
    elif results['best_success_rate'] > 30:
        print(f"   ✅ GOOD! 显著学习但受随机性挑战")
        print(f"   📊 考虑随机DOF变化，这是合理表现")
        print(f"   💡 Graph网络正在学习适应不同结构")
    else:
        print(f"   🤔 随机DOF + 大长度变化确实是挑战")
        print(f"   📊 可考虑微调GNN参数以适应高变化性")


if __name__ == "__main__":
    main()