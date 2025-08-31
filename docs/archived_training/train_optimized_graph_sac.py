#!/usr/bin/env python3
"""
优化Graph SAC训练脚本
使用简化的3维节点特征 + Goals分离设计
测试Graph网络的运动学推理能力
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# 使用优化的组件
from optimized_graph_her_wrapper import OptimizedGraphHERWrapper
from her_to_graph_actor import HERGraphActorNetwork

# HER buffer
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


class OptimizedGraphSACTrainer:
    """优化Graph SAC训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config['device']
        
        # 创建保存目录
        self.save_dir = Path(config.get('save_dir', './optimized_graph_sac_results'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_success_rate = -1.0
        
        print(f"🚀 优化Graph SAC训练器")
        print(f"   节点特征: 3维 [joint1, joint2, length]")
        print(f"   网络设计: 结构-空间分离处理")
        print(f"   目标: 测试运动学推理能力")
    
    def create_environment(self):
        """创建优化环境（使用HER包装器）"""
        env = OptimizedGraphHERWrapper(
            dof_range=self.config['dof_range'],
            segment_length_range=self.config['segment_length_range'],
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps']
        )
        return env
    
    def create_agent(self, env):
        """创建优化Graph SAC agent"""
        
        # Action representation
        max_dof = max(self.config['dof_range']) * 2
        action_rep_module = IdentityActionRepresentationModule(
            max_number_actions=max_dof,
            representation_dim=max_dof
        )
        
        # HER Buffer（使用最大DOF配置）
        her_buffer = create_variable_soft_arm_her_buffer(
            capacity=self.config['buffer_capacity'],
            joint_dim=max_dof,
            spatial_dim=3,
            n_segments=max(self.config['dof_range']),
            threshold=self.config['goal_threshold'],
            include_lengths_in_obs=False  # 我们有自己的格式
        )
        
        print(f"✅ HER Buffer: capacity={self.config['buffer_capacity']:,}, max_dof={max_dof}")
        
        # HER兼容的Graph Actor网络
        actor = HERGraphActorNetwork(
            action_dim=max_dof,
            dof_range=self.config['dof_range'],
            hidden_dim=self.config['hidden_dim'],
            num_graph_layers=self.config['num_graph_layers'],
            num_attention_heads=self.config['num_attention_heads']
        )
        
        print(f"✅ HER-Graph Actor: HER格式输入 + Graph处理")
        
        # SAC Policy Learner
        sac = ContinuousSoftActorCritic(
            state_dim=env.observation_space.shape[0],  # 动态维度
            action_space=env.action_space,
            actor_network_instance=actor,
            critic_hidden_dims=self.config['critic_hidden_dims'],
            action_representation_module=action_rep_module,
            training_rounds=self.config['training_rounds'],
            batch_size=self.config['batch_size']
        )
        
        # Pearl Agent
        agent = PearlAgent(
            policy_learner=sac,
            replay_buffer=her_buffer
        )
        
        print(f"✅ 优化Graph SAC Agent创建完成")
        return agent
    
    def train(self):
        """执行训练"""
        env = self.create_environment()
        agent = self.create_agent(env)
        
        print(f"\\n🎯 开始优化Graph SAC训练")
        print(f"🧠 核心测试：3维节点特征的运动学推理能力")
        print(f"🎮 预期：网络必须学会从关节角度推理空间行为")
        print(f"=" * 60)
        
        # 训练统计
        success_count = 0
        total_episodes = 0
        recent_successes = []
        total_steps = 0
        
        for episode in range(self.config['episodes']):
            # 重置环境
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
                
                # Agent观察
                agent.observe(result)
                
                # 学习
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
            
            # 定期评估
            if (episode + 1) % self.config['eval_every'] == 0:
                success_rate = sum(recent_successes) / len(recent_successes) * 100
                
                print(f"Episode {episode + 1:,}: 成功率 {success_rate:.1f}% "
                      f"(近100ep), 总成功率 {success_count/total_episodes*100:.1f}%")
                
                # 保存最佳模型
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self._save_checkpoint(agent, episode + 1, success_rate)
                    print(f"   🏆 新最佳成功率: {success_rate:.1f}%!")
        
        final_success_rate = success_count / total_episodes * 100
        print(f"\\n🎉 优化Graph SAC训练完成!")
        print(f"   最终成功率: {final_success_rate:.1f}%")
        print(f"   最佳成功率: {self.best_success_rate:.1f}%")
        print(f"   总episodes: {total_episodes:,}")
        print(f"   总steps: {total_steps:,}")
        
        return {
            'final_success_rate': final_success_rate,
            'best_success_rate': self.best_success_rate,
            'total_episodes': total_episodes,
            'total_steps': total_steps,
            'architecture': '3D-Node-Features + Structure-Spatial-Separation'
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
    """主训练函数"""
    
    # 🧪 快速测试配置（验证设计）
    config = {
        'dof_range': (2, 4),
        'segment_length_range': (0.168, 0.252),
        'goal_threshold': 0.15,
        'max_episode_steps': 200,
        
        # 快速验证配置
        'episodes': 1000,  # 快速测试
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_capacity': 50000,
        'batch_size': 128,
        'training_rounds': 25,
        'learning_starts': 500,
        'learn_every': 25,
        'eval_every': 100,
        
        # 网络配置
        'hidden_dim': 128,
        'num_graph_layers': 3,
        'num_attention_heads': 4,
        'critic_hidden_dims': [256, 256],
    }
    
    # 🚀 生产配置（完整训练）
    # config.update({
    #     'episodes': 10000,
    #     'buffer_capacity': 500000,
    #     'batch_size': 256,
    #     'training_rounds': 50,
    #     'learning_starts': 2000,
    #     'learn_every': 10,
    #     'hidden_dim': 256,
    # })
    
    print("🧪 优化Graph SAC训练")
    print("💡 节点特征: [joint1, joint2, length] - 3维")
    print("🎯 目标: 验证结构-空间分离的学习能力")
    print("🔬 核心假设: 网络必须学会运动学推理")
    print("=" * 60)
    
    trainer = OptimizedGraphSACTrainer(config)
    results = trainer.train()
    
    print(f"\\n📊 训练总结:")
    print(f"   架构: {results['architecture']}")
    print(f"   最终成功率: {results['final_success_rate']:.1f}%")
    print(f"   最佳成功率: {results['best_success_rate']:.1f}%")
    print(f"\\n🧠 关键验证:")
    if results['best_success_rate'] > 30:
        print(f"   ✅ 网络成功学会了运动学推理!")
        print(f"   ✅ 3维节点特征足以支持复杂控制!")
        print(f"   ✅ 结构-空间分离设计有效!")
    else:
        print(f"   ⚠️ 可能需要调整网络架构或训练参数")
        print(f"   💡 建议: 增加隐藏层维度或Graph层数")


if __name__ == "__main__":
    main()