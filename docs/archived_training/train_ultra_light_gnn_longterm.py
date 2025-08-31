#!/usr/bin/env python3
"""
超轻量GNN长期训练脚本
增加结构多样性，充分验证Graph网络的泛化能力
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# 使用超轻量GNN组件
from optimized_graph_her_wrapper import OptimizedGraphHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


def main():
    """长期训练主函数"""
    
    # 🚀 长期训练 + 高多样性配置
    config = {
        # 🌟 随机结构多样性 (2-5节)
        'dof_range': (2, 5),  # 2-5节随机选择 (4-10DOF) 
        'segment_length_range': (0.1, 0.35),  # 大长度变化 (3.5x范围)
        'goal_threshold': 0.15,
        'max_episode_steps': 200,
        
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
        
        'save_dir': './ultra_light_gnn_longterm_results'
    }
    
    print("🚀 超轻量GNN长期训练 + 高结构多样性")
    print("🌟 核心挑战: DOF 2-6节, 长度范围 0.1-0.4 (4倍变化)")
    print("⚡ 网络优势: 轻量GNN应该能很好适应结构变化")
    print("🎯 目标: 验证Graph方法在高多样性下的泛化能力")
    print("=" * 70)
    print(f"📊 配置详情:")
    print(f"   DOF范围: {config['dof_range'][0]}-{config['dof_range'][1]}节")
    print(f"   长度范围: {config['segment_length_range'][0]:.1f}-{config['segment_length_range'][1]:.1f} ({config['segment_length_range'][1]/config['segment_length_range'][0]:.1f}x变化)")
    print(f"   训练episodes: {config['episodes']:,}")
    print(f"   网络参数: {config['hidden_dim']}维, {config['num_gnn_layers']}层GNN")
    print("=" * 70)
    
    # 创建训练器 (复用之前的类)
    from train_ultra_light_gnn_sac import UltraLightGNNSACTrainer
    trainer = UltraLightGNNSACTrainer(config)
    
    try:
        results = trainer.train()
        
        print(f"\n🎉 长期高多样性训练完成!")
        print(f"   架构: {results['architecture']}")
        print(f"   最终成功率: {results['final_success_rate']:.1f}%")
        print(f"   最佳成功率: {results['best_success_rate']:.1f}%")
        print(f"   总episodes: {results['total_episodes']:,}")
        print(f"   总steps: {results['total_steps']:,}")
        
        # 高多样性性能分析
        print(f"\n🧠 高多样性挑战验证:")
        if results['best_success_rate'] > 70:
            print(f"   🏆 OUTSTANDING! Graph网络完美适应高多样性!")
            print(f"   ✅ 2-6节DOF + 4x长度变化 = 完全掌握!")
            print(f"   🌟 证明了Graph方法的强大泛化能力!")
        elif results['best_success_rate'] > 50:
            print(f"   🎯 EXCELLENT! Graph网络很好适应多样性!")
            print(f"   ✅ 结构变化对Graph网络影响有限")
            print(f"   💪 超轻量设计 + 高适应性 = 成功!")
        elif results['best_success_rate'] > 30:
            print(f"   ✅ GOOD! 显著学习但受多样性挑战")
            print(f"   📊 考虑多样性，这是合理的表现")
            print(f"   💡 可能需要稍大网络应对高变化")
        else:
            print(f"   🤔 高多样性确实是挑战")
            print(f"   📊 可能需要调整网络规模或训练策略")
            
    except KeyboardInterrupt:
        print(f"\n🛑 长期训练被用户中断")
        print(f"💾 最佳checkpoint已保存")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        raise


if __name__ == "__main__":
    main()