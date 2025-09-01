#!/usr/bin/env python3
"""
轻量级Graph SAC配置
参考成功MLP训练的网络规模，但保持Graph架构的核心优势
"""

from train_optimized_graph_sac import OptimizedGraphSACTrainer

def main():
    # 🪶 轻量级Graph SAC配置 (参考成功的MLP规模)
    lightweight_config = {
        'dof_range': (2, 4),
        'segment_length_range': (0.168, 0.252), 
        'goal_threshold': 0.15,
        'max_episode_steps': 200,
        
        # 快速训练参数 (参考成功配置)
        'episodes': 3000,  # 适中规模
        'device': 'cuda',
        'buffer_capacity': 100000,  # 10万经验 (vs 50万)
        'batch_size': 128,  # 小batch size快速迭代 (vs 256)
        'training_rounds': 25,  # 保守训练 (vs 50)
        'learning_starts': 1000,  # 快速开始 (vs 2000)
        'learn_every': 20,  # 适中频率 (vs 10)
        'eval_every': 100,  # 频繁评估看进度 (vs 250)
        
        # 🪶 轻量级网络配置 (保持Graph核心，减少复杂度)
        'hidden_dim': 128,  # 适中隐藏层 (vs 256)
        'num_graph_layers': 2,  # 只用2层Graph (vs 4层)
        'num_attention_heads': 4,  # 适中attention (vs 8头)
        'critic_hidden_dims': [256, 256],  # 保持标准Critic
        
        'save_dir': './lightweight_graph_sac_results'
    }
    
    print("🪶 轻量级Graph SAC训练")
    print("⚡ 配置: 3000 episodes, 快速迭代, 轻量Graph网络")
    print("📊 预期: 30次评估输出 (每100 episodes)，快速看到进度")
    print("🎯 目标: 验证3维节点特征，但用轻量化网络")
    print("=" * 60)
    
    trainer = OptimizedGraphSACTrainer(lightweight_config)
    
    try:
        results = trainer.train()
        
        print(f"\n🎉 轻量级训练完成!")
        print(f"   架构: {results['architecture']}")
        print(f"   最终成功率: {results['final_success_rate']:.1f}%")
        print(f"   最佳成功率: {results['best_success_rate']:.1f}%")
        print(f"   总episodes: {results['total_episodes']:,}")
        print(f"   总steps: {results['total_steps']:,}")
        
        # 性能分析
        if results['best_success_rate'] > 60:
            print(f"\n🏆 轻量Graph网络成功!")
            print(f"✅ 3维节点特征有效，无需重网络")
            print(f"🪶 轻量化设计优秀")
        elif results['best_success_rate'] > 30:
            print(f"\n✅ 显著学习效果")
            print(f"💡 轻量Graph > 重Graph (效率角度)")
        else:
            print(f"\n🤔 可能需要调整超参数")
            print(f"📊 但至少训练速度正常了")
            
    except KeyboardInterrupt:
        print(f"\n🛑 训练被用户中断")
        print(f"💾 最佳checkpoint已保存")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        raise


if __name__ == "__main__":
    main()