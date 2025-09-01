#!/usr/bin/env python3
"""
优化Graph SAC生产级训练脚本
长期训练配置，用于获得高质量结果
"""

from train_optimized_graph_sac import OptimizedGraphSACTrainer

def main():
    # 🚀 生产级长期训练配置
    production_config = {
        'dof_range': (2, 4),
        'segment_length_range': (0.168, 0.252), 
        'goal_threshold': 0.15,
        'max_episode_steps': 200,
        
        # 长期训练参数
        'episodes': 5000,  # 5000个episodes充分训练
        'device': 'cuda',
        'buffer_capacity': 500000,  # 50万经验容量
        'batch_size': 256,  # 大batch size稳定训练
        'training_rounds': 50,  # 充分学习
        'learning_starts': 2000,  # 充足warmup
        'learn_every': 10,  # 频繁更新
        'eval_every': 250,  # 每250个episode评估一次
        
        # 大网络配置
        'hidden_dim': 256,  # 大隐藏层
        'num_graph_layers': 4,  # 深Graph网络
        'num_attention_heads': 8,  # 多头注意力
        'critic_hidden_dims': [512, 512],  # 大Critic
        
        'save_dir': './optimized_graph_sac_results'
    }
    
    print("🚀 优化Graph SAC生产级训练")
    print("💪 配置: 5000 episodes, 500K buffer, 256 hidden, 4 graph layers")
    print("📊 预期: 20次评估输出，最终达到高成功率")
    print("🎯 目标: 验证3维节点特征的完整学习能力")
    print("=" * 80)
    
    trainer = OptimizedGraphSACTrainer(production_config)
    
    try:
        results = trainer.train()
        
        print(f"\n🎉 生产训练完成!")
        print(f"   架构: {results['architecture']}")
        print(f"   最终成功率: {results['final_success_rate']:.1f}%")
        print(f"   最佳成功率: {results['best_success_rate']:.1f}%")
        print(f"   总episodes: {results['total_episodes']:,}")
        print(f"   总steps: {results['total_steps']:,}")
        
        # 性能分析
        if results['best_success_rate'] > 80:
            print(f"\n🏆 EXCELLENT! 网络成功掌握了运动学推理!")
            print(f"✅ 3维节点特征充分有效")
            print(f"✅ Graph+Goal架构优秀")
        elif results['best_success_rate'] > 50:
            print(f"\n✅ GOOD! 显著学习效果")
            print(f"💡 可考虑进一步调优超参数")
        else:
            print(f"\n🤔 需要分析学习困难原因")
            print(f"💡 可能需要调整网络架构或训练策略")
            
    except KeyboardInterrupt:
        print(f"\n🛑 训练被用户中断")
        print(f"💾 最佳checkpoint已保存")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        raise


if __name__ == "__main__":
    main()