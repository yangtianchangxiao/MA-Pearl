#!/usr/bin/env python3
"""
对比软体臂训练前后的性能差异
随机策略 vs 训练好的Pearl SAC+HER
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

from pearl.utils.instantiations.environments import SoftArmReachEnvironment
from simple_soft_arm_demo import create_simple_agent, visualize_soft_arm_simple


def test_random_policy(env, num_episodes=5):
    """测试随机策略的表现"""
    print("Testing Random Policy...")
    
    results = []
    total_success = 0
    
    for episode in range(num_episodes):
        obs, action_space = env.reset()
        episode_reward = 0
        episode_data = []
        
        for step in range(200):
            # 随机动作
            action = action_space.sample() * 0.2  # 小幅随机动作
            
            result = env.step(action)
            episode_reward += result.reward.item()
            
            observation = result.observation.cpu().numpy() if hasattr(result.observation, 'cpu') else result.observation.numpy()
            joint_angles = observation[:6]
            achieved_goal = observation[6:9]
            desired_goal = observation[9:12]
            distance = np.linalg.norm(achieved_goal - desired_goal)
            
            episode_data.append({
                'step': step + 1,
                'distance': distance,
                'reward': result.reward.item()
            })
            
            if result.terminated or result.truncated:
                success = result.terminated.item() if hasattr(result.terminated, 'item') else result.terminated
                total_success += success
                
                print(f"Random Episode {episode+1}: {'SUCCESS' if success else 'TIMEOUT'}, Steps: {step+1}, Reward: {episode_reward:.2f}")
                break
        
        results.append({
            'episode': episode + 1,
            'success': success,
            'steps': step + 1,
            'reward': episode_reward,
            'final_distance': episode_data[-1]['distance'],
            'data': episode_data
        })
    
    random_success_rate = total_success / num_episodes * 100
    print(f"Random Policy Success Rate: {random_success_rate:.1f}%")
    
    return results, random_success_rate


def test_trained_policy(num_episodes=5):
    """测试训练好的策略"""
    print("Testing Trained Policy...")
    
    env, agent = create_simple_agent()
    results = []
    total_success = 0
    
    for episode in range(num_episodes):
        obs, action_space = env.reset()
        agent.reset(obs, action_space)
        episode_reward = 0
        episode_data = []
        
        for step in range(200):
            # 训练好的策略
            action = agent.act(exploit=True)
            
            result = env.step(action)
            episode_reward += result.reward.item()
            agent.observe(result)
            
            observation = result.observation.cpu().numpy()
            joint_angles = observation[:6]
            achieved_goal = observation[6:9]
            desired_goal = observation[9:12]
            distance = np.linalg.norm(achieved_goal - desired_goal)
            
            episode_data.append({
                'step': step + 1,
                'distance': distance,
                'reward': result.reward.item()
            })
            
            if result.terminated or result.truncated:
                success = result.terminated.item()
                total_success += success
                
                print(f"Trained Episode {episode+1}: {'SUCCESS' if success else 'TIMEOUT'}, Steps: {step+1}, Reward: {episode_reward:.2f}")
                break
        
        results.append({
            'episode': episode + 1,
            'success': success,
            'steps': step + 1,
            'reward': episode_reward,
            'final_distance': episode_data[-1]['distance'],
            'data': episode_data
        })
    
    trained_success_rate = total_success / num_episodes * 100
    print(f"Trained Policy Success Rate: {trained_success_rate:.1f}%")
    
    return results, trained_success_rate


def create_comparison_plot(random_results, trained_results, save_dir):
    """创建对比图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 成功率对比
    policies = ['Random Policy', 'Trained Policy (Pearl SAC+HER)']
    success_rates = [
        np.mean([r['success'] for r in random_results]) * 100,
        np.mean([r['success'] for r in trained_results]) * 100
    ]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax1.bar(policies, success_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # 添加数值标签
    for bar, rate in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. 最终距离对比
    random_distances = [r['final_distance'] for r in random_results]
    trained_distances = [r['final_distance'] for r in trained_results]
    
    ax2.boxplot([random_distances, trained_distances], labels=policies)
    ax2.axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Success Threshold (0.15m)')
    ax2.set_ylabel('Final Distance to Target (m)')
    ax2.set_title('Final Distance Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 距离随时间变化 (选择一个代表性episode)
    random_example = random_results[0]['data']
    trained_example = trained_results[0]['data']
    
    random_steps = [d['step'] for d in random_example]
    random_dist = [d['distance'] for d in random_example]
    trained_steps = [d['step'] for d in trained_example]
    trained_dist = [d['distance'] for d in trained_example]
    
    ax3.plot(random_steps, random_dist, 'r-', linewidth=2, label='Random Policy', alpha=0.8)
    ax3.plot(trained_steps, trained_dist, 'g-', linewidth=2, label='Trained Policy', alpha=0.8)
    ax3.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Success Threshold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Distance to Target (m)')
    ax3.set_title('Distance Evolution (Example Episode)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 奖励对比
    random_rewards = [r['reward'] for r in random_results]
    trained_rewards = [r['reward'] for r in trained_results]
    
    ax4.boxplot([random_rewards, trained_rewards], labels=policies)
    ax4.set_ylabel('Episode Reward')
    ax4.set_title('Episode Reward Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建统计表
    create_statistics_table(random_results, trained_results, save_dir)


def create_statistics_table(random_results, trained_results, save_dir):
    """创建统计表"""
    print("\n" + "="*60)
    print("SOFT ARM PERFORMANCE COMPARISON")
    print("="*60)
    
    # 计算统计数据
    random_success = np.mean([r['success'] for r in random_results]) * 100
    trained_success = np.mean([r['success'] for r in trained_results]) * 100
    
    random_reward = np.mean([r['reward'] for r in random_results])
    trained_reward = np.mean([r['reward'] for r in trained_results])
    
    random_distance = np.mean([r['final_distance'] for r in random_results])
    trained_distance = np.mean([r['final_distance'] for r in trained_results])
    
    random_steps = np.mean([r['steps'] for r in random_results])
    trained_steps = np.mean([r['steps'] for r in trained_results])
    
    print(f"{'Metric':<25} {'Random Policy':<15} {'Trained Policy':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Success Rate (%)':<25} {random_success:<15.1f} {trained_success:<15.1f} {trained_success-random_success:+.1f}")
    print(f"{'Avg Episode Reward':<25} {random_reward:<15.2f} {trained_reward:<15.2f} {trained_reward-random_reward:+.2f}")
    print(f"{'Avg Final Distance (m)':<25} {random_distance:<15.3f} {trained_distance:<15.3f} {trained_distance-random_distance:+.3f}")
    print(f"{'Avg Steps to Finish':<25} {random_steps:<15.1f} {trained_steps:<15.1f} {trained_steps-random_steps:+.1f}")
    print("="*70)
    
    improvement_factor = trained_success / max(random_success, 0.1)  # 避免除零
    print(f"SUCCESS RATE IMPROVEMENT: {improvement_factor:.1f}x")
    
    if trained_success >= 80:
        print("🎉 EXCELLENT PERFORMANCE!")
    elif trained_success >= 50:
        print("✅ GOOD PERFORMANCE!")
    elif trained_success >= 20:
        print("🔄 MODERATE IMPROVEMENT")
    else:
        print("❌ NEEDS MORE TRAINING")
    
    # 保存到文件
    with open(save_dir / 'performance_stats.txt', 'w') as f:
        f.write("SOFT ARM PERFORMANCE COMPARISON\n")
        f.write("="*60 + "\n")
        f.write(f"Random Policy Success Rate: {random_success:.1f}%\n")
        f.write(f"Trained Policy Success Rate: {trained_success:.1f}%\n")
        f.write(f"Improvement Factor: {improvement_factor:.1f}x\n")
        f.write(f"Average Reward Improvement: {trained_reward-random_reward:+.2f}\n")
        f.write(f"Average Distance Improvement: {trained_distance-random_distance:+.3f}m\n")


def main():
    """主函数"""
    print("🎯 Soft Arm Performance Comparison")
    print("Comparing Random Policy vs Trained Pearl SAC+HER")
    print("-" * 50)
    
    # 创建保存目录
    save_dir = Path('soft_arm_comparison')
    save_dir.mkdir(exist_ok=True)
    
    # 创建环境 (用于随机策略测试)
    env = SoftArmReachEnvironment(goal_threshold=0.15, max_steps=200)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 测试随机策略
    random_results, random_success_rate = test_random_policy(env, num_episodes=5)
    
    # 测试训练好的策略  
    trained_results, trained_success_rate = test_trained_policy(num_episodes=5)
    
    # 创建对比图表
    create_comparison_plot(random_results, trained_results, save_dir)
    
    print(f"\n📊 Results saved in: {save_dir}")
    print("📈 Check 'performance_comparison.png' for visual comparison")
    print("📋 Check 'performance_stats.txt' for detailed statistics")


if __name__ == "__main__":
    main()