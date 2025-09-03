#!/usr/bin/env python3
"""
分析变长节段对工作空间和目标可达性的影响
模拟真实训练环境中的情况
"""

import numpy as np
from complex_kinematics_environment import ComplexKinematicsSoftArmEnvironment

def test_variable_length_impact(n_tests=200):
    """测试变长节段对目标可达性的影响"""
    print("🔬 测试变长节段对目标可达性的影响")
    print("=" * 50)
    
    reachability_results = []
    length_configs = []
    
    for test in range(n_tests):
        # 创建环境（模拟真实训练）
        env = ComplexKinematicsSoftArmEnvironment(goal_threshold=0.15)
        
        # 模拟一次reset（随机DOF + 随机长度）
        obs, info = env.reset()
        
        # 记录当前配置
        current_dof = env.current_n_segments
        current_lengths = env.segment_lengths[:current_dof].copy()
        total_length = np.sum(current_lengths)
        
        # 当前目标采样算法
        max_reach_theory = total_length * 0.6
        
        # 快速采样一些可达点（100个）
        reachable_points = []
        for i in range(100):
            joint_angles = np.random.uniform(-np.pi, np.pi, current_dof * 2)
            env.joint_angles[:current_dof * 2] = joint_angles
            position = env._complex_forward_kinematics()
            reachable_points.append(position.copy())
        
        reachable_points = np.array(reachable_points)
        max_reach_actual = np.max(np.linalg.norm(reachable_points, axis=1))
        
        # 采样10个目标测试可达性
        reachable_count = 0
        for i in range(10):
            env._sample_goal()
            goal = env.goal_position
            
            # 检查是否在可达范围内（tolerance=0.15）
            distances = np.linalg.norm(reachable_points - goal, axis=1)
            if np.min(distances) <= 0.15:
                reachable_count += 1
        
        reachability_rate = reachable_count / 10 * 100
        
        # 记录结果
        reachability_results.append(reachability_rate)
        length_configs.append({
            'dof': current_dof,
            'lengths': current_lengths,
            'total_length': total_length,
            'max_reach_theory': max_reach_theory,
            'max_reach_actual': max_reach_actual,
            'reachability': reachability_rate
        })
        
        if (test + 1) % 50 == 0:
            print(f"   已测试 {test + 1}/{n_tests} 配置")
    
    return reachability_results, length_configs

def analyze_results(reachability_results, length_configs):
    """分析结果"""
    print(f"\n📊 变长节段目标可达性分析")
    print("=" * 50)
    
    # 总体统计
    avg_reachability = np.mean(reachability_results)
    min_reachability = np.min(reachability_results)
    max_reachability = np.max(reachability_results)
    
    # 按DOF分组统计
    dof_stats = {}
    for i, config in enumerate(length_configs):
        dof = config['dof']
        if dof not in dof_stats:
            dof_stats[dof] = []
        dof_stats[dof].append({
            'reachability': reachability_results[i],
            'total_length': config['total_length'],
            'theory_actual_ratio': config['max_reach_theory'] / config['max_reach_actual'] if config['max_reach_actual'] > 0 else 0
        })
    
    print(f"总体可达性:")
    print(f"  平均: {avg_reachability:.1f}%")
    print(f"  范围: {min_reachability:.1f}% - {max_reachability:.1f}%")
    print(f"  低于60%的配置: {sum(1 for r in reachability_results if r < 60)}/{len(reachability_results)}")
    
    print(f"\n按DOF分组:")
    for dof in sorted(dof_stats.keys()):
        stats = dof_stats[dof]
        avg_reach = np.mean([s['reachability'] for s in stats])
        min_reach = np.min([s['reachability'] for s in stats])
        count_low = sum(1 for s in stats if s['reachability'] < 60)
        avg_length = np.mean([s['total_length'] for s in stats])
        
        print(f"  {dof}节: 平均可达性{avg_reach:.1f}%, 最低{min_reach:.1f}%, "
              f"低于60%: {count_low}/{len(stats)}, 平均长度{avg_length:.2f}m")
    
    # 找出问题配置
    problem_configs = [
        (i, config) for i, config in enumerate(length_configs) 
        if reachability_results[i] < 60
    ]
    
    if problem_configs:
        print(f"\n⚠️  发现 {len(problem_configs)} 个问题配置 (可达性<60%):")
        for i, (idx, config) in enumerate(problem_configs[:5]):  # 只显示前5个
            print(f"  {i+1}. {config['dof']}节, 长度{config['lengths']}, "
                  f"可达性{reachability_results[idx]:.1f}%, "
                  f"理论/实际={config['max_reach_theory']:.2f}/{config['max_reach_actual']:.2f}")
    
    # 改进建议
    print(f"\n💡 改进建议:")
    
    # 计算更好的系数
    theory_actual_ratios = []
    for config in length_configs:
        if config['max_reach_actual'] > 0:
            ratio = config['max_reach_theory'] / config['max_reach_actual']
            theory_actual_ratios.append(ratio)
    
    if theory_actual_ratios:
        avg_ratio = np.mean(theory_actual_ratios)
        # 建议系数让理论reach约等于实际reach的0.8倍
        suggested_factor = 0.8 / avg_ratio * 0.6  # 当前0.6调整
        print(f"1. 建议调整reach系数: {0.6:.2f} → {suggested_factor:.2f}")
        print(f"2. 当前理论/实际平均比例: {avg_ratio:.2f}")
    
    if avg_reachability < 80:
        print(f"3. 整体可达性偏低({avg_reachability:.1f}%)，建议增大目标采样范围")
    
    return dof_stats

def main():
    reachability_results, length_configs = test_variable_length_impact(n_tests=300)
    dof_stats = analyze_results(reachability_results, length_configs)
    
    return reachability_results, length_configs, dof_stats

if __name__ == "__main__":
    results = main()