#!/usr/bin/env python3
"""
软体机械臂工作空间分析
分析目标设置是否合理，是否存在大量不可达目标
"""

import numpy as np
import matplotlib.pyplot as plt
from complex_kinematics_environment import ComplexKinematicsSoftArmEnvironment
import time

def sample_reachable_points(env, n_samples=1000):
    """通过随机采样估计可达工作空间"""
    reachable_points = []
    
    print(f"🔬 采样{n_samples}个随机配置的末端位置...")
    
    for i in range(n_samples):
        # 随机关节角度
        current_dof = env.current_n_segments * 2
        env.joint_angles[:current_dof] = np.random.uniform(-np.pi, np.pi, current_dof)
        
        # 计算末端位置
        position = env._complex_forward_kinematics()
        reachable_points.append(position.copy())
        
        if (i + 1) % 200 == 0:
            print(f"   已采样 {i + 1}/{n_samples}")
    
    return np.array(reachable_points)

def analyze_goal_sampling(env, n_goals=500):
    """分析目标采样分布"""
    goals = []
    
    print(f"🎯 分析{n_goals}个目标采样...")
    
    for i in range(n_goals):
        env._sample_goal()
        goals.append(env.goal_position.copy())
    
    return np.array(goals)

def check_reachability(env, goals, reachable_points, tolerance=0.15):
    """检查目标的可达性"""
    reachable_count = 0
    distances_to_workspace = []
    
    print(f"🔍 检查目标可达性 (tolerance={tolerance}m)...")
    
    for goal in goals:
        # 计算到最近可达点的距离
        distances = np.linalg.norm(reachable_points - goal, axis=1)
        min_distance = np.min(distances)
        distances_to_workspace.append(min_distance)
        
        if min_distance <= tolerance:
            reachable_count += 1
    
    reachability_rate = reachable_count / len(goals) * 100
    
    return reachability_rate, distances_to_workspace

def main():
    print("🤖 软体机械臂工作空间分析")
    print("=" * 50)
    
    # 测试不同DOF配置
    dof_configs = [
        (2, "2节4DOF"),
        (3, "3节6DOF"), 
        (4, "4节8DOF"),
        (5, "5节10DOF")
    ]
    
    results = {}
    
    for n_segments, name in dof_configs:
        print(f"\n📊 分析 {name}")
        print("-" * 30)
        
        # 创建环境
        env = ComplexKinematicsSoftArmEnvironment(goal_threshold=0.15)
        
        # 手动设置配置
        env.current_n_segments = n_segments
        env.segment_lengths = np.full(n_segments, 0.25)  # 固定0.25m每节
        
        # 初始化关节角度数组
        max_dof = 10  # 5节×2DOF
        env.joint_angles = np.zeros(max_dof)
        
        print(f"   总长度: {np.sum(env.segment_lengths):.2f}m")
        
        # 采样可达空间
        reachable_points = sample_reachable_points(env, n_samples=800)
        
        # 分析目标采样
        goals = analyze_goal_sampling(env, n_goals=300)
        
        # 检查可达性
        reachability_rate, distances = check_reachability(
            env, goals, reachable_points, tolerance=0.15
        )
        
        # 统计结果
        max_reach_actual = np.max(np.linalg.norm(reachable_points, axis=1))
        max_reach_theory = np.sum(env.segment_lengths) * 0.6  # 当前算法
        avg_distance_to_workspace = np.mean(distances)
        
        results[name] = {
            'reachability_rate': reachability_rate,
            'max_reach_actual': max_reach_actual,
            'max_reach_theory': max_reach_theory,
            'avg_distance': avg_distance_to_workspace,
            'reachable_points': reachable_points,
            'goals': goals
        }
        
        print(f"✅ {name} 结果:")
        print(f"   理论最大reach (0.6系数): {max_reach_theory:.2f}m")
        print(f"   实际最大reach: {max_reach_actual:.2f}m")
        print(f"   目标可达性: {reachability_rate:.1f}%")
        print(f"   平均到工作空间距离: {avg_distance_to_workspace:.3f}m")
        
        if reachability_rate < 70:
            print(f"   ⚠️  可达性过低！可能影响训练效果")
    
    # 总结分析
    print(f"\n📋 工作空间分析总结")
    print("=" * 50)
    
    for name, result in results.items():
        rate = result['reachability_rate']
        theory = result['max_reach_theory'] 
        actual = result['max_reach_actual']
        
        print(f"{name}:")
        print(f"  可达性: {rate:5.1f}% ", end="")
        if rate < 50:
            print("🔴 严重问题")
        elif rate < 70:
            print("🟡 需要改进")
        else:
            print("🟢 较好")
            
        print(f"  范围比较: 理论{theory:.2f}m vs 实际{actual:.2f}m")
        
        # 建议新的系数
        if actual > 0:
            suggested_factor = (actual * 0.9) / np.sum([0.25] * int(name[0]))
            print(f"  建议系数: {suggested_factor:.2f} (当前0.6)")
        print()
    
    # 可视化建议
    print("💡 改进建议:")
    print("1. 调整目标采样算法的reach系数")
    print("2. 考虑不同DOF的工作空间差异")
    print("3. 增加Z方向约束，避免过高目标")
    
    return results

if __name__ == "__main__":
    results = main()