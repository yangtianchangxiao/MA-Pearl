#!/usr/bin/env python3
"""
通用Pearl Agent可视化演示脚本
展示如何使用visualize_universal_agent.py来可视化任意网络和环境
"""

import subprocess
import sys
from pathlib import Path

def run_visualization(checkpoint_path, env_type, network_type, episodes=1, with_gif=True):
    """运行可视化演示"""
    cmd = [
        "/home/cx/miniconda3/envs/pytorch/bin/python", 
        "/home/cx/MA-Pearl/visualize_universal_agent.py",
        "--checkpoint", checkpoint_path if checkpoint_path else "",
        "--env-type", env_type,
        "--network-type", network_type,
        "--episodes", str(episodes)
    ]
    
    if not with_gif:
        cmd.append("--no-gif")
    
    print(f"🚀 执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    """演示各种配置的可视化"""
    
    print("🎬 Pearl通用Agent可视化演示")
    print("=" * 60)
    
    # 可用的配置
    demos = [
        {
            'name': 'Graph网络 - 变长软体臂 (有checkpoint)',
            'checkpoint': '/home/cx/MA-Pearl/graph_variable_arm_results/graph_variable_soft_arm_6dof/best_checkpoint.pt',
            'env_type': 'variable_soft_arm_6dof',
            'network_type': 'graph',
            'episodes': 1
        },
        {
            'name': 'MLP网络 - 变长软体臂 (随机初始化)',
            'checkpoint': None,
            'env_type': 'variable_soft_arm_6dof', 
            'network_type': 'mlp',
            'episodes': 1
        },
        {
            'name': 'MLP网络 - NDOF 3DOF (随机初始化)',
            'checkpoint': None,
            'env_type': 'ndof_3dof',
            'network_type': 'mlp', 
            'episodes': 1
        },
        {
            'name': 'Graph网络 - NDOF 3DOF (随机初始化)',
            'checkpoint': None,
            'env_type': 'ndof_3dof',
            'network_type': 'graph',
            'episodes': 1
        },
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"\n📍 演示 {i}: {demo['name']}")
        print("-" * 40)
        
        success = run_visualization(
            checkpoint_path=demo['checkpoint'],
            env_type=demo['env_type'],
            network_type=demo['network_type'],
            episodes=demo['episodes'],
            with_gif=False  # 不生成GIF，只保存截图
        )
        
        if success:
            print(f"✅ 演示 {i} 完成")
        else:
            print(f"❌ 演示 {i} 失败")
    
    print("\n🎉 所有演示完成!")
    print("\n📁 可视化结果保存在以下目录:")
    for viz_dir in Path(".").glob("visualization_*"):
        if viz_dir.is_dir():
            file_count = len(list(viz_dir.glob("*.png")))
            print(f"   {viz_dir.name}: {file_count} 个PNG文件")

if __name__ == "__main__":
    main()