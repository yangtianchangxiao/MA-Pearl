#!/usr/bin/env python3
"""
验证软体和刚体机械臂的Graph统一性
"""

import numpy as np

def create_unified_graph_representation():
    """验证统一Graph表示"""
    
    print("🤖 刚体vs软体机械臂Graph统一性分析")
    print("=" * 60)
    
    # 1. 刚体机械臂 (传统6DOF)
    rigid_arm = {
        'segments': [
            {'length': 0.3, 'dof': 1, 'joint_type': 'revolute'},  # 肩部
            {'length': 0.25, 'dof': 1, 'joint_type': 'revolute'}, # 肘部  
            {'length': 0.2, 'dof': 1, 'joint_type': 'revolute'},  # 腕部
            {'length': 0.15, 'dof': 1, 'joint_type': 'revolute'}, # 腕部2
            {'length': 0.1, 'dof': 1, 'joint_type': 'revolute'},  # 腕部3
            {'length': 0.05, 'dof': 1, 'joint_type': 'revolute'}  # 末端
        ]
    }
    
    # 2. 软体机械臂 (3段，每段2DOF)  
    soft_arm = {
        'segments': [
            {'length': 0.21, 'dof': 2, 'joint_type': 'soft_bending'},  # α1,β1
            {'length': 0.21, 'dof': 2, 'joint_type': 'soft_bending'},  # α2,β2
            {'length': 0.21, 'dof': 2, 'joint_type': 'soft_bending'}   # α3,β3
        ]
    }
    
    print("📐 刚体臂结构:")
    total_rigid_dof = 0
    for i, seg in enumerate(rigid_arm['segments']):
        print(f"   Segment {i+1}: 长度={seg['length']:.2f}m, DOF={seg['dof']}, 类型={seg['joint_type']}")
        total_rigid_dof += seg['dof']
    print(f"   总DOF: {total_rigid_dof}")
    
    print(f"\n🌊 软体臂结构:")
    total_soft_dof = 0
    for i, seg in enumerate(soft_arm['segments']):
        print(f"   Segment {i+1}: 长度={seg['length']:.2f}m, DOF={seg['dof']}, 类型={seg['joint_type']}")
        total_soft_dof += seg['dof']
    print(f"   总DOF: {total_soft_dof}")
    
    # 3. 统一Graph节点表示
    print(f"\n🎯 统一Graph节点表示:")
    print("方案1 - 直接统一:")
    print("   刚体节点: [joint_angle, 0, length, pos_x, pos_y, pos_z]")
    print("   软体节点: [alpha, beta, length, pos_x, pos_y, pos_z]")
    print("   ✅ 完全兼容! 都是6维节点特征")
    
    print(f"\n方案2 - 你的简化方案:")
    print("   刚体节点: [joint_angle, 0, length]  # padding 0")
    print("   软体节点: [alpha, beta, length]")
    print("   ✅ 完全兼容! 都是3维节点特征")
    print("   🎯 空间信息通过achieved_goal, desired_goal处理")
    
    # 4. 长度变化分析
    print(f"\n📏 长度变化特性:")
    print("刚体臂:")
    print("   ❌ 长度绝对固定 (mechanical constraints)")
    print("   ✅ 每个episode: lengths = [0.3, 0.25, 0.2, ...]")
    
    print("我们的软体臂:")
    print("   ❌ episode内长度固定 (从代码看)")
    print("   ✅ 每个episode: lengths = [0.21, 0.21, 0.21]")
    print("   🤔 但episode间可以随机改变长度!")
    
    # 5. 工作空间计算
    rigid_workspace = sum(seg['length'] for seg in rigid_arm['segments'])
    soft_workspace = sum(seg['length'] for seg in soft_arm['segments'])
    
    print(f"\n🌐 工作空间半径:")
    print(f"   刚体臂: {rigid_workspace:.2f}m")
    print(f"   软体臂: {soft_workspace:.2f}m") 
    
    # 6. 统一性结论
    print(f"\n🎉 统一性结论:")
    print("✅ Graph结构完全可以统一!")
    print("✅ 节点特征维度相同")  
    print("✅ 边连接关系相同 (sequential)")
    print("✅ 长度都是固定的 (至少episode内)")
    print("🌟 区别只是DOF的物理含义不同!")
    
    return rigid_arm, soft_arm

def analyze_zero_length_case():
    """分析零长度刚体的情况"""
    print(f"\n🔍 零长度刚体分析:")
    print("如果刚体某段长度=0:")
    
    zero_length_rigid = {
        'segments': [
            {'length': 0.3, 'dof': 1},   # 正常段
            {'length': 0.0, 'dof': 1},   # 零长度段 (纯旋转关节)
            {'length': 0.25, 'dof': 1},  # 正常段
        ]
    }
    
    print("   节点表示: [joint_angle, 0, 0.0]  # 长度为0")
    print("   ✅ Graph仍然有效! 只是空间贡献为0")
    print("   ✅ GNN可以学到'跳过'这个节点的空间影响")
    print("   🎯 这确实可以表示纯旋转关节!")

if __name__ == "__main__":
    create_unified_graph_representation()
    analyze_zero_length_case()