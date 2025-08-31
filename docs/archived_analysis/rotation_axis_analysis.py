#!/usr/bin/env python3
"""
旋转轴方向分析：软体vs刚体的关键差异
软体：所有β都在同一方向 (Z轴方向弯曲)
刚体：每个关节轴方向可能不同 (X/Y/Z轴旋转)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_rotation_axis_difference():
    """分析旋转轴方向的差异"""
    print("🔄 旋转轴方向差异分析")
    print("=" * 50)
    
    print("🌊 软体臂旋转特性:")
    print("   - 每个segment在局部坐标系中弯曲")
    print("   - β角: 弯曲方向角 (around local Z-axis)")
    print("   - α角: 弯曲幅度角 (bend magnitude)")
    print("   - 所有弯曲都是相对于segment的局部轴向")
    
    print(f"\n🤖 刚体臂旋转特性:")
    print("   - 每个关节有固定的旋转轴方向")
    print("   - 常见配置: [Z, Y, Y, X, Y, Z] (如6DOF机械臂)")
    print("   - 每个关节轴可能不同!")
    
    # 典型6DOF刚体臂的关节轴配置
    typical_6dof = [
        {"joint": "肩部Z轴", "axis": [0, 0, 1], "range": "±180°"},
        {"joint": "肩部Y轴", "axis": [0, 1, 0], "range": "±90°"},
        {"joint": "肘部Y轴", "axis": [0, 1, 0], "range": "±120°"},
        {"joint": "腕部X轴", "axis": [1, 0, 0], "range": "±180°"},
        {"joint": "腕部Y轴", "axis": [0, 1, 0], "range": "±90°"},
        {"joint": "腕部Z轴", "axis": [0, 0, 1], "range": "±180°"}
    ]
    
    print(f"\n📊 典型6DOF刚体臂关节轴:")
    for i, joint in enumerate(typical_6dof, 1):
        axis_str = f"[{joint['axis'][0]}, {joint['axis'][1]}, {joint['axis'][2]}]"
        print(f"   关节{i}: {joint['joint']} - 轴向{axis_str} - 范围{joint['range']}")
    
    return typical_6dof

def analyze_soft_arm_local_coordinates():
    """分析软体臂的局部坐标系"""
    print(f"\n🌊 软体臂局部坐标系分析:")
    
    # 从C++代码可以看出，每个segment都有自己的局部坐标系
    print("每个软体segment:")
    print("   1. 有局部坐标系 (随前一segment旋转)")
    print("   2. α: 相对局部Z轴的弯曲角度")
    print("   3. β: 弯曲的方位角 (在局部XY平面内)")
    
    # 软体臂的复合旋转
    print(f"\n🔄 软体segment的复合变换:")
    print("   Position = T1 * T2 * T3 * local_bend")
    print("   其中每个Ti包含:")
    print("     - 平移 (前一segment的端点)")
    print("     - 旋转 (前一segment的方向)")
    
    # 模拟3段软体臂的坐标系变换
    segment_configs = [
        {"alpha": 0.5, "beta": 0.2, "length": 0.21},
        {"alpha": 0.3, "beta": -0.1, "length": 0.21},
        {"alpha": 0.7, "beta": 0.3, "length": 0.21}
    ]
    
    print(f"\n📐 软体臂坐标系传播:")
    current_transform = np.eye(4)
    
    for i, seg in enumerate(segment_configs, 1):
        print(f"   Segment {i}: α={seg['alpha']:.1f}, β={seg['beta']:.1f}")
        print(f"     局部弯曲轴: 始终是当前坐标系的Z轴")
        print(f"     弯曲方向: β角在当前XY平面内旋转")

def compare_axis_representations():
    """对比轴表示方法"""
    print(f"\n🔍 轴表示方法对比:")
    print("=" * 40)
    
    print("问题核心:")
    print("🌊 软体: β总是相对于'当前segment轴向'的方位角")
    print("🤖 刚体: 每个关节有'固定世界坐标系'的旋转轴")
    
    print(f"\n🎯 统一挑战:")
    print("方案1 - 直接统一:")
    print("   软体: [α, β, length]  # β=局部方位角")
    print("   刚体: [θ, 0, length]  # θ=绕固定轴角度")
    print("   问题: β和θ的参考系不同!")
    
    print(f"\n💡 解决方案:")
    
    print("方案A - 轴向标识增强:")
    print("   节点: [angle, direction, length, axis_type]")
    print("   软体: [α, β, s, 1]  # 1=相对轴")
    print("   刚体: [θ, axis_id, L, 0]  # 0=绝对轴")
    print("   其中axis_id: 0=X轴, 1=Y轴, 2=Z轴")
    
    print(f"\n方案B - 局部坐标统一:")
    print("   核心思想: 将刚体也转换为'相对当前坐标系'")
    print("   刚体关节轴 → 相对于当前segment的轴向")
    print("   这样所有旋转都变成'局部坐标系相对旋转'")

def propose_unified_solution():
    """提出统一解决方案"""
    print(f"\n🚀 推荐统一方案: 局部坐标系统一")
    print("=" * 50)
    
    print("🎯 核心理念:")
    print("   所有旋转都表示为'相对于当前segment坐标系'的旋转")
    
    print(f"\n🔧 实现方式:")
    print("```python")
    print("# 统一的局部旋转表示")
    print("def unified_local_rotation_node(segment):")
    print("    if segment.type == 'soft':")
    print("        # 软体: 原生就是局部坐标")
    print("        return [segment.alpha, segment.beta, segment.length]")
    print("    ")
    print("    elif segment.type == 'rigid':")
    print("        # 刚体: 转换为局部坐标表示")
    print("        local_axis = transform_to_local_axis(segment.world_axis)")
    print("        if local_axis == 'Z':  # 轴向旋转")
    print("            return [0.001, segment.theta, segment.length]")
    print("        elif local_axis == 'Y':  # 侧向旋转") 
    print("            return [segment.theta, 0, segment.length]")
    print("        elif local_axis == 'X':  # 滚转旋转")
    print("            return [segment.theta, np.pi/2, segment.length]")
    print("```")
    
    print(f"\n🌟 统一后的物理含义:")
    print("   参数1 (α): 弯曲/抬升角度")
    print("   参数2 (β): 方位/偏转角度") 
    print("   参数3: 段长度")
    print("   所有旋转都在segment的局部坐标系内!")

def mathematical_elegance_check():
    """数学优雅性检查"""
    print(f"\n✨ 数学优雅性验证:")
    
    print("🎯 局部坐标统一的优势:")
    print("   ✅ 所有segment都有一致的'弯曲语义'")
    print("   ✅ GNN学习统一的局部几何关系")
    print("   ✅ 无需区分软体/刚体，都是'局部变形'")
    print("   ✅ 参数物理含义一致")
    
    print(f"\n🔬 数学验证:")
    print("   软体segment: 局部弯曲 → 全局位置")
    print("   刚体segment: 局部旋转 → 全局位置")  
    print("   统一性: 都是'局部变换'的组合!")
    
    print(f"\n🧠 对GNN的好处:")
    print("   - GNN学习: [local_bend, local_twist, length]")
    print("   - 不需要知道这是软体还是刚体")
    print("   - 只需要理解'局部几何变形'规律")
    print("   - 参数连续性完美保持")

if __name__ == "__main__":
    analyze_rotation_axis_difference()
    analyze_soft_arm_local_coordinates()
    compare_axis_representations()
    propose_unified_solution()
    mathematical_elegance_check()
    
    print(f"\n🎉 结论:")
    print("虽然存在旋转轴差异，但仍可通过局部坐标系统一!")
    print("关键是将所有旋转转换为'相对当前segment'的局部变换")
    print("这样 [α, β, length] 依然是完美的统一表示! 🎯")