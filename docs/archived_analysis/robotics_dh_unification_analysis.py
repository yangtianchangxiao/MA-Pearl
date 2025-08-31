#!/usr/bin/env python3
"""
机器人学DH参数统一分析
考虑刚体机械臂的完整几何描述：旋转轴、关节偏移、连杆长度、扭转角等
分析如何与软体臂统一表示
"""

import numpy as np

def analyze_dh_parameters():
    """分析DH参数的完整描述"""
    print("🤖 机器人学DH参数完整分析")
    print("=" * 50)
    
    print("📚 经典DH参数 (Denavit-Hartenberg):")
    print("   θᵢ: 关节角 (joint angle) - 绕zᵢ₋₁轴旋转")
    print("   dᵢ: 连杆偏移 (link offset) - 沿zᵢ₋₁轴平移")
    print("   aᵢ: 连杆长度 (link length) - 沿xᵢ轴平移")
    print("   αᵢ: 扭转角 (twist angle) - 绕xᵢ轴旋转")
    
    print(f"\n🔧 完整的刚体关节描述需要:")
    print("   1. 旋转轴方向: [X/Y/Z] 或任意轴 [ax, ay, az]")
    print("   2. 关节角度: θ (绕轴旋转)")
    print("   3. 连杆长度: a (segment长度)")
    print("   4. 偏移量: d (轴向偏移)")
    print("   5. 扭转角: α (连杆扭转)")
    
    # 典型6DOF机械臂的DH参数示例
    typical_6dof_dh = [
        {"joint": 1, "theta": "θ₁", "d": "d₁", "a": 0, "alpha": "π/2", "axis": "Z"},
        {"joint": 2, "theta": "θ₂", "d": 0, "a": "a₂", "alpha": 0, "axis": "Z"},
        {"joint": 3, "theta": "θ₃", "d": 0, "a": "a₃", "alpha": 0, "axis": "Z"}, 
        {"joint": 4, "theta": "θ₄", "d": "d₄", "a": 0, "alpha": "π/2", "axis": "Z"},
        {"joint": 5, "theta": "θ₅", "d": 0, "a": 0, "alpha": "-π/2", "axis": "Z"},
        {"joint": 6, "theta": "θ₆", "d": "d₆", "a": 0, "alpha": 0, "axis": "Z"}
    ]
    
    print(f"\n📊 典型6DOF机械臂DH表:")
    print("关节 | θ        | d    | a    | α      | 旋转轴")
    print("-" * 50)
    for joint in typical_6dof_dh:
        print(f"  {joint['joint']}  | {joint['theta']:8} | {joint['d']:4} | {joint['a']:4} | {joint['alpha']:6} | {joint['axis']:4}")
    
    return typical_6dof_dh

def analyze_soft_arm_equivalent():
    """分析软体臂的等价DH表示"""
    print(f"\n🌊 软体臂的DH等价表示")
    print("=" * 40)
    
    print("🤔 问题: 软体segment如何用DH参数描述？")
    print()
    
    print("💡 软体segment的'等价DH':")
    print("   θᵢ: α (弯曲角) - 等价于关节角")
    print("   dᵢ: 0 (无偏移) - 软体连续弯曲")
    print("   aᵢ: chord_length (弦长) - 等价于连杆长度")
    print("   αᵢ: β (方向角) - 等价于扭转角")
    print("   轴向: 总是相对于当前segment的局部轴")
    
    # 软体segment的DH等价计算
    def soft_segment_to_dh(alpha, beta, arc_length):
        """将软体segment转换为DH等价参数"""
        if alpha < 1e-6:  # 直线情况
            theta_eq = 0
            d_eq = 0
            a_eq = arc_length
            alpha_eq = beta
        else:  # 弯曲情况
            radius = arc_length / alpha
            chord_length = 2 * radius * np.sin(alpha/2)
            
            theta_eq = alpha  # 弯曲角等价为关节角
            d_eq = 0          # 无轴向偏移
            a_eq = chord_length  # 弦长等价为连杆长度
            alpha_eq = beta   # 方向角等价为扭转角
        
        return theta_eq, d_eq, a_eq, alpha_eq
    
    print(f"\n📊 软体segment DH转换示例:")
    print("软体参数     | 等价DH参数")
    print("-" * 40)
    
    soft_examples = [
        (0.5, 0.2, 0.21),   # 中等弯曲
        (1.0, -0.3, 0.21),  # 大弯曲
        (0.001, 0.1, 0.21)  # 接近直线
    ]
    
    for alpha, beta, arc_len in soft_examples:
        theta_eq, d_eq, a_eq, alpha_eq = soft_segment_to_dh(alpha, beta, arc_len)
        soft_str = f"[{alpha:.3f}, {beta:.1f}, {arc_len:.2f}]"
        dh_str = f"θ={theta_eq:.3f}, d={d_eq:.1f}, a={a_eq:.3f}, α={alpha_eq:.1f}"
        print(f"{soft_str:15} | {dh_str}")

def unified_dh_representation():
    """统一的DH参数表示"""
    print(f"\n🎯 统一DH参数方案")
    print("=" * 40)
    
    print("💡 核心思想: 扩展DH参数来统一描述")
    
    print(f"\n🔧 扩展DH参数: [θ, d, a, α, 轴类型]")
    print("   θ: 旋转角 (刚体关节角 or 软体弯曲角)")
    print("   d: 轴向偏移 (刚体偏移 or 软体=0)")
    print("   a: 连杆长度 (刚体直线长度 or 软体弦长)")
    print("   α: 扭转角 (刚体扭转 or 软体方向)")
    print("   轴类型: 旋转轴描述 (X/Y/Z or 局部)")
    
    print(f"\n🌟 统一映射规则:")
    
    print("🤖 刚体关节:")
    print("   输入: [joint_angle, offset, link_length, twist, axis_vector]")
    print("   DH: [θ=joint_angle, d=offset, a=link_length, α=twist, axis]")
    
    print(f"\n🌊 软体segment:")
    print("   输入: [bend_angle, 0, arc_length, direction, 'local']")
    print("   DH: [θ=bend_angle, d=0, a=chord_length, α=direction, 'local']")
    
    # 实现示例
    print(f"\n```python")
    print("def create_unified_dh_features(segments):")
    print("    nodes = []")
    print("    ")
    print("    for seg in segments:")
    print("        if seg.type == 'rigid':")
    print("            # 完整DH参数")
    print("            theta = seg.joint_angle")
    print("            d = seg.joint_offset") 
    print("            a = seg.link_length")
    print("            alpha = seg.twist_angle")
    print("            axis = seg.rotation_axis  # [1,0,0] or [0,1,0] or [0,0,1]")
    print("            ")
    print("        elif seg.type == 'soft':")
    print("            # 软体等价DH")
    print("            theta = seg.bend_angle  # α")
    print("            d = 0  # 无偏移")
    print("            r = seg.arc_length / max(seg.bend_angle, 1e-6)")
    print("            a = 2 * r * sin(seg.bend_angle/2)  # 弦长")
    print("            alpha = seg.direction_angle  # β")
    print("            axis = [0, 0, 1]  # 局部Z轴")
    print("        ")
    print("        # 统一节点特征: [θ, d, a, α, ax, ay, az]")
    print("        node = [theta, d, a, alpha, axis[0], axis[1], axis[2]]")
    print("        nodes.append(node)")
    print("    ")
    print("    return nodes")
    print("```")

def analyze_challenges_and_solutions():
    """分析挑战和解决方案"""
    print(f"\n⚠️ 统一DH方案的挑战")
    print("=" * 40)
    
    print("🔴 挑战1: 维度增加")
    print("   从3D [α,β,L] → 7D [θ,d,a,α,ax,ay,az]")
    print("   影响: 网络复杂度增加，训练难度上升")
    
    print(f"\n🔴 挑战2: 语义复杂性")
    print("   刚体θ: 真实关节角")
    print("   软体θ: 弯曲角，物理含义不同")
    print("   影响: 网络仍需学习区分")
    
    print(f"\n🔴 挑战3: 坐标系复杂性")
    print("   刚体: 世界坐标系的固定轴")
    print("   软体: 局部坐标系的相对轴")
    print("   影响: 轴向信息 [ax,ay,az] 参考系不同")
    
    print(f"\n💡 解决方案:")
    
    print("🎯 方案A: 分层特征")
    print("   核心特征: [θ, d, a, α] (4D)")
    print("   轴向特征: [ax, ay, az] (3D)")
    print("   类型特征: [is_soft] (1D)")
    print("   让网络分别处理几何和轴向信息")
    
    print(f"\n🎯 方案B: 局部坐标统一")
    print("   将所有轴向转换为'相对前一segment'")
    print("   刚体轴向: 相对于前一个连杆的局部轴")
    print("   软体轴向: 天然就是局部相对")
    print("   统一参考系，简化学习")

def final_recommendation():
    """最终推荐"""
    print(f"\n🏆 最终推荐")
    print("=" * 30)
    
    print("🎯 对于完整机械臂统一:")
    print("   **扩展DH + 局部坐标系**")
    print("   特征: [θ, d, a, α, local_axis_type]")
    print("   其中 local_axis_type ∈ {X_local, Y_local, Z_local}")
    
    print(f"\n🔧 实现策略:")
    print("   1. 预处理: 将所有轴向转换为局部坐标")
    print("   2. 统一DH: 刚体和软体都用相同DH格式")
    print("   3. 轴向编码: 简化的局部轴向类型")
    print("   4. 网络学习: 统一的DH几何关系")
    
    print(f"\n🎯 对于当前软体项目:")
    print("   **保持简化方案 [α, β, length]**")
    print("   理由: 纯软体系统无需复杂DH参数")
    print("   未来扩展时再考虑完整DH统一")
    
    print(f"\n🌟 核心洞察:")
    print("   你提到的问题确实是机械臂统一的关键难点!")
    print("   完整的刚体描述需要DH参数的所有4个维度")
    print("   真正的统一需要考虑偏移、扭转、任意轴向等")
    print("   这比简单的角度统一复杂得多!")

if __name__ == "__main__":
    analyze_dh_parameters()
    analyze_soft_arm_equivalent()
    unified_dh_representation()
    analyze_challenges_and_solutions()
    final_recommendation()
    
    print(f"\n🎉 总结:")
    print("你完全正确! 真正的机械臂统一需要考虑:")
    print("- 任意旋转轴 (X/Y/Z/任意方向)")
    print("- 关节偏移 (d参数)")
    print("- 连杆扭转 (α参数)")
    print("- 坐标系变换")
    print("这确实是个复杂的机器人学问题! 🤖")