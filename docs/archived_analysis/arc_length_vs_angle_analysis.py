#!/usr/bin/env python3
"""
弧长 vs 弧度分析 - 软体臂几何深度解析
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_arc_geometry():
    """分析弧长、弧度、半径的关系"""
    print("📐 弧长 vs 弧度关系分析")
    print("=" * 50)
    
    # 基本弧形几何公式
    print("🔬 弧形几何基础公式:")
    print("   弧长 s = r × θ  (r=半径, θ=弧度)")
    print("   弦长 c = 2r × sin(θ/2)")
    print("   弧高 h = r × (1 - cos(θ/2))")
    
    # 从软体臂运动学反推参数
    segment_length = 0.21  # 给定弧长
    
    print(f"\n🌊 软体臂参数分析:")
    print(f"   给定弧长: s = {segment_length}m")
    
    # 分析不同α角对应的几何参数
    alphas = [0.1, 0.5, 1.0, np.pi/2]  # 不同的α角
    
    print(f"\n📊 不同弯曲角α的几何分析:")
    print("α (rad)  |  α (deg)  |  半径r  |  弧度θ  |  弦长c  |  说明")
    print("-" * 70)
    
    for alpha in alphas:
        # 从软体臂运动学公式反推
        # 我们需要理解C++代码中alpha的物理含义
        
        # 方法1: 假设α就是弯曲的总角度 (弧度)
        if alpha > 0:
            radius_method1 = segment_length / alpha  # r = s/θ
            chord_method1 = 2 * radius_method1 * np.sin(alpha/2)
        else:
            radius_method1 = float('inf')
            chord_method1 = segment_length
        
        print(f"{alpha:6.3f}  |  {np.degrees(alpha):6.1f}°  |  {radius_method1:6.3f}  |  {alpha:6.3f}  |  {chord_method1:6.3f}  |  方法1")
    
    return segment_length

def analyze_cpp_kinematics():
    """分析C++运动学代码中的几何含义"""
    print(f"\n🔍 C++运动学代码分析:")
    print("从 config_to_translationMatrix 函数:")
    
    print("""
    x = arc_length/alpha * (1 - cos(alpha)) * sin(beta)
    y = arc_length/alpha * (1 - cos(alpha)) * cos(beta)  
    z = arc_length/alpha * sin(alpha)
    """)
    
    # 分析这个公式的几何含义
    segment_length = 0.21
    beta = 0  # 先忽略方向角
    
    print(f"📐 几何含义分析:")
    
    def cpp_translation(alpha, beta, arc_length):
        if alpha == 0:
            alpha = 0.000001
        x = arc_length/alpha * (1 - np.cos(alpha)) * np.sin(beta)
        y = arc_length/alpha * (1 - np.cos(alpha)) * np.cos(beta)
        z = arc_length/alpha * np.sin(alpha)
        return np.array([x, y, z])
    
    # 分析不同α值
    alphas = [0.001, 0.5, 1.0, np.pi/2, np.pi]
    
    print("\nα (rad)  |  α (deg)  |  公式项分析")
    print("-" * 50)
    
    for alpha in alphas:
        # 分析公式中的各项
        radius_term = segment_length / alpha  # arc_length/alpha
        height_factor = 1 - np.cos(alpha)     # (1 - cos(alpha))
        z_factor = np.sin(alpha)              # sin(alpha)
        
        pos = cpp_translation(alpha, 0, segment_length)
        
        print(f"{alpha:6.3f}  |  {np.degrees(alpha):6.1f}°  |  r={radius_term:.3f}, h_f={height_factor:.3f}, z={pos[2]:.3f}")
        
        # 对比标准弧形公式
        if alpha > 0:
            standard_radius = segment_length / alpha
            standard_chord = 2 * standard_radius * np.sin(alpha/2)
            actual_chord = np.linalg.norm(pos)
            
            print(f"          |          |  标准弦长={standard_chord:.3f}, 实际距离={actual_chord:.3f}")

def geometric_insight():
    """几何洞察"""
    print(f"\n🎯 关键洞察:")
    
    print("1. 弧长固定: s = 0.21m (segment_length)")
    print("2. α 是弯曲的总角度 (弧度)")
    print("3. 半径动态计算: r = s/α = 0.21/α") 
    print("4. 弧度就是α本身!")
    
    segment_length = 0.21
    
    print(f"\n📊 实例验证:")
    examples = [
        (0.001, "几乎直线"),
        (0.5, "轻微弯曲"), 
        (1.0, "中等弯曲"),
        (np.pi/2, "90度弯曲"),
        (np.pi, "半圆弯曲")
    ]
    
    for alpha, desc in examples:
        radius = segment_length / alpha
        arc_angle = alpha  # 弧度就是α
        chord = 2 * radius * np.sin(alpha/2)
        
        print(f"   {desc}:")
        print(f"      α = {alpha:.3f} rad = {np.degrees(alpha):.1f}°")
        print(f"      半径 r = {radius:.3f}m")  
        print(f"      弧度 θ = {arc_angle:.3f} rad (就是α!)")
        print(f"      弧长 s = r×θ = {radius*arc_angle:.3f}m ✓")
        print(f"      弦长 c = {chord:.3f}m")
        
        # 验证弧长公式
        calculated_arc = radius * arc_angle
        print(f"      验证: r×θ = {calculated_arc:.3f}m = {segment_length:.3f}m ✓")
        print()

def curvature_analysis():
    """曲率分析"""
    print(f"🌀 曲率分析:")
    print("曲率 κ = 1/r = α/s")
    
    segment_length = 0.21
    
    print(f"\nα (弧度)  |  半径r (m)  |  曲率κ (1/m)  |  含义")
    print("-" * 55)
    
    alphas = [0.1, 0.5, 1.0, 2.0, np.pi]
    
    for alpha in alphas:
        radius = segment_length / alpha
        curvature = 1 / radius  # κ = 1/r
        
        if curvature < 1:
            meaning = "低曲率 (接近直线)"
        elif curvature < 5:
            meaning = "中等曲率"
        else:
            meaning = "高曲率 (急转弯)"
            
        print(f"{alpha:8.3f}  |  {radius:9.3f}  |  {curvature:9.3f}  |  {meaning}")

if __name__ == "__main__":
    analyze_arc_geometry()
    analyze_cpp_kinematics()  
    geometric_insight()
    curvature_analysis()
    
    print(f"\n🎉 最终结论:")
    print("✅ 弧长 s = 0.21m (segment_length, 固定)")
    print("✅ 弧度 θ = α (控制参数，就是弯曲角度!)")  
    print("✅ 半径 r = s/θ = 0.21/α (动态计算)")
    print("🌟 α 既是控制参数，也是几何弧度!")
    print("💡 软体segment就是半径动态变化的圆弧!")