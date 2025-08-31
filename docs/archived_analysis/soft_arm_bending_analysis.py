#!/usr/bin/env python3
"""
软体机械臂弯曲机制深度分析
基于C++原始代码的运动学分析
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_soft_arm_kinematics():
    """分析软体臂的弯曲运动学"""
    print("🌊 软体机械臂弯曲机制分析")
    print("=" * 60)
    
    # 关键运动学函数 (从C++代码复制)
    def config_to_translation(alpha, beta, arc_length):
        """软体段的平移 - 关键是这里!"""
        if alpha == 0:
            alpha = 0.000001  # 避免除零
        
        # 🌟 核心公式: 弧形轨迹的数学表示
        x = arc_length/alpha * (1 - np.cos(alpha)) * np.sin(beta)
        y = arc_length/alpha * (1 - np.cos(alpha)) * np.cos(beta)  
        z = arc_length/alpha * np.sin(alpha)
        
        return np.array([x, y, z])
    
    def config_to_rotation(alpha, beta):
        """软体段的旋转矩阵"""
        if alpha == 0:
            alpha = 0.000001
        
        # Rodrigues旋转公式
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        cos_beta = np.cos(beta)
        sin_beta = np.sin(beta)
        
        R = np.array([
            [cos_beta*cos_beta*(1-cos_alpha) + cos_alpha, 
             -cos_beta*sin_beta*(1-cos_alpha), 
             sin_alpha*sin_beta],
            [-cos_beta*sin_beta*(1-cos_alpha), 
             sin_beta*sin_beta*(1-cos_alpha) + cos_alpha, 
             sin_alpha*cos_beta],
            [-sin_alpha*sin_beta, 
             -sin_alpha*cos_beta, 
             cos_alpha]
        ])
        return R
    
    print("🔬 弯曲机制核心理解:")
    print("1. α (alpha): 弯曲角度 - 控制segment的弯曲程度")
    print("2. β (beta): 方向角 - 控制弯曲的方向")
    print("3. arc_length: segment的弧长 - 固定不变!")
    
    # 🌟 关键洞察: 弯曲vs直线的区别
    segment_length = 0.21
    
    print(f"\n📐 直线 vs 弯曲的距离分析:")
    
    # 情况1: 直线 (α=0)
    alpha_straight = 0.001  # 几乎直线
    beta = 0
    pos_straight = config_to_translation(alpha_straight, beta, segment_length)
    distance_straight = np.linalg.norm(pos_straight)
    
    print(f"   直线状态 (α≈0°):")
    print(f"      位置: ({pos_straight[0]:.4f}, {pos_straight[1]:.4f}, {pos_straight[2]:.4f})")
    print(f"      端点距离: {distance_straight:.4f}m ≈ {segment_length:.2f}m (弧长)")
    
    # 情况2: 弯曲 (α=45°)
    alpha_bend = np.pi/4  # 45度弯曲
    pos_bend = config_to_translation(alpha_bend, beta, segment_length)
    distance_bend = np.linalg.norm(pos_bend)
    
    print(f"   弯曲状态 (α=45°):")
    print(f"      位置: ({pos_bend[0]:.4f}, {pos_bend[1]:.4f}, {pos_bend[2]:.4f})")
    print(f"      端点距离: {distance_bend:.4f}m < {segment_length:.2f}m (弧长)")
    
    print(f"\n💡 关键发现:")
    print(f"   🌟 弧长固定 = {segment_length}m (segment_length)")
    print(f"   📏 端点距离变化: {distance_straight:.3f}m → {distance_bend:.3f}m")
    print(f"   📊 收缩比例: {distance_bend/distance_straight:.1%}")
    
    # 多段组合分析
    print(f"\n🔗 多段软体臂的复合效果:")
    
    def forward_kinematics_3_segments(config, segment_length=0.21):
        """3段软体臂正向运动学"""
        positions = [np.array([0, 0, 0])]  # 基座
        
        # Segment 1
        alpha1, beta1 = config[0], config[1]
        pos1 = config_to_translation(alpha1, beta1, segment_length)
        rot1 = config_to_rotation(alpha1, beta1)
        positions.append(pos1)
        
        # Segment 2
        alpha2, beta2 = config[2], config[3] 
        local_pos2 = config_to_translation(alpha2, beta2, segment_length)
        pos2 = pos1 + rot1 @ local_pos2
        rot2 = rot1 @ config_to_rotation(alpha2, beta2)
        positions.append(pos2)
        
        # Segment 3
        alpha3, beta3 = config[4], config[5]
        local_pos3 = config_to_translation(alpha3, beta3, segment_length)
        pos3 = pos2 + rot2 @ local_pos3
        positions.append(pos3)
        
        return positions
    
    # 测试不同配置
    configs = {
        "全直线": [0.001, 0, 0.001, 0, 0.001, 0],
        "轻微弯曲": [0.2, 0, 0.2, 0, 0.2, 0],
        "大幅弯曲": [0.8, 0, 0.8, 0, 0.8, 0],
        "S形弯曲": [0.5, 0, -0.5, 0, 0.5, 0]
    }
    
    for name, config in configs.items():
        positions = forward_kinematics_3_segments(config)
        end_effector = positions[-1]
        total_distance = np.linalg.norm(end_effector)
        max_reach = 3 * segment_length  # 弧长总和
        
        print(f"   {name}:")
        print(f"      末端位置: ({end_effector[0]:.3f}, {end_effector[1]:.3f}, {end_effector[2]:.3f})")
        print(f"      到达距离: {total_distance:.3f}m (vs 最大{max_reach:.2f}m)")
        print(f"      利用率: {total_distance/max_reach:.1%}")
    
    return configs, forward_kinematics_3_segments

def compare_with_rigid_arm():
    """与刚体臂对比"""
    print(f"\n🤖 刚体 vs 🌊 软体机械臂对比:")
    print("=" * 40)
    
    # 刚体臂 (每个关节1DOF)
    def rigid_arm_fk(angles, lengths):
        """刚体臂正向运动学"""
        x, y, z = 0, 0, 0
        cumulative_angle = 0
        
        for i, (angle, length) in enumerate(zip(angles, lengths)):
            cumulative_angle += angle
            x += length * np.cos(cumulative_angle)
            z += length * np.sin(cumulative_angle)  # 简化为2D
        
        return np.array([x, y, z])
    
    # 软体臂参数
    soft_lengths = [0.21, 0.21, 0.21]
    soft_config = [0.5, 0, 0.3, 0, 0.2, 0]  # 示例配置
    
    # 刚体臂参数 (相同总长度)
    rigid_lengths = [0.21, 0.21, 0.21] 
    rigid_angles = [0.5, 0.3, 0.2]  # 示例角度
    
    # 计算末端位置
    _, soft_fk = analyze_soft_arm_kinematics()
    soft_positions = soft_fk(soft_config)
    soft_end = soft_positions[-1]
    
    rigid_end = rigid_arm_fk(rigid_angles, rigid_lengths)
    
    print(f"📏 几何特性对比:")
    print(f"   软体末端: ({soft_end[0]:.3f}, {soft_end[1]:.3f}, {soft_end[2]:.3f})")
    print(f"   刚体末端: ({rigid_end[0]:.3f}, {rigid_end[1]:.3f}, {rigid_end[2]:.3f})")
    
    print(f"\n🎯 核心区别:")
    print(f"   📐 刚体: segment间直线连接，长度=直线距离")
    print(f"   🌊 软体: segment内弧形弯曲，长度=弧长≠直线距离")
    print(f"   🔧 DOF: 刚体每segment 1DOF, 软体每segment 2DOF")
    print(f"   🎪 灵活性: 软体具有连续弯曲能力")

if __name__ == "__main__":
    configs, fk_func = analyze_soft_arm_kinematics()
    compare_with_rigid_arm()
    
    print(f"\n🎉 结论:")
    print(f"✅ 软体臂的'弯曲'是真实的弧形轨迹!")
    print(f"✅ segment_length是弧长，不是直线距离!")
    print(f"✅ 弯曲时端点距离 < 弧长，造成'收缩'效果")
    print(f"✅ α角控制弯曲程度，β角控制弯曲方向")
    print(f"🌟 这确实是软体和刚体的本质区别!")