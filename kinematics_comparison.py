#!/usr/bin/env python3
"""
运动学对比分析：简化 vs 复杂
分析两种运动学的控制能力和工作空间差异
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleKinematics:
    """简化运动学 (你目前训练环境用的)"""
    
    def __init__(self, segment_lengths):
        self.segment_lengths = segment_lengths
        self.n_segments = len(segment_lengths)
    
    def forward(self, joint_angles):
        """简化前向运动学"""
        position = np.array([0.0, 0.0, 0.0])
        cumulative_angle_xy = 0.0
        
        positions = [position.copy()]  # 记录每个segment的末端
        
        for i in range(self.n_segments):
            joint1 = joint_angles[i * 2]      # 弯曲角（Z方向）
            joint2 = joint_angles[i * 2 + 1]  # 方向角（XY平面）
            length = self.segment_lengths[i]
            
            cumulative_angle_xy += joint2
            
            # 简化：直线段近似
            segment_end = length * np.array([
                np.cos(cumulative_angle_xy) * np.cos(joint1),
                np.sin(cumulative_angle_xy) * np.cos(joint1),
                np.sin(joint1)
            ])
            
            position += segment_end
            positions.append(position.copy())
        
        return position, positions


class ComplexKinematics:
    """复杂运动学 (C++实际硬件用的)"""
    
    def __init__(self, segment_lengths):
        self.segment_lengths = segment_lengths
        self.n_segments = len(segment_lengths)
    
    def config_to_translation(self, alpha, beta, arc_length):
        """C++的复杂运动学公式"""
        if abs(alpha) < 1e-6:  # alpha不能为0
            alpha = 1e-6
        
        x = arc_length/alpha * (1 - np.cos(alpha)) * np.sin(beta)
        y = arc_length/alpha * (1 - np.cos(alpha)) * np.cos(beta)
        z = arc_length/alpha * np.sin(alpha)
        
        return np.array([x, y, z])
    
    def config_to_rotation(self, alpha, beta):
        """C++的复杂旋转矩阵"""
        if abs(alpha) < 1e-6:
            alpha = 1e-6
        
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        
        rotation = np.array([
            [cos_b*cos_b*(1-cos_a) + cos_a, -cos_b*sin_b*(1-cos_a), sin_a*sin_b],
            [-cos_b*sin_b*(1-cos_a), sin_b*sin_b*(1-cos_a) + cos_a, sin_a*cos_b],
            [-sin_a*sin_b, -sin_a*cos_b, cos_a]
        ])
        
        return rotation
    
    def forward(self, joint_angles):
        """复杂前向运动学 - 链式旋转变换"""
        position = np.array([0.0, 0.0, 0.0])
        rotation = np.eye(3)  # 累积旋转矩阵
        
        positions = [position.copy()]
        
        for i in range(self.n_segments):
            alpha = joint_angles[i * 2]      # 弯曲角 (对应C++的alpha)
            beta = joint_angles[i * 2 + 1]   # 方向角 (对应C++的beta)
            arc_length = self.segment_lengths[i]
            
            # 复杂：连续弯曲弧
            local_translation = self.config_to_translation(alpha, beta, arc_length)
            local_rotation = self.config_to_rotation(alpha, beta)
            
            # 链式变换
            position = position + rotation @ local_translation
            rotation = rotation @ local_rotation
            
            positions.append(position.copy())
        
        return position, positions


def compare_workspaces():
    """对比两种运动学的工作空间"""
    print("🔍 运动学工作空间对比分析")
    print("=" * 60)
    
    # 3节机械臂
    segment_lengths = [0.2, 0.2, 0.2]
    simple_kin = SimpleKinematics(segment_lengths)
    complex_kin = ComplexKinematics(segment_lengths)
    
    # 生成测试关节角度
    n_samples = 1000
    joint_ranges = [(-0.8, 0.8), (-np.pi, np.pi)]  # [弯曲角, 方向角]
    
    simple_endpoints = []
    complex_endpoints = []
    
    print(f"生成{n_samples}个随机配置...")
    
    for _ in range(n_samples):
        joint_angles = []
        for i in range(3):  # 3节
            joint1 = np.random.uniform(*joint_ranges[0])  # 弯曲角
            joint2 = np.random.uniform(*joint_ranges[1])  # 方向角
            joint_angles.extend([joint1, joint2])
        
        joint_angles = np.array(joint_angles)
        
        # 计算末端位置
        simple_end, _ = simple_kin.forward(joint_angles)
        complex_end, _ = complex_kin.forward(joint_angles)
        
        simple_endpoints.append(simple_end)
        complex_endpoints.append(complex_end)
    
    simple_endpoints = np.array(simple_endpoints)
    complex_endpoints = np.array(complex_endpoints)
    
    # 工作空间分析
    print(f"\n📊 工作空间统计:")
    
    simple_reach = np.linalg.norm(simple_endpoints, axis=1)
    complex_reach = np.linalg.norm(complex_endpoints, axis=1)
    
    print(f"简化运动学:")
    print(f"  最大到达距离: {simple_reach.max():.3f}m")
    print(f"  平均到达距离: {simple_reach.mean():.3f}m")
    print(f"  工作空间体积估计: {estimate_volume(simple_endpoints):.3f}m³")
    
    print(f"复杂运动学:")
    print(f"  最大到达距离: {complex_reach.max():.3f}m")
    print(f"  平均到达距离: {complex_reach.mean():.3f}m")
    print(f"  工作空间体积估计: {estimate_volume(complex_endpoints):.3f}m³")
    
    # 位置差异分析
    position_errors = np.linalg.norm(simple_endpoints - complex_endpoints, axis=1)
    print(f"\n🎯 控制精度差异:")
    print(f"  平均位置误差: {position_errors.mean():.3f}m")
    print(f"  最大位置误差: {position_errors.max():.3f}m")
    print(f"  误差标准差: {position_errors.std():.3f}m")
    
    # 可视化
    visualize_workspaces(simple_endpoints, complex_endpoints)
    
    return simple_endpoints, complex_endpoints, position_errors


def estimate_volume(points):
    """估计工作空间体积"""
    # 用边界盒体积估计
    ranges = points.max(axis=0) - points.min(axis=0)
    return np.prod(ranges)


def visualize_workspaces(simple_points, complex_points):
    """可视化工作空间对比"""
    fig = plt.figure(figsize=(15, 5))
    
    # 简化运动学工作空间
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(simple_points[:, 0], simple_points[:, 1], simple_points[:, 2], 
               c='blue', alpha=0.6, s=1)
    ax1.set_title('简化运动学工作空间')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    # 复杂运动学工作空间
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(complex_points[:, 0], complex_points[:, 1], complex_points[:, 2], 
               c='red', alpha=0.6, s=1)
    ax2.set_title('复杂运动学工作空间')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    
    # 对比图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(simple_points[:, 0], simple_points[:, 1], simple_points[:, 2], 
               c='blue', alpha=0.3, s=1, label='简化')
    ax3.scatter(complex_points[:, 0], complex_points[:, 1], complex_points[:, 2], 
               c='red', alpha=0.3, s=1, label='复杂')
    ax3.set_title('工作空间对比')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('kinematics_workspace_comparison.png', dpi=150, bbox_inches='tight')
    print(f"📊 工作空间对比图已保存: kinematics_workspace_comparison.png")


def test_specific_configurations():
    """测试特定配置的差异"""
    print(f"\n🧪 特定配置测试:")
    print("=" * 40)
    
    segment_lengths = [0.2, 0.2, 0.2]
    simple_kin = SimpleKinematics(segment_lengths)
    complex_kin = ComplexKinematics(segment_lengths)
    
    test_configs = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 直线
        [0.5, 0.0, 0.5, 0.0, 0.5, 0.0],      # Z方向弯曲
        [0.0, np.pi/2, 0.0, np.pi/2, 0.0, np.pi/2],  # Y方向
        [0.3, np.pi/4, -0.3, -np.pi/4, 0.3, np.pi/4], # S形
    ]
    
    config_names = ["直线配置", "Z向弯曲", "Y向转弯", "S形弯曲"]
    
    for config, name in zip(test_configs, config_names):
        joint_angles = np.array(config)
        
        simple_end, simple_path = simple_kin.forward(joint_angles)
        complex_end, complex_path = complex_kin.forward(joint_angles)
        
        error = np.linalg.norm(simple_end - complex_end)
        
        print(f"{name}:")
        print(f"  简化末端: [{simple_end[0]:.3f}, {simple_end[1]:.3f}, {simple_end[2]:.3f}]")
        print(f"  复杂末端: [{complex_end[0]:.3f}, {complex_end[1]:.3f}, {complex_end[2]:.3f}]")
        print(f"  位置误差: {error:.3f}m")
        print()


def main():
    """主函数"""
    print("🚀 运动学对比分析启动")
    print("分析简化运动学 vs 复杂运动学的差异")
    print()
    
    # 工作空间对比
    simple_points, complex_points, errors = compare_workspaces()
    
    # 特定配置测试
    test_specific_configurations()
    
    # 结论
    print("🎯 关键结论:")
    print("=" * 40)
    
    avg_error = errors.mean()
    max_error = errors.max()
    
    if avg_error > 0.05:  # 5cm
        print("❌ 两种运动学差异显著!")
        print(f"   平均误差: {avg_error:.3f}m > 5cm")
        print("   必须重新训练使用复杂运动学")
    elif avg_error > 0.02:  # 2cm
        print("⚠️ 两种运动学有一定差异")
        print(f"   平均误差: {avg_error:.3f}m (2-5cm)")
        print("   建议重新训练以提高精度")
    else:
        print("✅ 两种运动学相对接近")
        print(f"   平均误差: {avg_error:.3f}m < 2cm")
        print("   可能不需要重新训练")
    
    print(f"\n🔍 详细分析:")
    print(f"   最大误差: {max_error:.3f}m")
    print(f"   误差标准差: {errors.std():.3f}m")
    print(f"   >5cm误差的配置: {(errors > 0.05).sum()}/{len(errors)} ({(errors > 0.05).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()