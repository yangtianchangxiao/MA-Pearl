#!/usr/bin/env python3
"""
深入分析：了解弧度几何后，刚体+软体统一Graph方案的可行性
"""

import numpy as np

def analyze_unified_representation_viability():
    """分析统一表示的可行性"""
    print("🤔 统一Graph表示可行性分析")
    print("=" * 60)
    
    print("📊 原方案回顾:")
    print("   软体节点: [α, β, arc_length]")  
    print("   刚体节点: [θ, 0, link_length]")
    print("   ✅ 维度统一: 都是3维")
    
    print(f"\n🔬 深入几何分析后的新认知:")
    
    # 软体臂几何
    print("🌊 软体segment几何:")
    print("   - α: 弧度角 (0 to π)")
    print("   - β: 方向角 (-π to π)")  
    print("   - arc_length: 固定弧长")
    print("   - 半径: r = arc_length/α (动态)")
    print("   - 端点距离: 弦长 = 2r×sin(α/2)")
    
    # 刚体臂几何  
    print("\n🤖 刚体link几何:")
    print("   - θ: 关节角 (-π to π)")
    print("   - 0: 无第二DOF (padding)")
    print("   - link_length: 固定直线长度")
    print("   - 半径: 不适用 (直线)")
    print("   - 端点距离: = link_length (直线距离)")
    
    return True

def test_geometric_compatibility():
    """测试几何兼容性"""
    print(f"\n🧪 几何兼容性测试:")
    
    # 测试场景：相同总长度的臂
    total_length = 0.63  # 3段 × 0.21m
    
    # 软体臂配置
    soft_segments = [
        {"alpha": 0.5, "beta": 0.2, "arc_length": 0.21},
        {"alpha": 0.3, "beta": -0.1, "arc_length": 0.21}, 
        {"alpha": 0.7, "beta": 0.3, "arc_length": 0.21}
    ]
    
    # 等价刚体臂配置 (相同总长度)
    rigid_segments = [
        {"theta": 0.3, "padding": 0, "link_length": 0.21},
        {"theta": -0.2, "padding": 0, "link_length": 0.21},
        {"theta": 0.4, "padding": 0, "link_length": 0.21}
    ]
    
    print("📐 配置对比:")
    print("软体 | 刚体")
    print("-" * 30)
    for i, (soft, rigid) in enumerate(zip(soft_segments, rigid_segments)):
        print(f"段{i+1}: [{soft['alpha']:.1f}, {soft['beta']:.1f}, {soft['arc_length']:.2f}] | [{rigid['theta']:.1f}, {rigid['padding']}, {rigid['link_length']:.2f}]")
    
    # 计算工作空间
    def soft_workspace_estimate(segments):
        """软体臂工作空间估算"""
        total_arc = sum(s['arc_length'] for s in segments)
        # 最坏情况: 全部弯曲为半圆 (α=π)
        min_reach = sum(2*s['arc_length']/np.pi for s in segments)
        # 最好情况: 完全伸直 (α→0)
        max_reach = total_arc
        return min_reach, max_reach, total_arc
    
    def rigid_workspace(segments):
        """刚体臂工作空间"""
        total_length = sum(s['link_length'] for s in segments)
        return total_length  # 固定工作空间半径
    
    soft_min, soft_max, soft_total = soft_workspace_estimate(soft_segments)
    rigid_reach = rigid_workspace(rigid_segments)
    
    print(f"\n🌐 工作空间对比:")
    print(f"   软体臂: {soft_min:.3f}m ~ {soft_max:.3f}m (动态)")
    print(f"   刚体臂: {rigid_reach:.3f}m (固定)")
    print(f"   弧长总计: {soft_total:.3f}m")

def analyze_gnn_processing_differences():
    """分析GNN处理差异"""
    print(f"\n🧠 GNN处理差异分析:")
    
    print("🌊 软体节点的GNN理解:")
    print("   - [α, β, arc_length] → GNN学习弧形几何关系")
    print("   - α影响曲率 → 影响空间可达性")
    print("   - β控制方向 → 3D空间定向")
    print("   - arc_length约束 → 材料物理限制")
    
    print("\n🤖 刚体节点的GNN理解:")
    print("   - [θ, 0, link_length] → GNN学习线性几何关系") 
    print("   - θ只影响方向 → 不影响距离")
    print("   - 0 padding → 可能被GNN忽略")
    print("   - link_length固定 → 直线距离约束")
    
    print(f"\n⚠️  潜在问题:")
    print("1. 几何语义差异:")
    print("   - 软体α: 弧度角，影响曲率和距离")
    print("   - 刚体θ: 旋转角，只影响方向") 
    print("2. 空间效应差异:")
    print("   - 软体: 弯曲→收缩工作空间")
    print("   - 刚体: 旋转→保持工作空间")
    print("3. DOF利用差异:")
    print("   - 软体: 2DOF都有物理意义")
    print("   - 刚体: 1DOF + 1 padding")

def propose_solutions():
    """提出解决方案"""
    print(f"\n💡 统一方案选择:")
    
    print("🎯 方案A: 纯几何统一 (原方案)")
    print("   优点: ✅ 代码简单，维度统一")
    print("   缺点: ❌ 语义混淆，GNN可能学错")
    print("   适用: 🤷 小规模实验，概念验证")
    
    print("\n🎯 方案B: 类型标识增强")
    print("   节点表示: [param1, param2, length, type_flag]")
    print("   软体: [α, β, arc_length, 1.0]")
    print("   刚体: [θ, 0, link_length, 0.0]")
    print("   优点: ✅ 保持语义清晰，GNN可区分处理")
    print("   缺点: ❌ 增加维度，稍微复杂")
    print("   适用: 🎯 生产系统，混合机械臂")
    
    print("\n🎯 方案C: 统一物理表示")
    print("   核心思想: 将刚体也用弧度表示")
    print("   刚体→软体映射: θ → α=θ, β=0, r→∞")
    print("   节点表示: [curvature, direction, arc_length]")
    print("   软体: [α/arc_length, β, arc_length]  # 曲率表示")
    print("   刚体: [0, θ, link_length]           # 零曲率")
    print("   优点: ✅ 物理统一，数学优雅")
    print("   缺点: ❌ 概念复杂，需要验证")
    print("   适用: 🚀 研究项目，创新探索")
    
    print(f"\n🏆 推荐方案:")
    print("对于当前项目: **方案B (类型标识)**")
    print("理由:")
    print("✅ 保持现有代码兼容性")
    print("✅ GNN能明确区分不同类型")
    print("✅ 语义清晰，便于调试")
    print("✅ 扩展性好，支持混合系统")

def implementation_example():
    """实现示例"""
    print(f"\n🛠️  方案B实现示例:")
    
    print("```python")
    print("# 统一节点特征函数")
    print("def create_unified_node_features(arm_type, segments):")
    print("    nodes = []")
    print("    for seg in segments:")
    print("        if arm_type == 'soft':")
    print("            node = [seg.alpha, seg.beta, seg.arc_length, 1.0]")
    print("        elif arm_type == 'rigid':")
    print("            node = [seg.theta, 0.0, seg.link_length, 0.0]")
    print("        nodes.append(node)")
    print("    return torch.tensor(nodes)")
    print("")
    print("# GNN可以学习:")
    print("# if node[3] > 0.5:  # 软体")
    print("#     process_curved_geometry(node[:3])")
    print("# else:  # 刚体") 
    print("#     process_linear_geometry(node[:3])")
    print("```")

if __name__ == "__main__":
    analyze_unified_representation_viability()
    test_geometric_compatibility()
    analyze_gnn_processing_differences()
    propose_solutions()
    implementation_example()
    
    print(f"\n🎉 结论:")
    print("原统一方案在几何上可行，但语义上有风险")
    print("推荐增加类型标识，保证GNN正确理解不同几何类型")
    print("这样既保持了统一性，又避免了语义混淆！")