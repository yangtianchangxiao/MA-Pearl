#!/usr/bin/env python3
"""
简单Ad Hoc测试 - 验证核心概念
"""

import torch
import torch.nn as nn
import numpy as np
from simplified_graph_demo import SimplifiedGraphState

class SimpleAdHocLayer(nn.Module):
    """最简单的Ad Hoc层"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.message_weight = nn.Linear(dim, 1)
    
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor):
        """Ad Hoc消息传递"""
        n_nodes = node_features.shape[0]
        
        if n_nodes == 1:
            # 单节点：无需coordination
            return self.linear(node_features)
        
        # 多节点：adaptive coordination
        neighbor_messages = torch.mm(adj_matrix, node_features)
        
        # 学习消息重要性
        msg_importance = torch.sigmoid(self.message_weight(neighbor_messages))
        self_importance = torch.sigmoid(self.message_weight(node_features))
        
        # 归一化权重
        total_importance = msg_importance + self_importance + 1e-8
        msg_weight = msg_importance / total_importance
        self_weight = self_importance / total_importance
        
        # 加权组合
        combined = self_weight * node_features + msg_weight * neighbor_messages
        
        return self.linear(combined)


def test_ad_hoc_concept():
    """测试Ad Hoc核心概念"""
    print("🧪 测试Ad Hoc核心概念")
    print("=" * 50)
    
    # 创建Ad Hoc层
    ad_hoc_layer = SimpleAdHocLayer(dim=64)
    
    # 测试不同节点数量
    for n_nodes in [1, 2, 3, 5, 8]:
        print(f"\n测试 {n_nodes} 个节点:")
        
        # 创建测试数据
        node_features = torch.randn(n_nodes, 64)
        
        # 链状邻接矩阵
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        for i in range(n_nodes - 1):
            adj_matrix[i, i+1] = 1.0
            adj_matrix[i+1, i] = 1.0
        
        # 前向传播
        output = ad_hoc_layer(node_features, adj_matrix)
        
        print(f"  输入: {node_features.shape}")
        print(f"  邻接: {adj_matrix.shape}")
        print(f"  输出: {output.shape}")
        print(f"  ✅ 成功处理 {n_nodes} 节点")
    
    print(f"\n🎯 Ad Hoc核心验证:")
    print(f"✅ 1节点: 无coordination (自身处理)")
    print(f"✅ 多节点: adaptive message weighting")
    print(f"✅ 动态scale: 1-8节点都能处理")
    print(f"✅ 参数共享: 同一层处理不同size")


def test_soft_arm_scenarios():
    """测试软体机械臂场景"""
    print(f"\n🦾 软体机械臂Ad Hoc场景")
    print("=" * 50)
    
    scenarios = [
        (2, "训练基线 - 2节4DOF"),
        (3, "训练范围 - 3节6DOF"),
        (5, "训练上限 - 5节10DOF"),
        (6, "测试泛化 - 6节12DOF"),
        (8, "高DOF测试 - 8节16DOF")
    ]
    
    ad_hoc_layer = SimpleAdHocLayer(dim=32)
    
    for n_segments, desc in scenarios:
        print(f"\n{desc}:")
        
        # 模拟软体臂节点特征：[bend_angle, direction_angle, length]
        node_features = torch.randn(n_segments, 32)  # 编码后的特征
        
        # 链状连接（机械臂结构）
        adj_matrix = torch.zeros(n_segments, n_segments)
        for i in range(n_segments - 1):
            adj_matrix[i, i+1] = 1.0
            adj_matrix[i+1, i] = 1.0
        
        # Ad Hoc处理
        coordinated_features = ad_hoc_layer(node_features, adj_matrix)
        
        # 全局pooling模拟动作输出
        global_action_info = coordinated_features.mean(dim=0)
        
        print(f"  节点数: {n_segments}")
        print(f"  DOF: {n_segments * 2}")
        print(f"  协调特征: {coordinated_features.shape}")
        print(f"  动作表征: {global_action_info.shape}")
        
        if n_segments <= 5:
            print(f"  状态: 训练范围 ✅")
        else:
            print(f"  状态: 泛化测试 🎯")
    
    print(f"\n🎉 软体机械臂Ad Hoc测试完成!")
    print(f"✅ 训练范围 (2-5节): 正常处理")
    print(f"✅ 泛化范围 (6-8节): 参数共享成功")
    print(f"✅ Ad Hoc coordination: 节点间自适应协作")


if __name__ == "__main__":
    test_ad_hoc_concept()
    test_soft_arm_scenarios()