#!/usr/bin/env python3
"""
优化的Graph网络实现
节点特征简化至3维：[joint1, joint2, length]
完全去除冗余的位置信息，迫使网络学习运动学推理
"""

import torch
import torch.nn as nn
from typing import Tuple, List
from simplified_graph_demo import SimplifiedGraphState


class OptimizedGraphActorNetwork(nn.Module):
    """
    优化的Graph Actor网络
    
    核心改进：
    - 节点特征：3维 [joint1, joint2, length] - 纯结构信息
    - Goal信息：6维 [achieved_goal + desired_goal] - 纯空间信息  
    - 强制网络从结构推理空间行为
    """
    
    def __init__(
        self, 
        action_dim: int,
        hidden_dim: int = 128,
        num_graph_layers: int = 3,
        num_attention_heads: int = 4
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        print(f"🧠 优化Graph Actor网络:")
        print(f"   节点特征: 3维 [joint1, joint2, length]")
        print(f"   Goal信息: 6维 [achieved + desired]") 
        print(f"   隐藏层: {hidden_dim}维")
        print(f"   Graph层数: {num_graph_layers}")
        
        # 结构编码器：处理3维节点特征
        self.structure_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 关键：输入3维！
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph注意力层：理解节点间关系
        self.graph_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                batch_first=True
            ) for _ in range(num_graph_layers)
        ])
        
        # 空间编码器：处理Goal信息
        self.spatial_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # achieved(3) + desired(3)
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 结构-空间融合：关键决策层
        self.structure_spatial_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动作输出头
        self.action_mean_head = nn.Linear(hidden_dim, action_dim)
        self.action_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        输入: state_tensor [batch_size, tensor_dim] 或 [tensor_dim]
        """
        # 确保tensor是float32类型
        state_tensor = state_tensor.float()
        
        # 处理batch维度
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
            
        batch_size = state_tensor.shape[0]
        
        # 分别处理每个样本
        structure_features = []
        spatial_features = []
        
        for i in range(batch_size):
            tensor = state_tensor[i]
            
            # 解构Graph状态
            graph_state = SimplifiedGraphState.from_tensor(tensor)
            
            # 1. 结构特征处理
            node_features = graph_state.node_features  # [n_nodes, 3]
            structure_feat = self._process_structure(node_features)  # [hidden_dim]
            structure_features.append(structure_feat)
            
            # 2. 空间特征处理  
            spatial_info = torch.cat([
                graph_state.achieved_goal,  # [3]
                graph_state.desired_goal    # [3]
            ])  # [6]
            spatial_feat = self.spatial_encoder(spatial_info)  # [hidden_dim]
            spatial_features.append(spatial_feat)
        
        # Batch化
        structure_features = torch.stack(structure_features)  # [batch_size, hidden_dim]
        spatial_features = torch.stack(spatial_features)      # [batch_size, hidden_dim]
        
        # 结构-空间融合
        combined = torch.cat([structure_features, spatial_features], dim=1)  # [batch_size, 2*hidden_dim]
        fused_features = self.structure_spatial_fusion(combined)  # [batch_size, hidden_dim]
        
        # 动作输出 - 添加数值稳定性保护
        action_mean = self.action_mean_head(fused_features)
        action_mean = torch.clamp(action_mean, -10, 10)  # 防止mean过大
        
        action_log_std = self.action_std_head(fused_features) 
        action_log_std = torch.clamp(action_log_std, -10, 1)  # 更严格的std限制
        
        # 处理单样本情况
        if single_sample:
            action_mean = action_mean.squeeze(0)
            action_log_std = action_log_std.squeeze(0)
        
        return action_mean, action_log_std
    
    def _process_structure(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        处理机械臂结构信息
        输入: node_features [n_nodes, 3] - [joint1, joint2, length]
        输出: structure_feature [hidden_dim]
        """
        n_nodes = node_features.shape[0]
        
        # 编码每个节点
        encoded_nodes = self.structure_encoder(node_features)  # [n_nodes, hidden_dim]
        
        # Graph注意力传播：学习节点间依赖关系
        current_features = encoded_nodes
        layer_norm = torch.nn.LayerNorm(self.hidden_dim).to(current_features.device)
        
        for graph_layer in self.graph_layers:
            # 自注意力：每个节点关注其他节点
            attended, _ = graph_layer(
                current_features.unsqueeze(0),  # query
                current_features.unsqueeze(0),  # key  
                current_features.unsqueeze(0)   # value
            )
            attended = attended.squeeze(0)
            
            # 梯度裁剪防止爆炸
            attended = torch.clamp(attended, -5, 5)
            
            # 残差连接 + Layer Norm
            current_features = layer_norm(current_features + attended * 0.1)  # 缩放残差连接
        
        # 全局结构特征：所有节点信息的融合
        structure_feature = torch.mean(current_features, dim=0)  # [hidden_dim]
        
        # 数值稳定性检查
        if torch.isnan(structure_feature).any() or torch.isinf(structure_feature).any():
            print(f"⚠️ 检测到数值异常在structure processing")
            structure_feature = torch.zeros_like(structure_feature)
        
        return structure_feature
    
    def sample_action(self, state_tensor: torch.Tensor, get_log_prob: bool = False):
        """采样动作 - Pearl兼容接口"""
        action_mean, action_log_std = self.forward(state_tensor)
        action_std = action_log_std.exp()
        
        # 重参数化采样
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()
        
        if get_log_prob:
            log_prob = dist.log_prob(action)
            if action.dim() > 1:  # batch case
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:  # single sample case
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob
        else:
            return action


class OptimizedGraphCriticNetwork(nn.Module):
    """优化的Graph Critic网络"""
    
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 128,
        num_graph_layers: int = 2  # Critic可以简单一些
    ):
        super().__init__()
        
        # 复用Actor的结构处理部分
        self.structure_processor = OptimizedGraphActorNetwork(
            action_dim=1,  # dummy
            hidden_dim=hidden_dim,
            num_graph_layers=num_graph_layers
        )
        
        # Action编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Q值输出
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # structure+spatial + action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_tensor: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """计算Q值"""
        # 获取结构-空间特征
        with torch.no_grad():
            # 使用Actor的fusion层输出
            structure_spatial_feat = self._get_structure_spatial_features(state_tensor)
        
        # Action特征
        action_feat = self.action_encoder(action)
        
        # 组合
        combined = torch.cat([structure_spatial_feat, action_feat], dim=-1)
        q_value = self.q_head(combined).squeeze(-1)
        
        return q_value
    
    def _get_structure_spatial_features(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """获取结构-空间融合特征"""
        # 简化实现：直接使用Actor的前向传播到融合层
        # 在实际实现中应该共享encoder部分
        action_mean, _ = self.structure_processor.forward(state_tensor)
        return action_mean  # 作为特征使用


def test_optimized_graph_network():
    """测试优化的Graph网络"""
    print("🧪 测试优化Graph网络")
    print("💡 3维节点特征 + 6维Goal信息")
    print("🎯 强制网络学习运动学推理")
    print("=" * 50)
    
    # 测试不同DOF配置
    for n_segments in [2, 3, 4]:
        print(f"\\n测试 {n_segments}节机械臂 ({n_segments*2}DOF):")
        
        # 创建简化GraphState
        node_features = torch.randn(n_segments, 3)  # [joint1, joint2, length]
        adjacency_matrix = torch.eye(n_segments) + torch.roll(torch.eye(n_segments), 1, dims=1)
        achieved_goal = torch.randn(3) 
        desired_goal = torch.randn(3)
        
        graph_state = SimplifiedGraphState(
            node_features=node_features,
            adjacency_matrix=adjacency_matrix, 
            achieved_goal=achieved_goal,
            desired_goal=desired_goal
        )
        
        # 序列化为tensor
        state_tensor = graph_state.to_tensor()
        print(f"   状态tensor长度: {state_tensor.shape}")
        
        # 创建网络
        actor = OptimizedGraphActorNetwork(action_dim=n_segments*2)
        
        # 测试前向传播
        action_mean, action_log_std = actor.forward(state_tensor)
        print(f"   动作输出: mean{action_mean.shape}, log_std{action_log_std.shape}")
        
        # 测试批处理
        batch_tensor = state_tensor.unsqueeze(0).repeat(3, 1)  # batch_size=3
        batch_mean, batch_log_std = actor.forward(batch_tensor)
        print(f"   批处理输出: mean{batch_mean.shape}, log_std{batch_log_std.shape}")
        
        # 测试采样
        action = actor.sample_action(state_tensor)
        action_with_prob = actor.sample_action(state_tensor, get_log_prob=True)
        print(f"   采样动作: {action.shape}")
        print(f"   带概率采样: action{action_with_prob[0].shape}, log_prob{action_with_prob[1].shape}")
    
    print(f"\\n🎉 优化Graph网络测试完成！")
    print(f"✅ 节点特征：3维 [joint, joint, length]")
    print(f"✅ 空间信息：6维 [achieved + desired]")
    print(f"✅ 结构-空间分离处理")
    print(f"✅ 支持单样本和批处理")
    print(f"✅ Pearl SAC兼容接口")


if __name__ == "__main__":
    test_optimized_graph_network()