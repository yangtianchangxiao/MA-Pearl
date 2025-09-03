#!/usr/bin/env python3
"""
Graph曲率增量GNN训练脚本

核心创新:
- Graph-to-Graph架构: 输入Graph状态，输出Graph动作
- 曲率增量控制: 解决α≈0时β无效问题
- 真正任意DOF: 训练2-5节，可泛化到6-8节

作者: Claude Code
日期: 2025-09-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time
import json
from collections import deque

# 简化的GNN实现（不依赖PyG）
class SimpleGraphConv(nn.Module):
    """简单的图卷积层"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)  # [自己特征, 邻居特征]
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_nodes, in_dim]
        adj_matrix: [batch, n_nodes, n_nodes]
        """
        batch_size, n_nodes, in_dim = x.shape
        
        # 邻居聚合
        neighbor_sum = torch.bmm(adj_matrix, x)  # [batch, n_nodes, in_dim]
        
        # 拼接自己和邻居特征
        combined = torch.cat([x, neighbor_sum], dim=-1)  # [batch, n_nodes, in_dim*2]
        
        # 线性变换
        out = self.linear(combined)  # [batch, n_nodes, out_dim]
        
        return F.relu(out)


class GraphCurvatureGNNActor(nn.Module):
    """Graph曲率增量GNN Actor（无PyG依赖版本）"""
    
    def __init__(self, 
                 node_feature_dim: int = 13,
                 hidden_dim: int = 256,
                 num_gnn_layers: int = 3):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(SimpleGraphConv(hidden_dim, hidden_dim))
        
        # 输出头：每节点2维曲率增量
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
        print("🎯 GraphCurvatureGNNActor初始化", flush=True)
        print(f"   节点特征: {node_feature_dim}维", flush=True)
        print(f"   隐藏层: {hidden_dim}维", flush=True)
        print(f"   GNN层数: {num_gnn_layers}", flush=True)
        print(f"   参数量: {sum(p.numel() for p in self.parameters()):,}", flush=True)
    
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        node_features: [batch, n_nodes, feature_dim]
        adj_matrix: [batch, n_nodes, n_nodes]
        returns: [batch, n_nodes, 2] 曲率增量
        """
        x = self.input_projection(node_features)
        
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, adj_matrix)
        
        curvature_deltas = self.output_head(x)  # [batch, n_nodes, 2]
        
        return curvature_deltas
    
    def build_graph_state(self, obs: np.ndarray, n_segments: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """从观测构建图状态"""
        # 简化的特征构建
        # obs格式: [joint_angles(10), achieved_goal(3), desired_goal(3)]
        
        joint_angles = obs[:10]
        achieved_goal = obs[10:13]
        desired_goal = obs[13:16]
        
        node_features = []
        for i in range(n_segments):
            alpha = joint_angles[i * 2] if i * 2 < 10 else 0
            beta = joint_angles[i * 2 + 1] if i * 2 + 1 < 10 else 0
            
            # 节点特征：[sin(α), cos(α), sin(β), cos(β), 目标差, 位置编码]
            feature = [
                np.sin(alpha), np.cos(alpha),
                np.sin(beta), np.cos(beta),
                *(desired_goal - achieved_goal),  # 3维
                i / max(n_segments - 1, 1),  # 位置编码
                n_segments / 5.0,  # DOF归一化
                0, 0, 0, 0  # 填充到13维
            ][:self.node_feature_dim]
            
            node_features.append(feature)
        
        node_features = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0)  # [1, n_nodes, 13]
        
        # 构建邻接矩阵（链式连接）
        adj_matrix = torch.zeros(1, n_segments, n_segments)
        for i in range(n_segments - 1):
            adj_matrix[0, i, i + 1] = 1
            adj_matrix[0, i + 1, i] = 1
        
        return node_features, adj_matrix


class GraphCurvatureSAC:
    """简化的SAC用于Graph曲率训练"""
    
    def __init__(self, actor: GraphCurvatureGNNActor, config: Dict):
        self.actor = actor
        self.config = config
        self.device = config['device']
        
        # 简化的Critic（使用平铺状态）
        state_dim = 16  # 固定的观测维度
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + 10, 256),  # state + action(最大10维)
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=3e-4)
        
        # Target网络
        self.q1_target = self._copy_network(self.q1)
        self.q2_target = self._copy_network(self.q2)
        
        # 自动温度调节
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -10  # -action_dim
        
    def _copy_network(self, net):
        """复制网络"""
        import copy
        return copy.deepcopy(net)
    
    def update_targets(self, tau=0.005):
        """软更新目标网络"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class GraphCurvatureTrainer:
    """Graph曲率增量训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 保存目录
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(exist_ok=True)
        
        print("🚀 Graph曲率增量训练器初始化", flush=True)
        print("=" * 60, flush=True)
        print("配置:", flush=True)
        for key, value in config.items():
            print(f"  {key}: {value}", flush=True)
        
        self._setup_environment()
        self._setup_agent()
        
        # 经验回放
        self.replay_buffer = deque(maxlen=config['buffer_capacity'])
        
        # 训练统计
        self.best_success_rate = 0.0
        self.total_steps = 0
        
    def _setup_environment(self):
        """设置曲率环境"""
        from graph_curvature_environment import GraphCurvatureEnvironment
        
        self.env = GraphCurvatureEnvironment(
            dof_range=self.config['dof_range'],
            goal_threshold=self.config['goal_threshold'],
            curvature_step_size=self.config['curvature_step_size']
        )
        
        print("✅ Graph曲率环境创建完成", flush=True)
        
    def _setup_agent(self):
        """设置GNN Agent"""
        self.actor = GraphCurvatureGNNActor(
            node_feature_dim=self.config.get('node_feature_dim', 13),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_gnn_layers=self.config.get('num_gnn_layers', 3)
        ).to(self.device)
        
        self.sac = GraphCurvatureSAC(self.actor, self.config)
        
        print("✅ Graph GNN Agent创建完成", flush=True)
    
    def _run_episode(self) -> Tuple[float, bool, float]:
        """运行一个episode"""
        obs, info = self.env.reset()
        n_segments = info['n_segments']
        
        episode_reward = 0
        episode_success = False
        
        for step in range(self.config['max_episode_steps']):
            # 构建图状态
            node_features, adj_matrix = self.actor.build_graph_state(obs, n_segments)
            node_features = node_features.to(self.device)
            adj_matrix = adj_matrix.to(self.device)
            
            # 获取动作
            with torch.no_grad():
                curvature_deltas = self.actor(node_features, adj_matrix)  # [1, n_segments, 2]
                curvature_deltas = curvature_deltas.squeeze(0).cpu().numpy()  # [n_segments, 2]
            
            # 添加探索噪声
            if self.total_steps < self.config['learning_starts']:
                noise = np.random.randn(*curvature_deltas.shape) * 0.3
                curvature_deltas = np.clip(curvature_deltas + noise, -1, 1)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(curvature_deltas)
            
            # 存储转换
            transition = {
                'obs': obs,
                'action': curvature_deltas,
                'reward': reward,
                'next_obs': next_obs,
                'done': terminated or truncated,
                'n_segments': n_segments
            }
            self.replay_buffer.append(transition)
            
            episode_reward += reward
            self.total_steps += 1
            
            # 学习
            if self.total_steps >= self.config['learning_starts'] and \
               self.total_steps % self.config['learn_every'] == 0:
                self._learn()
            
            obs = next_obs
            
            if terminated:
                episode_success = True
                break
            if truncated:
                break
        
        final_distance = info.get('distance', float('inf'))
        
        return episode_reward, episode_success, final_distance
    
    def _learn(self):
        """SAC学习步骤"""
        if len(self.replay_buffer) < self.config['batch_size']:
            return
        
        # 采样批次
        batch_size = self.config['batch_size']
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # 转换为tensor
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for transition in batch:
            states.append(transition['obs'])
            # 将action平铺为固定10维
            action = transition['action']
            if isinstance(action, np.ndarray) and action.ndim == 2:
                action = action.flatten()
            padded_action = np.zeros(10)
            padded_action[:len(action)] = action[:10]
            actions.append(padded_action)
            rewards.append(transition['reward'])
            next_states.append(transition['next_obs'])
            dones.append(float(transition['done']))
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q函数更新
        with torch.no_grad():
            # 获取下一个动作和log_prob
            next_node_features, next_adj = [], []
            for next_state in next_states:
                nf, adj = self.actor.build_graph_state(next_state.cpu().numpy(), 
                                                        self.env.current_n_segments)
                next_node_features.append(nf)
                next_adj.append(adj)
            
            next_node_features = torch.cat(next_node_features, dim=0).to(self.device)
            next_adj = torch.cat(next_adj, dim=0).to(self.device)
            
            next_actions = self.actor(next_node_features, next_adj)
            next_actions_flat = next_actions.reshape(batch_size, -1)
            # 填充到10维
            next_actions_padded = torch.zeros(batch_size, 10).to(self.device)
            next_actions_padded[:, :next_actions_flat.shape[1]] = next_actions_flat[:, :10]
            
            # 计算目标Q值
            next_state_action = torch.cat([next_states, next_actions_padded], dim=1)
            q1_next = self.sac.q1_target(next_state_action)
            q2_next = self.sac.q2_target(next_state_action)
            min_q_next = torch.min(q1_next, q2_next)
            
            # SAC的目标Q值 (带entropy bonus)
            alpha = self.sac.log_alpha.exp()
            target_q = rewards + (1 - dones) * 0.99 * (min_q_next - alpha * 0.1)
        
        # 更新Q网络
        state_action = torch.cat([states, actions], dim=1)
        q1 = self.sac.q1(state_action)
        q2 = self.sac.q2(state_action)
        
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        
        self.sac.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.sac.q1_optimizer.step()
        
        self.sac.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.sac.q2_optimizer.step()
        
        # 更新策略网络 (每2次Q更新才更新一次策略)
        if self.total_steps % 100 == 0:
            # 构建当前state的graph
            node_features, adj = [], []
            for state in states:
                nf, adj_m = self.actor.build_graph_state(state.cpu().numpy(), 
                                                          self.env.current_n_segments)
                node_features.append(nf)
                adj.append(adj_m)
            
            node_features = torch.cat(node_features, dim=0).to(self.device)
            adj = torch.cat(adj, dim=0).to(self.device)
            
            # Actor loss
            new_actions = self.actor(node_features, adj)
            new_actions_flat = new_actions.reshape(batch_size, -1)
            new_actions_padded = torch.zeros(batch_size, 10).to(self.device)
            new_actions_padded[:, :new_actions_flat.shape[1]] = new_actions_flat[:, :10]
            
            state_new_action = torch.cat([states, new_actions_padded], dim=1)
            q1_new = self.sac.q1(state_new_action)
            q2_new = self.sac.q2(state_new_action)
            min_q_new = torch.min(q1_new, q2_new)
            
            actor_loss = -min_q_new.mean()
            
            self.sac.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.sac.actor_optimizer.step()
            
            # 更新温度
            alpha_loss = -(self.sac.log_alpha * (0.1 + self.sac.target_entropy)).mean()
            self.sac.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.sac.alpha_optimizer.step()
        
        # 软更新目标网络
        self.sac.update_targets()
    
    def train(self):
        """主训练循环"""
        print("\n🎓 开始Graph曲率增量训练", flush=True)
        print("=" * 60, flush=True)
        
        episode_rewards = []
        episode_successes = []
        episode_distances = []
        
        training_start_time = time.time()
        
        for episode in range(1, self.config['episodes'] + 1):
            episode_reward, episode_success, final_distance = self._run_episode()
            
            episode_rewards.append(episode_reward)
            episode_successes.append(episode_success)
            episode_distances.append(final_distance)
            
            # 定期评估
            if episode % self.config['eval_every'] == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                success_rate = np.mean(episode_successes[-100:]) * 100
                avg_distance = np.mean(episode_distances[-100:])
                
                elapsed_time = time.time() - training_start_time
                episodes_per_hour = episode / (elapsed_time / 3600)
                
                learning_status = "🎓学习中" if self.total_steps >= self.config['learning_starts'] else f"🔍探索中({self.total_steps}/{self.config['learning_starts']})"
                
                print(f"Episode {episode:4d} | "
                      f"成功率: {success_rate:5.1f}% | "
                      f"平均奖励: {avg_reward:7.1f} | "
                      f"平均距离: {avg_distance:.3f}m | "
                      f"速度: {episodes_per_hour:.1f} eps/h | {learning_status}", flush=True)
                
                # 保存最佳模型
                if success_rate > self.best_success_rate and \
                   self.total_steps >= self.config['learning_starts']:
                    self.best_success_rate = success_rate
                    self._save_checkpoint(episode, success_rate)
                    print(f"🏆 新纪录! Graph曲率成功率: {success_rate:.1f}%", flush=True)
        
        print(f"\n🎉 训练完成!", flush=True)
        print(f"   最佳成功率: {self.best_success_rate:.1f}%", flush=True)
        print(f"   总用时: {(time.time() - training_start_time)/3600:.1f}小时", flush=True)
    
    def _save_checkpoint(self, episode: int, success_rate: float):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'success_rate': success_rate,
            'actor_state_dict': self.actor.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = self.save_dir / "best_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        print(f"💾 保存模型: {checkpoint_path}", flush=True)


def main():
    """主函数"""
    config = {
        # 环境配置
        'dof_range': (2, 5),
        'goal_threshold': 0.10,
        'curvature_step_size': 0.1,
        'max_episode_steps': 200,
        
        # 网络配置
        'node_feature_dim': 13,
        'hidden_dim': 256,
        'num_gnn_layers': 3,
        
        # 训练配置
        'episodes': 5000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_capacity': 100000,
        'batch_size': 256,
        'learning_starts': 5000,
        'learn_every': 50,
        'eval_every': 10,
        
        # 保存配置
        'save_dir': 'graph_curvature_results'
    }
    
    print("🎯 Graph曲率增量GNN训练", flush=True)
    print("核心创新:", flush=True)
    print("  1. Graph-to-Graph架构", flush=True)
    print("  2. 曲率增量解决α≈0问题", flush=True)
    print("  3. 真正任意DOF支持", flush=True)
    print(flush=True)
    
    trainer = GraphCurvatureTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()