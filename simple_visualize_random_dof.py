#!/usr/bin/env python3
"""
简化的随机DOF GNN模型可视化脚本
直接加载网络权重进行推理，避免优化器状态问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from optimized_graph_her_wrapper import OptimizedGraphHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor


class SimpleGNNVisualizer:
    """简化的GNN可视化器"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device('cpu')  # 强制使用CPU以避免兼容性问题
        
        print(f"🎬 简化GNN可视化器")
        print(f"   Checkpoint: {self.checkpoint_path}")
        
        # 固定配置 (基于训练脚本)
        self.config = {
            'dof_range': (2, 5),
            'segment_length_range': (0.1, 0.35),
            'goal_threshold': 0.15,
            'max_episode_steps': 200,
            'hidden_dim': 128,
            'num_gnn_layers': 2
        }
        
        self._load_components()
        
    def _load_components(self):
        """加载环境和网络"""
        print("📦 加载组件...")
        
        # 创建环境
        self.env = OptimizedGraphHERWrapper(
            dof_range=self.config['dof_range'],
            segment_length_range=self.config['segment_length_range'],
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps']
        )
        
        # 创建GNN网络
        action_dim = max(self.config['dof_range']) * 2  # 最大DOF
        self.actor_network = UltraLightGNNActor(
            action_dim=action_dim,
            dof_range=self.config['dof_range'],
            hidden_dim=self.config['hidden_dim'],
            num_gnn_layers=self.config['num_gnn_layers']
        ).to(self.device)
        
        # 加载权重 (仅actor网络)
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 提取actor网络权重
            agent_state = checkpoint['agent_state']
            actor_weights = {}
            
            for key, value in agent_state.items():
                if key.startswith('actor_network.'):
                    new_key = key.replace('actor_network.', '')
                    actor_weights[new_key] = value
            
            # 加载到网络
            self.actor_network.load_state_dict(actor_weights, strict=False)
            self.actor_network.eval()
            
            print(f"✅ 网络权重加载成功")
            print(f"   训练Episode: {checkpoint.get('episode', 0)}")
            print(f"   成功率: {checkpoint.get('success_rate', 0):.1f}%")
        else:
            print("⚠️  未找到checkpoint，使用随机权重")
    
    def get_action(self, her_obs: torch.Tensor, current_dof: int):
        """使用GNN网络获取动作"""
        with torch.no_grad():
            # 从HER格式转换回Graph格式进行推理
            joint_angles = her_obs[:current_dof]
            achieved_goal = her_obs[-6:-3]
            desired_goal = her_obs[-3:]
            
            # 构造Graph状态 (简化版)
            n_segments = current_dof // 2
            
            # 创建节点特征 [joint1, joint2, length]
            node_features = []
            for i in range(n_segments):
                alpha = joint_angles[2*i]
                beta = joint_angles[2*i+1] 
                length = self.env.env.segment_lengths[i]
                node_features.append([alpha, beta, length])
            
            node_features = torch.tensor(node_features, dtype=torch.float32)
            
            # 创建简单的链式邻接矩阵
            adjacency = torch.zeros(n_segments, n_segments)
            for i in range(n_segments - 1):
                adjacency[i, i+1] = 1.0
                adjacency[i+1, i] = 1.0
            # 自连接
            for i in range(n_segments):
                adjacency[i, i] = 1.0
            
            # 创建goals
            goals = torch.cat([achieved_goal, desired_goal])
            
            # GNN推理
            action_mean, _ = self.actor_network(node_features, adjacency, goals)
            
            return action_mean[:current_dof].cpu().numpy()
    
    def test_episode(self, visualize=True):
        """测试单个episode"""
        print(f"\n🎮 开始测试")
        print("=" * 40)
        
        # 重置环境
        obs, _ = self.env.reset()
        current_dof = self.env.env.current_n_segments * 2
        
        print(f"🔧 当前配置:")
        print(f"   DOF: {current_dof} ({self.env.env.current_n_segments}节)")
        print(f"   Segment长度: {self.env.env.segment_lengths[:self.env.env.current_n_segments]}")
        print(f"   目标位置: {self.env.env.goal_position}")
        
        # 收集轨迹
        trajectory = []
        step = 0
        
        while step < self.config['max_episode_steps']:
            # 获取动作
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = self.get_action(obs_tensor, current_dof)
            
            # 执行动作
            result = self.env.step(action)
            
            # 记录轨迹
            current_ee_pos = self.env.env.get_end_effector_position()
            trajectory.append(current_ee_pos.copy())
            
            obs = result.observation
            step += 1
            
            if result.terminated or result.truncated:
                break
        
        # 计算结果
        final_distance = np.linalg.norm(current_ee_pos - self.env.env.goal_position)
        success = final_distance < self.config['goal_threshold']
        
        print(f"\n📊 测试结果:")
        print(f"   步数: {step}")
        print(f"   最终距离: {final_distance:.4f}m")
        print(f"   成功: {'✅ 是' if success else '❌ 否'}")
        
        if visualize and len(trajectory) > 0:
            self._plot_simple_trajectory(trajectory, current_dof)
        
        return {
            'success': success,
            'distance': final_distance,
            'steps': step,
            'dof': current_dof,
            'trajectory': trajectory
        }
    
    def _plot_simple_trajectory(self, trajectory, dof):
        """简单3D轨迹可视化"""
        trajectory = np.array(trajectory)
        target_pos = self.env.env.goal_position
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D轨迹
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, alpha=0.8, label='轨迹')
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                   c='green', s=100, label='起点', marker='o')
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                   c='blue', s=100, label='终点', marker='s')
        ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
                   c='red', s=200, marker='*', label='目标')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'随机DOF GNN: {dof}DOF 3D轨迹')
        ax1.legend()
        ax1.grid(True)
        
        # 距离曲线
        ax2 = fig.add_subplot(122)
        distances = [np.linalg.norm(pos - target_pos) for pos in trajectory]
        ax2.plot(distances, 'g-', linewidth=2)
        ax2.axhline(y=self.config['goal_threshold'], color='r', linestyle='--', 
                   label=f'目标阈值 ({self.config["goal_threshold"]}m)')
        ax2.set_xlabel('步数')
        ax2.set_ylabel('距离目标 (m)')
        ax2.set_title('距离变化')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.suptitle(f'随机DOF GNN模型可视化 - {dof}DOF', y=1.02)
        plt.show()
        
        return fig
    
    def batch_test(self, n_tests=10):
        """批量测试"""
        print(f"\n🧪 批量测试 ({n_tests}次)")
        print("=" * 40)
        
        results = []
        for i in range(n_tests):
            print(f"\n测试 {i+1}/{n_tests}:")
            result = self.test_episode(visualize=False)
            results.append(result)
            
            dof = result['dof']
            success = '✅' if result['success'] else '❌'
            print(f"   {dof}DOF: {success} (距离: {result['distance']:.4f}m)")
        
        # 统计
        total_success = sum(r['success'] for r in results)
        success_rate = (total_success / n_tests) * 100
        avg_distance = np.mean([r['distance'] for r in results])
        
        print(f"\n📈 批量测试统计:")
        print(f"   成功率: {success_rate:.1f}% ({total_success}/{n_tests})")
        print(f"   平均距离: {avg_distance:.4f}m")
        print(f"   目标阈值: {self.config['goal_threshold']}m")
        
        return results


def main():
    """主函数"""
    checkpoint_path = "./random_dof_gnn_results/best_checkpoint.pt"
    
    try:
        visualizer = SimpleGNNVisualizer(checkpoint_path)
        
        # 单次测试
        print("\n" + "="*50)
        print("单次可视化测试")
        print("="*50)
        visualizer.test_episode(visualize=True)
        
        # 批量测试
        print("\n" + "="*50)
        print("批量性能测试")
        print("="*50)
        visualizer.batch_test(n_tests=5)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()