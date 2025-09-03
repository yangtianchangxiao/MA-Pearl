#!/usr/bin/env python3
"""
随机DOF GNN模型可视化脚本
展示训练好的模型在不同DOF配置下的表现
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from pathlib import Path
import time

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from optimized_graph_her_wrapper import OptimizedGraphHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


class RandomDOFGNNVisualizer:
    """随机DOF GNN模型可视化器"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        
        print(f"🎬 随机DOF GNN可视化器")
        print(f"   Checkpoint: {self.checkpoint_path}")
        print(f"   设备: {self.device}")
        
        # 加载配置和模型
        self._load_agent()
        
    def _load_agent(self):
        """加载训练好的agent"""
        print("📦 加载checkpoint...")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint不存在: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 从checkpoint获取配置
        self.config = checkpoint.get('config', {
            'dof_range': (2, 5),
            'segment_length_range': (0.1, 0.3),
            'goal_threshold': 0.15,
            'max_episode_steps': 200,
            'hidden_dims': [256, 256],
            'learning_rate': 3e-4
        })
        
        # 确保所有必需的配置项都存在
        if 'hidden_dims' not in self.config:
            self.config['hidden_dims'] = [256, 256]
        
        # 创建环境
        self.env = OptimizedGraphHERWrapper(
            dof_range=self.config['dof_range'],
            segment_length_range=self.config['segment_length_range'],
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps']
        )
        
        # 创建Actor网络 (使用与训练时相同的配置)
        action_dim = max(self.config['dof_range']) * 2  # 最大DOF
        
        self.actor_network = UltraLightGNNActor(
            action_dim=action_dim,
            dof_range=self.config['dof_range'],
            hidden_dim=self.config.get('hidden_dim', 128),  # 使用训练时的配置
            num_gnn_layers=self.config.get('num_gnn_layers', 2)  # 使用训练时的配置
        ).to(self.device)
        
        # 创建SAC learner (使用默认学习率)
        state_dim = self.env.observation_space.shape[0]
        learner = ContinuousSoftActorCritic(
            action_space=self.env.action_space,
            state_dim=state_dim,
            actor_network_instance=self.actor_network,
            critic_hidden_dims=self.config.get('critic_hidden_dims', [512, 512]),  # 使用训练时的配置！
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4
        )
        
        # 创建HER buffer (虚拟的，不用于训练)
        replay_buffer = create_variable_soft_arm_her_buffer(capacity=1000)
        
        # 创建agent
        self.agent = PearlAgent(
            policy_learner=learner,
            replay_buffer=replay_buffer
        )
        
        # 设置action space (Pearl框架要求)
        self.agent._action_space = self.env.action_space
        
        # 加载完整的模型权重 - 修复checkpoint格式兼容性
        agent_state = checkpoint['agent_state']
        if 'policy_learner._actor.node_encoder.weight' in agent_state:
            # 新格式: agent.state_dict() 包含 policy_learner 前缀
            self.agent.load_state_dict(agent_state)
        else:
            # 旧格式: policy_learner.state_dict()
            self.agent.policy_learner.load_state_dict(agent_state)
        self.agent.policy_learner._actor.eval()  # 正确的Pearl框架访问方式
        
        # 训练统计信息
        self.episode_count = checkpoint.get('episode', 0)
        self.success_rate = checkpoint.get('success_rate', 0.0)
        
        print(f"✅ 模型加载成功")
        print(f"   训练Episode: {self.episode_count}")
        print(f"   成功率: {self.success_rate:.1f}%")
        print(f"   DOF范围: {self.config['dof_range'][0]}-{self.config['dof_range'][1]}节")
    
    def test_single_episode(self, target_dof: int = None, visualize: bool = True):
        """测试单个episode"""
        print(f"\n🎮 开始单次测试")
        print("=" * 50)
        
        # 重置环境 (如果指定DOF，需要在环境中设置)
        obs, _ = self.env.reset()
        
        current_dof = self.env.env.current_n_segments * 2
        segment_lengths = self.env.env.segment_lengths[:self.env.env.current_n_segments]
        
        print(f"🔧 当前配置:")
        print(f"   DOF: {current_dof} ({self.env.env.current_n_segments}节)")
        print(f"   Segment长度: {segment_lengths}")
        print(f"   目标位置: {self.env.env.goal_position}")
        
        # 收集轨迹数据
        trajectory = []
        rewards = []
        actions = []
        
        step = 0
        total_reward = 0
        
        while step < self.config['max_episode_steps']:
            # 获取动作
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                
                # 直接使用UltraLightGNNActor的forward方法
                action_mean, _ = self.agent.policy_learner._actor(obs_tensor)
                action = action_mean.cpu().numpy()
            
            # 执行动作
            result = self.env.step(action[:current_dof])  # 只使用当前DOF的动作
            
            # 记录数据
            current_ee_pos = self.env.env._forward_kinematics()
            trajectory.append(current_ee_pos.copy())
            rewards.append(float(result.reward))
            actions.append(action[:current_dof].copy())
            
            total_reward += result.reward
            obs = result.observation
            step += 1
            
            if result.terminated or result.truncated:
                break
        
        # 计算最终距离
        final_distance = np.linalg.norm(current_ee_pos - self.env.env.goal_position)
        success = final_distance < self.config['goal_threshold']
        
        print(f"\n📊 Episode结果:")
        print(f"   步数: {step}")
        print(f"   总奖励: {total_reward:.3f}")
        print(f"   最终距离: {final_distance:.4f}m")
        print(f"   成功: {'✅ 是' if success else '❌ 否'}")
        
        # 可视化轨迹
        if visualize and len(trajectory) > 0:
            self._plot_trajectory(trajectory, segment_lengths, current_dof)
        
        return {
            'success': success,
            'final_distance': final_distance,
            'total_reward': total_reward,
            'steps': step,
            'dof': current_dof,
            'segment_lengths': segment_lengths,
            'trajectory': trajectory,
            'actions': actions,
            'rewards': rewards
        }
    
    def _plot_trajectory(self, trajectory, segment_lengths, dof):
        """绘制3D轨迹"""
        trajectory = np.array(trajectory)
        target_pos = self.env.env.goal_position
        
        fig = plt.figure(figsize=(12, 8))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, alpha=0.7, label='End-effector轨迹')
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                   c='green', s=100, label='起始点')
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                   c='blue', s=100, label='结束点')
        ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
                   c='red', s=200, marker='*', label='目标点')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D轨迹 ({dof}DOF)')
        ax1.legend()
        ax1.grid(True)
        
        # XY平面投影
        ax2 = fig.add_subplot(222)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100)
        ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='blue', s=100)
        ax2.scatter(target_pos[0], target_pos[1], c='red', s=200, marker='*')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY平面投影')
        ax2.grid(True)
        ax2.axis('equal')
        
        # 距离变化曲线
        ax3 = fig.add_subplot(223)
        distances = [np.linalg.norm(pos - target_pos) for pos in trajectory]
        ax3.plot(distances, 'g-', linewidth=2)
        ax3.axhline(y=self.config['goal_threshold'], color='r', linestyle='--', 
                   label=f'目标阈值 ({self.config["goal_threshold"]}m)')
        ax3.set_xlabel('步数')
        ax3.set_ylabel('距离目标 (m)')
        ax3.set_title('距离变化')
        ax3.legend()
        ax3.grid(True)
        
        # 软体臂配置信息
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        info_text = f"""模型配置:
DOF: {dof} ({dof//2}节)
Segment长度: {[f'{l:.3f}' for l in segment_lengths]}
目标阈值: {self.config['goal_threshold']}m
训练Episode: {self.episode_count}
训练成功率: {self.success_rate:.1f}%

当前测试:
轨迹长度: {len(trajectory)}步
最终距离: {distances[-1]:.4f}m
成功: {'是' if distances[-1] < self.config['goal_threshold'] else '否'}
        """
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.suptitle(f'Random DOF GNN Visualization - {dof}DOF Test', y=0.98)
        
        # 保存图片文件而不是显示
        save_path = f'/home/cx/MA-Pearl/visualization_{dof}dof_test.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 可视化图片已保存到: {save_path}")
        plt.close()  # 关闭图形避免内存泄露
        
        return save_path
    
    def batch_evaluation(self, n_tests: int = 10, show_stats: bool = True):
        """批量评估不同DOF配置"""
        print(f"\n🧪 批量评估 ({n_tests}次测试)")
        print("=" * 50)
        
        results = []
        dof_stats = {}
        
        for test_id in range(n_tests):
            print(f"\n测试 {test_id + 1}/{n_tests}:")
            result = self.test_single_episode(visualize=False)
            results.append(result)
            
            # 按DOF分组统计
            dof = result['dof']
            if dof not in dof_stats:
                dof_stats[dof] = {'successes': 0, 'distances': [], 'rewards': []}
            
            dof_stats[dof]['successes'] += int(result['success'])
            dof_stats[dof]['distances'].append(result['final_distance'])
            dof_stats[dof]['rewards'].append(result['total_reward'])
        
        if show_stats:
            self._show_batch_stats(dof_stats, n_tests)
        
        return results, dof_stats
    
    def _show_batch_stats(self, dof_stats, total_tests):
        """显示批量统计结果"""
        print(f"\n📈 批量评估统计 ({total_tests}次测试)")
        print("=" * 60)
        
        overall_success = 0
        overall_distance = []
        
        for dof in sorted(dof_stats.keys()):
            stats = dof_stats[dof]
            count = len(stats['distances'])
            success_rate = (stats['successes'] / count) * 100
            avg_distance = np.mean(stats['distances'])
            avg_reward = np.mean(stats['rewards'])
            
            print(f"{dof}DOF ({count}次): 成功率 {success_rate:.1f}%, "
                  f"平均距离 {avg_distance:.4f}m, 平均奖励 {avg_reward:.2f}")
            
            overall_success += stats['successes']
            overall_distance.extend(stats['distances'])
        
        overall_success_rate = (overall_success / total_tests) * 100
        overall_avg_distance = np.mean(overall_distance)
        
        print(f"\n总体性能:")
        print(f"  成功率: {overall_success_rate:.1f}% ({overall_success}/{total_tests})")
        print(f"  平均距离: {overall_avg_distance:.4f}m")
        print(f"  目标阈值: {self.config['goal_threshold']}m")


def main():
    parser = argparse.ArgumentParser(description='随机DOF GNN模型可视化')
    parser.add_argument('--checkpoint', type=str, 
                      default='./random_dof_gnn_results/best_checkpoint.pt',
                      help='模型checkpoint路径')
    parser.add_argument('--device', type=str, default='cpu',
                      help='计算设备 (cpu/cuda)')
    parser.add_argument('--mode', type=str, default='single',
                      choices=['single', 'batch'],
                      help='运行模式: single(单次测试) 或 batch(批量评估)')
    parser.add_argument('--n_tests', type=int, default=10,
                      help='批量测试次数')
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器
        visualizer = RandomDOFGNNVisualizer(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        
        if args.mode == 'single':
            # 单次测试
            result = visualizer.test_single_episode(visualize=True)
            
        elif args.mode == 'batch':
            # 批量评估
            results, stats = visualizer.batch_evaluation(
                n_tests=args.n_tests, 
                show_stats=True
            )
        
    except Exception as e:
        print(f"❌ 可视化出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()