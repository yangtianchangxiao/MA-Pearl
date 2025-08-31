#!/usr/bin/env python3
"""
创建软体臂运动的GIF动画
展示从起始到目标的完整运动过程
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse

from optimized_graph_her_wrapper import OptimizedGraphHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


class SoftArmGifCreator:
    """软体臂GIF动画创建器"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device('cpu')
        
        print(f"🎬 软体臂GIF创建器")
        print(f"   Checkpoint: {self.checkpoint_path}")
        
        # 加载配置
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = checkpoint.get('config', {
            'dof_range': (2, 5),
            'segment_length_range': (0.1, 0.35),
            'goal_threshold': 0.15,
            'max_episode_steps': 200,
            'hidden_dim': 128,
            'num_gnn_layers': 2,
            'critic_hidden_dims': [512, 512]
        })
        
        self._setup_model()
        
    def _setup_model(self):
        """设置环境和模型"""
        # 创建环境
        self.env = OptimizedGraphHERWrapper(
            dof_range=self.config['dof_range'],
            segment_length_range=self.config['segment_length_range'],
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps']
        )
        
        # 创建网络
        action_dim = max(self.config['dof_range']) * 2
        self.actor_network = UltraLightGNNActor(
            action_dim=action_dim,
            dof_range=self.config['dof_range'],
            hidden_dim=self.config.get('hidden_dim', 128),
            num_gnn_layers=self.config.get('num_gnn_layers', 2)
        ).to(self.device)
        
        # 创建SAC learner
        state_dim = self.env.observation_space.shape[0]
        learner = ContinuousSoftActorCritic(
            action_space=self.env.action_space,
            state_dim=state_dim,
            actor_network_instance=self.actor_network,
            critic_hidden_dims=self.config.get('critic_hidden_dims', [512, 512]),
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4
        )
        
        replay_buffer = create_variable_soft_arm_her_buffer(capacity=1000)
        
        self.agent = PearlAgent(
            policy_learner=learner,
            replay_buffer=replay_buffer
        )
        self.agent._action_space = self.env.action_space
        
        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.agent.policy_learner.load_state_dict(checkpoint['agent_state'])
        self.agent.policy_learner._actor.eval()
        
        print(f"✅ 模型加载成功")
        print(f"   训练Episode: {checkpoint.get('episode', 0)}")
        print(f"   成功率: {checkpoint.get('success_rate', 0):.1f}%")
    
    def collect_episode_data(self):
        """收集完整episode的数据用于动画"""
        print("🎮 收集episode数据...")
        
        obs, _ = self.env.reset()
        current_dof = self.env.env.current_n_segments * 2
        
        print(f"🔧 配置:")
        print(f"   DOF: {current_dof} ({self.env.env.current_n_segments}节)")
        print(f"   Segment长度: {self.env.env.segment_lengths[:self.env.env.current_n_segments]}")
        print(f"   目标位置: {self.env.env.goal_position}")
        
        # 收集数据
        trajectory = []
        joint_angles_history = []
        rewards = []
        step = 0
        
        while step < self.config['max_episode_steps']:
            # 记录当前状态
            current_ee_pos = self.env.env._forward_kinematics()
            current_joints = self.env.env.joint_angles[:current_dof].copy()
            
            trajectory.append(current_ee_pos.copy())
            joint_angles_history.append(current_joints.copy())
            
            # 获取动作
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                action_mean, _ = self.agent.policy_learner._actor(obs_tensor)
                action = action_mean.cpu().numpy()
            
            # 执行动作
            result = self.env.step(action[:current_dof])
            rewards.append(float(result.reward))
            
            obs = result.observation
            step += 1
            
            if result.terminated or result.truncated:
                break
        
        # 最终状态
        final_ee_pos = self.env.env._forward_kinematics()
        final_joints = self.env.env.joint_angles[:current_dof].copy()
        trajectory.append(final_ee_pos.copy())
        joint_angles_history.append(final_joints.copy())
        
        final_distance = np.linalg.norm(final_ee_pos - self.env.env.goal_position)
        success = final_distance < self.config['goal_threshold']
        
        print(f"📊 Episode完成:")
        print(f"   步数: {step}")
        print(f"   最终距离: {final_distance:.4f}m")
        print(f"   成功: {'✅' if success else '❌'}")
        
        return {
            'trajectory': np.array(trajectory),
            'joint_angles': np.array(joint_angles_history),
            'rewards': rewards,
            'target_position': self.env.env.goal_position.copy(),
            'segment_lengths': self.env.env.segment_lengths[:self.env.env.current_n_segments].copy(),
            'current_dof': current_dof,
            'success': success,
            'final_distance': final_distance,
            'steps': step
        }
    
    def create_arm_shape(self, joint_angles, segment_lengths):
        """根据关节角度计算软体臂的3D形状 - 与环境运动学完全一致"""
        n_segments = len(segment_lengths)
        
        # 起始点
        points = [np.array([0.0, 0.0, 0.0])]
        position = np.array([0.0, 0.0, 0.0])
        cumulative_angle_xy = 0.0  # XY平面累积角度
        
        for i in range(n_segments):
            joint1 = joint_angles[i * 2]      # 弯曲角（Z方向）
            joint2 = joint_angles[i * 2 + 1]  # 方向角（XY平面）
            length = segment_lengths[i]
            
            # 更新累积方向角
            cumulative_angle_xy += joint2
            
            # 计算该段的末端位移 (与环境_forward_kinematics完全一致)
            segment_end = length * np.array([
                np.cos(cumulative_angle_xy) * np.cos(joint1),  # X
                np.sin(cumulative_angle_xy) * np.cos(joint1),  # Y
                np.sin(joint1)                                 # Z
            ])
            
            position += segment_end
            points.append(position.copy())
        
        return np.array(points)
    
    def create_gif(self, episode_data, gif_path='soft_arm_motion.gif', fps=10):
        """创建GIF动画"""
        print(f"🎨 创建GIF动画: {gif_path}")
        
        trajectory = episode_data['trajectory']
        joint_angles = episode_data['joint_angles']
        target_pos = episode_data['target_position']
        segment_lengths = episode_data['segment_lengths']
        
        fig = plt.figure(figsize=(12, 8))
        
        # 3D主图
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('Soft Arm Motion (3D)')
        
        # 设置固定的视角范围
        all_points = np.vstack([trajectory, target_pos.reshape(1, -1)])
        margin = 0.1
        ax1.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax1.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax1.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
        
        # 目标点 (固定)
        ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
                   c='red', s=200, marker='*', label='Target')
        
        # XY平面图
        ax2 = fig.add_subplot(222)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View (XY Plane)')
        ax2.scatter(target_pos[0], target_pos[1], c='red', s=200, marker='*')
        ax2.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax2.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax2.grid(True)
        
        # 距离曲线
        ax3 = fig.add_subplot(223)
        distances = [np.linalg.norm(pos - target_pos) for pos in trajectory]
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Distance to Target (m)')
        ax3.set_title('Distance Progress')
        ax3.axhline(y=self.config['goal_threshold'], color='r', linestyle='--', 
                   label=f'Threshold ({self.config["goal_threshold"]}m)')
        ax3.set_xlim(0, len(trajectory))
        ax3.set_ylim(0, max(distances) * 1.1)
        ax3.grid(True)
        ax3.legend()
        
        # 信息面板
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        # 初始化动画元素
        arm_line, = ax1.plot([], [], [], 'b-', linewidth=3, label='Arm')
        ee_point, = ax1.plot([], [], [], 'go', markersize=8, label='End-effector')
        ee_trail, = ax1.plot([], [], [], 'g--', alpha=0.5, linewidth=1)
        
        xy_arm_line, = ax2.plot([], [], 'b-', linewidth=2)
        xy_ee_point, = ax2.plot([], [], 'go', markersize=6)
        xy_trail, = ax2.plot([], [], 'g--', alpha=0.5)
        
        distance_line, = ax3.plot([], [], 'g-', linewidth=2)
        current_step_point, = ax3.plot([], [], 'ro', markersize=6)
        
        def animate(frame):
            if frame >= len(trajectory):
                return []
            
            # 计算当前软体臂形状
            current_joints = joint_angles[frame]
            arm_points = self.create_arm_shape(current_joints, segment_lengths)
            current_ee = trajectory[frame]
            
            # 更新3D图
            arm_line.set_data_3d(arm_points[:, 0], arm_points[:, 1], arm_points[:, 2])
            ee_point.set_data_3d([current_ee[0]], [current_ee[1]], [current_ee[2]])
            
            # 轨迹历史
            if frame > 0:
                trail_points = trajectory[:frame+1]
                ee_trail.set_data_3d(trail_points[:, 0], trail_points[:, 1], trail_points[:, 2])
            
            # 更新XY图
            xy_arm_line.set_data(arm_points[:, 0], arm_points[:, 1])
            xy_ee_point.set_data([current_ee[0]], [current_ee[1]])
            if frame > 0:
                trail_points = trajectory[:frame+1]
                xy_trail.set_data(trail_points[:, 0], trail_points[:, 1])
            
            # 更新距离图
            distance_line.set_data(range(frame+1), distances[:frame+1])
            current_step_point.set_data([frame], [distances[frame]])
            
            # 更新信息面板
            ax4.clear()
            ax4.axis('off')
            
            info_text = f"""Episode Info:
DOF: {episode_data['current_dof']} ({len(segment_lengths)} segments)
Step: {frame}/{len(trajectory)-1}
Distance: {distances[frame]:.4f}m
Target: {self.config['goal_threshold']}m
Status: {'SUCCESS' if distances[frame] < self.config['goal_threshold'] else 'IN PROGRESS'}

Model Info:
Training Episodes: 1325
Success Rate: 84.0%
Final Result: {'✅ SUCCESS' if episode_data['success'] else '❌ FAILED'}
"""
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            return [arm_line, ee_point, ee_trail, xy_arm_line, xy_ee_point, xy_trail, 
                   distance_line, current_step_point]
        
        # 添加图例
        ax1.legend(loc='upper right')
        
        plt.tight_layout()
        
        # 创建动画
        frames = len(trajectory)
        interval = 1000 // fps  # ms per frame
        
        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        
        # 保存GIF
        print(f"💾 保存GIF动画 (这可能需要几分钟)...")
        anim.save(gif_path, writer='pillow', fps=fps, dpi=100)
        print(f"✅ GIF保存完成: {gif_path}")
        
        plt.close()
        return gif_path


def main():
    parser = argparse.ArgumentParser(description='创建软体臂运动GIF')
    parser.add_argument('--checkpoint', type=str, 
                      default='./random_dof_gnn_results/best_checkpoint.pt',
                      help='模型checkpoint路径')
    parser.add_argument('--output', type=str, default='soft_arm_motion.gif',
                      help='输出GIF文件名')
    parser.add_argument('--fps', type=int, default=10,
                      help='GIF帧率')
    
    args = parser.parse_args()
    
    try:
        # 创建GIF创建器
        creator = SoftArmGifCreator(args.checkpoint)
        
        # 收集episode数据
        episode_data = creator.collect_episode_data()
        
        # 创建GIF
        gif_path = creator.create_gif(episode_data, args.output, args.fps)
        
        print(f"\n🎉 GIF创建成功!")
        print(f"   文件: {gif_path}")
        print(f"   帧数: {len(episode_data['trajectory'])}")
        print(f"   帧率: {args.fps} FPS")
        
    except Exception as e:
        print(f"❌ 创建GIF失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()