#!/usr/bin/env python3
"""
可视化训练好的软体臂Pearl SAC+HER agent
展示软体臂reaching任务的训练效果
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器环境使用
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
import sys
import os
import time
from pathlib import Path

# Pearl imports
from pearl.utils.instantiations.environments import SoftArmReachEnvironment
from pearl.utils.instantiations.environments.soft_arm_her_factory import create_soft_arm_her_buffer
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)


class SoftArmVisualizer:
    """专门用于软体臂reaching任务的可视化器"""
    
    def __init__(self, save_dir='soft_arm_videos'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 创建图形
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 历史轨迹
        self.ee_trajectory = []
        self.goal_history = []
        
        # 设置坐标轴
        self.ax.set_xlim([-0.6, 0.6])
        self.ax.set_ylim([-0.6, 0.6])
        self.ax.set_zlim([0, 0.6])
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        self.ax.set_title('软体机械臂Pearl训练效果演示', fontsize=14)
        
        # 录制设置
        self.writer = None
        self.recording = False
        self.frame_count = 0
        
    def visualize_soft_arm(self, joint_angles, segment_length=0.21):
        """根据关节角度计算并可视化软体臂形状"""
        positions = []
        positions.append([0, 0, 0])  # 基座
        
        current_pos = np.array([0, 0, 0], dtype=float)
        
        # 计算每段的位置
        for i in range(0, len(joint_angles), 2):
            alpha = joint_angles[i]      # 弯曲角度
            beta = joint_angles[i+1] if i+1 < len(joint_angles) else 0  # 方向角
            
            # 简化的正向运动学 
            # 每段向前延伸segment_length距离，带有alpha和beta的弯曲
            dx = segment_length * np.cos(alpha) * np.cos(beta)
            dy = segment_length * np.cos(alpha) * np.sin(beta)
            dz = segment_length * np.sin(alpha)
            
            current_pos += [dx, dy, dz]
            positions.append(current_pos.copy())
        
        return np.array(positions)
    
    def start_recording(self, filename='soft_arm_demo.gif', fps=10):
        """开始录制GIF动画"""
        if self.recording:
            return
            
        self.recording = True
        self.frame_count = 0
        self.frames = []
        self.gif_filename = self.save_dir / filename
        print(f"🎬 开始录制: {self.gif_filename}")
    
    def stop_recording(self):
        """停止录制并保存GIF"""
        if not self.recording:
            return
        
        print(f"💾 保存GIF动画: {self.gif_filename} ({len(self.frames)}帧)")
        
        # 创建动画并保存
        ani = animation.ArtistAnimation(self.fig, self.frames, interval=100, blit=False)
        ani.save(self.gif_filename, writer='pillow', fps=10)
        
        self.recording = False
        print(f"✅ GIF已保存: {self.gif_filename}")
    
    def update(self, observation, action, reward, terminated, step, episode=1):
        """更新可视化"""
        self.ax.cla()
        
        # 解析观测
        joint_angles = observation[:6]
        achieved_goal = observation[6:9] 
        desired_goal = observation[9:12]
        
        # 计算软体臂形状
        arm_positions = self.visualize_soft_arm(joint_angles)
        
        # 绘制软体臂
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 好看的颜色
        for i in range(len(arm_positions)-1):
            self.ax.plot([arm_positions[i,0], arm_positions[i+1,0]], 
                        [arm_positions[i,1], arm_positions[i+1,1]], 
                        [arm_positions[i,2], arm_positions[i+1,2]], 
                        color=colors[i % len(colors)], linewidth=4, 
                        label=f'Segment {i+1}' if i < 3 else '')
        
        # 绘制关节点
        for i, pos in enumerate(arm_positions):
            color = colors[i % len(colors)]
            size = 120 if i == 0 else (100 if i == len(arm_positions)-1 else 80)
            marker = 's' if i == 0 else ('*' if i == len(arm_positions)-1 else 'o')
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color=color, s=size, marker=marker, 
                          edgecolors='black', linewidth=2)
        
        # 绘制目标点
        self.ax.scatter([desired_goal[0]], [desired_goal[1]], [desired_goal[2]], 
                       color='gold', s=200, marker='*', 
                       edgecolors='orange', linewidth=2, label='Target')
        
        # 绘制当前末端位置
        self.ax.scatter([achieved_goal[0]], [achieved_goal[1]], [achieved_goal[2]], 
                       color='red', s=150, marker='o', 
                       edgecolors='darkred', linewidth=2, label='End Effector')
        
        # 添加轨迹
        self.ee_trajectory.append(achieved_goal.copy())
        if len(self.ee_trajectory) > 50:
            self.ee_trajectory.pop(0)
        
        if len(self.ee_trajectory) > 1:
            traj = np.array(self.ee_trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                        'r--', alpha=0.6, linewidth=2, label='EE Trajectory')
        
        # 绘制目标阈值球
        goal_threshold = 0.15
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = goal_threshold * np.outer(np.cos(u), np.sin(v)) + desired_goal[0]
        y_sphere = goal_threshold * np.outer(np.sin(u), np.sin(v)) + desired_goal[1]
        z_sphere = goal_threshold * np.outer(np.ones(np.size(u)), np.cos(v)) + desired_goal[2]
        self.ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='gold')
        
        # 计算距离
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        # 设置坐标轴
        self.ax.set_xlim([-0.6, 0.6])
        self.ax.set_ylim([-0.6, 0.6])
        self.ax.set_zlim([0, 0.6])
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)
        
        # 状态信息
        status = "🎯 SUCCESS!" if terminated else "🔄 Learning..."
        title = f"软体机械臂Pearl训练效果 | Episode {episode}, Step {step}\n{status}"
        self.ax.set_title(title, fontsize=14, pad=20)
        
        # 添加信息文本
        info_text = f"""距离目标: {distance:.3f}m
阈值: {goal_threshold:.3f}m  
奖励: {reward:.2f}
动作: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}...]"""
        
        self.ax.text2D(0.02, 0.98, info_text, transform=self.ax.transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 图例
        self.ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # 保存帧用于GIF录制
        if self.recording:
            # 复制当前艺术家对象用于动画
            artists = []
            for line in self.ax.lines:
                artists.append(line)
            for collection in self.ax.collections:
                artists.append(collection)
            self.frames.append(artists)
            self.frame_count += 1
        
        # 保存截图
        if step % 20 == 0:  # 每20步保存一次
            screenshot_path = self.save_dir / f'step_{episode}_{step:03d}.png'
            self.fig.savefig(screenshot_path, dpi=150, bbox_inches='tight')
    
    def close(self):
        """关闭可视化器"""
        if self.recording:
            self.stop_recording()
        plt.close(self.fig)


def create_trained_agent(model_type='50step'):
    """创建训练好的agent"""
    print(f"🤖 创建软体臂训练agent ({model_type})...")
    
    # 创建环境
    env = SoftArmReachEnvironment(
        goal_threshold=0.15,
        max_steps=200
    )
    
    # 创建HER buffer
    her_buffer = create_soft_arm_her_buffer(
        joint_dim=6,
        spatial_dim=3,
        capacity=500000,
        threshold=0.15
    )
    
    # 创建SAC learner
    sac_learner = ContinuousSoftActorCritic(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512],
        batch_size=512,
        training_rounds=25 if model_type == '50step' else 1,
    )
    
    # 创建agent
    agent = PearlAgent(
        policy_learner=sac_learner,
        replay_buffer=her_buffer,
    )
    
    print(f"✅ Agent创建完成: {env.observation_space.shape[0]}D观测, {env.action_space.shape[0]}D动作")
    return env, agent


def demonstrate_soft_arm_performance(num_episodes=3, record_gif=True):
    """演示软体臂训练效果"""
    print("🎬 软体臂Pearl训练效果演示开始...")
    
    # 创建训练好的agent
    env, agent = create_trained_agent()
    
    # 创建可视化器
    visualizer = SoftArmVisualizer()
    
    if record_gif:
        visualizer.start_recording('soft_arm_pearl_demo.gif')
    
    total_successes = 0
    
    for episode in range(num_episodes):
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        
        # 重置环境
        obs, action_space = env.reset()
        agent.reset(obs, action_space)
        
        episode_reward = 0
        step = 0
        
        for step in range(200):  # 最大200步
            # 获取动作 (exploit模式，展示训练效果)
            action = agent.act(exploit=True)
            
            # 执行动作
            result = env.step(action)
            episode_reward += result.reward.item()
            
            # 更新可视化
            visualizer.update(
                observation=result.observation.cpu().numpy(),
                action=action.cpu().numpy(),
                reward=result.reward.item(),
                terminated=result.terminated.item(),
                step=step + 1,
                episode=episode + 1
            )
            
            # agent观察结果
            agent.observe(result)
            
            # 检查终止
            if result.terminated or result.truncated:
                success = result.terminated.item()
                total_successes += success
                
                status = "🎯 SUCCESS!" if success else "⏱️ TIMEOUT"
                print(f"   {status} - Step: {step+1}, Reward: {episode_reward:.2f}")
                break
            
            time.sleep(0.1)  # 控制演示速度
    
    # 停止录制
    if record_gif:
        visualizer.stop_recording()
    
    # 保存最终截图
    final_screenshot = visualizer.save_dir / 'final_performance.png'
    visualizer.fig.savefig(final_screenshot, dpi=300, bbox_inches='tight')
    
    # 关闭可视化器
    visualizer.close()
    
    # 总结
    success_rate = total_successes / num_episodes * 100
    print(f"\n🎉 演示完成!")
    print(f"📊 成功率: {success_rate:.1f}% ({total_successes}/{num_episodes})")
    print(f"💾 文件保存在: {visualizer.save_dir}")
    
    return success_rate


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='软体机械臂训练效果可视化演示')
    parser.add_argument('--episodes', type=int, default=3, help='演示episodes数量')
    parser.add_argument('--no-gif', action='store_true', help='不录制GIF')
    
    args = parser.parse_args()
    
    # 设置环境
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始演示
    success_rate = demonstrate_soft_arm_performance(
        num_episodes=args.episodes,
        record_gif=not args.no_gif
    )
    
    print(f"✨ 软体臂展示完成，成功率: {success_rate:.1f}%")