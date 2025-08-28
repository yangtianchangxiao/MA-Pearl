#!/usr/bin/env python3
"""
简化的软体臂演示 - 快速展示训练效果
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from pathlib import Path

# Pearl imports
from pearl.utils.instantiations.environments import SoftArmReachEnvironment
from pearl.utils.instantiations.environments.soft_arm_her_factory import create_soft_arm_her_buffer
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)


def visualize_soft_arm_simple(joint_angles, segment_length=0.21):
    """简化的软体臂可视化 - 支持任意节数"""
    positions = []
    positions.append([0, 0, 0])  # 基座
    
    current_pos = np.array([0, 0, 0], dtype=float)
    n_segments = len(joint_angles) // 2  # 每节2个自由度
    
    # 计算每段的位置 (简化版正向运动学)
    for i in range(n_segments):
        alpha = joint_angles[i*2]      # 弯曲角度
        beta = joint_angles[i*2+1]     # 方向角
        
        # 每段向前延伸
        dx = segment_length * np.cos(alpha) * np.cos(beta)
        dy = segment_length * np.cos(alpha) * np.sin(beta)
        dz = segment_length * np.sin(alpha)
        
        current_pos += [dx, dy, dz]
        positions.append(current_pos.copy())
    
    return np.array(positions)


def create_simple_agent(n_segments=3):
    """创建简化的agent用于演示"""
    print(f"Creating {n_segments}-Segment Soft Arm Agent...")
    
    env = SoftArmReachEnvironment(n_segments=n_segments, goal_threshold=0.15, max_steps=200)
    
    her_buffer = create_soft_arm_her_buffer(
        joint_dim=env.dof, spatial_dim=3, capacity=10000, threshold=0.15
    )
    
    sac_learner = ContinuousSoftActorCritic(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        actor_hidden_dims=[512, 512],
        critic_hidden_dims=[512, 512],
        batch_size=256,
        training_rounds=25,
    )
    
    agent = PearlAgent(policy_learner=sac_learner, replay_buffer=her_buffer)
    
    return env, agent


def demonstrate_episode():
    """演示一个episode"""
    print("Soft Arm Pearl Demo Starting...")
    
    env, agent = create_simple_agent()
    save_dir = Path('soft_arm_demo')
    save_dir.mkdir(exist_ok=True)
    
    # 重置环境
    obs, action_space = env.reset()
    agent.reset(obs, action_space)
    
    episode_data = []
    episode_reward = 0
    
    for step in range(200):
        # 获取动作
        action = agent.act(exploit=True)
        
        # 执行动作
        result = env.step(action)
        episode_reward += result.reward.item()
        
        # 保存数据
        observation = result.observation.cpu().numpy()
        joint_angles = observation[:6]
        achieved_goal = observation[6:9]
        desired_goal = observation[9:12]
        
        episode_data.append({
            'step': step + 1,
            'joint_angles': joint_angles,
            'achieved_goal': achieved_goal,
            'desired_goal': desired_goal,
            'action': action.cpu().numpy(),
            'reward': result.reward.item(),
            'distance': np.linalg.norm(achieved_goal - desired_goal),
            'terminated': result.terminated.item()
        })
        
        # agent观察
        agent.observe(result)
        
        # 每20步保存一张图
        if step % 20 == 0:
            create_visualization(episode_data[-1], save_dir, step)
        
        # 检查终止
        if result.terminated or result.truncated:
            success = result.terminated.item()
            status = "SUCCESS!" if success else "TIMEOUT"
            print(f"{status} - Step: {step+1}, Reward: {episode_reward:.2f}")
            
            # 保存最终图像
            create_visualization(episode_data[-1], save_dir, f"final")
            break
    
    # 创建轨迹图
    create_trajectory_plot(episode_data, save_dir)
    
    success_rate = int(result.terminated.item()) * 100
    print(f"Demo Complete! Success Rate: {success_rate}%")
    print(f"Files saved in: {save_dir}")
    
    return episode_data


def create_visualization(data, save_dir, step):
    """创建单个可视化图像"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算软体臂形状
    arm_positions = visualize_soft_arm_simple(data['joint_angles'])
    
    # 绘制软体臂
    colors = ['blue', 'green', 'red', 'orange']
    for i in range(len(arm_positions)-1):
        ax.plot([arm_positions[i,0], arm_positions[i+1,0]], 
                [arm_positions[i,1], arm_positions[i+1,1]], 
                [arm_positions[i,2], arm_positions[i+1,2]], 
                color=colors[i % len(colors)], linewidth=4)
    
    # 绘制关节点
    for i, pos in enumerate(arm_positions):
        size = 120 if i == 0 else (100 if i == len(arm_positions)-1 else 80)
        marker = 's' if i == 0 else ('*' if i == len(arm_positions)-1 else 'o')
        ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                  color=colors[i % len(colors)], s=size, marker=marker)
    
    # 绘制目标
    ax.scatter([data['desired_goal'][0]], [data['desired_goal'][1]], [data['desired_goal'][2]], 
              color='gold', s=200, marker='*', label='Target')
    
    # 绘制当前末端位置
    ax.scatter([data['achieved_goal'][0]], [data['achieved_goal'][1]], [data['achieved_goal'][2]], 
              color='red', s=150, marker='o', label='End Effector')
    
    # 设置坐标轴
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([0, 0.6])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 标题和信息
    status = "SUCCESS!" if data['terminated'] else "Learning..."
    title = f"Soft Arm Pearl Demo - Step {data['step']} - {status}"
    ax.set_title(title)
    
    # 添加信息
    info_text = f"Distance: {data['distance']:.3f}m\nReward: {data['reward']:.2f}"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax.legend()
    
    # 保存
    if isinstance(step, str):
        filename = save_dir / f'soft_arm_step_{step}.png'
    else:
        filename = save_dir / f'soft_arm_step_{step:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def create_trajectory_plot(episode_data, save_dir):
    """创建轨迹和性能图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    steps = [d['step'] for d in episode_data]
    distances = [d['distance'] for d in episode_data]
    rewards = [d['reward'] for d in episode_data]
    
    # 距离变化
    ax1.plot(steps, distances, 'b-', linewidth=2)
    ax1.axhline(y=0.15, color='r', linestyle='--', label='Success Threshold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Distance to Target (m)')
    ax1.set_title('Distance Evolution')
    ax1.legend()
    ax1.grid(True)
    
    # 奖励变化
    ax2.plot(steps, rewards, 'g-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Evolution')
    ax2.grid(True)
    
    # 3D轨迹
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ee_positions = np.array([d['achieved_goal'] for d in episode_data])
    target_pos = episode_data[0]['desired_goal']
    
    ax3.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b-', linewidth=2, label='EE Trajectory')
    ax3.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], color='gold', s=200, marker='*', label='Target')
    ax3.scatter([ee_positions[0, 0]], [ee_positions[0, 1]], [ee_positions[0, 2]], color='green', s=100, marker='o', label='Start')
    ax3.scatter([ee_positions[-1, 0]], [ee_positions[-1, 1]], [ee_positions[-1, 2]], color='red', s=100, marker='o', label='End')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('End Effector Trajectory')
    ax3.legend()
    
    # 关节角度变化
    joint_angles = np.array([d['joint_angles'] for d in episode_data])
    n_joints = joint_angles.shape[1]
    for i in range(n_joints):
        ax4.plot(steps, joint_angles[:, i], label=f'Joint {i+1}')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Joint Angle (rad)')
    ax4.set_title('Joint Angles Evolution')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 演示
    episode_data = demonstrate_episode()
    
    print("Soft Arm Demo Complete!")