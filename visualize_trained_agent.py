#!/usr/bin/env python3
"""
可视化训练好的Pearl SAC+HER agent
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import sys
import os
sys.path.insert(0, '/home/cx/MA-Pearl')

from pearl.utils.instantiations.environments import NDOFArmEnvironment
from pearl.utils.instantiations.environments.arm_her_factory import create_arm_her_buffer
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)


def create_trained_agent():
    """加载真正训练好的agent"""
    env = NDOFArmEnvironment(dof=3, max_steps=50, goal_threshold=0.30)
    
    # 创建HER buffer
    her_buffer = create_arm_her_buffer(
        dof=3,
        spatial_dim=2,
        capacity=10000,
        threshold=0.30
    )
    
    # 读取保存的配置
    model_path = "/home/cx/MA-Pearl/pearl_arm_results/pearl_arm_3dof_model.pt"
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    
    # 创建与训练时相同的SAC learner
    sac_learner = ContinuousSoftActorCritic(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        actor_hidden_dims=config['actor_hidden_dims'],
        critic_hidden_dims=config['critic_hidden_dims'],
        batch_size=config['batch_size'],
        training_rounds=config['training_rounds'],
    )
    
    # 创建Pearl agent
    agent = PearlAgent(
        policy_learner=sac_learner,
        replay_buffer=her_buffer,
    )
    
    # 加载训练好的权重
    print(f"   加载模型权重: {model_path}")
    agent.policy_learner.load_state_dict(checkpoint['policy_learner_state_dict'])
    print(f"   模型成功率: {checkpoint['final_success_rate']:.1f}%")
    print(f"   训练episodes: {checkpoint['total_episodes']}")
    
    return agent, env


def visualize_agent_episodes(agent, env, num_episodes=5, save_path="trained_agent_demo.gif"):
    """可视化训练好的agent执行多个episodes"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('black')
    
    # 存储所有episode的数据
    all_episodes_data = []
    
    print(f"🎬 收集 {num_episodes} 个episodes的数据...")
    
    for episode in range(num_episodes):
        episode_data = {
            'arm_positions': [],
            'goals': [],
            'distances': [],
            'actions': [],
            'success': False,
            'steps': 0
        }
        
        obs, action_space = env.reset()
        agent.reset(obs, action_space)
        
        # 提取初始状态信息
        joint_angles = obs[:3].cpu().numpy()
        achieved_goal = obs[3:5].cpu().numpy()  
        desired_goal = obs[5:7].cpu().numpy()
        
        episode_data['goals'] = desired_goal
        
        for step in range(env.max_steps):
            # 获取agent动作
            action = agent.act(exploit=True)  # 使用训练好的策略
            result = env.step(action)
            
            # 记录数据
            joint_angles = result.observation[:3].cpu().numpy()
            achieved_goal = result.observation[3:5].cpu().numpy()
            
            # 计算arm positions
            positions = np.zeros((4, 2))
            angles_cumsum = np.cumsum([0] + joint_angles.tolist())
            for i in range(1, 4):
                positions[i, 0] = positions[i-1, 0] + np.cos(angles_cumsum[i])
                positions[i, 1] = positions[i-1, 1] + np.sin(angles_cumsum[i])
            
            episode_data['arm_positions'].append(positions.copy())
            distance = np.linalg.norm(achieved_goal - desired_goal)
            episode_data['distances'].append(distance)
            episode_data['actions'].append(action.cpu().numpy())
            episode_data['steps'] = step + 1
            
            # Agent观察结果并学习
            agent.observe(result)
            
            if result.terminated:
                episode_data['success'] = True
                print(f"Episode {episode+1}: ✅ 成功! 步数: {step+1}")
                break
            elif result.truncated:
                print(f"Episode {episode+1}: ❌ 超时失败! 步数: {step+1}")
                break
        
        all_episodes_data.append(episode_data)
    
    print("🎨 生成动画...")
    
    # 动画制作
    total_frames = sum(len(ep['arm_positions']) for ep in all_episodes_data)
    current_episode = 0
    current_step = 0
    
    def animate(frame):
        nonlocal current_episode, current_step
        
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('black')
        
        # 找到当前帧对应的episode和step
        frame_count = 0
        for ep_idx, episode_data in enumerate(all_episodes_data):
            if frame_count + len(episode_data['arm_positions']) > frame:
                current_episode = ep_idx
                current_step = frame - frame_count
                break
            frame_count += len(episode_data['arm_positions'])
        
        episode_data = all_episodes_data[current_episode]
        
        if current_step < len(episode_data['arm_positions']):
            positions = episode_data['arm_positions'][current_step]
            distance = episode_data['distances'][current_step]
            
            # 绘制arm
            colors = ['white', 'red', 'blue', 'green'] 
            for i in range(3):
                ax.plot([positions[i, 0], positions[i+1, 0]], 
                       [positions[i, 1], positions[i+1, 1]], 
                       color=colors[i+1], linewidth=6, solid_capstyle='round')
            
            # 绘制关节
            for i in range(4):
                ax.scatter(positions[i, 0], positions[i, 1], 
                          c=colors[i], s=150, zorder=5, edgecolors='black', linewidth=2)
            
            # 绘制目标
            goal = episode_data['goals']
            circle = plt.Circle(goal, 0.30, color='gold', alpha=0.7, linestyle='--', 
                              fill=False, linewidth=3)
            ax.add_patch(circle)
            ax.scatter(goal[0], goal[1], c='orange', s=200, zorder=6, 
                      marker='*', edgecolors='black', linewidth=2)
            
            # 状态显示
            status = "⏱️ Time-Optimized Agent" if current_step < len(episode_data['arm_positions']) - 1 else \
                    ("✅ SUCCESS!" if episode_data['success'] else "❌ TIMEOUT")
            
            ax.text(-2.8, 2.7, status, fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='green' if episode_data['success'] else 'red', alpha=0.8),
                   color='white')
            
            ax.text(2.8, 2.7, f"距离: {distance:.3f}m", fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.8),
                   color='white', ha='right')
            
            # Episode和Step信息
            ax.text(0, -2.7, f"Episode: {current_episode+1}/{num_episodes} | Step: {current_step+1}/{episode_data['steps']}", 
                   fontsize=12, ha='center', fontweight='bold', color='white')
            
            # 动作信息
            if current_step < len(episode_data['actions']):
                action = episode_data['actions'][current_step]
                action_text = f"θ₁: {action[0]:+.3f}\\nθ₂: {action[1]:+.3f}\\nθ₃: {action[2]:+.3f}"
                ax.text(-2.8, -2.2, action_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='cyan', alpha=0.8),
                       color='black', va='top')
        
        ax.set_title("⏱️ Pearl SAC+HER 3DOF机械臂 - 时间优化奖励", fontsize=16, fontweight='bold', color='white', pad=20)
    
    # 创建动画
    ani = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                interval=200, blit=False, repeat=True)
    
    # 保存GIF
    print(f"💾 保存动画到 {save_path}...")
    ani.save(save_path, writer='pillow', fps=5, dpi=100)
    plt.close()
    
    return save_path


def main():
    print("🤖 Pearl SAC+HER 可视化")
    print("=" * 50)
    
    # 创建训练好的agent
    print("🏗️  加载训练好的agent...")
    agent, env = create_trained_agent()
    
    print("⚡ 使用真实训练好的Pearl SAC+HER模型")
    
    # 可视化
    gif_path = visualize_agent_episodes(agent, env, num_episodes=5, 
                                       save_path="pearl_time_optimized_demo.gif")
    
    print(f"✅ 可视化完成!")
    print(f"📁 GIF已保存: {gif_path}")
    print(f"🎬 包含5个episodes的训练后agent表现")


if __name__ == "__main__":
    main()