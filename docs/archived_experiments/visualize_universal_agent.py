#!/usr/bin/env python3
"""
通用Pearl Agent可视化器 - 支持任意网络类型和环境
支持MLP、Graph网络，以及各种机械臂环境 (NDOF, SoftArm, VariableSoftArm等)
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器环境使用
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

# Pearl imports
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# Environment imports
from pearl.utils.instantiations.environments import (
    NDOFArmEnvironment,
    SoftArmReachEnvironment, 
    VariableSoftArmReachEnvironment
)

# Buffer imports
from pearl.utils.instantiations.environments.arm_her_factory import create_arm_her_buffer
from pearl.utils.instantiations.environments.soft_arm_her_factory import create_soft_arm_her_buffer
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer

# Graph network imports (conditional)
try:
    from pearl.neural_networks.sequential_decision_making.actor_networks import GraphActorNetwork
    from pearl.neural_networks.sequential_decision_making.q_value_networks import GraphQValueNetwork
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


class UniversalVisualizerConfig:
    """可视化配置类"""
    
    # 环境配置映射
    ENVIRONMENT_CONFIGS = {
        'ndof_3dof': {
            'class': NDOFArmEnvironment,
            'params': {'dof': 3, 'max_steps': 50, 'goal_threshold': 0.30},
            'buffer_factory': create_arm_her_buffer,
            'buffer_params': {'dof': 3, 'spatial_dim': 2, 'threshold': 0.30},
            'obs_parser': 'ndof',
            'visualizer': '2d'
        },
        'soft_arm_6dof': {
            'class': SoftArmReachEnvironment,
            'params': {'goal_threshold': 0.15, 'max_steps': 200},
            'buffer_factory': create_soft_arm_her_buffer,
            'buffer_params': {'joint_dim': 6, 'spatial_dim': 3, 'threshold': 0.15},
            'obs_parser': 'soft_arm',
            'visualizer': '3d'
        },
        'variable_soft_arm_6dof': {
            'class': VariableSoftArmReachEnvironment,
            'params': {
                'n_segments': 3, 'max_steps': 200, 'goal_threshold': 0.15,
                'segment_length_range': (0.168, 0.252), 'include_lengths_in_obs': True
            },
            'buffer_factory': create_variable_soft_arm_her_buffer,
            'buffer_params': {'joint_dim': 6, 'spatial_dim': 3, 'n_segments': 3, 'include_lengths_in_obs': True},
            'obs_parser': 'variable_soft_arm',
            'visualizer': '3d'
        }
    }


class UniversalAgentLoader:
    """通用Agent加载器"""
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """加载checkpoint文件"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
        
        print(f"📁 加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"✅ Checkpoint信息:")
        print(f"   Episode: {checkpoint.get('episode', 'N/A')}")
        print(f"   Success Rate: {checkpoint.get('success_rate', 'N/A'):.1f}%")
        return checkpoint
    
    @staticmethod
    def create_environment(env_type: str) -> Tuple[Any, Dict]:
        """创建环境"""
        if env_type not in UniversalVisualizerConfig.ENVIRONMENT_CONFIGS:
            raise ValueError(f"不支持的环境类型: {env_type}")
        
        config = UniversalVisualizerConfig.ENVIRONMENT_CONFIGS[env_type]
        env_class = config['class']
        env_params = config['params']
        
        print(f"🌍 创建环境: {env_type}")
        print(f"   参数: {env_params}")
        
        env = env_class(**env_params)
        return env, config
    
    @staticmethod
    def create_agent(env, env_config: Dict, network_type: str, config: Dict[str, Any]) -> PearlAgent:
        """创建Agent - 支持MLP和Graph网络"""
        print(f"🤖 创建Agent: {network_type}网络")
        
        # 创建action representation
        action_rep_module = IdentityActionRepresentationModule(
            max_number_actions=env.action_space.shape[0],
            representation_dim=env.action_space.shape[0]
        )
        
        # 创建HER buffer
        buffer_factory = env_config['buffer_factory']
        buffer_params = env_config['buffer_params'].copy()
        buffer_params['capacity'] = config.get('buffer_capacity', 500000)
        
        her_buffer = buffer_factory(**buffer_params)
        print(f"   HER Buffer: {buffer_params}")
        
        if network_type.lower() == 'graph':
            if not GRAPH_AVAILABLE:
                raise ImportError("Graph网络不可用，请检查导入")
            
            # 创建Graph网络
            actor = GraphActorNetwork(
                input_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                hidden_dims=config.get('actor_hidden_dims', [256, 256]),
                node_feature_dim=config.get('node_feature_dim', 8),
                num_graph_layers=config.get('num_graph_layers', 3),
                num_attention_heads=config.get('num_attention_heads', 4),
                use_kinematic_chain=config.get('use_kinematic_chain', True)
            )
            
            critic1 = GraphQValueNetwork(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                hidden_dims=config.get('critic_hidden_dims', [256, 256]),
                node_feature_dim=config.get('node_feature_dim', 8),
                num_graph_layers=config.get('num_graph_layers', 3),
                num_attention_heads=config.get('num_attention_heads', 4),
                use_kinematic_chain=config.get('use_kinematic_chain', True)
            )
            
            # SAC with Graph networks
            sac = ContinuousSoftActorCritic(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_network_instance=actor,
                critic_network_instance=critic1,
                action_representation_module=action_rep_module,
                training_rounds=config.get('training_rounds', 25),
                batch_size=config.get('batch_size', 512),
            )
            print(f"   Graph网络: {config.get('num_graph_layers', 3)}层, {config.get('num_attention_heads', 4)}头注意力")
            
        else:  # MLP network
            sac = ContinuousSoftActorCritic(
                state_dim=env.observation_space.shape[0],
                action_space=env.action_space,
                actor_hidden_dims=config.get('actor_hidden_dims', [512, 512]),
                critic_hidden_dims=config.get('critic_hidden_dims', [512, 512]),
                action_representation_module=action_rep_module,
                training_rounds=config.get('training_rounds', 25),
                batch_size=config.get('batch_size', 512),
            )
            print(f"   MLP网络: Actor {config.get('actor_hidden_dims', [512, 512])}, Critic {config.get('critic_hidden_dims', [512, 512])}")
        
        # 创建Pearl Agent
        agent = PearlAgent(
            policy_learner=sac,
            replay_buffer=her_buffer,
        )
        
        return agent


class UniversalObservationParser:
    """通用观测解析器"""
    
    @staticmethod
    def parse_observation(obs: torch.Tensor, parser_type: str) -> Dict[str, np.ndarray]:
        """解析不同类型环境的观测"""
        obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
        
        if parser_type == 'ndof':
            # NDOF: [joint_angles(3), achieved_goal(2), desired_goal(2)]
            return {
                'joint_angles': obs_np[:3],
                'achieved_goal': obs_np[3:5],
                'desired_goal': obs_np[5:7],
                'segment_lengths': np.ones(3),  # 默认长度
                'spatial_dim': 2
            }
        elif parser_type == 'soft_arm':
            # SoftArm: [joint_angles(6), achieved_goal(3), desired_goal(3)]
            return {
                'joint_angles': obs_np[:6],
                'achieved_goal': obs_np[6:9],
                'desired_goal': obs_np[9:12],
                'segment_lengths': np.array([0.21, 0.21, 0.21]),  # 固定长度
                'spatial_dim': 3
            }
        elif parser_type == 'variable_soft_arm':
            # VariableSoftArm: [joint_angles(6), segment_lengths(3), achieved_goal(3), desired_goal(3)]
            return {
                'joint_angles': obs_np[:6],
                'segment_lengths': obs_np[6:9],
                'achieved_goal': obs_np[9:12],
                'desired_goal': obs_np[12:15],
                'spatial_dim': 3
            }
        else:
            raise ValueError(f"不支持的观测解析类型: {parser_type}")


class UniversalVisualizer:
    """通用可视化器"""
    
    def __init__(self, visualizer_type: str, save_dir: str = 'visualization_output'):
        self.visualizer_type = visualizer_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 创建图形
        self.fig = plt.figure(figsize=(15, 10))
        if visualizer_type == '3d':
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim([-0.6, 0.6])
            self.ax.set_ylim([-0.6, 0.6])
            self.ax.set_zlim([0, 0.6])
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
        else:  # 2d
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim([-3, 3])
            self.ax.set_ylim([-3, 3])
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
        
        # 录制设置
        self.frames = []
        self.recording = False
        
        # 历史轨迹
        self.ee_trajectory = []
    
    def start_recording(self, filename: str = 'demo.gif'):
        """开始录制"""
        self.recording = True
        self.frames = []
        self.gif_filename = self.save_dir / filename
        print(f"🎬 开始录制: {self.gif_filename}")
    
    def stop_recording(self):
        """停止录制"""
        if self.recording:
            self.recording = False
            print(f"✅ 截图序列已保存到: {self.save_dir}")
    
    def compute_arm_positions(self, joint_angles: np.ndarray, segment_lengths: np.ndarray, spatial_dim: int) -> np.ndarray:
        """计算机械臂各段位置"""
        if spatial_dim == 2:
            # 2D机械臂 (NDOF)
            positions = np.zeros((len(joint_angles) + 1, 2))
            angles_cumsum = np.cumsum([0] + joint_angles.tolist())
            for i in range(1, len(positions)):
                positions[i, 0] = positions[i-1, 0] + np.cos(angles_cumsum[i])
                positions[i, 1] = positions[i-1, 1] + np.sin(angles_cumsum[i])
            return positions
        else:
            # 3D软体机械臂
            positions = []
            positions.append([0, 0, 0])  # 基座
            current_pos = np.array([0, 0, 0], dtype=float)
            
            # 每2个关节角度对应一段
            for i in range(0, len(joint_angles), 2):
                alpha = joint_angles[i]      # 弯曲角度
                beta = joint_angles[i+1] if i+1 < len(joint_angles) else 0  # 方向角
                segment_idx = i // 2
                segment_length = segment_lengths[segment_idx] if segment_idx < len(segment_lengths) else 0.21
                
                # 正向运动学计算
                dx = segment_length * np.cos(alpha) * np.cos(beta)
                dy = segment_length * np.cos(alpha) * np.sin(beta)
                dz = segment_length * np.sin(alpha)
                
                current_pos += [dx, dy, dz]
                positions.append(current_pos.copy())
            
            return np.array(positions)
    
    def update(self, parsed_obs: Dict, action: np.ndarray, reward: float, 
               terminated: bool, step: int, episode: int, info: Dict = None):
        """更新可视化"""
        self.ax.cla()
        
        joint_angles = parsed_obs['joint_angles']
        achieved_goal = parsed_obs['achieved_goal']
        desired_goal = parsed_obs['desired_goal']
        segment_lengths = parsed_obs['segment_lengths']
        spatial_dim = parsed_obs['spatial_dim']
        
        # 计算机械臂位置
        arm_positions = self.compute_arm_positions(joint_angles, segment_lengths, spatial_dim)
        
        if self.visualizer_type == '3d':
            self._update_3d(arm_positions, achieved_goal, desired_goal, 
                           action, reward, terminated, step, episode, info)
        else:
            self._update_2d(arm_positions, achieved_goal, desired_goal,
                           action, reward, terminated, step, episode, info)
        
        # 保存静态截图
        if self.recording and step % 10 == 0:  # 每10步保存一次
            screenshot_path = self.save_dir / f'episode_{episode}_step_{step:03d}.png'
            self.fig.savefig(screenshot_path, dpi=150, bbox_inches='tight')
    
    def _update_3d(self, arm_positions, achieved_goal, desired_goal, action, reward, terminated, step, episode, info):
        """3D可视化更新"""
        # 绘制机械臂
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i in range(len(arm_positions)-1):
            self.ax.plot([arm_positions[i,0], arm_positions[i+1,0]], 
                        [arm_positions[i,1], arm_positions[i+1,1]], 
                        [arm_positions[i,2], arm_positions[i+1,2]], 
                        color=colors[i % len(colors)], linewidth=4)
        
        # 关节点
        for i, pos in enumerate(arm_positions):
            color = colors[i % len(colors)]
            size = 120 if i == 0 else (100 if i == len(arm_positions)-1 else 80)
            marker = 's' if i == 0 else ('*' if i == len(arm_positions)-1 else 'o')
            self.ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color=color, s=size, marker=marker, 
                          edgecolors='black', linewidth=2)
        
        # 目标和末端执行器
        self.ax.scatter([desired_goal[0]], [desired_goal[1]], [desired_goal[2]], 
                       color='gold', s=200, marker='*', label='Target')
        self.ax.scatter([achieved_goal[0]], [achieved_goal[1]], [achieved_goal[2]], 
                       color='red', s=150, marker='o', label='End Effector')
        
        # 轨迹
        self.ee_trajectory.append(achieved_goal.copy())
        if len(self.ee_trajectory) > 50:
            self.ee_trajectory.pop(0)
        
        if len(self.ee_trajectory) > 1:
            traj = np.array(self.ee_trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r--', alpha=0.6, linewidth=2)
        
        # 设置坐标轴
        self.ax.set_xlim([-0.6, 0.6])
        self.ax.set_ylim([-0.6, 0.6])
        self.ax.set_zlim([0, 0.6])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 距离和状态
        distance = np.linalg.norm(achieved_goal - desired_goal)
        status = "🎯 SUCCESS!" if terminated else "🔄 Learning..."
        network_info = info.get('network_type', 'Unknown') if info else 'Unknown'
        
        title = f"{network_info}网络 Pearl训练效果 | Episode {episode}, Step {step}\n{status}"
        self.ax.set_title(title, fontsize=14, pad=20)
        
        # 信息文本
        info_text = f"""距离目标: {distance:.3f}m
奖励: {reward:.2f}
网络: {network_info}
动作: [{', '.join([f'{a:.2f}' for a in action[:3]])}...]"""
        
        self.ax.text2D(0.02, 0.98, info_text, transform=self.ax.transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def _update_2d(self, arm_positions, achieved_goal, desired_goal, action, reward, terminated, step, episode, info):
        """2D可视化更新"""
        # 绘制机械臂
        colors = ['white', 'red', 'blue', 'green']
        for i in range(len(arm_positions)-1):
            self.ax.plot([arm_positions[i, 0], arm_positions[i+1, 0]], 
                       [arm_positions[i, 1], arm_positions[i+1, 1]], 
                       color=colors[(i+1) % len(colors)], linewidth=6, solid_capstyle='round')
        
        # 关节点
        for i in range(len(arm_positions)):
            self.ax.scatter(arm_positions[i, 0], arm_positions[i, 1], 
                          c=colors[i % len(colors)], s=150, zorder=5, 
                          edgecolors='black', linewidth=2)
        
        # 目标圆和点
        circle = plt.Circle(desired_goal, 0.30, color='gold', alpha=0.7, 
                           linestyle='--', fill=False, linewidth=3)
        self.ax.add_patch(circle)
        self.ax.scatter(desired_goal[0], desired_goal[1], c='orange', s=200, 
                       zorder=6, marker='*', edgecolors='black', linewidth=2)
        
        # 设置坐标轴
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('black')
        
        # 状态信息
        distance = np.linalg.norm(achieved_goal - desired_goal)
        status = "✅ SUCCESS!" if terminated else "🔄 Learning..."
        network_info = info.get('network_type', 'Unknown') if info else 'Unknown'
        
        self.ax.text(-2.8, 2.7, status, fontsize=14, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor='green' if terminated else 'orange', alpha=0.8),
                    color='white')
        
        self.ax.text(2.8, 2.7, f"距离: {distance:.3f}m", fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.8),
                    color='white', ha='right')
        
        self.ax.text(0, -2.7, f"{network_info}网络 | Episode: {episode} | Step: {step}", 
                    fontsize=12, ha='center', fontweight='bold', color='white')
        
        self.ax.set_title(f"{network_info}网络 Pearl SAC+HER 机械臂演示", 
                         fontsize=16, fontweight='bold', color='white', pad=20)
    
    def close(self):
        """关闭可视化器"""
        if self.recording:
            self.stop_recording()
        plt.close(self.fig)


def demonstrate_agent(checkpoint_path: str, env_type: str, network_type: str, 
                     num_episodes: int = 3, record_gif: bool = True) -> float:
    """演示训练好的agent"""
    print("🎬 通用Pearl Agent可视化演示")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   环境类型: {env_type}")
    print(f"   网络类型: {network_type}")
    print("=" * 60)
    
    # 加载checkpoint和配置
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = UniversalAgentLoader.load_checkpoint(checkpoint_path)
            config = checkpoint.get('config', {})
        except Exception as e:
            print(f"⚠️  Checkpoint加载失败: {e}")
            print("⚠️  使用默认配置创建随机agent")
            checkpoint = {}
            config = {}
    else:
        print("⚠️  未指定checkpoint或文件不存在，创建随机agent演示")
        checkpoint = {}
        config = {}
    
    # 创建环境
    env, env_config = UniversalAgentLoader.create_environment(env_type)
    
    # 创建agent
    agent = UniversalAgentLoader.create_agent(env, env_config, network_type, config)
    
    # 加载权重
    try:
        if 'agent_state' in checkpoint:
            agent.policy_learner.load_state_dict(checkpoint['agent_state'])
            print(f"✅ 权重加载成功")
        elif 'policy_learner_state_dict' in checkpoint:
            agent.policy_learner.load_state_dict(checkpoint['policy_learner_state_dict'])
            print(f"✅ 权重加载成功 (legacy格式)")
        else:
            print("⚠️  未找到权重，使用随机初始化")
    except Exception as e:
        print(f"⚠️  权重加载失败: {e}")
        print("⚠️  使用随机初始化进行演示")
    
    # 创建可视化器
    visualizer = UniversalVisualizer(
        visualizer_type=env_config['visualizer'],
        save_dir=f'visualization_{env_type}_{network_type}'
    )
    
    if record_gif:
        screenshot_name = f'{env_type}_{network_type}_demo'
        visualizer.start_recording(screenshot_name)
    
    total_successes = 0
    parser_type = env_config['obs_parser']
    
    for episode in range(num_episodes):
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        
        # 重置环境和agent
        obs, action_space = env.reset()
        agent.reset(obs, action_space)
        
        episode_reward = 0
        max_steps = env_config['params'].get('max_steps', 200)
        
        for step in range(max_steps):
            # 获取动作 (exploit模式)
            action = agent.act(exploit=True)
            
            # 执行动作
            result = env.step(action)
            reward_value = result.reward.item() if torch.is_tensor(result.reward) else result.reward
            episode_reward += reward_value
            
            # 解析观测
            parsed_obs = UniversalObservationParser.parse_observation(
                result.observation, parser_type
            )
            
            # 更新可视化
            info = {
                'network_type': network_type,
                'env_type': env_type
            }
            visualizer.update(
                parsed_obs=parsed_obs,
                action=action.cpu().numpy(),
                reward=reward_value,
                terminated=result.terminated.item() if torch.is_tensor(result.terminated) else result.terminated,
                step=step + 1,
                episode=episode + 1,
                info=info
            )
            
            # Agent观察结果
            agent.observe(result)
            
            # 检查终止
            if result.terminated or result.truncated:
                success = result.terminated.item() if torch.is_tensor(result.terminated) else result.terminated
                total_successes += success
                
                status = "🎯 SUCCESS!" if success else "⏱️ TIMEOUT"
                print(f"   {status} - Step: {step+1}, Reward: {episode_reward:.2f}")
                break
            
            time.sleep(0.05)  # 控制演示速度
    
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


def main():
    parser = argparse.ArgumentParser(description='通用Pearl Agent可视化演示')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint文件路径')
    parser.add_argument('--env-type', type=str, required=True, 
                       choices=['ndof_3dof', 'soft_arm_6dof', 'variable_soft_arm_6dof'],
                       help='环境类型')
    parser.add_argument('--network-type', type=str, required=True,
                       choices=['mlp', 'graph'], help='网络类型')
    parser.add_argument('--episodes', type=int, default=3, help='演示episodes数量')
    parser.add_argument('--no-gif', action='store_true', help='不录制GIF')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始演示
    try:
        success_rate = demonstrate_agent(
            checkpoint_path=args.checkpoint,
            env_type=args.env_type,
            network_type=args.network_type,
            num_episodes=args.episodes,
            record_gif=not args.no_gif
        )
        
        print(f"\n✨ 演示完成，成功率: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()