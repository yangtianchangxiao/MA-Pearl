import numpy as np
import torch
from env.environment import RobotArmEnv
from env.visualizer import RobotArmVisualizer
from sac_optimized.sac import SAC
import argparse
import os
from datetime import datetime

def evaluate_and_visualize(model_path, num_episodes=5, save_video=True):
    # 环境配置
    env_config = {
        'max_steps': 200,
        'catch_radius': 0.1,
        'perception_radius': 1.0,
        'dt': 0.02,
    }
    
    # 创建环境
    env = RobotArmEnv(env_config)
    
    # 创建可视化器
    visualizer = RobotArmVisualizer(video_path='eval_videos')
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建并加载智能体
    agent = SAC(
        env_fn=lambda: env,
        num_envs=1,
        observation_space_shape=env.observation_space.shape,
        action_space_shape=env.action_space.shape,
        hidden_dims=[256, 256],
        device=device
    )
    agent.load_state_dict(checkpoint['agent_state_dict'])
    
    # 评估统计
    total_reward = 0
    success_count = 0
    
    # 开始评估
    for episode in range(num_episodes):
        # 为每个episode创建新的视频文件
        if save_video:
            video_filename = f'episode_{episode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            visualizer.start_recording(filename=video_filename)
        
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        while not done:
            # 获取动作
            action = agent.select_action(obs, evaluate=True)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # 获取机械臂和球的状态用于可视化
            robot_positions = env.robot.get_joint_positions()
            ball_pos = env.ball_state[:3] if env.ball_state is not None else None
            
            # 更新可视化
            visualizer.update(
                robot_positions=robot_positions,
                ball_pos=ball_pos,
                true_ball_pos=ball_pos,  # 在这个环境中观测值就是真实值
                info={
                    'success': info.get('success', False),
                    'episode': episode + 1,
                    'step': step,
                    'reward': episode_reward
                }
            )
            
            # 每10步保存一个截图
            if step % 10 == 0:
                visualizer.save_screenshot(f'episode_{episode}_step_{step}.png')
            
            if terminated and info.get('success', False):
                success_count += 1
                print(f"Success! Episode reward: {episode_reward:.2f}")
            elif done:
                print(f"Episode finished. Reward: {episode_reward:.2f}")
            
            obs = next_obs
            step += 1
        
        if save_video:
            visualizer.stop_recording()
        
        total_reward += episode_reward
    
    # 打印总体统计
    print("\nEvaluation Results:")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    print(f"Success Rate: {success_count/num_episodes:.2%}")
    
    visualizer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--no-video', action='store_true', help='Disable video recording')
    args = parser.parse_args()
    
    evaluate_and_visualize(args.model, args.episodes, not args.no_video)
