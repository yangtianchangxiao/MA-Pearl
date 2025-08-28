import numpy as np
import torch
from env.environment import RobotArmEnv
from sac_optimized.sac import SAC
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from gymnasium.vector import AsyncVectorEnv
import time
from collections import defaultdict

def evaluate(env, agent, num_episodes=50):
    total_reward = 0
    success_count = 0
    reward_components = {
        'distance': 0,
        'angle': 0,
        'action': 0,
        'catch': 0
    }
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            episode_reward += reward
            
            # 累计各个奖励组成部分
            if 'reward_info' in info:
                for k, v in info['reward_info'].items():
                    if k in reward_components:
                        reward_components[k] += v
            
            if terminated and info.get('success', False):
                success_count += 1
                
            obs = next_obs
            
        total_reward += episode_reward
    
    # 计算平均值
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes
    avg_components = {k: v / num_episodes for k, v in reward_components.items()}
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'reward_components': avg_components
    }

def save_checkpoint(agent, episode, avg_reward, avg_success, checkpoint_dir="checkpoints"):
    """保存模型checkpoint
    
    Args:
        agent: SAC代理
        episode: 当前回合数
        avg_reward: 平均奖励
        avg_success_rate: 平均成功率
        checkpoint_dir: checkpoint保存目录
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建checkpoint
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.state_dict(),
        'avg_reward': avg_reward,
        'avg_success_rate': avg_success,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存最新的模型
    latest_path = os.path.join(checkpoint_dir, "sac_model_latest.pth")
    torch.save(checkpoint, latest_path)
    print(f"\nSaved latest model to {latest_path}")
    
    # 如果是最佳模型，也保存一份
    best_path = os.path.join(checkpoint_dir, "sac_model_best.pth")
    if not os.path.exists(best_path) or avg_reward > torch.load(best_path)['avg_reward']:
        torch.save(checkpoint, best_path)
        print(f"New best model! Saved to {best_path}")
    
    # 保存带时间戳的版本
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_path = os.path.join(
        checkpoint_dir, 
        f"sac_model_reward_{avg_reward:.0f}_{timestamp}.pth"
    )
    torch.save(checkpoint, timestamped_path)

def log_training_info(total_steps, episode_num, success_history, episode_rewards, env_config, max_steps, writer=None):
    """记录训练信息到控制台和tensorboard
    Args:
        total_steps: 当前总步数
        episode_num: 当前回合数
        success_history: 成功历史记录
        episode_rewards: 回合奖励历史
        env_config: 环境配置
        max_steps: 最大训练步数
        writer: tensorboard writer
    """
    # 计算统计信息
    recent_success = success_history[-100:] if success_history else []
    success_rate = sum(1 for x in recent_success if x) / len(recent_success) if recent_success else 0
    recent_rewards = episode_rewards[-100:] if episode_rewards else []
    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
    
    # 计算训练速度和预计剩余时间
    if not hasattr(log_training_info, 'last_time'):
        log_training_info.last_time = time.time()
        log_training_info.last_steps = total_steps
    else:
        current_time = time.time()
        time_diff = current_time - log_training_info.last_time
        steps_diff = total_steps - log_training_info.last_steps
        steps_per_second = steps_diff / time_diff if time_diff > 0 else 0
        remaining_steps = max_steps - total_steps
        estimated_time = remaining_steps / steps_per_second if steps_per_second > 0 else float('inf')
        
        # 更新时间戳和步数
        log_training_info.last_time = current_time
        log_training_info.last_steps = total_steps
    
    # 打印训练进度
    print(f"\nEpisode {episode_num} | Steps: {total_steps}/{max_steps} | Success: {success_rate:.1%}")
    print(f"Average Reward (last 100): {avg_reward:.2f}")
    
    if hasattr(log_training_info, 'last_time'):
        steps_per_second = (total_steps - log_training_info.last_steps) / (time.time() - log_training_info.last_time)
        remaining_time = (max_steps - total_steps) / steps_per_second if steps_per_second > 0 else float('inf')
        print(f"Training Speed: {steps_per_second:.1f} steps/s")
        print(f"Estimated Time Remaining: {remaining_time/3600:.1f} hours")
    
    # 如果有课程学习配置，打印课程状态
    if 'curriculum' in env_config:
        curr = env_config['curriculum']
        print("\nCurriculum Status:")
        print(f"Level: {curr.get('current_level', 0)}")
        if 'ball' in curr:
            ball_config = curr['ball']
            print(f"Ball Distance: {ball_config['distance_range'][1]:.2f}m")
            print(f"Ball Speed: {ball_config['speed_range'][1]:.2f}m/s")
        print(f"Max Steps: {curr.get('max_steps', 'N/A')}")
    
    # 记录到tensorboard
    if writer is not None:
        writer.add_scalar('Train/Success_Rate', success_rate, total_steps)
        writer.add_scalar('Train/Average_Reward', avg_reward, total_steps)
        writer.add_scalar('Train/Episode_Number', episode_num, total_steps)
        if hasattr(log_training_info, 'last_time'):
            writer.add_scalar('Train/Steps_Per_Second', steps_per_second, total_steps)
        
        # 记录课程学习状态
        if 'curriculum' in env_config:
            curr = env_config['curriculum']
            writer.add_scalar('Curriculum/Level', curr.get('current_level', 0), total_steps)
            if 'ball' in curr:
                ball_config = curr['ball']
                writer.add_scalar('Curriculum/Ball_Distance', ball_config['distance_range'][1], total_steps)
                writer.add_scalar('Curriculum/Ball_Speed', ball_config['speed_range'][1], total_steps)
            writer.add_scalar('Curriculum/Max_Steps', curr.get('max_steps', 0), total_steps)

def run_evaluation(eval_env, agent, total_steps, episode_num, best_eval_reward, writer=None):
    """运行评估并更新课程学习
    Args:
        eval_env: 评估环境
        agent: SAC智能体
        total_steps: 当前总步数
        episode_num: 当前回合数
        best_eval_reward: 历史最佳评估奖励
        writer: tensorboard writer
    Returns:
        new_best_reward: 更新后的最佳奖励
        curriculum_updated: 课程是否更新
    """
    print("\nStarting evaluation...")
    eval_result = evaluate(eval_env, agent, num_eval_episodes)
    
    # 记录评估结果
    if writer is not None:
        writer.add_scalar('Eval/Average Reward', eval_result['avg_reward'], total_steps)
        writer.add_scalar('Eval/Success Rate', eval_result['success_rate'], total_steps)
        for component, value in eval_result['reward_components'].items():
            writer.add_scalar(f'Eval/Reward/{component}', value, total_steps)
    
    # 保存最佳模型
    if eval_result['avg_reward'] > best_eval_reward:
        best_eval_reward = eval_result['avg_reward']
        save_checkpoint(agent, episode_num, eval_result['avg_reward'], eval_result['success_rate'])
    
    # 更新课程学习
    curriculum_updated = update_curriculum(
        eval_env,
        eval_result['success_rate'],
        eval_result['avg_reward'],
        total_steps,
        eval_result['avg_reward']
    )
    
    if curriculum_updated:
        print("\nCurriculum updated based on evaluation results")
    
    return best_eval_reward, curriculum_updated

def train_episode(envs, agent, obs, episode_info, writer=None):
    """训练单个回合
    Args:
        envs: 训练环境
        agent: SAC智能体
        obs: 初始观察
        episode_info: 回合信息字典
        writer: tensorboard writer
    Returns:
        total_reward: 回合总奖励
        steps: 回合步数
        success: 是否成功
        next_obs: 最终观察
    """
    episode_reward = 0
    episode_steps = 0
    done = False
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = envs.step(action)
        
        # 使用any()来判断是否有任何环境结束
        done = terminated.any() or truncated.any()
        
        # 记录奖励组成部分
        if 'reward_info' in info:
            for k, v in info['reward_info'].items():
                if not k.startswith('_'):
                    episode_info['reward_components'][k] += v[0]
        
        # 检查任意环境是否成功
        if terminated.any() and info.get('success', np.array([False] * agent.num_envs)).any():
            episode_info['success'] = True
        
        # 存储所有环境的经验
        for env_idx in range(len(obs)):
            agent.replay_buffer.add(
                obs[env_idx],
                action[env_idx],
                reward[env_idx],
                next_obs[env_idx],
                done
            )
        
        obs = next_obs
        reward_mean = reward.mean().item()
        episode_reward += reward_mean
        episode_steps += 1
    
    return episode_reward, episode_steps, episode_info['success'], next_obs

def update_curriculum(env, success_rate, mean_reward, total_steps, eval_mean_reward):
    """根据成功率和奖励更新课程学习参数"""
    curr = env.env.curriculum
    step_curr = env.env.curriculum_config['step_curriculum']
    updated = False
    
    # 基于成功率的课程学习
    if success_rate >= env.env.curriculum_config['success_threshold']:
        curr_dist = curr['ball']['distance_range'][1]
        curr_speed = curr['ball']['speed_range'][1]
        
        # 确保在当前难度停留足够长的时间
        if total_steps % step_curr['min_steps_per_level'] == 0:
            # 逐步增加难度
            if curr_dist < env.env.curriculum_config['max_distance']:
                new_dist = min(curr_dist + env.env.curriculum_config['distance_increment'], 
                             env.env.curriculum_config['max_distance'])
                curr['ball']['distance_range'] = [env.env.curriculum_config['initial_distance'], new_dist]
                updated = True
                
            if curr_speed < env.env.curriculum_config['max_speed']:
                new_speed = min(curr_speed + env.env.curriculum_config['speed_increment'],
                              env.env.curriculum_config['max_speed'])
                curr['ball']['speed_range'] = [env.env.curriculum_config['initial_speed'], new_speed]
                updated = True
            
            if updated:
                curr['current_level'] += 1
        
    # 基于步数的课程学习
    if step_curr['enabled']:
        # 记录评估奖励
        if not hasattr(env.env, 'eval_history'):
            env.env.eval_history = {
                'max_reward': float('-inf'),
                'evals_since_max': 0,
                'eval_rewards': []
            }
        
        # 更新历史最大评估奖励
        eval_history = env.env.eval_history
        eval_history['eval_rewards'].append(eval_mean_reward)
        
        if eval_mean_reward > eval_history['max_reward']:
            eval_history['max_reward'] = eval_mean_reward
            eval_history['evals_since_max'] = 0
        else:
            eval_history['evals_since_max'] += 1
        
        # 如果连续几次评估都没有超过历史最大奖励，增加步数
        if (eval_history['evals_since_max'] >= step_curr['plateau_evals'] and
            curr['max_steps'] < step_curr['final_max_steps']):
            # 增加最大步数
            curr['max_steps'] = min(
                curr['max_steps'] + step_curr['step_increment'],
                step_curr['final_max_steps']
            )
            print(f"\nIncreasing max steps to {curr['max_steps']} (no improvement for {step_curr['plateau_evals']} evaluations)")
            print(f"Current max eval reward: {eval_history['max_reward']:.2f}")
            
            # 重置计数器，给新的步数限制一个机会
            eval_history['evals_since_max'] = 0
            updated = True
        
    return updated

def main():
    # 环境配置
    env_config = {
        'max_steps': 200,           # 每个回合的最大步数
        'catch_radius': 0.05,       # 抓取半径（5厘米）
        'perception_radius': 3.0,   # 感知半径（3米）
        'dt': 0.02,                # 时间步长（20毫秒）
        
        # 奖励设置
        'reward_catch': 100.0,      # 成功抓住的奖励
        'reward_distance': -0.5,    # 距离惩罚
        'reward_action': -0.05,     # 动作惩罚
        'reward_angle': -0.5,       # 角度惩罚
        'reward_improvement': {     # 改善奖励
            'distance': 1.0,        # 距离改善奖励
            'angle': 0.5           # 角度改善奖励
        },
        
        # 抓取判定
        'catch': {
            'radius': 0.05,         # 抓取半径（5厘米）
            'angle_tolerance': 0.3,  # 允许的最大角度偏差（约17度）
            'min_angle': 1.4,       # 最小垂直角度（约80度）
            'max_angle': 1.7,       # 最大垂直角度（约100度）
        },
        
        # 噪声设置（模拟现实世界的不确定性）
        'noise': {
            'action': 0.02,         # 动作噪声（2%）
            'observation': 0.005,    # 观测噪声（5毫米）
            'ball_init': 0.01,      # 球初始状态噪声（1厘米）
            'ball_vel': 0.05        # 球速度噪声（5%）
        },
        
        # 球的参数（模拟现实物理）
        'ball': {
            'g': 9.81,              # 重力加速度（米/秒²）
            'pos_range': {          # 初始位置范围（米）
                'x': [-1.0, -0.8],  # 距离机器人基座0.8-1米
                'y': [-0.3, 0.3],   # 左右30厘米范围
                'z': [1.0, 1.2]     # 高度1-1.2米
            },
            'vel_range': {          # 初始速度范围（米/秒）
                'x': [2.0, 3.0],    # 向机器人方向的速度
                'y': [-0.5, 0.5],   # 左右速度分量
                'z': [-0.5, 0.0]    # 略微向下的速度分量
            }
        },
        
        # 球体投掷器的配置
        'ball_config': {
            'g': 9.81,              # 重力加速度
            'pos_range': {
                'x': [-1.0, -0.8],  # 距离机器人基座0.8-1米
                'y': [-0.3, 0.3],   # 左右30厘米范围
                'z': [1.0, 1.2]     # 高度1-1.2米
            },
            'throw_config': {
                'speed': [2.0, 3.0],     # 出射速度范围 (m/s)
                'angle_h': [-15, 15],    # 水平角度范围 (度)
                'angle_v': [0, -15]      # 垂直角度范围 (度)
            }
        },
        
        # 课程学习状态
        'curriculum': {
            'enabled': True,
            'current_level': 0,
            'ball': {
                'distance_range': [0.5, 0.5],  # 初始距离范围
                'speed_range': [1.0, 1.0],     # 初始速度范围
                'height_range': [0.8, 0.8]     # 初始高度范围
            },
            'max_steps': 100,               # 初始最大步数
            'success_history': [],          # 成功历史记录
            'reward_history': [],           # 奖励历史记录
            'evaluation_counter': 0,        # 评估计数器
            'plateau_counter': 0,           # 性能平台计数器
            'best_success_rate': 0.0        # 最佳成功率
        },
        
        # 步数课程学习配置
        'step_curriculum': {
            'initial_steps': 100,           # 初始步数
            'max_steps': 500,              # 最大步数
            'final_max_steps': 500,        # 最终最大步数
            'step_increment': 20,          # 每次增加的步数
            'success_threshold': 0.8,      # 提升难度的成功率阈值
            'window_size': 50,             # 评估窗口大小
            'min_steps_per_level': 50000   # 每个难度级别的最小训练步数
        }
    }
    
    # 课程学习参数
    curriculum_config = {
        'initial_distance': 0.3,    # 初始球距离（米）
        'max_distance': 0.8,        # 最大球距离（米）
        'initial_speed': 0.5,       # 初始球速度（米/秒）
        'max_speed': 2.0,           # 最大球速度（米/秒）
        'distance_increment': 0.1,   # 每次增加的距离
        'speed_increment': 0.3,     # 每次增加的速度
        'success_threshold': 0.7,    # 进入下一阶段的成功率阈值
        'evaluation_window': 20,     # 评估窗口大小
        
        # 步数课程学习参数
        'step_curriculum': {
            'enabled': True,
            'initial_max_steps': 100,    # 初始最大步数
            'final_max_steps': 200,     # 最终最大步数
            'step_increment': 25,       # 每次增加的步数
            'plateau_evals': 5,         # 多少次评估没有提升就增加步数
            'min_steps_per_level': 50000  # 每个难度级别的最小步数
        }
    }
    
    # SAC配置
    sac_config = {
        'num_envs': 16,           # 增加并行环境数量
        'hidden_dims': [256, 256],
        'batch_size': 1024,       # 增加batch size
        'buffer_size': 1000000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # 优化器配置
        'optimizer': {
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'alpha_lr': 3e-4,
        },
        
        # 训练配置
        'gamma': 0.99,            # 折扣因子
        'tau': 0.005,             # 目标网络软更新系数
        'alpha': 0.2,             # 温度参数的初始值
        'reward_scale': 1.0,      # 奖励缩放
        'gradient_clip': 0.5,     # 梯度裁剪
        
        # 学习率调度
        'lr_schedule': {
            'enabled': True,
            'warmup_steps': 10000,
            'decay_steps': 1000000,
            'final_ratio': 0.1
        },
        
        # 目标熵
        'target_entropy': None,  # 如果为None，将自动设置为 -dim(A)
        
        # 训练稳定性
        'polyak_update_interval': 2,  # 目标网络更新间隔
        'normalize_observations': True,
        'normalize_rewards': False
    }
    
    # 课程学习参数
    curriculum_config = {
        'initial_distance': 0.3,    # 初始球距离（米）
        'max_distance': 0.8,        # 最大球距离（米）
        'initial_speed': 0.5,       # 初始球速度（米/秒）
        'max_speed': 2.0,           # 最大球速度（米/秒）
        'distance_increment': 0.1,   # 每次增加的距离
        'speed_increment': 0.3,     # 每次增加的速度
        'success_threshold': 0.7,    # 进入下一阶段的成功率阈值
        'evaluation_window': 20,     # 评估窗口大小
        
        # 步数课程学习参数
        'step_curriculum': {
            'enabled': True,
            'initial_max_steps': 100,    # 初始最大步数
            'final_max_steps': 200,     # 最终最大步数
            'step_increment': 25,       # 每次增加的步数
            'plateau_evals': 5,         # 多少次评估没有提升就增加步数
            'min_steps_per_level': 50000  # 每个难度级别的最小步数
        }
    }
    
    # 创建环境和智能体
    def make_env():
        return RobotArmEnv(env_config)
    
    # 使用异步向量化环境
    envs = AsyncVectorEnv([lambda: make_env() for _ in range(sac_config['num_envs'])])
    
    # 获取环境信息
    observation_space_shape = envs.single_observation_space.shape
    action_space_shape = envs.single_action_space.shape
    
    # 创建 TensorBoard writer
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'sac_optimized_{current_time}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log directory: {log_dir}")
    
    # 创建SAC智能体
    agent = SAC(
        env_fn=make_env,
        num_envs=sac_config['num_envs'],
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
        hidden_dims=sac_config['hidden_dims'],
        buffer_size=sac_config['buffer_size'],
        batch_size=sac_config['batch_size'],
        device=sac_config['device'],
        gamma=sac_config['gamma'],
        tau=sac_config['tau'],
        alpha=sac_config['alpha'],
        target_entropy=sac_config['target_entropy'],
        actor_lr=sac_config['optimizer']['actor_lr'],
        critic_lr=sac_config['optimizer']['critic_lr'],
        alpha_lr=sac_config['optimizer']['alpha_lr'],
        gradient_clip=sac_config['gradient_clip'],
        tensorboard_writer=writer,
        target_update_interval=sac_config['polyak_update_interval']
    )
    
    # 训练参数
    max_steps = int(1e6)
    eval_interval = 10000
    num_eval_episodes = 10
    
    # Warmup 参数 (使用相同步数便于管理)
    warmup_steps = 50000  # 总预训练步数
    random_steps = 10000  # 与warmup_steps相同
    
    # 更新环境配置
    env_config.update({
        'curriculum': {
            'enabled': True,
            'current_level': 0,
            'ball': {
                'distance_range': [curriculum_config['initial_distance'], curriculum_config['initial_distance']],
                'speed_range': [curriculum_config['initial_speed'], curriculum_config['initial_speed']]
            },
            'max_steps': curriculum_config['step_curriculum']['initial_max_steps']  # 添加最大步数限制
        },
        'success_history': [],  # 用于跟踪成功率
        'reward_history': []    # 用于跟踪奖励
    })
    
    try:
        # Warmup阶段
        print("Starting warmup phase...")
        steps = 0
        while steps < warmup_steps:
            obs, _ = envs.reset()
            dones = np.zeros(sac_config['num_envs'], dtype=bool)
            episode_steps = 0
            
            while not dones.all() and steps < warmup_steps:
                # 为每个环境生成随机动作
                actions = np.array([envs.single_action_space.sample() for _ in range(sac_config['num_envs'])])
                
                # 执行动作
                next_obs, rewards, terminated, truncated, infos = envs.step(actions)
                
                # 合并两种结束状态
                dones = np.logical_or(terminated, truncated)
                
                # 为每个环境添加经验
                for i in range(sac_config['num_envs']):
                    agent.replay_buffer.add(
                        obs[i].copy(),  # 确保数据是独立的副本
                        actions[i].copy(),
                        float(rewards[i]),  # 确保reward是标量
                        next_obs[i].copy(),
                        bool(dones[i])  # 确保done是布尔值
                    )
                
                obs = next_obs
                steps += sac_config['num_envs']  # 计数所有环境的步数
                episode_steps += 1
                
                if steps % 1000 == 0:
                    print(f"Warmup progress: {steps}/{warmup_steps} steps")
        
        print("Warmup phase completed. Starting training...")
        
        # 开始正式训练
        success_history = [[] for _ in range(sac_config['num_envs'])]  # 为每个环境创建独立的历史记录
        episode_rewards = [[] for _ in range(sac_config['num_envs'])]
        
        # 设置学习率调度器
        def lr_scheduler(step):
            if step % 1000 == 0:
                progress = min(step / optimizer_config['actor_lr_schedule']['decay_steps'], 1.0)
                new_actor_lr = optimizer_config['actor_lr_schedule']['initial'] + \
                    (optimizer_config['actor_lr_schedule']['final'] - 
                     optimizer_config['actor_lr_schedule']['initial']) * progress
                new_critic_lr = optimizer_config['critic_lr_schedule']['initial'] + \
                    (optimizer_config['critic_lr_schedule']['final'] - 
                     optimizer_config['critic_lr_schedule']['initial']) * progress
                
                for param_group in agent.actor_optim.param_groups:
                    param_group['lr'] = new_actor_lr
                for param_group in agent.critic1_optim.param_groups:
                    param_group['lr'] = new_critic_lr
                for param_group in agent.critic2_optim.param_groups:
                    param_group['lr'] = new_critic_lr
        
        # 注册回调函数
        agent.lr_scheduler = lr_scheduler
        
        # 训练循环
        total_steps = 0
        episode_num = 0
        best_eval_reward = float('-inf')
        eval_history = []
        
        while total_steps < max_steps:
            obs, _ = envs.reset()
            episode_info = {
                'reward_components': defaultdict(float),
                'success': False
            }
            episode_reward = 0  # 确保在每个episode开始时重置
            episode_steps = 0
            done = False
            
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = envs.step(action)
                
                # 使用any()来判断是否有任何环境结束
                done = terminated.any() or truncated.any()
                
                # 记录奖励组成部分
                if 'reward_info' in info:
                    for k, v in info['reward_info'].items():
                        # 跳过以下划线开头的键
                        if not k.startswith('_'):
                            episode_info['reward_components'][k] += v[0]  # 只取第一个环境的值
                
                # 检查任意环境是否成功
                if terminated.any() and info.get('success', np.array([False] * agent.num_envs)).any():
                    episode_info['success'] = True
                
                # 存储所有环境的经验
                for env_idx in range(len(obs)):
                    agent.replay_buffer.add(
                        obs[env_idx],
                        action[env_idx],
                        reward[env_idx],
                        next_obs[env_idx],
                        done
                    )
                
                obs = next_obs
                reward_mean = reward.mean().item()  # 确保获取标量值
                episode_reward += reward_mean  # 累加平均奖励
                episode_steps += 1
                total_steps += 1
            
            episode_num += 1
            
            # 记录训练信息
            if total_steps % 1000 == 0:
                log_training_info(
                    total_steps, episode_num,
                    success_history, episode_rewards,
                    env_config, max_steps, writer
                )
            
            # 评估和课程学习更新
            if total_steps % eval_interval == 0:
                eval_env = make_env()  # 创建单独的评估环境
                best_eval_reward, _ = run_evaluation(
                    eval_env, agent, total_steps,
                    episode_num, best_eval_reward, writer
                )
                eval_env.close()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        envs.close()
        if writer is not None:
            writer.close()

if __name__ == "__main__":
    import setproctitle
    # 设置一个容易识别的进程名称
    setproctitle.setproctitle("RobotArm_SAC")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # 关闭环境和 TensorBoard writer
        pass
