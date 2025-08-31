#!/usr/bin/env python3
"""
Stable-Baselines3 对照实验
使用相同的环境测试SAC+HER的标准实现
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch

# 导入我们的环境
from pearl.utils.instantiations.environments import NDOFArmEnvironment

class SB3ArmWrapper(gym.Env):
    """
    将我们的NDOFArmEnvironment包装成标准gym接口
    """
    def __init__(self, dof=3, max_steps=550, goal_threshold=0.30):
        self.env = NDOFArmEnvironment(dof=dof, max_steps=max_steps, goal_threshold=goal_threshold)
        self.episode_count = 0
        self.success_count = 0
        
        # 定义observation和action space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(dof,), dtype=np.float32
        )
        
        # 状态: [joint_angles, achieved_goal, desired_goal] - HER标准格式
        state_dim = dof + 4 + 4  # joints + achieved(2D) + desired(2D)，但需要适配HER
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(-np.inf, np.inf, shape=(dof,), dtype=np.float32),
            'achieved_goal': gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32), 
            'desired_goal': gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        })
        
    def reset(self, seed=None, **kwargs):
        obs, _ = self.env.reset(seed=seed)
        obs_np = obs.numpy()
        
        # 分解观测: [joint_angles(3), achieved_goal(2), desired_goal(2)] - 总共7维
        return {
            'observation': obs_np[:3],                    # joint angles
            'achieved_goal': obs_np[3:5],                # current position  
            'desired_goal': obs_np[5:7],                 # target position
        }, {}
    
    def step(self, action):
        # 将action转换为torch tensor
        action_tensor = torch.from_numpy(action.astype(np.float32))
        result = self.env.step(action_tensor)
        
        obs_np = result.observation.numpy()
        obs_dict = {
            'observation': obs_np[:3],
            'achieved_goal': obs_np[3:5], 
            'desired_goal': obs_np[5:7],
        }
        
        # HER需要基于achieved_goal vs desired_goal计算奖励
        achieved = obs_np[3:5]
        desired = obs_np[5:7] 
        distance = np.linalg.norm(achieved - desired)
        her_reward = 0.0 if distance <= self.env.goal_threshold else -1.0
        
        # 记录episode结束和成功
        if result.terminated or result.truncated:
            self.episode_count += 1
            if result.terminated:  # terminated=True意味着达成目标
                self.success_count += 1
            
            # 每10个episodes输出一次成功率
            if self.episode_count % 10 == 0:
                success_rate = (self.success_count / self.episode_count) * 100
                print(f"🎯 SB3 Episode {self.episode_count}: Success Rate = {success_rate:.1f}% ({self.success_count}/{self.episode_count})")
        
        return obs_dict, her_reward, result.terminated, result.truncated, {}
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """HER需要的奖励计算方法"""
        # 处理批量计算
        if len(achieved_goal.shape) == 2:  # 批量
            distances = np.linalg.norm(achieved_goal - desired_goal, axis=1)
            return np.where(distances <= self.env.goal_threshold, 0.0, -1.0)
        else:  # 单个样本
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return 0.0 if distance <= self.env.goal_threshold else -1.0

def test_sb3_baseline():
    """测试SB3基线性能"""
    print("🔬 Stable-Baselines3 基线测试")
    print("=" * 60)
    
    # 创建环境
    def make_env():
        return SB3ArmWrapper(dof=3, max_steps=550, goal_threshold=0.30)
    
    env = DummyVecEnv([make_env])
    
    print("环境信息:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # 测试环境
    obs = env.reset()
    print(f"  Initial obs type: {type(obs)}")
    print(f"  Initial obs shape: {len(obs) if isinstance(obs, (list, tuple)) else 'not sequence'}")
    if isinstance(obs, (list, tuple)) and len(obs) > 0:
        print(f"  Initial obs[0] keys: {obs[0].keys()}")
        print(f"  achieved_goal shape: {obs[0]['achieved_goal'].shape}")
        print(f"  desired_goal shape: {obs[0]['desired_goal'].shape}")
    
    # 配置HER + SAC
    print(f"\n配置 HER + SAC...")
    
    # 创建SAC模型配置 - 对标我们的Pearl配置
    model = SAC(
        'MultiInputPolicy',  # 支持Dict观察空间
        env,
        learning_rate=3e-4,
        batch_size=512,               # 对标我们的配置
        buffer_size=500000,           # 对标我们的配置
        learning_starts=50000,        # 对标我们的配置 (10%)
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy='future',
        ),
        verbose=1,
        device='cuda:0'
    )
    
    print("SB3配置:")
    print(f"  算法: HER + SAC")
    print(f"  Buffer size: 500,000")
    print(f"  Batch size: 512")
    print(f"  Learning starts: 50,000")
    print(f"  Device: cuda:0")
    
    # 训练 - 公平对比：完整550K步
    print(f"\n🚀 开始公平对照训练...")
    total_timesteps = 550 * 1000  # 1000 episodes = 550,000 timesteps
    print(f"   目标训练量: {total_timesteps:,} timesteps")
    print(f"   预估episodes: ~1000")
    
    # 评估回调
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./sb3_arm_best/', 
        log_path='./sb3_arm_logs/',
        eval_freq=5500,  # 每10个episodes评估
        n_eval_episodes=10,
        deterministic=True, 
        render=False,
        verbose=1
    )
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False
        )
        
        print(f"\n🎉 SB3训练完成!")
        
        # 最终评估 - 使用正确的方法
        print(f"进行最终评估...")
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100, deterministic=True)
        print(f"SB3最终表现:")
        print(f"  平均奖励: {mean_reward:.3f} ± {std_reward:.3f}")
        
        # 由于我们用的是{0, -1}奖励，成功率 ≈ (mean_reward + 1) * 100
        success_rate = max(0, (mean_reward + 1) * 100)
        print(f"  估算成功率: {success_rate:.1f}%")
        
    except KeyboardInterrupt:
        print(f"\n🛑 训练被中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        raise

if __name__ == "__main__":
    test_sb3_baseline()