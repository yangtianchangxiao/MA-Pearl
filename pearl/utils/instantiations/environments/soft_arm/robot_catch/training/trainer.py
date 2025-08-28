from collections import defaultdict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os

class Trainer:
    def __init__(self, agent, envs, logger, evaluator, curriculum_manager, config):
        self.agent = agent
        self.envs = envs
        self.logger = logger
        self.evaluator = evaluator
        self.curriculum_manager = curriculum_manager
        self.config = config
        
        # 创建模型保存目录
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"\n创建模型保存目录: {self.checkpoint_dir}")
        
        # 先将环境引用传递给课程学习管理器
        self.curriculum_manager.envs = self.envs
        
        # 获取当前课程配置
        current_config = self.curriculum_manager.get_current_config()
        print("\n初始课程配置：")
        print(current_config)
        
        # 更新所有训练环境的配置
        if hasattr(self.envs, 'env_fns'):
            print("\n更新训练环境配置...")
            for env_fn in self.envs.env_fns:
                env = env_fn()
                env.update_curriculum_config(current_config)
                env.close()  # 确保环境被正确关闭
        
        # 创建评估环境并更新其配置
        print("\n创建并更新评估环境...")
        self.eval_env = self.envs.env_fns[0]()
        self.eval_env.update_curriculum_config(current_config)
        
        print("\nTrainer初始化完成")
        print(f"当前课程阶段：{self.curriculum_manager.current_stage}")
        print(f"评估环境reward_scale：{self.eval_env.config['reward_scale']}")
        
        # 确保配置中包含必要的��数
        self.max_steps = config.get('max_steps', 1_000_000)
        self.eval_interval = config.get('eval_interval', 10000)
        self.log_interval = config.get('log_interval', 1000)
        self.num_updates = config.get('num_updates', 50)
        self.update_interval = config.get('update_interval', 50)
        self.warmup_steps = config.get('warmup_steps', 10000)
        
        # 添加经验回放缓冲区统计
        self.buffer_stats = {
            'size': 0,
            'transitions_per_step': 0,
            'updates_per_step': 0
        }
        
        # 训练状态
        self.total_steps = 0
        self.episode_num = 0
        self.best_eval_reward = float('-inf')
        self.success_history = []
        self.episode_rewards = []
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(comment=f"_SAC_RobotArm")
    
    def __del__(self):
        """析构函数，确保环境被正确关闭"""
        if hasattr(self, 'eval_env'):
            self.eval_env.close()
    
    def run_evaluation(self):
        """运行评估"""
        print("\nRunning evaluation...")
        eval_results = self.evaluator.evaluate(self.eval_env, self.agent)
        
        # 从eval_stats字典中获取结果
        mean_reward = eval_results['avg_reward']
        success_rate = eval_results['success_rate']
        
        # 记录评估结果
        self.logger.log_evaluation_results(
            self.total_steps,
            mean_reward,
            success_rate,
            eval_results.get('reward_components', {})
        )
        
        # 更新最佳奖励并保存模型
        if mean_reward > self.best_eval_reward:
            self.best_eval_reward = mean_reward
            print(f"\nNew best evaluation reward: {self.best_eval_reward:.2f}")
            # 保存最佳模型
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            self.agent.save_model(best_model_path)
            print(f"Saved best model to: {best_model_path}")
        
        # 保存最新的评估模型
        latest_model_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        self.agent.save_model(latest_model_path)
        print(f"Saved latest model to: {latest_model_path}")
        
        return mean_reward, success_rate
    
    def train(self):
        """运行训练循环"""
        obs, _ = self.envs.reset()
        training_iter = 0
        samples_since_update = 0
        episode_start_step = self.total_steps  # 记录episode开始时的步数
        
        # Warmup阶段
        print(f"\nStarting warmup phase for {self.warmup_steps} steps...")
        while self.total_steps < self.warmup_steps:
            # 1. 与环境交互一步
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.envs.step(action)
            done = terminated.any() or truncated.any()
            
            # 2. 只存储经验
            for env_idx in range(len(obs)):
                if not np.isnan(reward[env_idx]):
                    self.agent.replay_buffer.add(
                        obs[env_idx],
                        action[env_idx],
                        reward[env_idx],
                        next_obs[env_idx],
                        done
                    )
                    samples_since_update += 1
            
            # 3. 更新状态
            if done:
                obs, _ = self.envs.reset()
            else:
                obs = next_obs
            
            self.total_steps += 1
            
            # 只打印进度
            if self.total_steps % 1000 == 0:
                print(f"Warmup progress: {self.total_steps}/{self.warmup_steps} steps")
        
        print("\nWarmup phase completed. Starting training...")
        
        # 正式训练阶段
        try:
            while self.total_steps < self.max_steps:
                # 1. 与环境交互一步
                action = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.envs.step(action)
                done = terminated.any() or truncated.any()
                
                # 2. 存储经验
                for env_idx in range(len(obs)):
                    if not np.isnan(reward[env_idx]):
                        self.agent.replay_buffer.add(
                            obs[env_idx],
                            action[env_idx],
                            reward[env_idx],
                            next_obs[env_idx],
                            done
                        )
                        samples_since_update += 1
                
                # 3. 记录和更新
                if training_iter % self.log_interval == 0:
                    self._log_training_info(obs, reward, info)
                
                # 4. 更新策略
                if samples_since_update >= self.update_interval:
                    for _ in range(self.num_updates):
                        self.agent.update()
                    samples_since_update = 0
                
                # 5. 更新状态
                if done:
                    obs, _ = self.envs.reset()
                    self.episode_num += 1
                    # 记录episode长度
                    episode_length = self.total_steps - episode_start_step
                    self.logger.episode_lengths.append(episode_length)
                    episode_start_step = self.total_steps  # 重置开始时间
                    
                    # 记录其他信息
                    if isinstance(info, (list, tuple)):
                        for env_info in info:
                            if isinstance(env_info, dict):
                                if 'catch_distance' in env_info:
                                    self.logger.catch_distances.append(env_info['catch_distance'])
                                if 'reaction_time' in env_info:
                                    self.logger.reaction_times.append(env_info['reaction_time'])
                else:
                    obs = next_obs
                
                self.total_steps += 1
                training_iter += 1
                
                # 6. 评估
                if self.total_steps % self.eval_interval == 0:
                    self.run_evaluation()
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            self.envs.close()
            self.writer.close()
    
    def _log_training_info(self, obs, reward, info):
        """记录训练信息"""
        successes = []
        rewards = []
        reward_components = {
            'result_distance': 0,
            'result_angle': 0,
            'intent_distance': 0,
            'intent_angle': 0,
            'action_smooth': 0,
            'action_magnitude': 0,
            'jerk': 0,
            'catch': 0
        }
        
        # 处理reward
        for env_idx in range(len(obs)):
            rewards.append(
                reward[env_idx].item() if isinstance(reward[env_idx], (np.ndarray, np.float32, np.float64))
                else float(reward[env_idx])
            )
        
        # 处理reward_components
        if isinstance(info, dict) and 'reward_info' in info:
            reward_info = info['reward_info']
            for key in reward_components:
                if key in reward_info:
                    # 获取所有环境的平均值
                    values = reward_info[key]
                    if isinstance(values, np.ndarray):
                        reward_components[key] = float(np.mean(values))
        
        # 记录训练信息
        self.logger.log_training_info(
            self.total_steps,
            self.episode_num,
            successes,
            rewards,
            None,
            self.max_steps,
            reward_components
        )
        
        # 记录平均奖励
        step_reward = reward.mean().item()
        self.writer.add_scalar('train/step_reward', step_reward, self.total_steps)
        
        # 更新logger的last_reward_info
        if isinstance(info, (list, tuple)) and len(info) > 0:
            if isinstance(info[0], dict) and 'reward_info' in info[0]:
                self.logger.last_reward_info = info[0]['reward_info']
    
