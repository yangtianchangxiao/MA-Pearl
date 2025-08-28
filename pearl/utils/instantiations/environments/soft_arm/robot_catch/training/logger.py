import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

class TrainingLogger:
    def __init__(self, log_dir="logs", writer=None):
        """初始化训练日志记录器
        Args:
            log_dir: 日志目录
            writer: 可选的TensorBoard writer实例
        """
        self.log_dir = log_dir
        if writer is None:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = writer
        
        # 添加历史记录
        self.episode_lengths = []  # 记录每个episode的长度
        self.catch_distances = []  # 记录抓取距离
        self.reaction_times = []   # 记录反应时间
    
    def log_training_info(self, total_steps: int, episode_num: int, 
                         success_history: list, episode_rewards: list, 
                         env_config: dict, max_steps: int,
                         reward_components: dict = None):
        """记录训练信息"""
        window_size = 100  # 可以作为配置参数
        
        # 计算基本指标
        metrics = {
            'success_rate': np.mean(success_history) if success_history else 0,
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'avg_episode_length': np.mean(self.episode_lengths[-window_size:]) if self.episode_lengths else 0,
        }
        
        # 打印训练进度
        print("\n" + "="*50)
        print(f"Training Progress - Step {total_steps}/{max_steps}")
        print(f"Episodes Completed: {episode_num}")
        
        # 打印基本指标
        print("\nBasic Metrics:")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Average Reward: {metrics['avg_reward']:.2f}")
        print(f"Average Episode Length: {metrics['avg_episode_length']:.1f}")
        
        # 打印奖励组成
        if reward_components:
            print("\nReward Components:")
            for key, value in reward_components.items():
                print(f"  {key}: {value:.3f}")
        
        # 记录到TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, total_steps)
        
        # 记录奖励组成到TensorBoard
        if reward_components:
            for key, value in reward_components.items():
                self.writer.add_scalar(f'train/reward_{key}', value, total_steps)
    
    def log_evaluation_results(self, total_steps, avg_reward, success_rate, 
                             reward_components):
        """记录评估结果
        Args:
            total_steps: 当前总步数
            avg_reward: 平均奖励
            success_rate: 成功率
            reward_components: 奖励组成部分
        """
        self.writer.add_scalar('Eval/Average Reward', avg_reward, total_steps)
        self.writer.add_scalar('Eval/Success Rate', success_rate, total_steps)
        
        # 记录奖励组成部分
        if reward_components:
            for component, value in reward_components.items():
                if not component.startswith('_'):
                    self.writer.add_scalar(f'Eval/Reward/{component}', value, total_steps)
    
    def close(self):
        """关闭日志记录器"""
        self.writer.close()
