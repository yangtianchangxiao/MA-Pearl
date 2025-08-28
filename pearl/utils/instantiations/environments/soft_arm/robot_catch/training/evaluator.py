import numpy as np
import torch
import os
from datetime import datetime
from collections import defaultdict

class Evaluator:
    def __init__(self, num_eval_episodes=50):
        """初始化评估器
        Args:
            num_eval_episodes: 评估回合数
        """
        self.num_eval_episodes = num_eval_episodes
        
    def evaluate_policy(self, policy, env, num_episodes=10):
        """评估策略的性能"""
        total_rewards = []
        success_count = 0
        episode_lengths = []
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            while not (done or truncated):
                # 转换状态到正确的格式
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                # 使用策略选择动作
                with torch.no_grad():
                    action, _ = policy(obs_tensor, deterministic=True)
                action = action.squeeze(0).cpu().numpy()
                
                # 执行动作
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # 检查是否成功抓住球
                if info.get('success', False):
                    success_count += 1
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # 计算评估指标
        mean_reward = np.mean(total_rewards)
        success_rate = success_count / num_episodes
        mean_episode_length = np.mean(episode_lengths)
        
        return {
            'mean_reward': mean_reward,
            'success_rate': success_rate,
            'mean_episode_length': mean_episode_length,
            'total_episodes': num_episodes
        }
    
    def evaluate(self, env, agent, num_episodes=50):
        """评估智能体性能"""
        metrics = defaultdict(list)
        reward_components_sum = defaultdict(float)
        
        for episode in range(num_episodes):
            episode_metrics = self._evaluate_episode(env, agent)
            
            # 收集每个episode的指标
            for key, value in episode_metrics.items():
                if key == 'reward_components':
                    # 累积奖励组成
                    for comp_key, comp_value in value.items():
                        reward_components_sum[comp_key] += comp_value
                else:
                    metrics[key].append(value)
        
        # 计算平均值
        num_episodes = float(len(metrics['reward']))
        reward_components_avg = {
            k: v / num_episodes for k, v in reward_components_sum.items()
        }
        
        # 计算统计信息
        eval_stats = {
            'success_rate': np.mean(metrics['success']),
            'avg_reward': np.mean(metrics['reward']),
            'reward_components': reward_components_avg
        }
        
        # 打印详细的评估结果
        self._print_eval_results(eval_stats)
        
        return eval_stats
    
    def _evaluate_episode(self, env, agent):
        """评估单个回合的性能"""
        obs, _ = env.reset()
        episode_reward = 0
        done = False
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
        
        while not done:
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 更新奖励组成部分
            if 'reward_info' in info:
                for key, value in info['reward_info'].items():
                    if key in reward_components:  # 只记录我们关心的组成部分
                        reward_components[key] += value
            
            episode_reward += reward
            obs = next_obs
        
        return {
            'reward': episode_reward,
            'success': info.get('success', False),
            'reward_components': reward_components
        }
    
    def _calculate_smoothness(self, trajectory):
        """计算轨迹平滑度"""
        # 实现平滑度计算逻辑
        pass
    
    def _print_eval_results(self, eval_stats):
        """打印详细的评估结果"""
        print("\n" + "="*50)
        print("Evaluation Results")
        print(f"Success Rate: {eval_stats['success_rate']:.2%}")
        print(f"Average Reward: {eval_stats['avg_reward']:.2f}")
        
        # 打印奖励组成
        if 'reward_components' in eval_stats:
            print("\nReward Components:")
            for key, value in eval_stats['reward_components'].items():
                print(f"  {key}: {value:.3f}")
    
    def save_checkpoint(self, agent, episode, avg_reward, avg_success_rate, 
                       checkpoint_dir='checkpoints'):
        """保存模型检查点
        Args:
            agent: SAC智能体
            episode: 当前回合数
            avg_reward: 平均奖励
            avg_success_rate: 平均成功率
            checkpoint_dir: 保存目录
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 创建checkpoint
        checkpoint = {
            'actor_state_dict': agent.actor.state_dict(),
            'critic1_state_dict': agent.critic1.state_dict(),
            'critic2_state_dict': agent.critic2.state_dict(),
            'episode': episode,
            'avg_reward': avg_reward,
            'avg_success_rate': avg_success_rate
        }
        
        # 使用时间戳创建唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = os.path.join(
            checkpoint_dir, 
            f"sac_model_reward_{avg_reward:.0f}_{timestamp}.pth"
        )
        torch.save(checkpoint, timestamped_path)
