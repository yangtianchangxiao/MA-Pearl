import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import numpy as np
from typing import Tuple, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from .networks import PolicyNetwork, QNetwork, TempNetwork
from .replay_buffer import ReplayBuffer

class SAC:
    """Soft Actor-Critic implementation."""
    
    def __init__(self,
                 observation_space_shape,
                 action_space_shape,
                 num_envs: int = 8,
                 hidden_dims: list[int] = [256, 256],
                 buffer_size: int = int(1e6),
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 alpha_lr: float = 3e-4,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 device: str = "cuda",
                 tensorboard_writer: Optional[SummaryWriter] = None,
                 target_update_interval: int = 1,
                 gradient_clip: float = 0.5,
                 prioritized_replay: bool = False,
                 per_alpha: float = 0.6,
                 per_beta: float = 0.4,
                 per_eps: float = 1e-6,
                 optimizer_config: dict = None,
                 lr_schedule_config: dict = None):
        
        self.num_envs = num_envs
        self.obs_dim = observation_space_shape[0]
        self.act_dim = action_space_shape[0]
        
        # Set target entropy to -dim(A)
        self.target_entropy = -np.prod(action_space_shape)
        
        # Initialize networks
        self.actor = PolicyNetwork(self.obs_dim, self.act_dim, hidden_dims).to(device)
        self.critic1 = QNetwork(self.obs_dim, self.act_dim, hidden_dims).to(device)
        self.critic2 = QNetwork(self.obs_dim, self.act_dim, hidden_dims).to(device)
        self.target_critic1 = QNetwork(self.obs_dim, self.act_dim, hidden_dims).to(device)
        self.target_critic2 = QNetwork(self.obs_dim, self.act_dim, hidden_dims).to(device)
        self.log_temp = TempNetwork(alpha).to(device)
        
        # Copy critic parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Initialize optimizers with weight decay
        optimizer_config = optimizer_config or {}
        weight_decay = float(optimizer_config.get('weight_decay', 1e-4))
        amsgrad = bool(optimizer_config.get('amsgrad', False))
        
        self.actor_optim = optim.AdamW(
            self.actor.parameters(), 
            lr=actor_lr,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        self.critic1_optim = optim.AdamW(
            self.critic1.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        self.critic2_optim = optim.AdamW(
            self.critic2.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        self.temp_optim = optim.AdamW(
            self.log_temp.parameters(),
            lr=alpha_lr,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            size=buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=device,
            prioritized=prioritized_replay,
            alpha=per_alpha,
            beta=per_beta,
            eps=per_eps
        )
        
        # Set hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.target_update_interval = target_update_interval
        self.gradient_clip = gradient_clip
        
        # Initialize mixed precision training
        self.actor_scaler = torch.amp.GradScaler('cuda')
        self.critic_scaler = torch.amp.GradScaler('cuda')
        self.temp_scaler = torch.amp.GradScaler('cuda')
        
        # Initialize TensorBoard writer
        self.writer = tensorboard_writer
        self.train_steps = 0
        
        # 配置学习率调度
        self.lr_schedule_config = lr_schedule_config or {
            'enabled': False,
            'warmup_steps': 10000,
            'decay_steps': 1000000,
            'final_ratio': 0.1
        }
        
        if self.lr_schedule_config['enabled']:
            self.actor_scheduler = self._create_lr_scheduler(self.actor_optim)
            self.critic1_scheduler = self._create_lr_scheduler(self.critic1_optim)
            self.critic2_scheduler = self._create_lr_scheduler(self.critic2_optim)
            self.temp_scheduler = self._create_lr_scheduler(self.temp_optim)
    
    def _create_lr_scheduler(self, optimizer):
        """创建学习率调度器"""
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self._get_lr_lambda
        )
    
    def _get_lr_lambda(self, step):
        """计算学习率调度因子"""
        config = self.lr_schedule_config
        if step < config['warmup_steps']:
            # 预热阶段
            return step / config['warmup_steps']
        else:
            # 衰减阶段
            progress = (step - config['warmup_steps']) / (config['decay_steps'] - config['warmup_steps'])
            progress = min(1.0, max(0.0, progress))
            return 1.0 - (1.0 - config['final_ratio']) * progress
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从经验回放中采样
        if self.replay_buffer.prioritized:
            batch, indices, weights = self.replay_buffer.sample(self.batch_size)
            # 使用重要性权重
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            weights = 1.0
        
        # 从ReplayBufferSamples对象中获取数据
        state = batch.observations
        action = batch.actions
        reward = batch.rewards.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
        next_state = batch.next_observations
        done = batch.dones.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]

        # 更新critic网络
        with torch.no_grad():
            next_action, next_log_pi = self.actor(next_state)  # next_log_pi: [batch_size]
            next_log_pi = next_log_pi.unsqueeze(-1)  # [batch_size] -> [batch_size, 1]
            
            target_q1 = self.target_critic1(next_state, next_action)  # [batch_size, 1]
            target_q2 = self.target_critic2(next_state, next_action)  # [batch_size, 1]
            target_q = torch.min(target_q1, target_q2)  # [batch_size, 1]
            
            # 计算温度调节后的Q值
            temp = self.log_temp().exp()  # [1]
            target_q = target_q - temp * next_log_pi  # [batch_size, 1]
            
            # 计算目标Q值
            target_q = reward + (1 - done) * self.gamma * target_q  # [batch_size, 1]

        # Critic loss
        current_q1 = self.critic1(state, action)  # [batch_size, 1]
        current_q2 = self.critic2(state, action)  # [batch_size, 1]
        
        # 确保维度匹配
        assert current_q1.shape == target_q.shape, f"Shape mismatch: current_q1 {current_q1.shape} vs target_q {target_q.shape}"
        assert current_q2.shape == target_q.shape, f"Shape mismatch: current_q2 {current_q2.shape} vs target_q {target_q.shape}"
        
        critic1_loss = F.mse_loss(current_q1, target_q.detach())
        critic2_loss = F.mse_loss(current_q2, target_q.detach())

        # 更新critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # 更新actor
        new_action, log_pi = self.actor(state)
        q1 = self.critic1(state, new_action)
        q2 = self.critic2(state, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.log_temp() * log_pi - q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 更新alpha
        alpha_loss = -(self.log_temp() * (log_pi + self.target_entropy).detach()).mean()
        self.temp_optim.zero_grad()
        alpha_loss.backward()
        self.temp_optim.step()

        # 软更新目标网络
        if self.train_steps % self.target_update_interval == 0:
            for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.train_steps += 1

        # 如果使用PER，更新优先级
        if self.replay_buffer.prioritized:
            td_errors = torch.abs(current_q1 - target_q).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # 应用梯度裁剪
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.gradient_clip)
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.gradient_clip)
        
        # 更新学习率
        if self.lr_schedule_config['enabled']:
            self.actor_scheduler.step()
            self.critic1_scheduler.step()
            self.critic2_scheduler.step()
            self.temp_scheduler.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'alpha': self.log_temp().item(),
            'alpha_loss': alpha_loss.item()
        }
    
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """选择动作
        Args:
            obs: 观察值，可能是单个观察(obs_dim,)或批量观察(batch_size, obs_dim)
            evaluate: 是否是评估模式
        Returns:
            action: 动作值，与输入观察的batch维度对应
        """
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                # 确保obs是浮点类型
                obs = obs.astype(np.float32)
                
                # 检查是否需要添加batch维度
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)
                
                obs = torch.FloatTensor(obs).to(self.device)
            
            # 获取动作
            action, _ = self.actor(obs, deterministic=evaluate)
            action = action.cpu().numpy()
            
            # 如果输入是单个观察，返回单个动作
            if len(action) == 1:
                action = action.squeeze(0)
            
            return action
    
    def evaluate(self, env, num_episodes: int = 10) -> Tuple[float, float, float]:
        """评估智能体
        Args:
            env: 评估环境
            num_episodes: 评估回合数
        Returns:
            mean_reward: 平均奖励
            success_rate: 成功率
            mean_episode_length: 平均回合长度
        """
        total_rewards = []
        success_count = 0
        episode_lengths = []
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action = self.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                if info.get('success', False):
                    success_count += 1
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        mean_reward = np.mean(total_rewards)
        success_rate = success_count / num_episodes
        mean_episode_length = np.mean(episode_lengths)
        
        return mean_reward, success_rate, mean_episode_length
    
    def state_dict(self):
        """返回模型的状态字典"""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optim.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optim.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optim.state_dict(),
            'log_alpha': self.log_temp.state_dict(),
            'alpha_optimizer_state_dict': self.temp_optim.state_dict()
        }

    def load_state_dict(self, state_dict):
        """加载模型的状态字典"""
        self.actor.load_state_dict(state_dict['actor_state_dict'])
        self.critic1.load_state_dict(state_dict['critic1_state_dict'])
        self.critic2.load_state_dict(state_dict['critic2_state_dict'])
        self.actor_optim.load_state_dict(state_dict['actor_optimizer_state_dict'])
        self.critic1_optim.load_state_dict(state_dict['critic1_optimizer_state_dict'])
        self.critic2_optim.load_state_dict(state_dict['critic2_optimizer_state_dict'])
        self.log_temp.load_state_dict(state_dict['log_alpha'])
        self.temp_optim.load_state_dict(state_dict['alpha_optimizer_state_dict'])

    def save_model(self, path):
        """保存模型到指定路径"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'log_temp_state_dict': self.log_temp.state_dict(),
            'actor_optimizer_state_dict': self.actor_optim.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optim.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optim.state_dict(),
            'temp_optimizer_state_dict': self.temp_optim.state_dict(),
            'train_steps': self.train_steps
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """从指定路径加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.log_temp.load_state_dict(checkpoint['log_temp_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optim.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optim.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.temp_optim.load_state_dict(checkpoint['temp_optimizer_state_dict'])
        self.train_steps = checkpoint['train_steps']
        print(f"Model loaded from {path}")
