import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Tuple, Any

@dataclass
class ReplayBufferSamples:
    """Container for replay buffer samples."""
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(
        self,
        size: int,
        obs_dim: int,
        act_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6
    ):
        """Initialize replay buffer.
        
        Args:
            size: Maximum number of transitions to store
            obs_dim: Dimension of observation space
            act_dim: Dimension of action space
            device: Device to store tensors on
            prioritized: Whether to use prioritized experience replay
            alpha: Alpha parameter for prioritized experience replay
            beta: Beta parameter for prioritized experience replay
            eps: Epsilon parameter for prioritized experience replay
        """
        # Pre-allocate numpy arrays for better performance
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        
        # Priority replay related
        self.priorities = np.zeros(size, dtype=np.float32)
        self.max_priority = 1.0
        
        self.max_size = size
        self.ptr = 0
        self._size = 0
        self.full = False
        self.device = device
        
        # PER parameters
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer.
        
        Args:
            obs: Observation
            action: Action
            reward: Reward
            next_obs: Next observation
            done: Done flag
        """
        # 确保输入数组的维度正确
        obs = np.array(obs, dtype=np.float32)
        next_obs = np.array(next_obs, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        
        # 打印维度信息进行调试
        # print(f"Buffer add - obs shape: {obs.shape}, next_obs shape: {next_obs.shape}")
        
        # assert obs.shape == self.observation_space_shape, \
        #     f"Observation shape mismatch. Expected {self.observation_space_shape}, got {obs.shape}"
        # assert next_obs.shape == self.observation_space_shape, \
        #     f"Next observation shape mismatch. Expected {self.observation_space_shape}, got {next_obs.shape}"
        
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.acts[self.ptr] = action
        self.rews[self.ptr] = reward
        self.done[self.ptr] = done
        
        self.ptr += 1
        if self.ptr == self.max_size:
            self.ptr = 0
            self.full = True
    
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions
        """
        upper_bound = self.max_size if self.full else self.ptr
        batch_indices = np.random.randint(0, upper_bound, size=batch_size)
        
        # Convert numpy arrays to tensors and move to device
        observations = torch.FloatTensor(self.obs[batch_indices]).to(self.device)
        next_observations = torch.FloatTensor(self.next_obs[batch_indices]).to(self.device)
        actions = torch.FloatTensor(self.acts[batch_indices]).to(self.device)
        rewards = torch.FloatTensor(self.rews[batch_indices]).to(self.device)
        dones = torch.FloatTensor(self.done[batch_indices]).to(self.device)
        
        return ReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards
        )
    
    @property
    def size(self) -> int:
        """Return current size of buffer."""
        return self.max_size if self.full else self.ptr
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return self._size
