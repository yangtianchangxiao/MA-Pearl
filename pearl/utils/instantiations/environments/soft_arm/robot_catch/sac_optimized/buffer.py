import numpy as np
import torch
from typing import Dict, Tuple

class ReplayBuffer:
    def __init__(self, 
                 size: int,
                 obs_dim: int,
                 act_dim: int,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 prioritized: bool = False,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 eps: float = 1e-6):
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
        self.size = 0
        self.device = device
        
        # Prefetch related
        self._prefetch_batch = None
        self._prefetch_indices = None
        
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
    def add(self, 
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool) -> None:
        # Direct array assignment for better performance
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, 
               batch_size: int,
               beta: float = 0.4) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        # Priority sampling
        if self.size < batch_size:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
            indices = np.random.choice(self.size, batch_size, p=probs)
            
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Prepare batch using vectorized operations
        batch = {
            "obs": torch.FloatTensor(self.obs[indices]).to(self.device),
            "acts": torch.FloatTensor(self.acts[indices]).to(self.device),
            "rews": torch.FloatTensor(self.rews[indices]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[indices]).to(self.device),
            "done": torch.FloatTensor(self.done[indices]).to(self.device),
            "weights": torch.FloatTensor(weights).to(self.device)
        }
        
        return batch, indices, weights
    
    def update_priorities(self, 
                         indices: np.ndarray,
                         priorities: np.ndarray) -> None:
        self.priorities[indices] = priorities + 1e-6  # Small constant for stability
        self.max_priority = max(self.max_priority, priorities.max())
        
    def prefetch_batch(self, batch_size: int) -> None:
        """Prefetch next batch for faster training"""
        if self._prefetch_batch is None:
            self._prefetch_batch, self._prefetch_indices, _ = self.sample(batch_size)
            
    def get_prefetched_batch(self) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """Get prefetched batch and trigger next prefetch"""
        batch = self._prefetch_batch
        indices = self._prefetch_indices
        self._prefetch_batch = None
        self._prefetch_indices = None
        return batch, indices
