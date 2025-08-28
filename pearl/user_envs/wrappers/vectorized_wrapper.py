#!/usr/bin/env python3
"""
Simple vectorized wrapper for Pearl environments.
Following Linus principle: "Good code has no special cases"
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Union
import gym
from gym.vector import AsyncVectorEnv, SyncVectorEnv

from pearl.api.environment import Environment
from pearl.api.action_result import ActionResult
from pearl.api.action_space import ActionSpace


class VectorizedEnvironmentWrapper(Environment):
    """
    Wrapper to parallelize Pearl environments.
    
    Design Philosophy (Linus-style):
    - No special cases: same interface as single env
    - Simple data structure: list of envs
    - Clear ownership: wrapper owns all envs
    """
    
    def __init__(
        self, 
        env_factory, 
        num_envs: int = 4,
        use_async: bool = True
    ):
        """
        Args:
            env_factory: Function that returns a Pearl environment
            num_envs: Number of parallel environments  
            use_async: Whether to use async (multiprocess) or sync execution
        """
        self.num_envs = num_envs
        self.env_factory = env_factory
        
        # Create reference environment for spaces
        self._ref_env = env_factory()
        
        # Create gym wrapper functions
        def make_gym_env():
            env = env_factory()
            return PearlToGymAdapter(env)
        
        # Create vectorized gym environments
        if use_async:
            self._vec_env = AsyncVectorEnv([make_gym_env for _ in range(num_envs)])
        else:
            self._vec_env = SyncVectorEnv([make_gym_env for _ in range(num_envs)])
        
        # Track which environments are done
        self._env_dones = np.zeros(num_envs, dtype=bool)
        self._episode_rewards = np.zeros(num_envs)
        self._episode_lengths = np.zeros(num_envs, dtype=int)
        
        # Store last observations for agent resets
        self._last_observations = None
    
    @property 
    def action_space(self) -> ActionSpace:
        """Return action space of reference environment."""
        return self._ref_env.action_space
    
    @property
    def observation_space(self) -> gym.Space:
        """Return observation space of reference environment.""" 
        return self._ref_env.observation_space
    
    def reset(self, seed: int = None) -> Tuple[np.ndarray, ActionSpace]:
        """Reset all environments and return first observations."""
        observations = self._vec_env.reset(seed=seed)
        self._env_dones.fill(False)
        self._episode_rewards.fill(0)
        self._episode_lengths.fill(0)
        self._last_observations = observations
        
        return observations, self.action_space
    
    def step(self, actions: Union[np.ndarray, List]) -> List[ActionResult]:
        """
        Step all environments with given actions.
        
        Returns list of ActionResult objects, one per environment.
        """
        # Ensure actions is numpy array
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        # Step vectorized environments
        observations, rewards, dones, infos = self._vec_env.step(actions)
        
        # Update tracking
        self._episode_rewards += rewards  
        self._episode_lengths += 1
        
        # Convert to Pearl ActionResult format
        results = []
        for i in range(self.num_envs):
            result = ActionResult(
                observation=observations[i],
                reward=rewards[i],
                terminated=dones[i],
                truncated=False,  # Gym vector envs handle this internally
                info=infos[i] if i < len(infos) else {}
            )
            results.append(result)
        
        # Store for potential agent resets
        self._last_observations = observations
        self._env_dones = dones
        
        return results
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for completed episodes."""
        completed_mask = self._env_dones
        if not np.any(completed_mask):
            return {}
        
        completed_rewards = self._episode_rewards[completed_mask]
        completed_lengths = self._episode_lengths[completed_mask] 
        
        return {
            'completed_episodes': np.sum(completed_mask),
            'avg_reward': np.mean(completed_rewards),
            'avg_length': np.mean(completed_lengths),
            'success_rate': np.mean(completed_rewards > 0.0) * 100
        }


class PearlToGymAdapter(gym.Env):
    """
    Adapter to make Pearl environment compatible with gym.vector.
    
    Pure adapter pattern - no business logic, just interface conversion.
    """
    
    def __init__(self, pearl_env: Environment):
        self.pearl_env = pearl_env
        
        # Copy spaces
        self.action_space = self._convert_action_space(pearl_env.action_space)
        self.observation_space = pearl_env.observation_space
        
    def _convert_action_space(self, pearl_action_space: ActionSpace) -> gym.Space:
        """Convert Pearl ActionSpace to gym Space."""
        # For continuous control - assume Box space
        if hasattr(pearl_action_space, 'low') and hasattr(pearl_action_space, 'high'):
            return gym.spaces.Box(
                low=pearl_action_space.low,
                high=pearl_action_space.high,
                dtype=np.float32
            )
        else:
            # Fallback - inspect the space
            return pearl_action_space
    
    def reset(self, seed=None):
        """Reset Pearl environment."""
        if seed is not None:
            np.random.seed(seed)
        
        obs, _ = self.pearl_env.reset()
        return obs
    
    def step(self, action):
        """Step Pearl environment.""" 
        result = self.pearl_env.step(action)
        
        return (
            result.observation,
            result.reward, 
            result.terminated or result.truncated,
            result.info
        )