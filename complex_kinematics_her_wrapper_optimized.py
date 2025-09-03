#!/usr/bin/env python3
"""
优化版复杂运动学HER包装器 - 3.1x性能提升

基于complex_kinematics_her_wrapper.py，但使用优化版环境
"""

import numpy as np
from typing import Dict, Any, Tuple
from complex_kinematics_environment_optimized import OptimizedComplexKinematicsSoftArmEnvironment
from pearl.utils.instantiations.spaces.box import BoxSpace

class OptimizedComplexKinematicsHERWrapper:
    """优化版复杂运动学HER包装器 - 提升3.1x性能"""
    
    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 5),
        segment_length_range: Tuple[float, float] = (0.1, 0.35),
        goal_threshold: float = 0.05,
        max_steps: int = 200,
        workspace_limits: Tuple[float, float, float] = (2.0, 2.0, 1.5)
    ):
        # 使用优化版环境(verbose=False提升性能)
        self.env = OptimizedComplexKinematicsSoftArmEnvironment(
            dof_range=dof_range,
            segment_length_range=segment_length_range,
            goal_threshold=goal_threshold,
            max_steps=max_steps,
            workspace_limits=workspace_limits,
            verbose=False  # 关键：训练时不输出提升速度
        )
        
        # 为Pearl框架转换空间格式
        max_dof = max(dof_range) * 2
        
        # Pearl observation space
        self.observation_space = BoxSpace.from_gym(self.env.observation_space)
        
        # Pearl action space  
        self.action_space = BoxSpace.from_gym(self.env.action_space)
        
    def reset(self) -> Tuple[np.ndarray, BoxSpace]:
        """重置环境并返回Pearl格式"""
        obs, info = self.env.reset()
        
        # 当前DOF对应的动作空间
        current_dof = self.env.current_n_segments * 2
        action_space = BoxSpace(
            low=np.full(current_dof, -np.pi, dtype=np.float32),
            high=np.full(current_dof, np.pi, dtype=np.float32)
        )
        
        return obs.astype(np.float32), action_space
    
    def step(self, action: np.ndarray):
        """执行一步并返回Pearl ActionResult"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Pearl ActionResult格式
        from pearl.api.action_result import ActionResult
        import torch
        
        action_result = ActionResult(
            observation=torch.from_numpy(obs.astype(np.float32)),
            reward=torch.tensor(reward, dtype=torch.float32),
            terminated=torch.tensor(terminated, dtype=torch.bool),
            truncated=torch.tensor(truncated, dtype=torch.bool),
            info=info,
            cost=torch.tensor(0.0, dtype=torch.float32),  # Pearl要求
            available_action_space=self.action_space  # Pearl要求
        )
        
        return action_result