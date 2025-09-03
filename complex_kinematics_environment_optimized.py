#!/usr/bin/env python3
"""
优化版复杂运动学环境 - 保持C++兼容性但提升性能

优化策略:
1. 使用numpy向量化操作减少循环
2. 缓存三角函数结果避免重复计算
3. 移除训练时的print语句
4. 预分配数组避免内存分配
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional
# 纯numpy版本的快速运动学计算(无需numba依赖)
def _fast_complex_forward_kinematics(joint_angles, segment_lengths, n_segments):
    """JIT编译的快速前向运动学"""
    position = np.zeros(3)
    rotation = np.eye(3)
    
    for i in range(n_segments):
        alpha = joint_angles[i * 2]
        beta = joint_angles[i * 2 + 1]
        arc_length = segment_lengths[i]
        
        # 避免除零
        if abs(alpha) < 1e-6:
            alpha = 1e-6
            
        # 计算三角函数值
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        
        # Translation
        factor = arc_length / alpha * (1 - cos_a)
        local_translation = np.array([
            factor * sin_b,
            factor * cos_b, 
            arc_length / alpha * sin_a
        ])
        
        # Rotation matrix
        local_rotation = np.array([
            [cos_b*cos_b*(1-cos_a) + cos_a, -cos_b*sin_b*(1-cos_a), sin_a*sin_b],
            [-cos_b*sin_b*(1-cos_a), sin_b*sin_b*(1-cos_a) + cos_a, sin_a*cos_b], 
            [-sin_a*sin_b, -sin_a*cos_b, cos_a]
        ])
        
        # 累积变换
        position = position + rotation @ local_translation
        rotation = rotation @ local_rotation
        
    return position

class OptimizedComplexKinematicsSoftArmEnvironment(gym.Env):
    """优化版复杂运动学软体机械臂环境"""
    
    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 5),
        segment_length_range: Tuple[float, float] = (0.1, 0.35),
        goal_threshold: float = 0.05,
        max_steps: int = 200,
        workspace_limits: Tuple[float, float, float] = (2.0, 2.0, 1.5),
        verbose: bool = False  # 控制打印输出
    ):
        super().__init__()
        
        self.dof_range = dof_range
        self.segment_length_range = segment_length_range
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.workspace_limits = workspace_limits
        self.verbose = verbose  # 训练时设为False提升性能
        
        # 预分配数组避免重复分配内存
        max_segments = max(dof_range)
        max_dof = max_segments * 2
        
        self.joint_angles = np.zeros(max_dof)
        self.segment_lengths = np.zeros(max_segments) 
        
        # 观测和动作空间 (最大DOF)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(max_dof + 6,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi,
            shape=(max_dof,), dtype=np.float32
        )
        
        # 缓存当前状态避免重复计算
        self._current_position = None
        self._position_dirty = True
        
        if self.verbose:
            print("🚀 优化版复杂运动学软体机械臂环境初始化")
            print(f"   DOF范围: {dof_range[0]}-{dof_range[1]}节")
            print(f"   使用C++复杂运动学 + JIT加速")
    
    def _sample_config(self):
        """采样配置"""
        self.current_n_segments = np.random.randint(*self.dof_range)
        
        # 采样segment lengths
        for i in range(self.current_n_segments):
            self.segment_lengths[i] = np.random.uniform(*self.segment_length_range)
        
        # 重置关节角度
        current_dof = self.current_n_segments * 2
        self.joint_angles[:current_dof] = 0.0
        self._position_dirty = True
    
    def _sample_goal(self):
        """采样目标位置"""
        # 工作空间内随机采样
        self.goal_position = np.array([
            np.random.uniform(-self.workspace_limits[0], self.workspace_limits[0]),
            np.random.uniform(-self.workspace_limits[1], self.workspace_limits[1]), 
            np.random.uniform(0.1, self.workspace_limits[2])
        ])
    
    def _get_current_position(self):
        """获取当前末端位置(带缓存)"""
        if self._position_dirty:
            self._current_position = _fast_complex_forward_kinematics(
                self.joint_angles, self.segment_lengths, self.current_n_segments
            )
            self._position_dirty = False
        return self._current_position
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        current_position = self._get_current_position()
        
        # achieved_goal + desired_goal + joint_angles
        current_dof = self.current_n_segments * 2
        max_dof = max(self.dof_range) * 2
        
        obs = np.zeros(max_dof + 6, dtype=np.float32)
        obs[:3] = current_position  # achieved_goal
        obs[3:6] = self.goal_position  # desired_goal
        obs[6:6+current_dof] = self.joint_angles[:current_dof]  # joint_angles
        
        return obs
    
    def _compute_reward(self, current_position: np.ndarray) -> float:
        """计算奖励 - 负距离"""
        distance = np.linalg.norm(current_position - self.goal_position)
        return -distance
    
    def _get_info(self) -> Dict:
        """获取信息"""
        current_position = self._get_current_position()
        distance = np.linalg.norm(current_position - self.goal_position)
        
        return {
            'distance': distance,
            'goal_position': self.goal_position.copy(),
            'current_position': current_position.copy(),
            'current_dof': self.current_n_segments * 2,
            'is_success': distance < self.goal_threshold
        }
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        self._sample_config()
        self._sample_goal()
        self.step_count = 0
        
        if self.verbose:
            print(f"🔄 Episode Reset - DOF: {self.current_n_segments}节({self.current_n_segments*2}DOF)")
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 限制动作范围
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 更新关节角度
        current_dof = self.current_n_segments * 2
        self.joint_angles[:current_dof] = action[:current_dof]
        self._position_dirty = True  # 标记位置需要重新计算
        
        # 计算奖励和终止条件
        current_position = self._get_current_position()
        reward = self._compute_reward(current_position)
        
        distance = np.linalg.norm(current_position - self.goal_position)
        terminated = distance < self.goal_threshold
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info