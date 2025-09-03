#!/usr/bin/env python3
"""
复杂运动学软体机械臂环境 - 与C++硬件一致

基于optimized_graph_environment.py，但使用C++的复杂运动学
这样训练出的模型可以直接部署到真实硬件上
"""

import numpy as np
import torch
import random
from typing import Tuple, Optional, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

class ComplexKinematicsSoftArmEnvironment(gym.Env):
    """
    复杂运动学软体机械臂环境
    
    关键改进：使用与C++硬件完全一致的复杂运动学
    - 连续弯曲弧而不是直线段近似
    - 链式旋转变换而不是简单累加
    - 真正的continuum robot数学模型
    """
    
    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 5),
        segment_length_range: Tuple[float, float] = (0.1, 0.35),
        goal_threshold: float = 0.15,
        max_steps: int = 200,
        reward_type: str = "dense"
    ):
        super().__init__()
        
        self.dof_range = dof_range
        self.segment_length_range = segment_length_range
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.reward_type = reward_type
        
        # 当前episode配置
        self.current_n_segments = 0
        self.segment_lengths = None
        self.joint_angles = None
        self.goal_position = None
        self.step_count = 0
        
        print("🚀 复杂运动学软体机械臂环境初始化")
        print(f"   DOF范围: {dof_range[0]}-{dof_range[1]}节")
        print(f"   使用C++复杂运动学 (continuum robot)")
        print(f"   节点特征: 3维 [joint1, joint2, length]") 
        print(f"   Goals: achieved(3) + desired(3)")
        print(f"   阈值: {goal_threshold}")
        
        self._setup_spaces()
        
    def _setup_spaces(self):
        """设置动作和观测空间"""
        max_dof = max(self.dof_range) * 2
        
        # 动作空间：关节角度
        self.action_space = spaces.Box(
            low=-np.pi/2, high=np.pi/2,
            shape=(max_dof,), dtype=np.float32
        )
        
        # 观测空间：关节角度 + achieved_goal + desired_goal
        obs_dim = max_dof + 6  # goals (3+3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 随机选择节数
        self.current_n_segments = random.randint(*self.dof_range)
        current_dof = self.current_n_segments * 2
        
        # 随机segment长度
        self.segment_lengths = np.random.uniform(
            *self.segment_length_range, 
            size=self.current_n_segments
        )
        
        # 初始化关节角度
        max_dof = max(self.dof_range) * 2
        self.joint_angles = np.zeros(max_dof)
        
        # 采样目标位置
        self._sample_goal()
        
        self.step_count = 0
        
        # print(f"🔄 Episode Reset - DOF: {self.current_n_segments}节({current_dof}DOF)")
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 限制动作范围
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 只使用当前DOF对应的动作
        current_dof = self.current_n_segments * 2
        self.joint_angles[:current_dof] = action[:current_dof]
        
        # 计算当前末端位置 (使用复杂运动学)
        current_position = self._complex_forward_kinematics()
        
        # 计算奖励
        reward = self._compute_reward(current_position)
        
        # 检查终止条件
        distance = np.linalg.norm(current_position - self.goal_position)
        terminated = distance < self.goal_threshold
        
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_observation()
        info = self._get_info()
        info['distance'] = distance
        
        return obs, reward, terminated, truncated, info
    
    def _complex_forward_kinematics(self) -> np.ndarray:
        """
        复杂前向运动学 - 与C++硬件完全一致
        使用continuum robot的数学模型
        """
        position = np.array([0.0, 0.0, 0.0])
        rotation = np.eye(3)  # 累积旋转矩阵
        
        for i in range(self.current_n_segments):
            alpha = self.joint_angles[i * 2]      # 弯曲角 (对应C++的alpha)
            beta = self.joint_angles[i * 2 + 1]   # 方向角 (对应C++的beta)  
            arc_length = self.segment_lengths[i]
            
            # C++的复杂运动学公式
            local_translation = self._config_to_translation(alpha, beta, arc_length)
            local_rotation = self._config_to_rotation(alpha, beta)
            
            # 链式变换 (关键：旋转累积)
            position = position + rotation @ local_translation
            rotation = rotation @ local_rotation
            
        return position
    
    def _config_to_translation(self, alpha: float, beta: float, arc_length: float) -> np.ndarray:
        """C++的translation公式"""
        if abs(alpha) < 1e-6:  # alpha不能为0
            alpha = 1e-6
            
        x = arc_length/alpha * (1 - np.cos(alpha)) * np.sin(beta)
        y = arc_length/alpha * (1 - np.cos(alpha)) * np.cos(beta)
        z = arc_length/alpha * np.sin(alpha)
        
        return np.array([x, y, z])
    
    def _config_to_rotation(self, alpha: float, beta: float) -> np.ndarray:
        """C++的rotation matrix公式"""
        if abs(alpha) < 1e-6:
            alpha = 1e-6
            
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        
        rotation = np.array([
            [cos_b*cos_b*(1-cos_a) + cos_a, -cos_b*sin_b*(1-cos_a), sin_a*sin_b],
            [-cos_b*sin_b*(1-cos_a), sin_b*sin_b*(1-cos_a) + cos_a, sin_a*cos_b], 
            [-sin_a*sin_b, -sin_a*cos_b, cos_a]
        ])
        
        return rotation
    
    def _sample_goal(self):
        """在复杂运动学工作空间内采样目标位置"""
        # 保守估计工作空间
        max_reach = np.sum(self.segment_lengths) * 0.6  # 复杂运动学工作空间更紧凑
        
        # 在球面内随机采样
        while True:
            goal = np.random.uniform(-max_reach, max_reach, 3)
            goal[2] = abs(goal[2])  # Z向上
            
            if np.linalg.norm(goal) <= max_reach:
                self.goal_position = goal
                break
    
    def _compute_reward(self, current_position: np.ndarray) -> float:
        """计算奖励"""
        distance = np.linalg.norm(current_position - self.goal_position)
        
        if self.reward_type == "sparse":
            return 1.0 if distance < self.goal_threshold else 0.0
        else:  # dense
            # 密集奖励：距离越近奖励越高
            if distance < self.goal_threshold:
                return 50.0  # 成功奖励
            else:
                return -distance  # 距离惩罚
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        max_dof = max(self.dof_range) * 2
        
        # 当前末端位置 (复杂运动学)
        achieved_goal = self._complex_forward_kinematics()
        
        # 组装观测：joint_angles + achieved_goal + desired_goal
        obs = np.concatenate([
            self.joint_angles[:max_dof],  # 关节角度 (padding到max_dof)
            achieved_goal,                # 当前位置
            self.goal_position           # 目标位置
        ])
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """获取信息"""
        current_position = self._complex_forward_kinematics()
        distance = np.linalg.norm(current_position - self.goal_position)
        
        return {
            'current_n_segments': self.current_n_segments,
            'current_position': current_position.copy(),
            'goal_position': self.goal_position.copy(),
            'distance': distance,
            'success': distance < self.goal_threshold,
            'segment_lengths': self.segment_lengths.copy()
        }
    
    def render(self, mode='human'):
        """渲染(可选实现)"""
        pass
    
    def get_arm_shape(self):
        """获取机械臂形状 - 用于可视化"""
        positions = [np.array([0.0, 0.0, 0.0])]
        current_pos = np.array([0.0, 0.0, 0.0])
        current_rot = np.eye(3)
        
        for i in range(self.current_n_segments):
            alpha = self.joint_angles[i * 2]
            beta = self.joint_angles[i * 2 + 1]
            arc_length = self.segment_lengths[i]
            
            local_trans = self._config_to_translation(alpha, beta, arc_length)
            local_rot = self._config_to_rotation(alpha, beta)
            
            current_pos = current_pos + current_rot @ local_trans
            current_rot = current_rot @ local_rot
            
            positions.append(current_pos.copy())
        
        return np.array(positions)


def test_complex_kinematics_environment():
    """测试复杂运动学环境"""
    print("🧪 测试复杂运动学环境")
    print("=" * 50)
    
    env = ComplexKinematicsSoftArmEnvironment(
        dof_range=(2, 4),
        max_steps=10
    )
    
    # 测试几个episode
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset(seed=episode)
        
        print(f"  节数: {info['current_n_segments']}")
        print(f"  目标: [{info['goal_position'][0]:.3f}, {info['goal_position'][1]:.3f}, {info['goal_position'][2]:.3f}]")
        
        for step in range(5):
            action = np.random.randn(env.action_space.shape[0]) * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"    Step {step}: 距离={info['distance']:.3f}m, 奖励={reward:.1f}")
            
            if terminated:
                print(f"    ✅ 成功!")
                break
        
        if not terminated:
            print(f"    Episode结束，最终距离: {info['distance']:.3f}m")
    
    print(f"\n✅ 复杂运动学环境测试完成")
    print(f"✅ 使用与C++硬件一致的运动学")
    print(f"✅ 支持随机DOF配置")


if __name__ == "__main__":
    test_complex_kinematics_environment()