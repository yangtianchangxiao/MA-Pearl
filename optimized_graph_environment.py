#!/usr/bin/env python3
"""
优化的Graph环境实现
使用3维节点特征 + achieved/desired goals
完全去除冗余位置信息
"""

import torch
import numpy as np
import random
from typing import Tuple, Dict, Any
from simplified_graph_demo import SimplifiedGraphState
from pearl.api.environment import Environment
from pearl.api.action_result import ActionResult
from pearl.api.observation import Observation


class OptimizedGraphSoftArmEnvironment(Environment):
    """
    优化的Graph软体机械臂环境
    
    核心改进：
    - 节点特征：3维 [joint1, joint2, length] - 纯结构信息
    - Goals：achieved_goal(3) + desired_goal(3) - 纯空间信息
    - 强制网络学习运动学推理
    """
    
    def __init__(
        self,
        dof_range: Tuple[int, int] = (2, 4),
        base_segment_length: float = 0.21,
        segment_length_range: Tuple[float, float] = (0.168, 0.252),
        goal_threshold: float = 0.15,
        max_steps: int = 200,
        workspace_radius: float = 1.5,
    ):
        self.dof_range = dof_range
        self.base_segment_length = base_segment_length
        self.segment_length_range = segment_length_range
        self.goal_threshold = goal_threshold
        self.max_steps = max_steps
        self.workspace_radius = workspace_radius
        
        # 当前episode状态
        self.current_n_segments = None
        self.segment_lengths = None
        self.joint_angles = None
        self.goal_position = None
        self.step_count = 0
        
        print(f"🚀 优化Graph软体机械臂环境初始化")
        print(f"   DOF范围: {dof_range[0]}-{dof_range[1]}节")
        print(f"   节点特征: 3维 [joint1, joint2, length]")
        print(f"   Goals: achieved(3) + desired(3)")
        print(f"   阈值: {goal_threshold}")
    
    def reset(self, seed: int = None) -> Tuple[Observation, Any]:
        """重置环境"""
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 随机选择DOF配置
        self.current_n_segments = random.randint(*self.dof_range)
        current_dof = self.current_n_segments * 2
        
        # 随机化segment长度
        self.segment_lengths = np.random.uniform(
            *self.segment_length_range, 
            size=self.current_n_segments
        ).astype(np.float32)
        
        # 初始化关节角度
        self.joint_angles = np.random.uniform(
            -np.pi/4, np.pi/4,
            size=current_dof
        ).astype(np.float32)
        
        # 随机目标位置
        self._sample_goal()
        
        # 重置计数
        self.step_count = 0
        
        # 创建Graph状态
        graph_state = self._create_graph_state()
        
        print(f"🔄 Episode Reset - DOF: {self.current_n_segments}节({current_dof}DOF)")
        
        # 返回tensor和action space信息
        return graph_state.to_tensor(), self._get_action_space()
    
    def step(self, action: torch.Tensor) -> ActionResult:
        """执行动作"""
        self.step_count += 1
        
        # 应用动作到关节角度
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = action
        
        # 只使用当前DOF对应的动作
        current_dof = self.current_n_segments * 2
        valid_action = action_np[:current_dof] * 0.01  # 速度缩放
        
        # 更新关节角度
        self.joint_angles += valid_action
        self.joint_angles = np.clip(self.joint_angles, -np.pi/2, np.pi/2)
        
        # 创建新的Graph状态
        graph_state = self._create_graph_state()
        
        # 计算奖励
        current_position = self._forward_kinematics()
        distance = np.linalg.norm(current_position - self.goal_position)
        
        # 稀疏奖励
        if distance <= self.goal_threshold:
            reward = torch.tensor(50.0)
            terminated = torch.tensor(True)
        else:
            reward = torch.tensor(-1.0)
            terminated = torch.tensor(False)
        
        # 截断条件
        truncated = torch.tensor(self.step_count >= self.max_steps)
        
        return ActionResult(
            observation=graph_state.to_tensor(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            available_action_space=self._get_action_space()
        )
    
    def _create_graph_state(self) -> SimplifiedGraphState:
        """创建优化的Graph状态"""
        
        # 创建节点特征：[joint1, joint2, length] for each segment
        node_features_list = []
        for i in range(self.current_n_segments):
            joint1 = self.joint_angles[i * 2]      # 弯曲角
            joint2 = self.joint_angles[i * 2 + 1]  # 方向角  
            length = self.segment_lengths[i]        # 段长度
            
            node_feature = torch.tensor([joint1, joint2, length], dtype=torch.float32)
            node_features_list.append(node_feature)
        
        node_features = torch.stack(node_features_list)  # [n_segments, 3]
        
        # 创建邻接矩阵（链状连接）
        adjacency_matrix = torch.zeros(self.current_n_segments, self.current_n_segments, dtype=torch.float32)
        for i in range(self.current_n_segments - 1):
            adjacency_matrix[i, i+1] = 1.0  # 前向连接
            adjacency_matrix[i+1, i] = 1.0  # 后向连接
        
        # 计算当前末端位置（achieved goal）
        achieved_goal = torch.tensor(self._forward_kinematics(), dtype=torch.float32)
        
        # 目标位置（desired goal）
        desired_goal = torch.tensor(self.goal_position, dtype=torch.float32)
        
        return SimplifiedGraphState(
            node_features=node_features,
            adjacency_matrix=adjacency_matrix,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal
        )
    
    def _forward_kinematics(self) -> np.ndarray:
        """前向运动学：从关节角度计算末端位置"""
        position = np.array([0.0, 0.0, 0.0])
        cumulative_angle_xy = 0.0  # XY平面累积角度
        
        for i in range(self.current_n_segments):
            joint1 = self.joint_angles[i * 2]      # 弯曲角（Z方向）
            joint2 = self.joint_angles[i * 2 + 1]  # 方向角（XY平面）
            length = self.segment_lengths[i]
            
            # 更新累积方向角
            cumulative_angle_xy += joint2
            
            # 计算该段的末端位移
            segment_end = length * np.array([
                np.cos(cumulative_angle_xy) * np.cos(joint1),  # X
                np.sin(cumulative_angle_xy) * np.cos(joint1),  # Y
                np.sin(joint1)                                 # Z
            ])
            
            position += segment_end
        
        return position
    
    def _sample_goal(self):
        """在工作空间内采样目标位置"""
        # 估算工作空间半径
        max_reach = np.sum(self.segment_lengths) * 0.8  # 80%安全范围
        
        # 球坐标采样
        r = np.random.uniform(0.2, max_reach)  # 避免太近的目标
        theta = np.random.uniform(0, np.pi)     # 极角
        phi = np.random.uniform(0, 2*np.pi)    # 方位角
        
        self.goal_position = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            max(0.05, r * np.cos(theta))  # 避免地面以下
        ]).astype(np.float32)
    
    def _get_action_space(self):
        """获取动作空间信息"""
        from pearl.utils.instantiations.spaces.box_action import BoxActionSpace
        
        # 使用最大DOF定义action space（用于Pearl兼容性）
        max_dof = max(self.dof_range) * 2
        return BoxActionSpace(
            low=torch.full((max_dof,), -1.0),
            high=torch.full((max_dof,), 1.0)
        )
    
    @property
    def action_space(self):
        """Pearl环境接口"""
        return self._get_action_space()
    
    @property
    def observation_space(self):
        """Pearl环境接口"""
        from gymnasium.spaces import Box
        import numpy as np
        
        # 观测空间大小是动态的，这里给个粗略估计
        # 最大可能的tensor长度：max_nodes*3 + max_nodes^2 + 6 + 3
        max_nodes = max(self.dof_range)
        max_tensor_dim = max_nodes * 3 + max_nodes * max_nodes + 6 + 3
        
        return Box(low=-np.inf, high=np.inf, shape=(max_tensor_dim,))


def test_optimized_graph_environment():
    """测试优化的Graph环境"""
    print("🧪 测试优化Graph环境")
    print("💡 3维节点 + Goals分离")
    print("🎯 强制学习运动学推理")
    print("=" * 50)
    
    env = OptimizedGraphSoftArmEnvironment(
        dof_range=(2, 4),
        max_steps=10
    )
    
    # 测试多个episode
    for episode in range(3):
        state_tensor, action_space = env.reset()
        
        # 解析状态
        graph_state = SimplifiedGraphState.from_tensor(state_tensor)
        
        print(f"\\nEpisode {episode+1}:")
        print(f"   节点特征: {graph_state.node_features.shape}")
        print(f"   邻接矩阵: {graph_state.adjacency_matrix.shape}")
        print(f"   当前位置: {graph_state.achieved_goal.numpy()}")
        print(f"   目标位置: {graph_state.desired_goal.numpy()}")
        print(f"   初始距离: {torch.norm(graph_state.achieved_goal - graph_state.desired_goal):.3f}")
        
        # 执行几步
        for step in range(5):
            # 随机动作
            action = torch.randn(action_space.shape[0]) * 0.1
            result = env.step(action)
            
            # 解析新状态
            new_graph_state = SimplifiedGraphState.from_tensor(result.observation)
            distance = torch.norm(new_graph_state.achieved_goal - new_graph_state.desired_goal)
            
            print(f"     Step {step+1}: reward={result.reward.item():.1f}, distance={distance:.3f}")
            
            if result.terminated or result.truncated:
                success_msg = "SUCCESS!" if result.terminated else "TIMEOUT"
                print(f"     {success_msg}")
                break
    
    print(f"\\n🎉 优化Graph环境测试完成！")
    print(f"✅ 节点特征：纯结构信息")
    print(f"✅ Goals：纯空间信息")
    print(f"✅ 动态DOF支持")
    print(f"✅ Pearl框架兼容")


if __name__ == "__main__":
    test_optimized_graph_environment()