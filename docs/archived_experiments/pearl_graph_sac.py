"""
Pearl框架Graph SAC集成适配器
将机械臂Graph Transformer集成到Pearl SAC训练流程
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from torch_geometric.data import Data, Batch

from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic
)
from pearl.api.action_space import ActionSpace
from pearl.action_representation_modules.action_representation_module import ActionRepresentationModule

from robot_graph_transformer import RobotGraphTransformer, create_robot_graph
from robot_distributions import GraphSACPolicy, create_action_mask


class GraphStateConverter:
    """
    将Pearl环境的状态转换为图表示
    支持固定长度和变长软体机械臂
    """
    def __init__(self, env_type: str = "variable", max_dof: int = 9):
        """
        Args:
            env_type: 环境类型 ("fixed", "variable", "ndof")
            max_dof: 最大自由度数量
        """
        self.env_type = env_type
        self.max_dof = max_dof
    
    def parse_observation(self, obs: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        解析不同环境的观测格式
        
        Returns:
            dict包含: joint_angles, segment_lengths, current_pos, goal_pos
        """
        obs_np = obs.detach().cpu().numpy()
        
        if self.env_type == "fixed":
            # 固定长度软体臂: [joint_angles(6), current_pos(3), goal_pos(3)]
            joint_angles = obs_np[:6]
            current_pos = obs_np[6:9]
            goal_pos = obs_np[9:12]
            segment_lengths = np.array([0.21, 0.21, 0.105])  # 默认长度
            
        elif self.env_type == "variable":
            # 变长软体臂: [joint_angles(6), segment_lengths(3), current_pos(3), goal_pos(3)]
            joint_angles = obs_np[:6]
            segment_lengths = obs_np[6:9]
            current_pos = obs_np[9:12]
            goal_pos = obs_np[12:15]
            
        elif self.env_type == "ndof":
            # NDOF硬体臂: [joint_angles(N), current_pos(2/3), goal_pos(2/3)]
            # 需要推断DOF数量
            if len(obs_np) == 7:  # 3DOF, 2D
                joint_angles = obs_np[:3]
                current_pos = obs_np[3:5]
                goal_pos = obs_np[5:7]
                segment_lengths = np.ones(3) * 0.21
            elif len(obs_np) == 12:  # 6DOF, 3D
                joint_angles = obs_np[:6]
                current_pos = obs_np[6:9]
                goal_pos = obs_np[9:12]
                segment_lengths = np.ones(6) * 0.21
            else:
                raise ValueError(f"未知的NDOF观测维度: {len(obs_np)}")
        else:
            raise ValueError(f"未知的环境类型: {self.env_type}")
        
        return {
            "joint_angles": joint_angles,
            "segment_lengths": segment_lengths,
            "current_pos": current_pos,
            "goal_pos": goal_pos
        }
    
    def observation_to_graph(self, obs: torch.Tensor, joint_types: np.ndarray = None) -> Data:
        """
        将观测转换为图数据
        """
        parsed = self.parse_observation(obs)
        
        # 确定关节类型
        if joint_types is None:
            n_joints = len(parsed["joint_angles"])
            if self.env_type in ["fixed", "variable"]:
                joint_types = np.zeros(n_joints)  # 软体关节
            else:
                joint_types = np.ones(n_joints)   # 硬体关节
        
        # 创建图
        graph = create_robot_graph(
            joint_angles=parsed["joint_angles"],
            segment_lengths=parsed["segment_lengths"],
            joint_types=joint_types
        )
        
        # 添加目标信息作为图级属性
        graph.goal_pos = torch.tensor(parsed["goal_pos"], dtype=torch.float32)
        graph.current_pos = torch.tensor(parsed["current_pos"], dtype=torch.float32)
        
        return graph
    
    def batch_observations_to_graphs(self, observations: torch.Tensor) -> Tuple[Batch, torch.Tensor]:
        """
        批量转换观测为图，并生成action mask
        
        Args:
            observations: [batch_size, obs_dim] 批量观测
            
        Returns:
            batch: PyG Batch对象
            action_mask: [batch_size, max_dof] 动作mask
        """
        graphs = []
        dof_list = []
        
        for obs in observations:
            graph = self.observation_to_graph(obs)
            graphs.append(graph)
            dof_list.append(len(graph.x))
        
        batch = Batch.from_data_list(graphs)
        action_mask = create_action_mask(dof_list, self.max_dof)
        
        return batch, action_mask


class GraphContinuousSoftActorCritic(ContinuousSoftActorCritic):
    """
    Graph版本的连续SAC
    继承Pearl的ContinuousSoftActorCritic，替换网络为Graph Transformer
    """
    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        env_type: str = "variable",
        max_dof: int = 9,
        joint_feature_dim: int = 4,
        joint_types: int = 4,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        **kwargs
    ):
        # 保存Graph相关参数
        self.env_type = env_type
        self.max_dof = max_dof
        self.state_converter = GraphStateConverter(env_type, max_dof)
        
        # 初始化父类 (使用临时网络)
        super().__init__(
            state_dim=state_dim,
            action_space=action_space,
            **kwargs
        )
        
        # 替换为Graph网络
        self.graph_transformer = RobotGraphTransformer(
            joint_feature_dim=joint_feature_dim,
            joint_types=joint_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=hidden_dim
        )
        
        # Graph版本的actor
        self.graph_actor = GraphSACPolicy(
            graph_transformer=self,
            max_action_dim=self.max_dof
        )
        
        # Graph版本的critics
        critic_input_dim = hidden_dim + self.max_dof
        self.graph_critic1 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.graph_critic2 = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 替换原有网络
        self._actor_network = self.graph_actor
        self._critic1_network = self.graph_critic1
        self._critic2_network = self.graph_critic2
    
    def _state_to_graph_batch(self, state_batch: torch.Tensor) -> Tuple[Batch, torch.Tensor]:
        """
        将状态批次转换为图批次
        """
        return self.state_converter.batch_observations_to_graphs(state_batch)
    
    def act(
        self,
        state: torch.Tensor,
        action_representation: ActionRepresentationModule,
        exploit: bool = False
    ) -> torch.Tensor:
        """
        重写act方法，使用Graph网络
        """
        # 转换为图
        graph_batch, action_mask = self._state_to_graph_batch(state.unsqueeze(0))
        
        # 使用Graph actor
        with torch.no_grad():
            action, _, _ = self.graph_actor(graph_batch, action_mask, deterministic=exploit)
        
        # 只返回有效动作维度
        if action_mask is not None:
            dof = int(action_mask[0].sum().item())
            action = action[0, :dof]
        else:
            action = action[0]
        
        return action
    
    def get_action_prob(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        action_representation: ActionRepresentationModule
    ) -> torch.Tensor:
        """
        计算动作概率，使用Graph网络
        """
        graph_batch, action_mask = self._state_to_graph_batch(state_batch)
        
        # 需要将action_batch填充到max_dof维度
        batch_size = action_batch.shape[0]
        padded_actions = torch.zeros(batch_size, self.max_dof, device=action_batch.device)
        
        for i, action in enumerate(action_batch):
            dof = len(action)
            padded_actions[i, :dof] = action
        
        log_prob, _ = self.graph_actor.evaluate_actions(graph_batch, padded_actions, action_mask)
        return log_prob
    
    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        action_representation: ActionRepresentationModule
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算Q值，使用Graph网络
        """
        graph_batch, action_mask = self._state_to_graph_batch(state_batch)
        
        # 获取图特征
        _, graph_features, _ = self.graph_transformer(graph_batch)
        
        # 填充动作
        batch_size = action_batch.shape[0]
        padded_actions = torch.zeros(batch_size, self.max_dof, device=action_batch.device)
        
        for i, action in enumerate(action_batch):
            dof = len(action)
            padded_actions[i, :dof] = action
        
        # 应用mask
        if action_mask is not None:
            padded_actions = padded_actions * action_mask
        
        # 计算Q值
        state_action = torch.cat([graph_features, padded_actions], dim=1)
        q1 = self.graph_critic1(state_action)
        q2 = self.graph_critic2(state_action)
        
        return q1, q2


def create_graph_sac_agent(env, config: Dict[str, Any]):
    """
    创建Graph SAC agent的工厂函数
    
    Args:
        env: 机械臂环境
        config: 配置字典
        
    Returns:
        Graph SAC agent
    """
    from pearl.pearl_agent import PearlAgent
    from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
    
    # 确定环境类型
    env_name = env.__class__.__name__
    if "Variable" in env_name:
        env_type = "variable"
        max_dof = 6
    elif "Soft" in env_name:
        env_type = "fixed"
        max_dof = 6
    elif "NDOF" in env_name:
        env_type = "ndof"
        max_dof = 9
    else:
        env_type = "variable"  # 默认
        max_dof = 9
    
    print(f"🔧 创建Graph SAC Agent:")
    print(f"   环境类型: {env_type}")
    print(f"   最大DOF: {max_dof}")
    print(f"   观测维度: {env.observation_space.shape[0]}")
    print(f"   动作维度: {env.action_space.shape[0]}")
    
    # Action representation module
    action_rep_module = IdentityActionRepresentationModule(
        max_number_actions=max_dof,
        representation_dim=max_dof
    )
    
    # Graph SAC policy learner
    graph_sac = GraphContinuousSoftActorCritic(
        state_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        env_type=env_type,
        max_dof=max_dof,
        action_representation_module=action_rep_module,
        **config.get('sac_params', {})
    )
    
    # Pearl Agent
    agent = PearlAgent(
        policy_learner=graph_sac,
        replay_buffer=config.get('replay_buffer', None)
    )
    
    return agent


# 测试代码
if __name__ == "__main__":
    print("=== Pearl Graph SAC集成测试 ===")
    
    # 模拟环境和配置
    class MockActionSpace:
        def __init__(self, dim):
            self.shape = (dim,)
    
    class MockObsSpace:
        def __init__(self, dim):
            self.shape = (dim,)
    
    class MockEnv:
        def __init__(self, obs_dim, action_dim):
            self.observation_space = MockObsSpace(obs_dim)
            self.action_space = MockActionSpace(action_dim)
    
    # 测试不同环境
    test_envs = [
        ("VariableSoftArmReachEnvironment", MockEnv(15, 6)),  # 变长软体臂
        ("SoftArmReachEnvironment", MockEnv(12, 6)),         # 固定软体臂
        ("NDOFArmEnvironment", MockEnv(7, 3))                # NDOF硬体臂
    ]
    
    for env_name, mock_env in test_envs:
        print(f"\n=== 测试 {env_name} ===")
        
        # 创建状态转换器
        if "Variable" in env_name:
            converter = GraphStateConverter("variable", 6)
        elif "Soft" in env_name:
            converter = GraphStateConverter("fixed", 6)
        else:
            converter = GraphStateConverter("ndof", 9)
        
        # 创建测试观测
        if env_name == "VariableSoftArmReachEnvironment":
            obs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  # joint_angles
                               0.21, 0.18, 0.25,                 # segment_lengths
                               1.0, 1.0, 1.0,                    # current_pos
                               1.5, 1.5, 1.5], dtype=torch.float32)  # goal_pos
        elif env_name == "SoftArmReachEnvironment":
            obs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  # joint_angles
                               1.0, 1.0, 1.0,                    # current_pos
                               1.5, 1.5, 1.5], dtype=torch.float32)  # goal_pos
        else:  # NDOF
            obs = torch.tensor([0.1, 0.2, 0.3,      # joint_angles
                               1.0, 1.0,             # current_pos
                               1.5, 1.5], dtype=torch.float32)  # goal_pos
        
        # 测试状态转换
        try:
            parsed = converter.parse_observation(obs)
            print(f"✅ 观测解析成功:")
            print(f"   关节角度: {len(parsed['joint_angles'])}维")
            print(f"   段长度: {len(parsed['segment_lengths'])}维")
            
            # 测试图转换
            graph = converter.observation_to_graph(obs)
            print(f"✅ 图转换成功:")
            print(f"   节点数: {graph.x.shape[0]}")
            print(f"   节点特征维度: {graph.x.shape[1]}")
            print(f"   边数: {graph.edge_index.shape[1]}")
            
        except Exception as e:
            print(f"❌ 转换失败: {e}")
    
    print("\n✅ Pearl Graph SAC集成测试完成!")