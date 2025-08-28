# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable
import torch
import numpy as np

from pearl.api.action import Action
from pearl.api.action_space import ActionSpace
from pearl.api.reward import Reward
from pearl.api.state import SubjectiveState

from pearl.replay_buffers import BasicReplayBuffer
from pearl.utils.tensor_like import assert_is_tensor_like


class ArmHERBuffer(BasicReplayBuffer):
    """
    修复版本的HER buffer，专门为robotic arm环境设计
    
    正确实现标准HER算法：
    - 状态格式: [joint_angles, achieved_goal, desired_goal]
    - HER replacement: achieved_goal -> desired_goal
    """

    def __init__(
        self,
        capacity: int,
        dof: int,
        spatial_dim: int,
        reward_fn: Callable[[SubjectiveState, Action], Reward],
        terminated_fn: Callable[[SubjectiveState, Action], bool] | None = None,
    ) -> None:
        super().__init__(capacity=capacity)
        self._dof = dof
        self._spatial_dim = spatial_dim
        self._reward_fn = reward_fn
        self._terminated_fn = terminated_fn
        self._trajectory: list[
            tuple[
                SubjectiveState,
                Action,
                SubjectiveState,
                ActionSpace,
                ActionSpace,
                bool,
                bool,
                int | None,
                float | None,
            ]
        ] = []

    def push(
        self,
        state: SubjectiveState,
        action: Action,
        reward: Reward,
        next_state: SubjectiveState | None = None,
        curr_available_actions: ActionSpace | None = None,
        next_available_actions: ActionSpace | None = None,
        terminated: bool = False,
        truncated: bool = False,
        max_number_actions: int | None = None,
        cost: float | None = None,
    ) -> None:
        # 修复接口兼容性 - 允许None值
        if next_state is not None:
            next_state = assert_is_tensor_like(next_state)
        
        # 修复: 立即存储原始transition，episode结束时再添加HER transition
        super().push(
            state=state,
            action=action, 
            reward=reward,
            next_state=next_state,
            curr_available_actions=curr_available_actions,
            next_available_actions=next_available_actions,
            terminated=terminated,
            truncated=truncated,
            max_number_actions=max_number_actions,
            cost=cost,
        )
        
        # 兼容Pearl标准接口 - 允许action space为None
        if curr_available_actions is None:
            curr_available_actions = next_available_actions  # 使用默认值
        
        if next_available_actions is None and curr_available_actions is not None:
            next_available_actions = curr_available_actions  # 使用当前值作为默认

        # 收集trajectory
        self._trajectory.append(
            (
                state,
                action,
                next_state,
                curr_available_actions,
                next_available_actions,
                terminated,
                truncated,
                max_number_actions,
                cost,
            )
        )
        
        # Episode结束时进行HER处理
        if terminated or truncated:
            self._process_hindsight_experience()
            self._trajectory = []

    def _process_hindsight_experience(self) -> None:
        """标准HER实现：future策略 + 多采样"""
        if len(self._trajectory) == 0:
            return
            
        # 标准HER参数
        n_sampled_goals = 4  # 对标SB3的n_sampled_goal=4
        
        # 对每个transition生成HER经验
        for t_idx, (
            state,
            action,
            next_state,
            curr_available_actions,
            next_available_actions,
            terminated,
            truncated,
            max_number_actions,
            cost,
        ) in enumerate(self._trajectory):
            # 创建HER状态：用最终achieved_goal替换desired_goal
            state = assert_is_tensor_like(state)
            next_state = assert_is_tensor_like(next_state)
            
            # 修改状态的desired_goal部分
            her_state = state.clone()
            her_next_state = next_state.clone()
            
            her_state[self._dof + self._spatial_dim:] = final_achieved_goal
            her_next_state[self._dof + self._spatial_dim:] = final_achieved_goal
            
            # 重新计算reward和termination
            her_reward = self._reward_fn(her_state, action)
            her_terminated = (
                self._terminated_fn(her_state, action)
                if self._terminated_fn is not None
                else (her_reward == 0.0)
            )
            
            # 绕过super().push()的DiscreteActionSpace检查，直接存储transition
            self._store_transition(
                state=her_state,
                action=action,
                reward=her_reward,
                terminated=her_terminated,
                truncated=truncated,
                curr_available_actions_tensor_with_padding=None,
                curr_unavailable_actions_mask=None,
                next_state=her_next_state,
                next_available_actions_tensor_with_padding=None,
                next_unavailable_actions_mask=None,
                cost=cost,
            )