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


class FixedArmHERBuffer(BasicReplayBuffer):
    """
    修复的标准HER buffer实现
    
    实现真正的HER算法：
    - Future策略：从future steps采样achieved_goal作为新desired_goal
    - 多采样：每个transition产生多个HER经验
    - 标准实现：对标SB3的HerReplayBuffer
    """

    def __init__(
        self,
        capacity: int,
        dof: int,
        spatial_dim: int,
        reward_fn: Callable[[SubjectiveState, Action], Reward],
        terminated_fn: Callable[[SubjectiveState, Action], bool] | None = None,
        n_sampled_goals: int = 4,  # 对标SB3
    ) -> None:
        super().__init__(capacity=capacity)
        self._dof = dof
        self._spatial_dim = spatial_dim
        self._reward_fn = reward_fn
        self._terminated_fn = terminated_fn
        self._n_sampled_goals = n_sampled_goals
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
        # 修复接口兼容性
        if next_state is not None:
            next_state = assert_is_tensor_like(next_state)
        
        # 直接存储原始transition，避免ActionSpace检查
        self._store_transition(
            state=state,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            curr_available_actions_tensor_with_padding=None,
            curr_unavailable_actions_mask=None,
            next_state=next_state,
            next_available_actions_tensor_with_padding=None,
            next_unavailable_actions_mask=None,
            cost=cost,
        )
        
        # 处理action space默认值
        if curr_available_actions is None:
            curr_available_actions = next_available_actions
        if next_available_actions is None and curr_available_actions is not None:
            next_available_actions = curr_available_actions

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
        
        # Episode结束时进行标准HER处理
        if terminated or truncated:
            self._process_standard_her_experience()
            self._trajectory = []

    def _process_standard_her_experience(self) -> None:
        """标准HER实现：future策略 + 多采样"""
        if len(self._trajectory) == 0:
            return
            
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
            
            # 标准HER: 从当前时间步的future中采样goals
            future_indices = list(range(t_idx + 1, len(self._trajectory)))
            if len(future_indices) == 0:
                continue  # 最后一步没有future
                
            # 随机采样future goals (标准HER策略)
            n_goals_to_sample = min(self._n_sampled_goals, len(future_indices))
            sampled_future_indices = np.random.choice(
                future_indices, 
                size=n_goals_to_sample, 
                replace=False
            )
            
            state = assert_is_tensor_like(state)
            next_state = assert_is_tensor_like(next_state)
            
            # 为每个采样的future goal创建HER transition
            for future_idx in sampled_future_indices:
                future_state = self._trajectory[future_idx][2]  # next_state of future step
                future_state = assert_is_tensor_like(future_state)
                
                # 提取future achieved_goal作为新的desired_goal
                future_achieved_goal = future_state[self._dof:self._dof + self._spatial_dim]
                
                # 创建HER状态
                her_state = state.clone()
                her_next_state = next_state.clone()
                
                # 替换desired_goal部分
                her_state[self._dof + self._spatial_dim:] = future_achieved_goal
                her_next_state[self._dof + self._spatial_dim:] = future_achieved_goal
                
                # 重新计算reward和termination
                her_reward = self._reward_fn(her_state, action)
                her_terminated = (
                    self._terminated_fn(her_state, action)
                    if self._terminated_fn is not None
                    else (her_reward > 0.0)  # 基于奖励判断
                )
                
                # 直接存储HER transition，避免递归
                self._store_transition(
                    state=her_state,
                    action=action,
                    reward=her_reward,
                    terminated=her_terminated,
                    truncated=False,  # HER transitions不应该被truncated
                    curr_available_actions_tensor_with_padding=None,
                    curr_unavailable_actions_mask=None,
                    next_state=her_next_state,
                    next_available_actions_tensor_with_padding=None,
                    next_unavailable_actions_mask=None,
                    cost=cost,
                )