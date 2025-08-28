# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

# Import vector environment wrappers first (no external dependencies)
from .subprocess_vector_env import SubprocVectorEnv
from .vectorized_wrapper import VectorizedEnvironmentWrapper, PearlToGymAdapter

# Then import existing wrappers (may have external dependencies)
try:
    from .atari_wrappers import EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv
except ImportError:
    pass  # Skip if gymnasium not available

from .dynamic_action_env import DynamicActionSpaceWrapper
from .gym_avg_torque_cost import GymAvgTorqueWrapper
from .partial_observability import (
    AcrobotPartialObservableWrapper,
    CartPolePartialObservableWrapper,
    MountainCarPartialObservableWrapper,
    PartialObservableWrapper,
    PendulumPartialObservableWrapper,
    PuckWorldPartialObservableWrapper,
)
from .safety import PuckWorldSafetyWrapper
from .sparse_reward import (
    AcrobotSparseRewardWrapper,
    MountainCarSparseRewardWrapper,
    PendulumSparseRewardWrapper,
    PuckWorldSparseRewardWrapper,
)

__all__ = [
    "SubprocVectorEnv",
    "VectorizedEnvironmentWrapper",
    "PearlToGymAdapter", 
    "AcrobotPartialObservableWrapper",
    "CartPolePartialObservableWrapper",
    "MountainCarPartialObservableWrapper",
    "PendulumPartialObservableWrapper",
    "PuckWorldPartialObservableWrapper",
    "PuckWorldSafetyWrapper",
    "PuckWorldSparseRewardWrapper",
    "AcrobotSparseRewardWrapper",
    "MountainCarSparseRewardWrapper",
    "PendulumSparseRewardWrapper",
    "PartialObservableWrapper",
    "GymAvgTorqueWrapper",
    "DynamicActionSpaceWrapper",
    "NoopResetEnv",
    "FireResetEnv",
    "EpisodicLifeEnv",
    "MaxAndSkipEnv",
]
