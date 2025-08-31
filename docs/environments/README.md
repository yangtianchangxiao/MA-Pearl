# Pearl机械臂环境说明

本文档介绍Pearl框架中实现的机械臂强化学习环境，包括定长和变长两种软体机械臂环境。

## 📂 环境列表

| 环境名称 | 文件位置 | 描述 | 状态 |
|---------|----------|------|------|
| [固定长度软体臂](#固定长度软体臂) | `SoftArmReachEnvironment` | 3DOF固定长度软体机械臂 | ✅ 稳定 |
| [变长软体臂](#变长软体臂) | `VariableSoftArmReachEnvironment` | 6DOF变长软体机械臂 | ✅ 新实现 |
| [NDOFArmEnvironment](#ndof硬体臂) | `NDOFArmEnvironment` | N自由度硬体机械臂 | ✅ 经典环境 |

## 🎯 环境特点对比

| 特征 | 固定长度软体臂 | 变长软体臂 | NDOF硬体臂 |
|------|---------------|-----------|------------|
| **自由度** | 3DOF (1.5节) | 6DOF (3节) | 3/6/9DOF |
| **段长度** | 固定0.21m | 动态0.168-0.252m | N/A |
| **工作空间** | 固定 | 动态计算 | 固定 |
| **观测维度** | 9维 | 15维 | 动态 |
| **奖励类型** | 稀疏 | 稀疏 | 稀疏 |
| **算法适配** | SAC+HER | SAC+HER | SAC+HER |

## 📋 详细说明

点击环境名称查看详细文档：

- **[固定长度软体臂环境](./fixed_length_soft_arm.md)** - 经过验证的基础环境
- **[变长软体臂环境](./variable_length_soft_arm.md)** - 增强泛化能力的新环境  
- **[环境比较分析](./environment_comparison.md)** - 三种环境的详细对比

## 🚀 快速开始

### 固定长度环境
```python
from pearl.utils.instantiations.environments import SoftArmReachEnvironment

env = SoftArmReachEnvironment(
    n_segments=3,           # 1.5节 -> 3DOF
    goal_threshold=0.30,    # 目标阈值
    max_steps=200          # 最大步数
)
```

### 变长环境
```python
from pearl.utils.instantiations.environments import VariableSoftArmReachEnvironment

env = VariableSoftArmReachEnvironment(
    n_segments=3,                              # 3节 -> 6DOF
    segment_length_range=(0.168, 0.252),       # ±20%变化
    goal_threshold=0.15,                       # 更精确的阈值
    max_steps=50,                             # 更短的episode
    include_lengths_in_obs=True               # 观测包含长度信息
)
```

## 🔗 相关文档

- [Pearl框架tensor处理修复说明](../framework_fixes/tensor_processing_fix.md)
- [HER算法实现细节](../algorithms/her_implementation.md)
- [训练脚本使用指南](../training/training_scripts.md)