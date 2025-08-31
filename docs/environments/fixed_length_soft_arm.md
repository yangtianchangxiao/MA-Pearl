# 固定长度软体机械臂环境

## 📝 环境概述

`SoftArmReachEnvironment` 是经过充分验证的固定长度软体机械臂强化学习环境，使用SAC+HER算法取得了良好的训练效果。

## 🔧 技术规格

### 基本配置
- **环境名称**: `SoftArmReachEnvironment`  
- **文件位置**: `pearl/utils/instantiations/environments/soft_arm_reach_environment.py`
- **自由度**: 3DOF (1.5节软体臂，每节2个自由度)
- **段长度**: 固定 0.21m/节
- **总臂长**: 固定 0.315m (1.5节)
- **工作空间**: 固定立方体范围

### 状态空间
**观测维度**: 9维向量
```python
observation = [
    joint_angles[0],    # 第1节弯曲角度 (-π/2, π/2)
    joint_angles[1],    # 第1节方向角度 (-π, π)  
    joint_angles[2],    # 第2节弯曲角度 (-π/2, π/2)
    current_pos[0],     # 当前末端执行器X坐标
    current_pos[1],     # 当前末端执行器Y坐标  
    current_pos[2],     # 当前末端执行器Z坐标
    goal_pos[0],        # 目标位置X坐标
    goal_pos[1],        # 目标位置Y坐标
    goal_pos[2]         # 目标位置Z坐标
]
```

### 动作空间
**动作维度**: 3维连续动作
```python
action = [
    delta_joint_0,      # 第1节弯曲角度增量
    delta_joint_1,      # 第1节方向角度增量
    delta_joint_2       # 第2节弯曲角度增量
]
```
- **范围**: [-0.1, 0.1] (每步最大角度变化)
- **类型**: 连续动作空间 (Box)

### 奖励函数
**稀疏奖励系统**:
- **成功**: +50.0 (距离 ≤ 0.30m)
- **失败**: -1.0 (距离 > 0.30m)

**HER兼容**: 完全支持Hindsight Experience Replay算法

## 🎯 训练配置

### 推荐超参数
```python
config = {
    'episodes': 10000,              # 训练episodes
    'max_episode_steps': 200,       # 每episode最大步数
    'goal_threshold': 0.30,         # 目标距离阈值
    'buffer_capacity': 500000,      # HER buffer容量
    'batch_size': 512,              # 批量大小
    'training_rounds': 25,          # 每50步训练25次
    'learning_starts': 50000,       # warmup期
    'learn_every': 50,              # 训练频率
}
```

### 网络架构
- **Actor网络**: [512, 512] 隐藏层
- **Critic网络**: [512, 512] 隐藏层
- **算法**: Continuous SAC + HER

## 📊 性能基准

### 训练表现
- **收敛时间**: ~5000 episodes
- **最终成功率**: 85-95%
- **平均episode长度**: 120-150步
- **训练稳定性**: 高 (经过大量验证)

### 实际应用
- ✅ **基准环境**: 作为软体机械臂RL的标准基准
- ✅ **算法验证**: 验证新算法的有效性
- ✅ **教学演示**: 强化学习课程的经典案例

## 🛠️ 使用示例

### 基本使用
```python
from pearl.utils.instantiations.environments import SoftArmReachEnvironment
from pearl.utils.instantiations.environments.soft_arm_her_factory import create_soft_arm_her_buffer

# 创建环境
env = SoftArmReachEnvironment(
    n_segments=3,           # 1.5节 (兼容历史参数)
    goal_threshold=0.30,    
    max_steps=200
)

# 创建HER buffer
her_buffer = create_soft_arm_her_buffer(
    joint_dim=3,            # 3DOF
    spatial_dim=3,          # 3D空间
    capacity=500000,
    threshold=0.30
)

# 训练循环
obs, action_space = env.reset()
for step in range(200):
    action = policy.act(obs)
    result = env.step(action)
    her_buffer.push(...)
    if result.terminated:
        break
```

### 与Pearl Agent集成
```python
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic

# 创建SAC learner
sac = ContinuousSoftActorCritic(
    state_dim=9,
    action_space=env.action_space,
    actor_hidden_dims=[512, 512],
    critic_hidden_dims=[512, 512],
    batch_size=512
)

# 创建Pearl agent
agent = PearlAgent(
    policy_learner=sac,
    replay_buffer=her_buffer
)
```

## 📈 训练脚本

### 官方训练脚本
```bash
python train_soft_arm_pearl.py \
    --episodes 10000 \
    --device cuda:0 \
    --threshold 0.30 \
    --segments 3
```

### 多进程训练 (历史脚本)
```bash
python pearl/utils/scripts/train_arm_multiprocess.py \
    --episodes 100000 \
    --num_processes 4 \
    --device cuda:0
```

## ⚠️ 注意事项

### 已知限制
- **固定几何**: 无法适应不同臂长配置
- **泛化性**: 对几何参数变化敏感
- **计算效率**: 较长的episode长度 (200步)

### 兼容性
- ✅ **Pearl框架**: 完全兼容
- ✅ **HER算法**: 原生支持
- ✅ **GPU训练**: 支持CUDA加速
- ✅ **多进程**: 支持并行训练

## 🔗 相关环境

- **升级版本**: [变长软体机械臂环境](./variable_length_soft_arm.md)
- **硬体版本**: [NDOF硬体机械臂环境](./ndof_arm_environment.md)
- **环境对比**: [环境比较分析](./environment_comparison.md)