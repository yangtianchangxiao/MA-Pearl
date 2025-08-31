# 变长软体机械臂环境

## 📝 环境概述

`VariableSoftArmReachEnvironment` 是在固定长度环境基础上开发的增强版软体机械臂环境，通过**每episode随机化段长度**来提升agent的泛化能力，适用于需要处理多种机械臂配置的实际应用场景。

## 🔧 技术规格

### 基本配置
- **环境名称**: `VariableSoftArmReachEnvironment`
- **文件位置**: `pearl/utils/instantiations/environments/variable_soft_arm_environment.py`
- **自由度**: 6DOF (3节软体臂，每节2个自由度)
- **段长度**: **动态变化** 0.168m-0.252m (±20%)
- **总臂长**: **动态变化** 0.504m-0.756m
- **工作空间**: **动态计算** (基于当前臂长的80%)

### 🎲 动态特性
**每episode随机化**:
- **Segment长度**: 每节独立随机，范围 [0.168m, 0.252m]
- **总臂长**: 3节总长度 [0.504m, 0.756m]  
- **工作空间**: 基于总长度×80%动态计算 [0.403m, 0.605m]
- **目标位置**: 在动态工作空间内随机生成

### 状态空间
**观测维度**: 15维向量 (比固定长度增加6维)
```python
observation = [
    # 关节角度 (6维)
    joint_angles[0],     # 第1节弯曲角度 (-π/2, π/2)
    joint_angles[1],     # 第1节方向角度 (-π, π)
    joint_angles[2],     # 第2节弯曲角度 (-π/2, π/2)  
    joint_angles[3],     # 第2节方向角度 (-π, π)
    joint_angles[4],     # 第3节弯曲角度 (-π/2, π/2)
    joint_angles[5],     # 第3节方向角度 (-π, π)
    
    # 当前段长度 (3维) - 关键新增信息
    segment_lengths[0],  # 第1节当前长度 [0.168, 0.252]
    segment_lengths[1],  # 第2节当前长度 [0.168, 0.252]
    segment_lengths[2],  # 第3节当前长度 [0.168, 0.252]
    
    # 末端执行器位置 (3维)
    current_pos[0],      # 当前末端执行器X坐标
    current_pos[1],      # 当前末端执行器Y坐标
    current_pos[2],      # 当前末端执行器Z坐标
    
    # 目标位置 (3维)
    goal_pos[0],         # 目标位置X坐标  
    goal_pos[1],         # 目标位置Y坐标
    goal_pos[2]          # 目标位置Z坐标
]
```

### 动作空间
**动作维度**: 6维连续动作
```python
action = [
    delta_joint_0,       # 第1节弯曲角度增量
    delta_joint_1,       # 第1节方向角度增量
    delta_joint_2,       # 第2节弯曲角度增量
    delta_joint_3,       # 第2节方向角度增量
    delta_joint_4,       # 第3节弯曲角度增量
    delta_joint_5        # 第3节方向角度增量
]
```
- **范围**: [-0.1, 0.1] (每步最大角度变化)
- **类型**: 连续动作空间 (Box)

### 奖励函数
**稀疏奖励系统** (保持与固定长度一致):
- **成功**: +50.0 (距离 ≤ 0.15m，更精确的阈值)
- **失败**: -1.0 (距离 > 0.15m)

**HER完全兼容**: 支持变长观测格式的HER算法

## 🎯 核心创新点

### 1. 动态几何配置
```python
# 每episode开始时随机化
min_len, max_len = self.segment_length_range  # (0.168, 0.252)
self.current_segment_lengths = np.random.uniform(
    min_len, max_len, size=self.n_segments
).astype(np.float32)

# 动态计算工作空间
total_length = np.sum(self.current_segment_lengths)
actual_safe_reach = total_length * 0.8  # 80%安全边界
```

### 2. 增强观测空间  
- **几何感知**: 观测中包含当前段长度信息
- **适应性**: Agent可以根据当前几何配置调整策略
- **泛化性**: 训练期间接触到多种配置组合

### 3. 专用HER Buffer
```python
# 支持变长观测格式的HER buffer
class VariableArmHERBuffer(BasicReplayBuffer):
    def _extract_goals_from_state(self, state):
        # 正确解析变长观测: [joints, lengths, achieved, desired]
        if self._include_lengths_in_obs:
            config_dim = self._dof + self._n_segments
            achieved_goal = state[config_dim:config_dim + self._spatial_dim]
            desired_goal = state[config_dim + self._spatial_dim:]
        # ...
```

## 📊 训练配置

### 推荐超参数 (针对变长环境优化)
```python
config = {
    # 环境配置
    'n_segments': 3,                              # 3节机械臂
    'segment_length_range': (0.168, 0.252),       # ±20%变化范围
    'goal_threshold': 0.15,                       # 更精确阈值
    'max_episode_steps': 50,                      # 更短episode
    
    # 训练配置
    'episodes': 100000,                           # 更多episodes
    'batch_size': 512,                           # 保持一致
    'training_rounds': 50,                       # 50步训练25次
    'learning_starts': 2000,                     # 更快启动
    'learn_every': 25,                           # 更频繁学习
    
    # 网络配置
    'actor_hidden_dims': [512, 512],             # 更大网络
    'critic_hidden_dims': [512, 512],
    'buffer_capacity': 1000000,                  # 更大buffer
}
```

### 关键设计决策
- **更短episode**: 50步 vs 200步，加快训练速度
- **更精确阈值**: 0.15m vs 0.30m，提高控制精度  
- **更频繁学习**: 每25步 vs 每50步，适应动态环境
- **更大buffer**: 100万 vs 50万，存储更多配置样本

## 🚀 使用示例

### 基本使用
```python
from pearl.utils.instantiations.environments import VariableSoftArmReachEnvironment
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer

# 创建变长环境
env = VariableSoftArmReachEnvironment(
    n_segments=3,                              # 3节
    segment_length_range=(0.168, 0.252),       # ±20%变化
    goal_threshold=0.15,                       # 精确阈值
    max_steps=50,                             # 短episode
    include_lengths_in_obs=True               # 包含长度信息
)

# 创建专用HER buffer
her_buffer = create_variable_soft_arm_her_buffer(
    joint_dim=6,            # 6DOF
    spatial_dim=3,          # 3D空间
    n_segments=3,           # 3节
    capacity=1000000,       # 大容量
    threshold=0.15,         # 匹配环境阈值
    include_lengths_in_obs=True
)

# 训练循环
for episode in range(100000):
    obs, action_space = env.reset()  # 每次reset都会随机化段长度
    print(f"当前段长度: {env.current_segment_lengths}")
    print(f"工作空间大小: {env.current_workspace_bounds}")
    
    for step in range(50):
        action = agent.act(obs)
        result = env.step(action)
        agent.observe(result)  # 自动处理变长观测
        if result.terminated:
            break
```

### Pearl Agent集成
```python
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule

# 创建action representation module
action_rep_module = IdentityActionRepresentationModule(
    max_number_actions=6,  # 6DOF
    representation_dim=6
)

# 创建Continuous SAC learner
sac = ContinuousSoftActorCritic(
    state_dim=15,                    # 变长观测15维
    action_space=env.action_space,   # 6维连续动作
    actor_hidden_dims=[512, 512],
    critic_hidden_dims=[512, 512],
    action_representation_module=action_rep_module,
    training_rounds=50,
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
python train_variable_soft_arm_official.py
```

脚本会自动：
- 设置进程名为 `Variable-Arm-Pearl-SAC` (nvidia-smi可见)
- 创建tmux会话 `variable_arm_training`
- 每500episodes显示详细进度
- 自动保存最佳checkpoint

### 训练监控
```bash
# 查看训练进度
tmux attach -t variable_arm_training

# 查看GPU使用
nvidia-smi  # 查找 Variable-Arm-Pearl-SAC

# 查看训练日志
ls variable_arm_results/
```

## 📊 性能预期

### 训练特点
- **收敛速度**: 预期比固定长度慢 (更复杂的状态空间)
- **最终性能**: 预期略低但泛化性更强
- **训练稳定性**: 需要更大的buffer和更多episodes
- **样本效率**: HER算法显著提升样本利用率

### 预期指标
- **目标成功率**: 70-85% (在多种配置下)
- **收敛时间**: ~10000-15000 episodes  
- **平均episode长度**: 25-40步
- **泛化能力**: 能适应训练时未见的段长度组合

## ⚠️ 注意事项

### 实现细节
- **Pearl框架修复**: 需要我们修复的tensor处理bug才能正常运行
- **观测格式**: 15维观测，前6维关节角+3维长度+3维当前位置+3维目标
- **HER兼容**: 专用HER buffer正确处理变长观测格式

### 已知限制  
- **计算开销**: 每episode需要重新计算工作空间
- **内存需求**: 更大的观测空间和buffer容量
- **调试复杂**: 动态配置增加了调试难度

### 性能优化
- ✅ **动态workspace计算**: O(1)复杂度
- ✅ **观测空间优化**: 包含最少必要信息
- ✅ **HER效率**: 专门优化的HER buffer实现

## 🔬 实验验证

### A/B测试设计
1. **基线对比**: vs 固定长度环境在相同配置下的性能
2. **泛化测试**: 在训练时未见的段长度配置下测试  
3. **消融研究**: 移除段长度信息对性能的影响
4. **鲁棒性**: 对极端配置组合的适应能力

### 关键度量
- **成功率**: 不同段长度配置下的任务完成率
- **样本效率**: 达到目标性能所需的episodes数量
- **泛化误差**: 训练配置vs测试配置的性能差异
- **稳定性**: 多次运行的方差

## 🔗 相关环境

- **基础版本**: [固定长度软体机械臂环境](./fixed_length_soft_arm.md)
- **硬体版本**: [NDOF硬体机械臂环境](./ndof_arm_environment.md)  
- **详细对比**: [环境比较分析](./environment_comparison.md)
- **技术细节**: [Pearl框架修复说明](../framework_fixes/tensor_processing_fix.md)