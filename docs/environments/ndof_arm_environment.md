# NDOF硬体机械臂环境

## 📝 环境概述

`NDOFArmEnvironment` 是Pearl框架中的经典硬体机械臂强化学习环境，支持可配置自由度的刚性关节机械臂，广泛用于多DOF强化学习算法的验证和基准测试。

## 🔧 技术规格

### 基本配置
- **环境名称**: `NDOFArmEnvironment`
- **文件位置**: `pearl/utils/instantiations/environments/arm_environment.py`
- **自由度**: 可配置 (通常3DOF, 6DOF, 9DOF)
- **关节类型**: 旋转关节 (刚性连接)
- **工作空间**: 固定球形/椭球形区域
- **控制方式**: 关节角度增量控制

### 可配置参数
```python
NDOFArmEnvironment(
    dof=3,                    # 自由度数量 (3, 6, 9)
    max_steps=200,           # 每episode最大步数
    goal_threshold=0.30,     # 目标距离阈值
    workspace_bounds=None    # 工作空间边界 (可选)
)
```

### 状态空间
**观测维度**: DOF + 4/6 维 (取决于2D/3D工作空间)

#### 3DOF配置 (最常用)
```python
observation = [
    joint_angles[0],    # 关节1角度 (-π, π)
    joint_angles[1],    # 关节2角度 (-π, π)  
    joint_angles[2],    # 关节3角度 (-π, π)
    current_pos[0],     # 当前末端执行器X坐标
    current_pos[1],     # 当前末端执行器Y坐标
    goal_pos[0],        # 目标位置X坐标 (2D工作空间)
    goal_pos[1]         # 目标位置Y坐标
]
```
**总维度**: 7维 (3关节 + 2当前位置 + 2目标位置)

#### 6DOF配置
```python
observation = [
    joint_angles[0:6],   # 6个关节角度
    current_pos[0:3],    # 3D当前位置
    goal_pos[0:3]        # 3D目标位置  
]
```
**总维度**: 12维 (6关节 + 3当前位置 + 3目标位置)

### 动作空间
**动作维度**: 等于DOF数量
```python
action = [
    delta_joint_0,      # 关节1角度增量
    delta_joint_1,      # 关节2角度增量
    # ... 更多关节
    delta_joint_n       # 关节n角度增量
]
```
- **范围**: 通常 [-0.1, 0.1] 弧度/步
- **类型**: 连续动作空间 (Box)

### 奖励函数
**稀疏奖励系统**:
- **成功**: 0.0 (距离 ≤ threshold)  
- **失败**: -1.0 (距离 > threshold)

**HER兼容**: 原生支持Hindsight Experience Replay

## 🎯 配置变种

### 1. 3DOF平面臂 (最经典)
```python
env = NDOFArmEnvironment(
    dof=3,
    max_steps=200,
    goal_threshold=0.30,
    # 2D平面工作空间
)
```
- **应用**: 基础RL算法验证
- **复杂度**: 低，适合快速实验
- **收敛性**: 好，训练稳定

### 2. 6DOF空间臂  
```python
env = NDOFArmEnvironment(
    dof=6,
    max_steps=250,
    goal_threshold=0.25,
    # 3D空间工作空间
)
```
- **应用**: 工业机器人仿真
- **复杂度**: 中等，需要更大网络
- **收敛性**: 中等，需要更多样本

### 3. 9DOF冗余臂
```python
env = NDOFArmEnvironment(
    dof=9,
    max_steps=300,
    goal_threshold=0.20,
    # 高冗余度配置
)
```
- **应用**: 冗余度研究，障碍物回避
- **复杂度**: 高，挑战性强
- **收敛性**: 需要先进算法和调参

## 📊 训练配置

### 3DOF推荐配置
```python
config_3dof = {
    'episodes': 50000,
    'max_episode_steps': 200,
    'goal_threshold': 0.30,
    'buffer_capacity': 500000,
    'batch_size': 512,
    'training_rounds': 50,
    'actor_hidden_dims': [256, 256],    # 较小网络即可
    'critic_hidden_dims': [256, 256],
    'learning_starts': 10000,
    'learn_every': 25
}
```

### 6DOF推荐配置  
```python
config_6dof = {
    'episodes': 100000,             # 更多episodes
    'max_episode_steps': 250,       # 更长episode
    'goal_threshold': 0.25,         # 更严格阈值
    'buffer_capacity': 1000000,     # 更大buffer
    'batch_size': 512,
    'training_rounds': 50,
    'actor_hidden_dims': [512, 512], # 更大网络
    'critic_hidden_dims': [512, 512],
    'learning_starts': 20000,        # 更长warmup
    'learn_every': 25
}
```

## 🚀 使用示例

### 基本使用
```python
from pearl.utils.instantiations.environments import NDOFArmEnvironment
from pearl.utils.instantiations.environments.arm_her_factory import create_arm_her_buffer

# 创建3DOF环境
env = NDOFArmEnvironment(
    dof=3,
    max_steps=200,
    goal_threshold=0.30
)

# 创建专用HER buffer
her_buffer = create_arm_her_buffer(
    joint_dim=3,
    spatial_dim=2,          # 2D工作空间
    capacity=500000,
    threshold=0.30
)

# 训练循环
obs, action_space = env.reset()
for step in range(200):
    action = agent.act(obs)
    result = env.step(action)
    agent.observe(result)
    if result.terminated:
        print(f"Success in {step} steps!")
        break
```

### 多进程训练
```python
from pearl.user_envs.wrappers.subprocess_vector_env import SubprocVectorEnv

# 创建多进程环境
def make_env():
    return NDOFArmEnvironment(dof=6, max_steps=250)

vec_env = SubprocVectorEnv([make_env for _ in range(4)])

# 并行收集经验
observations = vec_env.reset()
for step in range(1000):
    actions = [agent.act(obs) for obs in observations]
    results = vec_env.step(actions)
    # 处理批量结果...
```

## 📈 性能基准

### 不同DOF的训练表现

| DOF | 收敛Episodes | 最终成功率 | 平均Episode长度 | 训练时间 |
|-----|-------------|------------|----------------|----------|
| 3DOF | 8,000 | 92-96% | 45-60步 | 2-3小时 |
| 6DOF | 25,000 | 85-92% | 80-120步 | 8-12小时 |
| 9DOF | 60,000+ | 75-85% | 120-180步 | 24+小时 |

### 算法对比 (3DOF基准)

| 算法 | 收敛Speed | 最终性能 | 样本效率 | 稳定性 |
|------|-----------|----------|----------|--------|
| SAC+HER | 快 | 95% | 高 | 优秀 |
| TD3+HER | 中 | 93% | 中等 | 好 |
| PPO | 慢 | 78% | 低 | 中等 |
| DDPG+HER | 中 | 89% | 中等 | 中等 |

## 🔧 高级功能

### 自定义工作空间
```python
# 定义矩形工作空间
custom_bounds = [
    -1.0, 1.0,  # X范围
    -0.5, 1.5   # Y范围  
]

env = NDOFArmEnvironment(
    dof=3,
    workspace_bounds=custom_bounds
)
```

### 障碍物回避 (9DOF)
```python
# 高DOF用于障碍物回避
env = NDOFArmEnvironment(
    dof=9,
    max_steps=400,
    goal_threshold=0.15,    # 更精确控制
    # 可以结合自定义reward function
)

def obstacle_reward(state, action, next_state):
    # 自定义奖励函数，惩罚进入障碍物区域
    base_reward = env._compute_reward(state)
    obstacle_penalty = check_obstacle_collision(next_state)
    return base_reward - obstacle_penalty
```

### 多目标任务
```python
# 连续多点到达任务
class MultiTargetNDOFArm(NDOFArmEnvironment):
    def __init__(self, targets, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = targets
        self.current_target_idx = 0
    
    def _compute_reward(self, state):
        current_target = self.targets[self.current_target_idx]
        distance = np.linalg.norm(state['ee_pos'] - current_target)
        
        if distance <= self.goal_threshold:
            self.current_target_idx = (self.current_target_idx + 1) % len(self.targets)
            return 10.0  # 到达奖励
        return -1.0
```

## ⚠️ 注意事项

### 已知限制
- **奇异性问题**: 某些关节配置可能导致奇异性
- **局部最优**: 高DOF容易陷入局部最优解
- **计算复杂度**: DOF增加导致指数级复杂度增长

### 调试技巧
```python
# 可视化关节配置
def visualize_arm_config(joint_angles):
    positions = env._forward_kinematics(joint_angles)
    # 绘制机械臂配置...
    
# 分析失败case
def analyze_failures(failed_episodes):
    for episode in failed_episodes:
        final_distance = episode['final_distance']
        joint_limits_hit = episode['joint_limits']
        # 分析失败原因...
```

### 性能优化
- **网络大小**: DOF增加时适当增大网络
- **Buffer容量**: 高DOF需要更大experience buffer
- **批量大小**: 复杂任务受益于更大batch size
- **学习率**: 高DOF可能需要更小学习率

## 🔗 扩展应用

### 工业应用
- **装配任务**: 精确定位和插入
- **焊接路径**: 连续轨迹跟踪  
- **拾取放置**: 抓取和放置物体

### 研究应用  
- **冗余度利用**: 9DOF以上的冗余关节控制
- **多任务学习**: 同时学习多种操作技能
- **迁移学习**: 从仿真到现实的知识迁移

### 教学应用
- **RL基础**: 经典的连续控制问题
- **多DOF控制**: 高维动作空间的挑战
- **HER算法**: 稀疏奖励环境的解决方案

## 📋 故障排除

### 常见问题

#### 训练不收敛
```python
# 检查点1: 网络容量
if dof >= 6:
    actor_hidden_dims = [512, 512]  # 增大网络
    
# 检查点2: 学习率  
if dof >= 9:
    actor_learning_rate = 1e-4  # 降低学习率
    
# 检查点3: Buffer大小
buffer_capacity = min(1000000, dof * 100000)
```

#### 成功率低
```python
# 增加HER采样
her_n_goals = min(8, dof)  # DOF越高，越多HER goals

# 调整阈值
if dof >= 6:
    goal_threshold *= 1.2  # 稍微放宽阈值
```

#### 训练不稳定
```python
# 增加探索
if dof >= 9:
    exploration_noise = 0.2  # 增加动作噪声
    
# 稳定性配置
training_rounds = min(100, dof * 10)
batch_size = max(256, dof * 64)
```

## 📖 相关文档

- [环境比较分析](./environment_comparison.md)
- [固定长度软体臂环境](./fixed_length_soft_arm.md)
- [变长软体臂环境](./variable_length_soft_arm.md)
- [HER算法实现](../algorithms/her_implementation.md)
- [多进程训练指南](../training/multiprocess_training.md)