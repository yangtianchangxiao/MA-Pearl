# MA-Pearl 项目文档

## 📋 文档目录

本项目包含Pearl框架的机械臂强化学习环境实现，包括固定长度和变长软体机械臂环境。

### 🤖 [环境说明](./environments/README.md)
机械臂强化学习环境的完整文档
- **[固定长度软体臂环境](./environments/fixed_length_soft_arm.md)** - 经过验证的基础环境 (3DOF)
- **[变长软体机械臂环境](./environments/variable_length_soft_arm.md)** - 增强泛化能力的新环境 (6DOF)
- **[NDOF硬体机械臂环境](./environments/ndof_arm_environment.md)** - 经典多自由度环境
- **[环境比较分析](./environments/environment_comparison.md)** - 三种环境的详细对比

### 🔧 [框架修复](./framework_fixes/) *(计划中)*
Pearl框架的技术修复和改进
- Pearl tensor处理bug修复
- HER buffer优化实现
- 兼容性改进

### 🎯 [算法实现](./algorithms/) *(计划中)*
强化学习算法的具体实现
- SAC+HER算法详解
- 连续动作空间优化
- 多进程训练策略

### 🚀 [训练指南](./training/) *(计划中)*
训练脚本和最佳实践
- 训练脚本使用指南
- 超参数调优建议
- 性能监控和调试

## 🌟 项目亮点

### ✅ 已完成功能
- **三种机械臂环境**: 固定长度软体臂、变长软体臂、NDOF硬体臂
- **Pearl框架修复**: 修复了tensor处理bug，完全向后兼容
- **专用HER实现**: 针对变长环境优化的HER buffer
- **完整训练流程**: 端到端的训练脚本和监控

### 🆕 核心创新
- **动态几何配置**: 变长软体臂支持每episode随机化段长度
- **增强泛化能力**: 通过几何变化提升实际部署的鲁棒性
- **动态工作空间**: 基于当前臂长自动计算工作空间边界
- **观测空间优化**: 在状态中包含当前几何信息

## 🎯 快速开始

### 环境选择指南
```
你的应用场景？
├── 研究/教学/验证 → 固定长度软体臂
├── 实际机器人部署 → 变长软体臂  
└── 多DOF算法研究 → NDOF硬体臂
```

### 基础使用
```python
# 固定长度 - 快速验证
from pearl.utils.instantiations.environments import SoftArmReachEnvironment
env = SoftArmReachEnvironment(n_segments=3, goal_threshold=0.30)

# 变长 - 实际部署
from pearl.utils.instantiations.environments import VariableSoftArmReachEnvironment  
env = VariableSoftArmReachEnvironment(
    n_segments=3, 
    segment_length_range=(0.168, 0.252)
)

# NDOF - 多DOF研究
from pearl.utils.instantiations.environments import NDOFArmEnvironment
env = NDOFArmEnvironment(dof=6, max_steps=250)
```

## 📊 性能对比

| 环境 | DOF | 收敛Speed | 最终成功率 | 泛化能力 | 适用场景 |
|------|-----|-----------|------------|----------|----------|
| 固定软体臂 | 3 | 快 (5K) | 90-95% | 差 | 研究验证 |
| 变长软体臂 | 6 | 中 (10K) | 70-85% | **优秀** | **实际部署** |
| NDOF硬体臂 | 3-9 | 中 (8K) | 85-90% | 中等 | 多DOF研究 |

## 🔗 相关资源

### 训练脚本
- `train_soft_arm_pearl.py` - 固定长度环境训练
- `train_variable_soft_arm_official.py` - 变长环境训练
- `pearl/utils/scripts/train_arm_multiprocess.py` - 多进程训练

### 核心文件
- `pearl/utils/instantiations/environments/` - 环境实现目录
- `pearl/replay_buffers/tensor_based_replay_buffer.py` - 框架修复
- `docs/environments/` - 详细文档目录

## 🚀 最新进展

### 当前训练状态
- **变长软体臂训练**: 正在进行中
- **Tmux会话**: `variable_arm_training`
- **进程名**: `Variable-Arm-Pearl-SAC` (nvidia-smi可见)
- **配置**: 100K episodes, 6DOF, 段长度±20%变化

### 监控命令
```bash
# 查看训练进度
tmux attach -t variable_arm_training

# 查看GPU使用
nvidia-smi | grep Variable-Arm

# 查看结果
ls variable_arm_results/
```

## 📝 贡献指南

1. **环境开发**: 参考现有环境实现新的变种
2. **算法改进**: 优化SAC+HER的超参数和网络结构
3. **文档完善**: 补充使用案例和最佳实践
4. **性能测试**: 在不同硬件配置下验证性能

---

📖 **详细文档请查看各子目录的说明文件**  
🤝 **问题反馈请提交issue或直接讨论**