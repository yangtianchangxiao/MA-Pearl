# Pearl通用Agent可视化器

## 概述

`visualize_universal_agent.py` 是一个通用的Pearl Agent可视化工具，支持：

- **任意网络类型**: MLP, Graph Transformer
- **多种环境**: NDOF机械臂, 软体机械臂, 变长软体机械臂
- **灵活输入**: 支持有checkpoint和无checkpoint的演示
- **多种输出**: 静态截图序列、最终性能图

## 支持的配置

### 环境类型
- `ndof_3dof`: 3自由度刚体机械臂 (2D可视化)
- `soft_arm_6dof`: 6自由度软体机械臂 (3D可视化) 
- `variable_soft_arm_6dof`: 变长6自由度软体机械臂 (3D可视化)

### 网络类型
- `mlp`: 标准多层感知机网络
- `graph`: Graph Transformer网络 (需要Graph组件可用)

## 使用方法

### 基础用法

```bash
# 使用checkpoint演示训练好的Graph网络
python visualize_universal_agent.py \
    --checkpoint /path/to/checkpoint.pt \
    --env-type variable_soft_arm_6dof \
    --network-type graph \
    --episodes 3

# 无checkpoint演示(随机初始化)
python visualize_universal_agent.py \
    --checkpoint "" \
    --env-type ndof_3dof \
    --network-type mlp \
    --episodes 2 \
    --no-gif
```

### 参数说明

- `--checkpoint`: Checkpoint文件路径 (可选，为空则使用随机初始化)
- `--env-type`: 环境类型 (`ndof_3dof`, `soft_arm_6dof`, `variable_soft_arm_6dof`)
- `--network-type`: 网络类型 (`mlp`, `graph`) 
- `--episodes`: 演示episodes数量 (默认: 3)
- `--no-gif`: 不录制GIF，只保存静态截图

### 示例命令

```bash
# 1. Graph网络 + 变长软体臂 + checkpoint
python visualize_universal_agent.py \
    --checkpoint ./graph_variable_arm_results/graph_variable_soft_arm_6dof/best_checkpoint.pt \
    --env-type variable_soft_arm_6dof \
    --network-type graph \
    --episodes 1

# 2. MLP网络 + NDOF环境 + 随机初始化  
python visualize_universal_agent.py \
    --checkpoint "" \
    --env-type ndof_3dof \
    --network-type mlp \
    --episodes 2 \
    --no-gif

# 3. 快速演示所有配置
python demo_visualization.py
```

## 输出文件

可视化器会创建以下输出：

```
visualization_{env_type}_{network_type}/
├── episode_1_step_010.png    # 每10步的截图
├── episode_1_step_020.png
├── ...
└── final_performance.png     # 最终性能截图
```

## 架构特点

### 通用设计
- **环境抽象**: 统一的环境配置映射
- **观测解析**: 自动解析不同环境的观测格式
- **网络兼容**: 无缝支持MLP和Graph网络
- **错误处理**: 优雅处理checkpoint加载失败等异常

### 可视化功能
- **2D/3D渲染**: 根据环境自动选择可视化方式
- **轨迹追踪**: 显示末端执行器运动轨迹
- **状态信息**: 实时显示距离、奖励、网络类型等
- **多格式输出**: 支持PNG截图序列

### 扩展性
- **新环境**: 在`ENVIRONMENT_CONFIGS`中添加配置
- **新网络**: 在`UniversalAgentLoader.create_agent()`中添加支持
- **新可视化**: 继承`UniversalVisualizer`类自定义渲染

## 环境配置详解

每个环境配置包含：

```python
'environment_name': {
    'class': EnvironmentClass,              # 环境类
    'params': {...},                        # 环境参数
    'buffer_factory': create_buffer_func,   # HER buffer工厂
    'buffer_params': {...},                 # Buffer参数
    'obs_parser': 'parser_type',           # 观测解析器
    'visualizer': '2d' or '3d'             # 可视化类型
}
```

## 故障排除

### 常见问题

1. **Checkpoint加载失败**
   - 检查文件路径是否正确
   - 确认checkpoint格式兼容
   - 使用`weights_only=False`参数

2. **Graph网络不可用**
   - 确认Graph组件已正确导入
   - 检查Pearl框架版本

3. **中文字体警告**
   - 不影响功能，可忽略
   - 或安装中文字体支持

### 调试模式

可以修改代码添加更多调试信息：

```python
# 在demonstrate_agent()函数中添加
print(f"Debug: checkpoint keys: {checkpoint.keys()}")
print(f"Debug: config: {config}")
```

## 与现有可视化器对比

| 特性 | 通用可视化器 | 专用可视化器 |
|------|------------|------------|
| 环境支持 | 多种环境 | 单一环境 |
| 网络支持 | MLP + Graph | 仅MLP |
| checkpoint | 灵活加载 | 固定格式 |  
| 扩展性 | 高 | 低 |
| 使用复杂度 | 中等 | 低 |

## 最佳实践

1. **性能比较**: 使用相同参数比较不同网络类型
2. **调试训练**: 使用无checkpoint模式验证环境设置
3. **结果展示**: 生成多个episode的截图序列
4. **错误诊断**: 观察随机初始化agent的行为模式