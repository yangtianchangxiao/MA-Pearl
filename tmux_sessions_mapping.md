# Tmux会话映射记录

## 当前运行的训练会话

### 1. `complex_kinematics_training` 
- **阈值**: 0.05 (最难)
- **网络**: 大网络版本 (hidden_dim=256, num_gnn_layers=3, batch_size=512)
- **状态**: 刚重启，使用更大网络配置

### 2. `complex_kinematics_0_1`
- **阈值**: 0.10 (中等)  
- **网络**: 大网络版本 (hidden_dim=256, num_gnn_layers=3, batch_size=512)
- **状态**: 刚重启，使用更大网络配置

### 3. `complex_kinematics_original`
- **阈值**: 0.15 (最易)
- **网络**: 轻量版本 (hidden_dim=128, num_gnn_layers=2, batch_size=256) 
- **状态**: 继续运行，作为对照组 (30.0%成功率)

## 实验设计
- **对比1**: 0.05 vs 0.10 vs 0.15 阈值难度对学习的影响
- **对比2**: 大网络 vs 小网络 对困难任务的处理能力
- **所有训练**: 都使用复杂运动学 (C++硬件兼容)

## 检查命令
```bash
tmux capture-pane -t complex_kinematics_training -p | tail -5  # 0.05大网络
tmux capture-pane -t complex_kinematics_0_1 -p | tail -5       # 0.10大网络  
tmux capture-pane -t complex_kinematics_original -p | tail -5   # 0.15轻量
```