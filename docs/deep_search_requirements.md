# 国萌教授研究深度搜索需求清单
## 用于指导AI深度搜索的关键技术内容

**目标**: 学习国萌教授的多智能体协同方法，应用到软体臂博弈层设计

---

## 🎯 **优先级1: 核心数学方法**

### 1.1 协同推送论文的数学建模
**搜索关键词**: `"Collaborative Planar Pushing" mathematical formulation contact forces optimization`

**需要获取的具体内容**:
```
- 多机器人协同的目标函数设计
- 接触力约束的数学表示
- 分布式优化的求解算法
- 混合系统的状态切换逻辑
- 实时控制的计算复杂度分析
```

### 1.2 时间最小化与在线同步的优化方法
**搜索关键词**: `"Time minimization online synchronization" branch and bound temporal logic`

**需要获取的具体内容**:
```
- Branch and Bound搜索算法的具体实现
- 任务分解的偏序分析方法
- 在线适应算法的数学公式
- 时间复杂度和最优性保证
- 动态重分配的触发机制
```

---

## 🎯 **优先级2: 工程实现细节**

### 2.1 分布式控制架构
**搜索关键词**: `"multi-agent distributed control" NMPC mode switching real-time`

**需要获取的具体内容**:
```
- 模式切换策略的具体算法
- NMPC控制器的参数设置
- 重规划模块的触发条件
- 实时性能的具体指标
- 通信协议和延迟处理
```

### 2.2 约束处理和碰撞避免
**搜索关键词**: `"polytopic objects" contact mode generation collision avoidance`

**需要获取的具体内容**:
```
- 多方向可行性分析的算法
- 线性规划求解器的选择
- 接触模式生成的具体步骤
- 碰撞检测和避免的实现
- 约束违背时的恢复机制
```

---

## 🎯 **优先级3: 理论基础**

### 3.1 多智能体系统理论
**搜索关键词**: `"multi-agent systems" consensus theory distributed optimization game theory`

**需要获取的具体内容**:
```
- 一致性理论在协同控制中的应用
- 分布式优化的收敛性证明
- 博弈论均衡解的存在性条件
- 网络拓扑对性能的影响
- 鲁棒性分析方法
```

### 3.2 时态逻辑和形式化方法
**搜索关键词**: `"Linear Temporal Logic" LTL automaton task decomposition`

**需要获取的具体内容**:
```
- LTL公式到自动机的转换
- 任务自动机的偏序分析
- 协同任务的形式化表示
- 可达性和安全性验证
- 在线适应的正确性保证
```

---

## 🎯 **优先级4: 与软体臂的对应关系**

### 4.1 机器人vs关节的对应
**搜索关键词**: `"robot coordination" "joint coordination" "manipulator control" distributed`

**需要获取的具体内容**:
```
- 多机器人协调 → 多关节协调的映射关系
- 接触力约束 → 关节力矩约束的转换
- 任务分解 → 运动分解的方法
- 模式切换 → 控制模式切换的对应
- 在线适应 → 参数在线调整的策略
```

### 4.2 软体臂特有的挑战
**搜索关键词**: `"soft robotics" "continuum robot" "distributed control" "compliance"`

**需要获取的具体内容**:
```
- 软体材料的分布式建模方法
- 连续性约束的处理技术
- α,β耦合的数学表示
- 弹性变形的实时补偿
- 软体动力学的简化模型
```

---

## 📚 **具体论文和资源**

### 必读论文清单
```
1. "Collaborative Planar Pushing of Polytopic Objects with Multiple Robots in Complex Scenes" (2024)
   - ArXiv: 2405.07908
   - 重点: 数学建模、优化算法、实验结果

2. "Time Minimization and Online Synchronization for Multi-agent Systems under Collaborative Temporal Logic Tasks" 
   - ArXiv: 2208.07756
   - 重点: BnB算法、在线适应、理论分析

3. "Fast and Adaptive Multi-Agent Planning under Collaborative Temporal Logic Tasks via Poset Products"
   - 重点: 快速规划、偏序分析、算法优化
```

### 相关技术资源
```
1. 国萌教授的GitHub代码库 (如果有公开代码)
2. 北大工学院相关技术报告
3. 多智能体系统的经典教材章节
4. 分布式优化的综述论文
```

---

## 🎯 **搜索策略建议**

### 搜索优先级
1. **先搜索具体算法**: 数学公式、伪代码、参数设置
2. **再搜索工程细节**: 实现架构、性能指标、调试方法  
3. **最后搜索理论基础**: 收敛性证明、稳定性分析

### 搜索关键信息
```
✅ 优先级最高: 具体的数学公式和算法步骤
✅ 重要: 参数设置和调优经验
✅ 有用: 实验结果和性能对比
✅ 参考: 理论分析和证明过程
```

---

## 💡 **预期输出格式**

对每个搜索主题，希望得到：
```
1. **数学公式**: 核心优化目标函数、约束条件
2. **算法流程**: 具体步骤的伪代码或描述
3. **参数设置**: 关键参数的推荐值和调优方法
4. **实现要点**: 工程实现的关键技巧和注意事项
5. **性能指标**: 计算复杂度、实时性、精度等指标
```

**最终目标**: 将这些方法转化为适用于软体臂多关节协同控制的博弈层设计方案。

---

**备注**: 这个搜索清单按照重要性排序，可以根据时间和资源情况选择性深入搜索。