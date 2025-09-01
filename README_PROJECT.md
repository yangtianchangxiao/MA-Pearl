# MA-Pearl: Variable DOF Soft Arm with Graph Neural Networks
## Multi-Agent Reinforcement Learning for Soft Robotics

🎯 **Project Goal**: Develop scalable multi-agent RL algorithms for soft robotic arms with variable degrees of freedom (DOF), enabling generalization from training on small configurations (2-5 segments) to arbitrary segment numbers.

## 🏆 **Current Achievements**

### ✅ Random DOF GNN Training (Latest)
- **Model**: Ultra-light GNN Actor (69K parameters, 2 layers)
- **Training Range**: 2-5 segments (4-10 DOF)
- **Success Rate**: 84.0% (1325 training episodes)
- **Cross-DOF Generalization**: 70% success rate across unseen configurations
- **Demo**: 8DOF soft arm completing task in 17 steps with 0.115m accuracy

### ✅ Comprehensive Visualization System
- **Static Analysis**: `visualize_random_dof_gnn.py` - Complete performance analysis
- **Animated GIF**: `create_gif_visualization.py` - Real-time motion visualization  
- **Simplified Testing**: `simple_visualize_random_dof.py` - Quick validation
- **Demo Output**: `corrected_soft_arm_demo.gif` - Working 8DOF demonstration

### ✅ Technical Infrastructure
- **Graph-HER Integration**: `optimized_graph_her_wrapper.py` - Seamless HER buffer compatibility
- **Ultra-Light Architecture**: `lightweight_gnn_actor.py` - 5-10x faster than standard approaches
- **Correct Kinematics**: Fixed forward kinematics for accurate visualization
- **Modular Design**: Clean separation between training, inference, and visualization

## 🚀 **Quick Start**

### View Current Model Performance
```bash
# Generate animated GIF demonstration
python create_gif_visualization.py --output demo.gif --fps 10

# Run comprehensive analysis
python visualize_random_dof_gnn.py --mode batch --n_tests 10

# Quick single test
python simple_visualize_random_dof.py
```

### Train New Model
```bash
# Train ultra-light GNN on random DOF configurations
python train_ultra_light_gnn_random_dof.py
```

## 📊 **Performance Metrics**

| Configuration | Success Rate | Avg Steps | Avg Distance | Status |
|--------------|-------------|-----------|--------------|---------|
| 4DOF (2 segments) | 100% | 15-25 | 0.14m | ✅ Excellent |
| 6DOF (3 segments) | Variable | 20-200 | 0.20m | ⚠️ Inconsistent |
| 8DOF (4 segments) | 100% | 17-47 | 0.14m | ✅ Excellent |
| 10DOF (5 segments) | 75% | 18-26 | 0.15m | ⭐ Good |
| **Overall** | **70%** | **Variable** | **0.16m** | **🎯 Target for improvement** |

## 🔬 **Research Direction: Game Theory Layer**

### Current Focus: Ad-Hoc Multi-Agent Coordination
We're developing a **game theory optimization layer** on top of the RL base policy:

```python
# Two-stage approach
μ = rl_pretrained_policy(state)          # Stage 1: RL base solution
a* = game_layer_optimization(μ, constraints)  # Stage 2: Multi-agent coordination
```

**Goal**: Improve generalization from 70% → 85%+ success rate for unseen DOF configurations.

### Research Resources
- **Search Strategy**: `docs/adhoc_multiagent_search_strategy.md` - Comprehensive literature search framework
- **Feasibility Analysis**: `docs/game_theory_feasibility_analysis.md` - Technical viability assessment  
- **Implementation Framework**: `docs/research_to_implementation_framework.md` - Research-to-code pipeline

## 📁 **Project Structure**

```
MA-Pearl/
├── 🎬 Visualization & Demo
│   ├── visualize_random_dof_gnn.py      # Complete analysis tool
│   ├── create_gif_visualization.py      # GIF animation creator  
│   ├── simple_visualize_random_dof.py   # Quick testing
│   └── corrected_soft_arm_demo.gif      # Current demo
│
├── 🧠 Core Training & Networks
│   ├── train_ultra_light_gnn_random_dof.py  # Main training script
│   ├── lightweight_gnn_actor.py             # Ultra-light GNN architecture
│   ├── optimized_graph_environment.py       # Graph-based environment
│   └── optimized_graph_her_wrapper.py       # HER integration
│
├── 📚 Documentation & Research  
│   ├── docs/adhoc_multiagent_search_strategy.md  # MARL+Game theory search
│   ├── docs/game_theory_feasibility_analysis.md  # Technical analysis
│   ├── docs/eudh_concept/                         # Extended DH framework
│   └── docs/archived_*/                           # Historical experiments
│
├── 🗃️ Results & Models
│   ├── random_dof_gnn_results/           # Current best model (84% success)
│   ├── soft_arm_pearl_results/           # Historical results
│   └── */results/                        # Other experiments
│
└── 🔧 Pearl Framework (Base)
    └── pearl/                            # Modified Pearl RL library
```

## 🎯 **Next Steps**

### Immediate Goals
1. **Game Theory Layer Implementation** - Add multi-agent coordination on top of RL base
2. **Higher DOF Testing** - Validate performance on 12+ DOF configurations  
3. **Performance Optimization** - Target 85%+ success rate across all DOF ranges

### Research Questions
- Can game theory optimization improve 6DOF performance from 0% → 60%+?
- How does the approach scale to 15+ segment configurations?
- What are the optimal cooperation strategies for soft arm coordination?

## 📖 **Documentation**

- **Visualization Guide**: `README_visualization.md` - Complete visualization usage
- **Project Overview**: `docs/PROJECT_OVERVIEW.md` - Comprehensive project summary
- **Research Framework**: `docs/research_to_implementation_framework.md` - Academic to code pipeline

## 🤝 **Contributing**

This is an active research project. Key areas for contribution:
- **Game Theory Integration**: Implementing multi-agent coordination algorithms
- **Higher DOF Scaling**: Testing and optimizing for 12+ segment configurations
- **Visualization Improvements**: Enhanced analysis and demonstration tools

---

## 📄 **Original Pearl Framework**

This project builds upon Meta's Pearl RL library. For original Pearl documentation, see the sections below.

---
