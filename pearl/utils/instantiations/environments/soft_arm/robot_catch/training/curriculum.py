from collections import deque
import numpy as np

class CurriculumConfig:
    def __init__(self):
        # 课程学习基本配置
        self.curriculum_enabled = True
        self.min_success_rate = 0.6     # 进入下一阶段所需的最小成功率
        self.eval_window = 100          # 评估窗口大小
        self.stage_steps = 50000        # 每个阶段的最小步数
        
        # 课程阶段配置
        self.stages = {
            1: {  # 第一阶段：固定位置，低速
                'ball_config': {
                    'pos_range': {
                        'x': [-0.9, -0.9],
                        'y': [0.0, 0.0],
                        'z': [1.1, 1.1]
                    },
                    'throw_config': {
                        'speed': [2.0, 2.0],
                        'angle_h': [0, 0],
                        'angle_v': [-10, -10]
                    }
                }
            },
            2: {  # 第二阶段：固定高度，变化位置
                'ball_config': {
                    'pos_range': {
                        'x': [-1.0, -0.8],
                        'y': [-0.2, 0.2],
                        'z': [1.1, 1.1]
                    },
                    'throw_config': {
                        'speed': [2.0, 2.5],
                        'angle_h': [-10, 10],
                        'angle_v': [-10, -10]
                    }
                }
            },
            3: {  # 第三阶段：完全随机
                'ball_config': {
                    'pos_range': {
                        'x': [-1.0, -0.8],
                        'y': [-0.3, 0.3],
                        'z': [1.0, 1.2]
                    },
                    'throw_config': {
                        'speed': [2.0, 3.0],
                        'angle_h': [-15, 15],
                        'angle_v': [-15, 0]
                    }
                }
            }
        }

class CurriculumManager:
    def __init__(self, config):
        """初始化课程学习管理器
        Args:
            config: 课程学习配置，包含stages等信息
        """
        # 检查课程学习是否启用
        self.enabled = config.get('enabled', True)
        if not self.enabled:
            print("课程学习已禁用")
            return
            
        # 基本参数
        self.success_threshold = config.get('success_threshold', 0.7)
        self.min_episodes = config.get('min_episodes', 100)
        self.stages = config.get('stages', {})
        
        print("\n课程学习初始配置：")
        print(f"Stages: {self.stages}")
        
        if not self.stages:
            print("警告：未找到课程学习阶段配置")
            self.enabled = False
            return
            
        self.current_stage = 1
        self.steps_in_stage = 0
        self.total_steps = 0
        
        # 性能指标
        self.metrics = {
            'success_rate': deque(maxlen=100),
            'catch_distances': deque(maxlen=100),
            'reaction_times': deque(maxlen=100)
        }
        
        # 记录每个阶段的表现
        self.stage_history = []
        print(f"课程学习已启用，共{len(self.stages)}个阶段")
        print(f"当前阶段配置：{self.get_current_config()}")
        
    def update(self, info):
        """更新课程学习状态"""
        if not self.enabled:
            return False
            
        # 更新步数
        self.steps_in_stage += 1
        self.total_steps += 1
        
        # 收集性能指标
        self.metrics['success_rate'].append(float(info['success']))
        self.metrics['catch_distances'].append(info.get('catch_distance', 0.0))
        self.metrics['reaction_times'].append(info.get('reaction_time', 0.0))
        
        # 评估是否应该进入下一阶段
        if self._should_advance():
            self._advance_stage()
            return True
        return False
        
    def _should_advance(self):
        """判断是否应该进入下一阶段"""
        if not self.enabled:
            return False
            
        # 检查是否有足够的样本
        if len(self.metrics['success_rate']) < self.min_episodes:
            return False
            
        # 检查是否达到最小步数要求
        if self.steps_in_stage < 50000:  # 每个阶段至少训练50000步
            return False
            
        # 计算成功率
        success_rate = np.mean(self.metrics['success_rate'])
        
        # 检查是否达到进入下一阶段的条件
        return (success_rate >= self.success_threshold and 
                self.current_stage < len(self.stages))
    
    def _advance_stage(self):
        """进入下一个课程阶段"""
        if not self.enabled:
            return
            
        # 保存当前阶段的表现
        self.stage_history.append({
            'stage': self.current_stage,
            'steps': self.steps_in_stage,
            'success_rate': np.mean(self.metrics['success_rate']),
            'avg_distance': np.mean(self.metrics['catch_distances']),
            'avg_reaction_time': np.mean(self.metrics['reaction_times'])
        })
        
        # 进入下一阶段
        self.current_stage += 1
        self.steps_in_stage = 0
        
        # 重置指标
        self.metrics = {
            'success_rate': deque(maxlen=100),
            'catch_distances': deque(maxlen=100),
            'reaction_times': deque(maxlen=100)
        }
        
        print(f"\n进入课程学习第{self.current_stage}阶段")
        
        # 获取新阶段的配置
        new_config = self.get_current_config()
        print(f"新阶段配置: {new_config}")
        
        # 更新所有环境的配置
        if hasattr(self, 'envs'):
            # 如果是向量化环境
            if hasattr(self.envs, 'env_fns'):
                for env_fn in self.envs.env_fns:
                    env = env_fn()
                    env.update_curriculum_config(new_config)
            # 如果是单个环境
            else:
                self.envs.update_curriculum_config(new_config)
        
        return new_config
    
    def get_current_config(self):
        """获取当前阶段的配置"""
        if not self.enabled or self.current_stage > len(self.stages):
            return {}
        
        # 使用整数键而不是字符串键
        stage_config = self.stages.get(self.current_stage, {})
        
        print(f"\n获取第{self.current_stage}阶段配置：")
        print(f"Stage config: {stage_config}")
        
        # 确保返回的是配置的深拷贝，避免意外修改
        config = {
            'reward_scale': stage_config.get('reward_scale', {}),
            'ball_config': stage_config.get('ball_config', {})
        }
        print(f"返回配置：{config}")
        return config
    
    def get_stats(self):
        """获取当前课程学习状态统计"""
        success_rate = 0
        if self.metrics['success_rate']:
            success_rate = np.mean(self.metrics['success_rate'])
            
        return {
            'current_stage': self.current_stage,
            'steps_in_stage': self.steps_in_stage,
            'total_steps': self.total_steps,
            'success_rate': success_rate,
            'window_size': len(self.metrics['success_rate'])
        }
    
    def should_increase_steps(self):
        """检查是否应该增加最大步数"""
        # 直接返回False，禁用步数调整
        return False