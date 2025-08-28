import os
import torch
from gymnasium.vector import AsyncVectorEnv
from env.environment import RobotArmEnv
from env.env_factory import EnvFactory
from sac_optimized.sac import SAC
from training.logger import TrainingLogger
from torch.utils.tensorboard import SummaryWriter
from training.evaluator import Evaluator
from training.curriculum import CurriculumManager
from training.trainer import Trainer
from config.config_manager import ConfigManager
from agent.agent_factory import AgentFactory
from training.training_system import TrainingSystem
import setproctitle

def main():
    # 设置一个容易识别的进程名称
    setproctitle.setproctitle("RobotArm_SAC")
    
    # 加载配置
    config_manager = ConfigManager('config/default.yaml')
    
    # 获取训练配置
    training_config = config_manager.get_training_config()
    
    # 获取环境配置并确保包含所有必要字段
    env_config = config_manager.get_env_config()
    
    # 确保基本字段存在
    if 'ball_config' not in env_config:
        env_config['ball_config'] = {
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
    
    if 'reward_scale' not in env_config:
        env_config['reward_scale'] = {
            'distance': 0.5,
            'distance_intention': 0.25,
            'angle': 0.5,
            'angle_intention': 0.25,
            'catch': 100.0,
            'smooth': 0.1,
            'magnitude': 0.05,
            'jerk': 0.05
        }
    
    # 获取第一阶段的课程学习配置
    if training_config.get('curriculum', {}).get('enabled', False):
        curriculum_stages = training_config['curriculum'].get('stages', {})
        if curriculum_stages and 1 in curriculum_stages:
            stage_one_config = curriculum_stages[1]
            print("\n获取第一阶段课程学习配置：")
            print(stage_one_config)
            
            # 更新环境配置
            if 'reward_scale' in stage_one_config:
                env_config['reward_scale'] = stage_one_config['reward_scale'].copy()
            if 'ball_config' in stage_one_config:
                # 递归更新ball_config
                def update_dict_recursive(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict):
                            d[k] = update_dict_recursive(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d
                update_dict_recursive(env_config['ball_config'], stage_one_config['ball_config'])
            
            print("\n更新后的环境配置：")
            print(f"reward_scale: {env_config['reward_scale']}")
            print(f"ball_config: {env_config['ball_config']}")
    
    # 创建环境工厂和环境
    env_factory = EnvFactory(env_config)
    train_envs = env_factory.create_vector_env(num_envs=8)
    eval_env = env_factory.create_single_env()
    
    # 创建智能体
    agent = AgentFactory.create(
        agent_type='sac',
        config=config_manager.get_agent_config(),
        env=train_envs
    )
    
    # 创建训练系统
    training_system = TrainingSystem(
        agent=agent,
        train_envs=train_envs,
        eval_env=eval_env,
        config=training_config
    )
    
    # 开始训练
    training_system.train()

if __name__ == "__main__":
    main()
