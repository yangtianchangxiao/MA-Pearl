from typing import Callable, List
from gymnasium.vector import AsyncVectorEnv
from env.environment import RobotArmEnv

class EnvFactory:
    def __init__(self, env_config):
        self.env_config = env_config
        # 获取课程学习配置
        self.curriculum_config = env_config.get('curriculum', {})
        if self.curriculum_config.get('enabled', False):
            # 如果启用了课程学习，使用第一阶段的配置
            stages = self.curriculum_config.get('stages', {})
            if stages and 1 in stages:
                stage_one_config = stages[1]
                # 更新环境配置
                if 'reward_scale' in stage_one_config:
                    self.env_config['reward_scale'] = stage_one_config['reward_scale'].copy()
                if 'ball_config' in stage_one_config:
                    self.env_config['ball_config'] = stage_one_config['ball_config'].copy()
                print("\nEnvFactory: 使用第一阶段课程学习配置")
                print(f"reward_scale: {self.env_config['reward_scale']}")

    def create_single_env(self) -> RobotArmEnv:
        """创建单个环境实例"""
        env = RobotArmEnv(self.env_config)
        print("\n环境创建完成，最终配置：")
        print(f"reward_scale: {env.config['reward_scale']}")
        print(f"ball_config: {env.config['ball_config']}")
        return env

    def create_vector_env(self, num_envs: int) -> AsyncVectorEnv:
        """创建向量化环境"""
        def make_env() -> Callable:
            def _init() -> RobotArmEnv:
                env = RobotArmEnv(self.env_config)
                return env
            return _init

        envs = AsyncVectorEnv(
            [make_env() for _ in range(num_envs)],
            copy=False  # 避免不必要的环境复制
        )
        print("\n向量化环境创建完成，每个子环境使用相同配置")
        return envs 