from sac_optimized.sac import SAC

class AgentFactory:
    @staticmethod
    def create(agent_type: str, config: dict, env):
        """创建智能体
        Args:
            agent_type: 智能体类型 ('sac')
            config: 智能体配置，必须包含：
                hidden_dims: list[int]
                buffer_size: int
                batch_size: int
                gamma: float
                tau: float
                alpha: float
                lr: float
            env: 训练环境
        """
        if agent_type.lower() == 'sac':
            sac_config = config.copy()
            
            # 移除不需要的参数
            sac_config.pop('type', None)
            sac_config.pop('device', None)
            sac_config.pop('num_envs', None)  # 移除配置中的num_envs，因为我们会从env获取
            
            # 保留高级配置
            optimizer_config = sac_config.pop('optimizer', None)
            lr_schedule_config = sac_config.pop('lr_schedule', None)
            
            # 处理学习率参数
            lr = sac_config.pop('lr')
            sac_config.update({
                'actor_lr': lr,
                'critic_lr': lr,
                'alpha_lr': lr,
                'optimizer_config': optimizer_config,
                'lr_schedule_config': lr_schedule_config,
                'num_envs': env.num_envs  # 从env获取num_envs
            })
            
            return SAC(
                observation_space_shape=env.single_observation_space.shape,
                action_space_shape=env.single_action_space.shape,
                **sac_config  # num_envs已经在sac_config中了
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}") 