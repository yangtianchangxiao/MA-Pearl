from torch.utils.tensorboard import SummaryWriter
from training.logger import TrainingLogger
from training.evaluator import Evaluator
from training.curriculum import CurriculumManager
from training.trainer import Trainer

class TrainingSystem:
    def __init__(self, agent, train_envs, eval_env, config):
        """
        Args:
            config: 训练配置，必须包含：
                max_steps: int
                eval_interval: int
                log_interval: int
                num_updates: int
                update_interval: int
                curriculum: dict
        """
        self.agent = agent
        self.train_envs = train_envs
        self.eval_env = eval_env
        
        # 验证配置完整性
        required_keys = ['max_steps', 'eval_interval', 'log_interval']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        self.config = config
        self.writer = SummaryWriter(comment="_SAC_RobotArm")
        
        # 创建训练组件
        self.logger = TrainingLogger(writer=self.writer)
        self.evaluator = Evaluator()
        self.curriculum_manager = CurriculumManager(config.get('curriculum', {}))
        
        self.trainer = Trainer(
            agent=agent,
            envs=train_envs,
            logger=self.logger,
            evaluator=self.evaluator,
            curriculum_manager=self.curriculum_manager,
            config=config
        )
        
        # 添加训练控制参数
        self.early_stopping = {
            'enabled': config.get('early_stopping', {}).get('enabled', False),
            'patience': config.get('early_stopping', {}).get('patience', 10),
            'min_delta': config.get('early_stopping', {}).get('min_delta', 0.0),
            'best_reward': float('-inf'),
            'no_improve_count': 0
        }
        
        self.checkpoint = {
            'save_freq': config.get('checkpoint', {}).get('save_freq', 10000),
            'keep_best': config.get('checkpoint', {}).get('keep_best', True),
            'max_to_keep': config.get('checkpoint', {}).get('max_to_keep', 5),
            'dir': 'checkpoints'
        }
    
    def train(self):
        """开始训练"""
        try:
            self.trainer.train()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            self.train_envs.close()
            self.eval_env.close()
            self.logger.close() 