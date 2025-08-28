import yaml
import os

class ConfigManager:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.validate_config()
    
    def load_config(self, config_path):
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def validate_config(self):
        """验证配置完整性"""
        required_sections = ['agent_config', 'env_config', 'training_config']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get_env_config(self):
        """获取环境配置"""
        return self.config['env_config']
    
    def get_training_config(self):
        """获取训练配置"""
        return self.config['training_config']
    
    def get_agent_config(self):
        """获取智能体配置"""
        return self.config['agent_config'] 