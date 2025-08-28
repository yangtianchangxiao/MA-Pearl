import numpy as np

class RobotArm:
    """纯Python版本的软体机械臂，用于测试"""
    
    def __init__(self, n_segments=3, segment_length=0.21, config=None):
        """初始化机械臂"""
        self.n_segments = n_segments
        self.segment_length = segment_length
        
        # 使用配置或默认值
        self.config = config or {
            'joint_limits': {
                'alpha': [-np.pi/2, np.pi/2],  # alpha角度范围：-90度到90度
                'beta': [-np.pi, np.pi]        # beta角度范围：-180度到180度
            },
            'velocity_limits': {
                'alpha': np.pi/2,  # alpha最大速度：90度/秒
                'beta': np.pi      # beta最大速度：180度/秒
            }
        }
        
        # 设置关节限制
        self.joint_limits = self.config['joint_limits']
        self.velocity_limits = self.config['velocity_limits']
        
        # 初始化状态
        self.config_state = np.zeros(2 * n_segments)  # [alpha1, beta1, alpha2, beta2, ...]
        self.last_action = np.zeros(2 * n_segments)  # 记录最后一次执行的动作
        self.last_ee_position = None  # 记录上一时刻末端执行器位置
        
        # 初始化机械臂C++对象（模拟）
        self.robot = self._create_mock_robot()
        
    def _create_mock_robot(self):
        """创建模拟机械臂对象"""
        class MockRobot:
            def forward_kinematics(self, angles):
                # 简化的正向运动学
                x, y, z = 0.0, 0.0, 0.0
                for i in range(0, len(angles), 2):
                    if i+1 < len(angles):
                        alpha = angles[i]
                        beta = angles[i+1] if i+1 < len(angles) else 0
                        # 简化计算
                        x += 0.21 * np.cos(alpha) * np.cos(beta)
                        y += 0.21 * np.cos(alpha) * np.sin(beta)  
                        z += 0.21 * np.sin(alpha)
                return [x, y, z]
                
        return MockRobot()
    
    def step(self, action, dt):
        """执行动作并更新状态"""
        # 应用速度限制
        clamped_action = np.zeros_like(action)
        for i in range(0, len(action), 2):
            # Alpha速度限制
            clamped_action[i] = np.clip(action[i], -self.velocity_limits['alpha'], self.velocity_limits['alpha'])
            # Beta速度限制  
            if i+1 < len(action):
                clamped_action[i+1] = np.clip(action[i+1], -self.velocity_limits['beta'], self.velocity_limits['beta'])
        
        # 更新配置
        self.config_state += clamped_action * dt
        
        # 应用关节限制
        for i in range(0, len(self.config_state), 2):
            # Alpha角度限制
            self.config_state[i] = np.clip(self.config_state[i], 
                                         self.joint_limits['alpha'][0], 
                                         self.joint_limits['alpha'][1])
            # Beta角度限制
            if i+1 < len(self.config_state):
                self.config_state[i+1] = np.clip(self.config_state[i+1],
                                                self.joint_limits['beta'][0], 
                                                self.joint_limits['beta'][1])
        
        # 记录动作
        self.last_action = clamped_action.copy()
        
        # 更新末端执行器位置
        self.last_ee_position = self.get_ee_position()
        
        return self.config_state.copy()
    
    def get_ee_position(self):
        """获取末端执行器位置"""
        return self.robot.forward_kinematics(self.config_state)
    
    def get_ee_direction(self):
        """获取末端执行器方向（简化版本）"""
        if len(self.config_state) >= 2:
            alpha = self.config_state[-2]  # 最后一段的alpha
            beta = self.config_state[-1] if len(self.config_state) > 1 else 0  # 最后一段的beta
            # 计算方向向量
            direction = [
                np.cos(alpha) * np.cos(beta),
                np.cos(alpha) * np.sin(beta),
                np.sin(alpha)
            ]
            return direction
        return [1.0, 0.0, 0.0]
    
    def reset(self):
        """重置机械臂状态"""
        self.config_state = np.zeros(2 * self.n_segments)
        self.last_action = np.zeros(2 * self.n_segments)
        self.last_ee_position = None
    
    def get_state(self):
        """获取当前状态"""
        return {
            'config': self.config_state.copy(),
            'ee_position': self.get_ee_position(),
            'ee_direction': self.get_ee_direction(),
            'last_action': self.last_action.copy()
        }