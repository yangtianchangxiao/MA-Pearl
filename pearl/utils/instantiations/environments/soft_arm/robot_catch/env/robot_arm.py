import numpy as np
import cpp_robot_arm

class RobotArm:
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
        
        print("RobotArm initialized with {} segments and segment length {}".format(n_segments, segment_length))
        # 创建C++机械臂实例
        self.robot = cpp_robot_arm.RobotArm(n_segments, segment_length)
        
        self.dt = 0.02  # 时间步长
    
    def get_action_dim(self):
        print(f"Action dimension: {self.n_segments * 2}")  # 每个关节2个自由度
        return self.n_segments * 2
    
    def get_config_dim(self):
        """获取构型空间维度"""
        return 2 * self.n_segments  # 每个关节2个角度（alpha和beta）
    
    def get_positions_dim(self):
        """获取位置空间维度"""
        return 3 * (self.n_segments + 1)  # 每个关节和末端执行器的3D位置

    def reset(self, config=None):
        """重置机械臂状态"""
        if config is None:
            config = np.zeros(2 * self.n_segments)
        self.set_config(config)
        self.last_action = np.zeros(2 * self.n_segments)
        self.last_ee_position = np.array(self.robot.get_end_effector_position())
        return self.get_state()
    
    def set_config(self, config):
        """设置机械臂构型，带关节限制"""
        # 确保构型在限制范围内
        limited_config = []
        for i in range(0, 2 * self.n_segments, 2):
            alpha = np.clip(config[i], *self.joint_limits['alpha'])
            beta = np.clip(config[i+1], *self.joint_limits['beta'])
            limited_config.extend([alpha, beta])
        
        self.config_state = np.array(limited_config)
        self.robot.set_config(limited_config)
    
    def apply_action(self, action):
        """应用动作（角速度），带速度限制"""
        # 确保action是numpy数组并且形状正确
        action = np.array(action).flatten()  # 先展平
        if len(action) != 2 * self.n_segments:
            raise ValueError(f"Action dimension mismatch. Expected {2 * self.n_segments}, got {len(action)}")
        
        # 记录当前末端执行器位置
        self.last_ee_position = np.array(self.robot.get_end_effector_position())
        
        # 将[-1, 1]的动作直接映射到速度范围
        scaled_action = []
        for i in range(0, 2 * self.n_segments, 2):
            alpha_vel = action[i] * self.velocity_limits['alpha']  # 直接乘以最大速度
            beta_vel = action[i+1] * self.velocity_limits['beta']
            scaled_action.extend([alpha_vel, beta_vel])
        
        # 记录动作
        self.last_action = np.array(scaled_action)
        
        # 更新关节角度
        new_config = self.config_state + np.array(scaled_action) * self.dt
        self.set_config(new_config)  # 这里会自动应用关节限制
    
    def get_state(self):
        """获取机械臂的当前状态"""
        current_ee_position = np.array(self.robot.get_end_effector_position())
        
        # 计算末端执行器速度
        if self.last_ee_position is None:
            ee_velocity = np.zeros(3)
        else:
            ee_velocity = (current_ee_position - self.last_ee_position) / self.dt
        
        return {
            'config': self.config_state.copy(),
            'positions': np.array(self.robot.get_positions()),
            'end_effector': {
                'position': current_ee_position,
                'direction': np.array(self.robot.get_end_effector_direction()),
                'velocity': ee_velocity
            }
        }
    
    def get_last_action(self):
        """获取最后一次执行的动作"""
        return self.last_action.copy()
    
    def check_collision(self, ball_pos, catch_radius=0.05):
        """检查是否抓住球"""
        ee_pos = np.array(self.robot.get_end_effector_position())
        distance = np.linalg.norm(ee_pos - ball_pos)
        return distance <= catch_radius

    def step(self, action):
        """执行一步动作并返回新状态
        Args:
            action: 动作向量，包含每个关节的角速度
        Returns:
            dict: 包含机械臂当前状态的字典
        """
        self.apply_action(action)
        return self.get_state()

    def predict_next_state(self, action):
        """预测给定动作后的下一个状态
        Args:
            action: 动作数组，shape=(2*n_segments,)，每个关节的alpha和beta速度
        Returns:
            predicted_state: 预测的下一个状态
        """
        # 确保action是numpy数组并且形状正确
        action = np.array(action).flatten()
        if len(action) != 2 * self.n_segments:
            raise ValueError(f"Action dimension mismatch. Expected {2 * self.n_segments}, got {len(action)}")
        
        # 将[-1, 1]的动作直接映射到速度范围
        scaled_action = []
        for i in range(0, 2 * self.n_segments, 2):
            alpha_vel = action[i] * self.velocity_limits['alpha']
            beta_vel = action[i+1] * self.velocity_limits['beta']
            scaled_action.extend([alpha_vel, beta_vel])
        
        # 预测下一个构型
        next_config = self.config_state + np.array(scaled_action) * self.dt
        
        # 确保构型在限制范围内
        limited_next_config = []
        for i in range(0, 2 * self.n_segments, 2):
            alpha = np.clip(next_config[i], *self.joint_limits['alpha'])
            beta = np.clip(next_config[i+1], *self.joint_limits['beta'])
            limited_next_config.extend([alpha, beta])
        
        # 临时设置下一个构型来获取状态
        original_config = self.config_state.copy()
        self.set_config(limited_next_config)
        predicted_state = self.get_state()
        # 恢复原始构型
        self.set_config(original_config)
        
        return predicted_state
