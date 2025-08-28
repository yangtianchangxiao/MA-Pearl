import numpy as np

class BallThrower:
    def __init__(self, config=None):
        self.config = config or {
            'g': 9.81,              # 重力加速度
            'pos_range': {
                'x': [-0.3, -0.2],  # 初始x位置范围
                'y': [-0.1, 0.1],   # 初始y位置范围
                'z': [0.3, 0.4]     # 初始z位置范围
            },
            'throw_config': {
                'speed': [2.0, 3.0],     # 出射速度范围 (m/s)
                'angle_h': [-20, 20],    # 水平角度范围 (度)
                'angle_v': [30, 45]      # 垂直角度范围 (度)
            },
            'target_range': {       # 期望落点范围
                'x': [0.3, 0.6],    # 目标x范围
                'y': [-0.3, 0.3]    # 目标y范围
            }
        }
        self.g = self.config['g']  # 重力加速度
        self.dt = 0.02  # 时间步长
        
        self.pos = None
        self.velocity = None
        
    def update_config(self, new_config):
        """更新配置
        Args:
            new_config: dict, 新的配置
        """
        # 递归更新配置
        def update_dict_recursive(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict_recursive(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        update_dict_recursive(self.config, new_config)
        print("\nBallThrower配置已更新：")
        print(f"pos_range: {self.config['pos_range']}")
        print(f"throw_config: {self.config['throw_config']}")
        
        # 重置状态
        self.reset()
        
    def calculate_velocity(self, speed, angle_h, angle_v):
        """根据速度和角度计算三个方向的速度分量"""
        # 转换为弧度
        angle_h_rad = np.deg2rad(angle_h)
        angle_v_rad = np.deg2rad(angle_v)
        
        # 计算速度分量
        v_xy = speed * np.cos(angle_v_rad)  # 水平面投影速度
        vx = v_xy * np.cos(angle_h_rad)     # x方向速度
        vy = v_xy * np.sin(angle_h_rad)     # y方向速度
        vz = speed * np.sin(angle_v_rad)    # z方向速度
        
        return np.array([vx, vy, vz])
    
    def estimate_landing_point(self, pos, velocity):
        """估计落点位置"""
        # 解二次方程：z = z0 + vz*t - 0.5*g*t^2 = 0
        # at^2 + bt + c = 0, where:
        a = -0.5 * self.g
        b = velocity[2]
        c = pos[2]
        
        # 求解时间
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
        
        t = (-b + np.sqrt(discriminant)) / (2*a)
        
        # 计算落点
        landing_x = pos[0] + velocity[0] * t
        landing_y = pos[1] + velocity[1] * t
        
        return np.array([landing_x, landing_y])
    
    def reset(self, pos=None, velocity=None):
        """重置球的状态，可以指定位置和速度，否则随机生成"""
        if pos is None:
            pos = [
                np.random.uniform(*self.config['pos_range']['x']),
                np.random.uniform(*self.config['pos_range']['y']),
                np.random.uniform(*self.config['pos_range']['z'])
            ]
        
        if velocity is None:
            # 随机生成投掷参数
            speed = np.random.uniform(*self.config['throw_config']['speed'])
            angle_h = np.random.uniform(*self.config['throw_config']['angle_h'])
            angle_v = np.random.uniform(*self.config['throw_config']['angle_v'])
            
            # 计算速度分量
            velocity = self.calculate_velocity(speed, angle_h, angle_v)
            
            # 验证落点是否在目标范围内
            landing_point = self.estimate_landing_point(np.array(pos), velocity)
            
            # 如果落点不在目标范围内，调整参数重试
            max_attempts = 10
            attempt = 0
            while landing_point is not None and attempt < max_attempts:
                in_x_range = (self.config['target_range']['x'][0] <= landing_point[0] <= 
                            self.config['target_range']['x'][1])
                in_y_range = (self.config['target_range']['y'][0] <= landing_point[1] <= 
                            self.config['target_range']['y'][1])
                
                if in_x_range and in_y_range:
                    break
                
                # 重新生成参数
                speed = np.random.uniform(*self.config['throw_config']['speed'])
                angle_h = np.random.uniform(*self.config['throw_config']['angle_h'])
                angle_v = np.random.uniform(*self.config['throw_config']['angle_v'])
                velocity = self.calculate_velocity(speed, angle_h, angle_v)
                landing_point = self.estimate_landing_point(np.array(pos), velocity)
                attempt += 1
        
        self.pos = np.array(pos)
        self.velocity = np.array(velocity)
        return self.get_state()
    
    def step(self):
        """更新球的状态"""
        if self.pos is None:
            return None
        
        # 更新速度和位置
        self.velocity[2] -= self.g * self.dt
        self.pos += self.velocity * self.dt
        
        # 如果球落地，返回None
        if self.pos[2] < 0:
            self.pos = None
            self.velocity = None
            return None
        
        return self.get_state()
    
    def get_state(self):
        """获取球的当前状态"""
        if self.pos is None:
            return None
        return {
            'position': self.pos.copy(),
            'velocity': self.velocity.copy()
        }
    
    def update_ball_state(self, ball_state, dt):
        """更新球的状态"""
        # 提取当前状态
        position = ball_state['position']
        velocity = ball_state['velocity']
        
        # 更新位置
        new_position = position + velocity * dt
        
        # 更新速度（考虑重力）
        new_velocity = velocity.copy()
        new_velocity[2] -= self.g * dt  # 只在z方向上受重力影响
        
        return {
            'position': new_position,
            'velocity': new_velocity
        }
