"""机械臂和环境的配置参数"""

# 机械臂参数
ROBOT_CONFIG = {
    'n_segments': 3,              # 段数
    'segment_length': 0.21,       # 每段长度（米）
    'max_angle': 1.57,           # 最大关节角度（弧度，约90度）
    'min_angle': -1.57,          # 最小关节角度（弧度，约-90度）
    'max_angle_velocity': 0.1,    # 最大角速度（弧度/步）
}

# 投掷物体参数
PROJECTILE_CONFIG = {
    'mass': 0.1,                 # 质量（千克）
    'gravity': [0, 0, -9.81],    # 重力加速度（米/秒²）
    'init_pos_range': {          # 初始位置范围
        'x': [1.5, 2.5],
        'y': [1.5, 2.5],
        'z': [1.5, 2.5]
    },
    'init_vel_range': {          # 初始速度范围
        'x': [-3.0, -1.0],
        'y': [-3.0, -1.0],
        'z': [-0.5, 0.5]
    }
}

# 环境参数
ENV_CONFIG = {
    'dt': 0.033,                # 时间步长（秒）
    'max_steps': 300,           # 最大步数
    'catch_distance': 0.1,      # 接住物体的距离阈值（米）
    'catch_angle': 0.785,       # 接住物体的角度阈值（弧度，约45度）
    'reward_weights': {
        'distance': 1.0,        # 距离奖励权重
        'angle': 0.5,           # 角度奖励权重
        'catch': 100.0,         # 接住奖励
    }
}
