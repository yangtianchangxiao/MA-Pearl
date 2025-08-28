def __init__(self):
    # 定义观测空间（小球的信息）
    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(6,),  # 小球的位置和速度信息
        dtype=np.float32
    )
    
    # 定义动作空间（机器人关节）
    self.action_space = spaces.Box(
        low=-1,
        high=1,
        shape=(3,),  # 假设是3个关节
        dtype=np.float32
    ) 