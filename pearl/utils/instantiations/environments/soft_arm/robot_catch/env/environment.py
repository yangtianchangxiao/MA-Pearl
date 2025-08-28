import numpy as np
import sys
import gymnasium as gym
from gymnasium import spaces
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from .robot_arm import RobotArm
except ImportError:
    print("⚠️ C++扩展导入失败，使用Python版本")
    from .robot_arm_python import RobotArm
from .ball_thrower import BallThrower

class RobotArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, config=None):
        super().__init__()
        
        # 环境配置
        self.config = {
            'max_steps': 200,           # 每个回合的最大步数
            'catch_radius': 0.1,       # 抓取半径
            'perception_radius': 1.0,   # 感知半径
            'dt': 0.02,                # 时间步长
            
            # 奖励设置
            'reward_scale': {
                'distance': 1.0,           # 实际距离改善的奖励
                'distance_intention': 0.5,  # 动作意图的奖励
                'angle': 1.0,              # 实际角度改善的奖励
                'angle_intention': 0.5,    # 动作意图的奖励
                'catch': 100.0,            # 抓住的奖励
                'smooth': 0.1,             # 动作平滑度系数
                'magnitude': 0.05,         # 动作幅度系数
                'jerk': 0.05              # 加加速度系数
            },
            
            # 抓取判定
            'catch': {
                'radius': 0.1,         # 抓取半径
                'angle_tolerance': 0.5,  # 允许的最大角度偏差
                'min_angle': 1.2,       # 最小垂直角度
                'max_angle': 1.9,       # 最大垂直角度
            },
            
            # 噪声设置
            'noise': {
                'action': 0.05,         # 动作噪声（相对于动作范围）
                'observation': 0.02,    # 观测噪声（米）
                'ball_init': 0.02,      # 球初始状态噪声（米）
                'ball_vel': 0.1         # 球速度噪声（相对于速度）
            },
            
            # 球的参数
            'ball': {
                'g': 9.81,              # 重力加速度（米/秒²）
                'pos_range': {          # 初始位置范围（米）
                    'x': [-1.0, -0.8],  # 距离机器人基座0.8-1米
                    'y': [-0.3, 0.3],   # 左右30厘米范围
                    'z': [1.0, 1.2]     # 高度1-1.2米
                },
                'velocity_range': {     # 初始速度范围（米/
                    'x': [2.0, 3.0],    # 向机器人方向的速度
                    'y': [-0.5, 0.5],   # 左右速度分量
                    'z': [-0.5, 0.0]    # 略微向下的速度分量
                }
            },
            
            # 球体投掷器的配置
            'ball_config': {
                'g': 9.81,              # 重力加速度
                'pos_range': {
                    'x': [-1.0, -0.8],  # 距离机器人基座0.8-1米
                    'y': [-0.3, 0.3],   # 左右30厘米范围
                    'z': [1.0, 1.2]     # 高度1-1.2米
                },
                'throw_config': {
                    'speed': [2.0, 3.0],     # 出射速度范围 (m/s)
                    'angle_h': [-15, 15],    # 水平角度范围 (度)
                    'angle_v': [0, -15]      # 垂直角度范围 (度)
                },
                'target_range': {       # 期望落点范围（可选）
                    'x': [0.3, 0.6],    # 目标x范围
                    'y': [-0.3, 0.3]    # 目标y范围
                }
            },
            
            # 机械臂参数
            'robot': {
                'n_segments': 3,        # 机械臂段数
                'segment_length': 0.21, # 每段长度（米）
                'joint_limits': {       # 关节角限（���）
                    'alpha': [-np.pi/2, np.pi/2],  # alpha角度范围-90度到90度
                    'beta': [-np.pi, np.pi]        # beta角度范围：-180度到180度
                },
                'velocity_limits': {    # 关节速度限制（弧度/秒）
                    'alpha': np.pi/2,   # alpha最大速度：90度/秒
                    'beta': np.pi       # beta最大速度：180度/秒
                }
            },
            
            # 添加通用阈值到配置中
            'thresholds': {
                'velocity': 1e-6,  # 速度阈值
                'smooth': 0.5,     # 平滑度阈值
                'magnitude': 0.8,  # 动作幅度阈值
                'jerk': 0.3       # 加加速度阈值
            }
        }
        if config:
            # 递归更新配置
            def update_dict_recursive(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_dict_recursive(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            update_dict_recursive(self.config, config)
        
        print("\n当前环境reward_scale参数：")
        print(self.config['reward_scale'])
        
        # 创建械臂和球体投掷器实，传入相应配��
        self.robot = RobotArm(
            n_segments=self.config['robot']['n_segments'],
            segment_length=self.config['robot']['segment_length'],
            config=self.config['robot']
        )
        self.ball_thrower = BallThrower(config=self.config['ball_config'])
        
        self.current_step = 0
        self.total_reward = 0
        self.ball_state = None
        self.steps = 0
        self.ball_entered = False  # 新增：标记球是否已进入感知范围
        
        # OpenRL需要的属性
        self.agent_num = 1  # 单智能体环境
        
        # 动作空间：每个关节2个自由度（alpha和beta）
        action_dim = self.robot.get_action_dim()
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(19,),  # 6 + 3 + 3 + 3 + 3 + 1
            dtype=np.float32
        )
        
        # 添加用于存储上一时刻状态的变量
        self.last_distance = None
        self.last_angle = None
        self.last_action = None
        self.last_action_diff = None
        self.last_reward_info = {
            'result_distance': 0,
            'result_angle': 0,
            'intent_distance': 0,
            'intent_angle': 0,
            'action_smooth': 0,
            'action_magnitude': 0,
            'jerk': 0,
            'catch': 0
        }
    
    def reset(self, seed=None, options=None):
        """重置环境
        Args:
            seed: 随机种子
            options: 可选的置，可以包含curriculum_config
        Returns:
            obs: 初始观察
            info: 信息字典
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 如果提供了curriculum_config，更新配置
        if options and 'curriculum_config' in options:
            curriculum_config = options['curriculum_config']
            if curriculum_config:  # 只在有配置时更新
                if 'reward_scale' in curriculum_config:
                    self.config['reward_scale'] = curriculum_config['reward_scale'].copy()
                if 'ball_config' in curriculum_config:
                    self.config['ball_config'] = curriculum_config['ball_config'].copy()
                    self.ball_thrower.update_config(curriculum_config['ball_config'].copy())
                print(f"应用课程学配置: {curriculum_config}")
            
        # 重置机械臂和球体
        robot_state = self.robot.reset()
        self.ball_state = self.ball_thrower.reset()
        
        # 重置状态记录
        self.current_step = 0
        self.total_reward = 0
        self.last_distance = None
        self.last_angle = None
        self.last_action = None
        self.last_action_diff = None
        self.ball_entered = False
        self.last_reward_info = {
            'result_distance': 0,
            'result_angle': 0,
            'intent_distance': 0,
            'intent_angle': 0,
            'action_smooth': 0,
            'action_magnitude': 0,
            'jerk': 0,
            'catch': 0
        }
        
        # 仿真直到球进入感知范围，添加最大尝试次数限制
        max_attempts = 100
        attempt = 0
        while attempt < max_attempts:
            distance_to_ball = np.linalg.norm(
                np.array(self.ball_state['position']) - 
                np.array(robot_state['end_effector']['position'])
            )
            
            if distance_to_ball <= self.config['perception_radius']:
                self.ball_entered = True
                break
            
            # 如果不在感知范围内，继续物理仿真
            self.ball_state = self.ball_thrower.update_ball_state(
                self.ball_state, 
                self.config['dt']
            )
            
            # 如果球已经离开可能的捕捉区域，重新投掷
            if self._is_ball_unreachable():
                self.ball_state = self.ball_thrower.reset()
                attempt += 1
        
        # 如果达到最大尝试次数仍未成功，强制重置到可见位置
        if attempt >= max_attempts:
            print(f"Warning: Failed to place ball in perception radius after {max_attempts} attempts")
            # 强制将球放在感知范围内
            robot_pos = np.array(robot_state['end_effector']['position'])
            self.ball_state['position'] = robot_pos + np.array([-0.5, 0, 0.5])  # 在机器人前方0.5米处
            self.ball_entered = True
        
        # 获取观察
        obs = self._get_observation(robot_state, self.ball_state)
        
        # 设置info字典
        info = {
            'success': False,
            'total_reward': 0.0,
            'ball_visible': True,
            'distance': distance_to_ball,
            'catch_distance': distance_to_ball,
            'reaction_time': 0.0
        }
        
        return obs, info
    
    def _is_ball_unreachable(self):
        """检查球是否已经离开可的捕捉区域"""
        if self.ball_state is None:
            return True
        
        ball_pos = self.ball_state['position']
        ball_vel = self.ball_state['velocity']
        
        # 检查球是否已经落地
        if ball_pos[2] <= 0:
            return True
        
        # 检查球是否已经远离机器人太远
        if np.linalg.norm(ball_pos) > self.config['perception_radius'] * 2:
            return True
        
        # 检查球是否正在远离机器人
        if ball_pos[0] > 0 and ball_vel[0] > 0:  # 假设机器人在原点
            return True
        
        return False
    
    def step(self, action):
        """执行一步动作"""
        # 执行动作
        robot_state = self.robot.step(action)
        current_ee_pos = np.array(robot_state['end_effector']['position'])
        
        # 更新球的状态
        if self.ball_state is not None:
            self.ball_state = self.ball_thrower.update_ball_state(
                self.ball_state, 
                self.config['dt']
            )
            
            # 检查球是否已经落地
            ball_landed = self.ball_state['position'][2] <= 0
            
            # 计算当前距离（用于info）
            distance = np.linalg.norm(
                np.array(self.ball_state['position']) - current_ee_pos
            )
            
            # 只有球落地才结束回合
            done = ball_landed
        else:
            done = True
            distance = float('inf')
            ball_landed = False
        
        # 获取观察和计算奖励
        obs = self._get_observation(robot_state, self.ball_state)
        reward, reward_info = self._compute_reward(obs, action, {})
        
        # 更新计数器和历史信息
        self.current_step += 1
        self.total_reward += reward
        
        # 检查是否抓住球
        caught = self._check_catch(obs)
        if caught:
            done = True
            reward_info['catch'] = self.config['reward_scale']['catch']
        
        # 设置info
        info = {
            'success': caught,
            'total_reward': self.total_reward,
            'steps': self.current_step,
            'distance': distance,
            'reward_info': reward_info,
            'ball_landed': ball_landed,
            'catch_info': {
                'distance': distance,
                'radius': self.config['catch']['radius'],
                'min_angle': self.config['catch']['min_angle'],
                'max_angle': self.config['catch']['max_angle']
            } if caught else None
        }
        
        return obs, reward, done, False, info
    
    def _get_observation(self, robot_state, ball_state):
        """获取观察
        
        观察空间包含:
        - robot_joint_angles (6): 实际关节角度
        - ee_position (3): 末端执行器位置
        - ee_velocity (3): 末端执行器速度
        - ball_position (3): 球的位置
        - ball_velocity (3): 球的速度
        - ball_visible (1): 球是否可见（总是1，因为reset确保了球可见）
        """
        # 获取机器人状态
        robot_config = robot_state['config'].flatten()  # 关节角度 (6)
        ee_position = robot_state['end_effector']['position'].flatten()  # 末端位置 (3)
        ee_velocity = robot_state['end_effector']['velocity'].flatten()  # 末端速度 (3)
        
        # 获取球的状态（reset确保了球总是可见的）
        ball_position = ball_state['position']
        ball_velocity = ball_state['velocity']
        ball_visible = np.array([1.0])  # 球总是可见
        
        # 组合所有观察分量
        observation = np.concatenate([
            robot_config,      # (6,)
            ee_position,       # (3,)
            ee_velocity,       # (3,)
            ball_position,     # (3,)
            ball_velocity,     # (3,)
            ball_visible       # (1,)
        ]).astype(np.float32)
        
        return observation
    
    def _compute_angle_reward(self, obs):
        """计算末端方向与球体速度方向的角度奖励"""
        # 获取末端速度和球速度
        end_effector_vel = obs[9:12]
        ball_vel = obs[15:18]
        
        # 计算速度方向
        end_vel_norm = np.linalg.norm(end_effector_vel)
        ball_vel_norm = np.linalg.norm(ball_vel)
        
        if ball_vel_norm < self.config['thresholds']['velocity'] or end_vel_norm < self.config['thresholds']['velocity']:
            return 0.0
            
        # 计算速度方向的夹角
        end_vel_dir = end_effector_vel / end_vel_norm
        ball_vel_dir = ball_vel / ball_vel_norm
        cos_angle = np.clip(np.dot(end_vel_dir, ball_vel_dir), -1.0, 1.0)
        current_angle = np.arccos(cos_angle)
        
        # 如果角度差异小于容忍度，给予奖励
        if abs(current_angle - np.pi/2) <= self.config['catch']['angle_tolerance']:
            return 0.0
        else:
            return self.config['reward_angle'] * (abs(current_angle - np.pi/2) - self.config['catch']['angle_tolerance'])
    
    def _check_catch(self, obs):
        """检查是否抓住球"""
        # 获取末端执行器位置和球位置
        end_pos = obs[6:9]  # 末端位置
        ball_pos = obs[12:15]  # 球位置
        
        # 检查距离
        distance = np.linalg.norm(ball_pos - end_pos)
        if distance > self.config['catch']['radius']:
            return False
            
        # 如果配置中不需要检查角度（第一阶段），直接返回True
        if self.config['catch']['min_angle'] == 0 and self.config['catch']['max_angle'] == 180:
            return True
            
        # 后续阶段需要检查速度和角度
        # 获取末端执行器速度和球速度
        end_effector_vel = obs[9:12]  # 末端速度
        ball_vel = obs[15:18]  # 球速度
        
        # 计算速度方向
        end_vel_norm = np.linalg.norm(end_effector_vel)
        ball_vel_norm = np.linalg.norm(ball_vel)
        
        # 降低速度阈值要求
        velocity_threshold = 0.01  # 降低到0.01
        if ball_vel_norm < velocity_threshold or end_vel_norm < velocity_threshold:
            return False
            
        # 计算速度方向的夹角
        end_vel_dir = end_effector_vel / end_vel_norm
        ball_vel_dir = ball_vel / ball_vel_norm
        cos_angle = np.clip(np.dot(end_vel_dir, ball_vel_dir), -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi  # 转换为角度
        
        # 检查角度是否在允许范围内
        return (self.config['catch']['min_angle'] <= angle <= self.config['catch']['max_angle'])
    
    def _compute_reward(self, obs, action, info):
        """计算奖励值"""
        reward_info = {
            'result_distance': 0,
            'intent_distance': 0,
            'result_angle': 0,
            'intent_angle': 0,
            'action_smooth': 0,
            'action_magnitude': 0,
            'jerk': 0,
            'catch': 0
        }
        
        # 获取当前状态
        ball_pos = obs[12:15]
        end_pos = obs[6:9]
        ball_vel = obs[15:18]
        end_vel = obs[9:12]
        
        # 1. 计算实际距离改善
        current_distance = np.linalg.norm(ball_pos - end_pos)
        if self.last_distance is not None:
            distance_improvement = self.last_distance - current_distance
            reward_info['result_distance'] = (
                self.config['reward_scale']['distance'] * distance_improvement
            )
        self.last_distance = current_distance

        # 2. 计算意图距离改善
        dt = self.config['dt']
        prediction_horizon = 0.5  # 增加到0.5秒，给机器人更多的规划时间
        robot_state = self.robot.get_state()
        next_robot_state = self.robot.predict_next_state(action)
        intended_pos = np.array(next_robot_state['end_effector']['position'])
        
        # 预测球的位置（考虑重力加速度的影响）
        t = prediction_horizon
        predicted_ball_pos = np.zeros(3)
        predicted_ball_pos[0] = ball_pos[0] + ball_vel[0] * t  # x方向匀速运动
        predicted_ball_pos[1] = ball_pos[1] + ball_vel[1] * t  # y方向匀速运动
        predicted_ball_pos[2] = ball_pos[2] + ball_vel[2] * t - 4.905 * t * t  # z方向考虑重力
        
        # 果预测位置在地面以下，调整到地面位置
        if predicted_ball_pos[2] < 0:
            # 计算落地时间
            a = -4.905
            b = ball_vel[2]
            c = ball_pos[2]
            # 求解二次方程 at^2 + bt + c = 0
            t = (-b - np.sqrt(b*b - 4*a*c))/(2*a)
            # 使用落地时间重新计算x,y位置
            predicted_ball_pos[0] = ball_pos[0] + ball_vel[0] * t
            predicted_ball_pos[1] = ball_pos[1] + ball_vel[1] * t
            predicted_ball_pos[2] = 0
        
        # 计算距离改善
        current_to_predicted = np.linalg.norm(predicted_ball_pos - end_pos)
        intended_to_predicted = np.linalg.norm(predicted_ball_pos - intended_pos)
        distance_improvement = current_to_predicted - intended_to_predicted
        
        # 添加距离归一化
        max_distance = 3.0  # 假设最大离为3米
        normalized_improvement = distance_improvement / max_distance
        reward_info['intent_distance'] = (
            self.config['reward_scale']['distance_intention'] * normalized_improvement
        )
        
        # 3. 计算角度奖励（如果配置中启用）
        if self.config['reward_scale']['angle'] > 0 or self.config['reward_scale']['angle_intention'] > 0:
            ball_vel_norm = np.linalg.norm(ball_vel)
            end_vel_norm = np.linalg.norm(end_vel)
            
            if ball_vel_norm > self.config['thresholds']['velocity'] and end_vel_norm > self.config['thresholds']['velocity']:
                # 计算当前角度
                current_angle = np.dot(end_vel, ball_vel) / (ball_vel_norm * end_vel_norm)
                current_angle = np.clip(current_angle, -1.0, 1.0)
                
                # 计算实际角度改善
                if hasattr(self, 'last_angle') and self.last_angle is not None:
                    angle_improvement = float(abs(self.last_angle)) - float(abs(current_angle))
                    reward_info['result_angle'] = (
                        self.config['reward_scale']['angle'] * angle_improvement
                    )
                self.last_angle = float(current_angle)
                
                # 计算意图角度
                intended_vel = np.array(next_robot_state['end_effector']['velocity'])
                intended_vel_norm = np.linalg.norm(intended_vel)
                
                if intended_vel_norm > self.config['thresholds']['velocity']:
                    intended_angle = np.dot(intended_vel, ball_vel) / (intended_vel_norm * ball_vel_norm)
                    intended_angle = np.clip(intended_angle, -1.0, 1.0)
                    angle_improvement = float(abs(current_angle)) - float(abs(intended_angle))
                    reward_info['intent_angle'] = (
                        self.config['reward_scale']['angle_intention'] * angle_improvement
                    )
        
        # 4. 动作平滑度相关的奖励（如果配置中启用）
        if any(abs(self.config['reward_scale'][key]) > 0 for key in ['smooth', 'magnitude', 'jerk']):
            if hasattr(self, 'last_action') and self.last_action is not None:
                action_diff = action - self.last_action
                action_diff_magnitude = np.linalg.norm(action_diff)
                
                # 动作平滑度
                if action_diff_magnitude > self.config['thresholds']['smooth']:
                    reward_info['action_smooth'] = -abs(self.config['reward_scale']['smooth'])
                
                # 动作幅度
                action_magnitude = np.linalg.norm(action)
                if action_magnitude > self.config['thresholds']['magnitude']:
                    reward_info['action_magnitude'] = -abs(self.config['reward_scale']['magnitude'])
                
                # Jerk（加加速度）
                if hasattr(self, 'last_action_diff') and self.last_action_diff is not None:
                    jerk = action_diff - self.last_action_diff
                    jerk_magnitude = np.linalg.norm(jerk)
                    if jerk_magnitude > self.config['thresholds']['jerk']:
                        reward_info['jerk'] = -abs(self.config['reward_scale']['jerk'])
                self.last_action_diff = action_diff
        
        # 5. 抓住奖励
        caught = self._check_catch(obs)
        if caught:
            reward_info['catch'] = self.config['reward_scale']['catch']
        
        # 更新历史信息
        self.last_action = action.copy()
        self.last_reward_info = reward_info
        
        # 计算总奖励
        reward = sum(reward_info.values())
        return reward, reward_info
    
    def _is_done(self, obs):
        """检查是否结束"""
        if self.ball_state is None:
            return True
        
        # 如果球还没进入感知围，继续
        if not self.ball_entered:
            return False
            
        # 如果球已经被抓住，结束
        if self._check_catch(obs):
            return True
            
        # 如果球已经落地离开可能的抓取范围，结束
        if self.ball_state['position'][2] <= 0:
            return True
            
        return False
    
    def get_robot_positions(self):
        """获取机器人各关节的位置"""
        # 直接从机器人对象获取位置信息
        robot_state = self.robot.get_state()
        positions = np.array(robot_state['positions']).reshape(-1, 3)
        return positions

    def get_ball_position(self):
        """获球的当前位置"""
        if self.ball_state is None:
            return np.zeros(3)
        return self.ball_state['position'].copy()

    def close(self):
        """关闭环境"""
        pass

    def update_curriculum_config(self, curriculum_config):
        """更新环境的课程学习配置"""
        print("\n收到课程学习配置更新：")
        print(curriculum_config)
        
        if 'reward_scale' in curriculum_config:
            self.config['reward_scale'] = curriculum_config['reward_scale'].copy()
            
        if 'ball_config' in curriculum_config:
            import copy
            self.config['ball_config'] = copy.deepcopy(curriculum_config['ball_config'])
            self.ball_thrower.update_config(copy.deepcopy(curriculum_config['ball_config']))
            
        if 'catch' in curriculum_config:
            self.config['catch'] = curriculum_config['catch'].copy()
            
        print("\n更新后的环境配置：")
        print(f"reward_scale: {self.config['reward_scale']}")
        print(f"ball_config: {self.config['ball_config']}")
        print(f"catch: {self.config['catch']}")
