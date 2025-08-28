from environment import RobotArmEnv
from visualizer import RobotArmVisualizer
import numpy as np

def main():
    # 完整的环境配置
    config = {
        # 基础环境参数
        'max_steps': 200,           # 每个回合的最大步数
        'catch_radius': 0.08,       # 抓取半径（米）
        'perception_radius': 0.5,   # 感知半径（米），球在这个范围内才能被感知到
        'dt': 0.02,                # 时间步长（秒）
        
        # 奖励设置
        'reward_catch': 200.0,      # 成功抓住的奖励
        'reward_distance': -0.5,    # 距离惩罚系数
        'reward_action': -0.05,     # 动作惩罚系数
        
        # 噪声设置
        'noise': {
            'action': 0.1,          # 动作噪声（相对于动作范围的10%）
            'observation': 0.03,    # 观测噪声（3厘米）
            'ball_init': 0.02,      # 球初始状态噪声（2厘米）
            'ball_vel': 0.15        # 球速度噪声（相对于速度的15%）
        },
        
        # 球的参数
        'ball': {
            'g': 9.81,              # 重力加速度（米/秒²）
            'pos_range': {          # 初始位置范围（米）
                'x': [-0.5, -0.3],  # 初始x位置范围
                'y': [-0.2, 0.2],   # 初始y位置范围
                'z': [0.3, 0.5]     # 初始z位置范围
            },
            'throw_config': {       # 投掷参数
                'speed': [1.0, 2.0],     # 出射速度范围 (m/s)
                'angle_h': [-30, 45],    # 水平角度范围 (度)
                'angle_v': [15, 45]      # 垂直角度范围 (度)
            },
            'target_range': {       # 期望落点范围
                'x': [0.3, 0.6],    # 目标x范围
                'y': [-0.3, 0.3]    # 目标y范围
            }
        },
        
        # 机械臂参数
        'robot': {
            'n_segments': 3,        # 机械臂段数
            'segment_length': 0.21, # 每段长度（米）
            'joint_limits': {       # 关节角度限制（弧度）
                'alpha': [-np.pi/2, np.pi/2],  # alpha角度范围：-90度到90度
                'beta': [-np.pi, np.pi]        # beta角度范围：-180度到180度
            },
            'velocity_limits': {    # 关节速度限制（弧度/秒）
                'alpha': np.pi/2,   # alpha最大速度：90度/秒
                'beta': np.pi       # beta最大速度：180度/秒
            }
        }
    }
    
    # 创建环境和可视化器
    env = RobotArmEnv(config)
    vis = RobotArmVisualizer()
    
    # 运行5个回合
    for episode in range(5):
        print(f"\nEpisode {episode + 1}")
        
        # 重置环境
        obs = env.reset()
        initial_pos = obs['ball_position']
        print(f"Initial ball position: {initial_pos}")
        print(f"Initial ball velocity: {env.ball_thrower.velocity}")
        
        episode_reward = 0
        done = False
        
        while not done:
            # 简单的启发式控制策略
            action = np.zeros(6)  # 3个关节，每个关节2个角度
            
            # 更新环境
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # 更新可视化
            vis.update(
                robot_positions=obs['positions'],
                ball_pos=obs['ball_position'],      # 观测位置
                true_ball_pos=env.ball_state['position'] if env.ball_state else None,  # 真实位置
                info=info
            )
            
            # 打印当前状态和噪声的影响
            if env.ball_state:
                true_pos = env.ball_state['position']
                obs_pos = obs['ball_position']
                obs_error = np.linalg.norm(obs_pos - true_pos)
                print(f"\rStep reward: {reward:.2f} | "
                      f"Total reward: {episode_reward:.2f} | "
                      f"Observation error: {obs_error:.3f}m", 
                      end="")
        
        print(f"\nEpisode {episode + 1} finished: {info}")
    
    vis.close()

if __name__ == "__main__":
    main()