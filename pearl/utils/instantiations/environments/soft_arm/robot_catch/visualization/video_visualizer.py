import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from env.environment import RobotArmEnv
from env.env_factory import EnvFactory
from config.config_manager import ConfigManager
from agent.agent_factory import AgentFactory

class VideoVisualizer:
    def __init__(self, model_path, config_path='config/default.yaml'):
        """初始化可视化器"""
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.env_config = self.config_manager.get_env_config()
        self.agent_config = self.config_manager.get_agent_config()
        
        # 修改为第一阶段的简单配置
        self.env_config.update({
            'reward_scale': {
                'distance': 5.0,
                'distance_intention': 5.0,
                'angle': 0.0,
                'angle_intention': 0.0,
                'catch': 100.0,
                'smooth': 0.0,
                'magnitude': 0.0,
                'jerk': 0.0
            },
            'ball_config': {
                'g': 9.81,
                'pos_range': {
                    'x': [-0.9, -0.9],
                    'y': [0.0, 0.0],
                    'z': [1.1, 1.1]
                },
                'throw_config': {
                    'speed': [2.0, 2.0],
                    'angle_h': [0, 0],
                    'angle_v': [-10, -10]
                }
            },
            'catch': {
                'radius': 0.15,    # 大幅增加抓取半径
                'min_angle': 0,    # 不考虑角度要求
                'max_angle': 180
            }
        })
        
        # 创建环境
        env_factory = EnvFactory(self.env_config)
        self.env = env_factory.create_single_env()
        
        # 修改agent_config，添加num_envs=1
        self.agent_config['num_envs'] = 1
        
        # 创建智能体
        self.agent = AgentFactory.create(
            agent_type='sac',
            config=self.agent_config,
            env=type('DummyEnv', (), {
                'single_observation_space': self.env.observation_space,
                'single_action_space': self.env.action_space,
                'num_envs': 1
            })()
        )
        
        # 加载模型
        self.agent.load_model(model_path)
        print(f"Loaded model from: {model_path}")
        
        # 创建图形
        plt.style.use('dark_background')  # 使用深色主题
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 存储轨迹数据
        self.robot_positions = []
        self.ball_positions = []
        
        print("\n使用第一阶段简单配置：")
        print(f"reward_scale: {self.env_config['reward_scale']}")
        print(f"ball_config: {self.env_config['ball_config']}")
        print(f"catch: {self.env_config['catch']}")
    
    def collect_episode_data(self):
        """收集一个回合的数据"""
        obs, _ = self.env.reset()
        done = False
        self.robot_positions = []
        self.ball_positions = []
        
        print("\n开始收集数据...")
        print(f"球的初始位置: {obs[12:15]}")
        print(f"末端执行器初始位置: {obs[6:9]}")
        
        while not done:
            # 记录当前状态
            self.robot_positions.append(self.env.get_robot_positions())
            self.ball_positions.append(self.env.get_ball_position())
            
            # 执行动作
            action = self.agent.select_action(obs)
            obs, reward, done, _, info = self.env.step(action)
            
            # 打印调试信息
            if len(self.robot_positions) % 10 == 0:  # 每10帧打印一次
                ee_pos = obs[6:9]
                ball_pos = obs[12:15]
                distance = np.linalg.norm(ball_pos - ee_pos)
                print(f"Frame {len(self.robot_positions)}")
                print(f"末端执行器位置: {ee_pos}")
                print(f"球的位置: {ball_pos}")
                print(f"距离: {distance:.3f}")
                if 'reward_info' in info:
                    print(f"奖励信息: {info['reward_info']}")
        
        success = info.get('success', False)
        print(f"\n回合结束，是否成功: {success}")
        return success
    
    def update_plot(self, frame):
        """更新图形"""
        self.ax.cla()  # 清除当前帧
        
        # 设置坐标轴范围和标签
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([0, 2])
        self.ax.set_xlabel('X', color='white')
        self.ax.set_ylabel('Y', color='white')
        self.ax.set_zlabel('Z', color='white')
        
        # 设置背景颜色
        self.ax.xaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        self.ax.yaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        self.ax.zaxis.set_pane_color((0.2, 0.2, 0.2, 1.0))
        
        # 设置网格
        self.ax.grid(True, color='gray', alpha=0.3)
        
        # 设置固定视角 - 从侧面45度角观察
        self.ax.view_init(elev=20, azim=45)
        
        # 绘制机械臂
        robot_pos = self.robot_positions[frame]  # 这是一个包含4个点的数组：基座点和3个关节点
        
        # 绘制机械臂的三个段
        colors = ['cyan', 'lightblue', 'blue']  # 为每个段设置不同的颜色
        for i in range(len(robot_pos)-1):
            self.ax.plot3D(
                [robot_pos[i][0], robot_pos[i+1][0]],
                [robot_pos[i][1], robot_pos[i+1][1]],
                [robot_pos[i][2], robot_pos[i+1][2]],
                color=colors[i], linewidth=3, label=f'Joint {i+1}'
            )
            
            # 在每个关节位置画一个球
            self.ax.scatter(
                robot_pos[i][0],
                robot_pos[i][1],
                robot_pos[i][2],
                color=colors[i],
                s=100
            )
        
        # 绘制末端执行器
        self.ax.scatter(
            robot_pos[-1][0],
            robot_pos[-1][1],
            robot_pos[-1][2],
            color='red',
            s=150,
            label='End Effector'
        )
        
        # 绘制球
        ball_pos = self.ball_positions[frame]
        self.ax.scatter(
            ball_pos[0],
            ball_pos[1],
            ball_pos[2],
            color='yellow',
            s=150,
            label='Ball'
        )
        
        # 如果不是第一帧，绘制球的轨迹
        if frame > 0:
            ball_trajectory = np.array(self.ball_positions[:frame+1])
            self.ax.plot3D(
                ball_trajectory[:, 0],
                ball_trajectory[:, 1],
                ball_trajectory[:, 2],
                'y--',  # 黄色虚线
                alpha=0.5,
                linewidth=1,
                label='Ball Trajectory' if frame == 1 else ""
            )
        
        # 添加图例
        self.ax.legend(loc='upper right', framealpha=0.8)
        
        # 添加标题和帧信息
        self.ax.set_title(f'Frame {frame}\nBall Position: ({ball_pos[0]:.2f}, {ball_pos[1]:.2f}, {ball_pos[2]:.2f})', 
                         color='white', pad=20)
    
    def create_animation(self, save_path):
        """创建并保存动画"""
        print("Creating animation...")
        
        # 确保保存为GIF格式
        save_path = os.path.splitext(save_path)[0] + '.gif'
        
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=len(self.robot_positions),
            interval=50,  # 20 FPS
            blit=False
        )
        
        # 使用PillowWriter保存为GIF
        writer = PillowWriter(fps=20)
        
        try:
            anim.save(save_path, writer=writer)
            print(f"Saved animation to: {save_path}")
        except Exception as e:
            print(f"Error saving animation: {str(e)}")
        finally:
            plt.close()

def main():
    # 设置模型路径和输出目录的绝对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'checkpoints', 'best_model.pt')
    videos_dir = os.path.join(base_dir, 'videos')
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return
    
    # 创建可视化器
    visualizer = VideoVisualizer(model_path)
    
    # 创建视频保存目录
    os.makedirs(videos_dir, exist_ok=True)
    
    # 运行多个回合
    num_episodes = 5
    successes = 0
    
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode+1}/{num_episodes}")
        success = visualizer.collect_episode_data()
        if success:
            successes += 1
        
        # 保存动画
        video_path = os.path.join(videos_dir, f'robot_catch_episode_{episode+1}.gif')
        visualizer.create_animation(video_path)
    
    print(f"\nSuccess rate: {successes/num_episodes:.2%}")

if __name__ == "__main__":
    main() 