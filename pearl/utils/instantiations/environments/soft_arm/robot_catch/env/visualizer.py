import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg，适合服务器环境
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.animation import FFMpegWriter

class RobotArmVisualizer:
    def __init__(self, video_path='videos'):
        # 创建视频保存目录
        self.video_path = video_path
        os.makedirs(video_path, exist_ok=True)
        
        # 创建图形对象
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 存储历史轨迹
        self.ball_history = []
        self.ball_obs_history = []
        
        # 设置坐标轴范围
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([0, 1])
        
        # 添加标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Robot Arm Simulation')
        
        # 视频写入器
        self.writer = None
        self.recording = False
        self.frame_count = 0
        
    def start_recording(self, filename='simulation.mp4', fps=30):
        """开始录制视频"""
        if self.recording:
            return
            
        self.recording = True
        self.frame_count = 0
        
        # 设置视频写入器
        self.writer = FFMpegWriter(
            fps=fps,
            metadata=dict(title='Robot Arm Simulation'),
            bitrate=2000
        )
        
        # 开始写入视频
        self.writer.setup(self.fig, os.path.join(self.video_path, filename))
        
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
            
        self.writer.finish()
        self.recording = False
        print(f"Video saved to {self.video_path}")
        
    def update(self, robot_positions, ball_pos, true_ball_pos=None, info=None):
        """更新可视化"""
        self.ax.cla()  # 清除当前帧
        
        # 绘制机械臂
        positions = np.array(robot_positions).reshape(-1, 3)
        
        # 使用不同颜色和粗细绘制机械臂段
        colors = ['b', 'g', 'r']
        for i in range(len(positions)-1):
            # 绘制主要段
            self.ax.plot([positions[i,0], positions[i+1,0]], 
                        [positions[i,1], positions[i+1,1]], 
                        [positions[i,2], positions[i+1,2]], 
                        color=colors[i], linewidth=3, label=f'Segment {i+1}')
            
            # 绘制关节球
            self.ax.scatter([positions[i,0]], [positions[i,1]], [positions[i,2]], 
                           color=colors[i], s=100)
        
        # 绘制末端执行器
        self.ax.scatter([positions[-1,0]], [positions[-1,1]], [positions[-1,2]], 
                       color='r', s=150, marker='*', label='End Effector')
        
        # 更新并绘制球的观测轨迹
        if ball_pos is not None:
            self.ball_obs_history.append(ball_pos)
            if len(self.ball_obs_history) > 50:
                self.ball_obs_history.pop(0)
            
            ball_obs_trace = np.array(self.ball_obs_history)
            self.ax.plot(ball_obs_trace[:, 0], ball_obs_trace[:, 1], ball_obs_trace[:, 2], 
                        'r:', alpha=0.5, label='Observed Trajectory')
            self.ax.scatter(ball_pos[0], ball_pos[1], ball_pos[2], 
                          c='r', marker='o', s=50, label='Observed Ball')
        
        # 更新并绘制球的真实轨迹
        if true_ball_pos is not None:
            self.ball_history.append(true_ball_pos)
            if len(self.ball_history) > 50:
                self.ball_history.pop(0)
            
            ball_trace = np.array(self.ball_history)
            self.ax.plot(ball_trace[:, 0], ball_trace[:, 1], ball_trace[:, 2], 
                        'g-', alpha=0.8, label='True Trajectory')
            self.ax.scatter(true_ball_pos[0], true_ball_pos[1], true_ball_pos[2], 
                          c='g', marker='o', s=50, label='True Ball')
        
        # 设置坐标轴范围和标签
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([0, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Robot Arm Simulation')
        self.ax.legend()
        
        # 显示额外信息
        if info:
            status_text = []
            if 'success' in info:
                status = "Success!" if info['success'] else "Trying..."
                status_text.append(status)
            if 'episode' in info:
                status_text.append(f"Episode: {info['episode']}")
            if 'reward' in info:
                status_text.append(f"Reward: {info['reward']:.2f}")
            
            status_str = ' | '.join(status_text)
            self.ax.text2D(0.02, 0.98, status_str, transform=self.ax.transAxes)
        
        # 如果正在录制，保存当前帧
        if self.recording:
            self.writer.grab_frame()
            self.frame_count += 1
        
        # 在非服务器环境中，可以实时显示
        if matplotlib.get_backend() != 'Agg':
            plt.draw()
            plt.pause(0.001)
    
    def save_screenshot(self, filename='screenshot.png'):
        """保存当前帧为图片"""
        self.fig.savefig(os.path.join(self.video_path, filename))
    
    def close(self):
        """关闭可视化器"""
        if self.recording:
            self.stop_recording()
        plt.close(self.fig)
