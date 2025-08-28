import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RobotArmVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def update_plot(self, obs):
        """更新3D图像
        
        Args:
            obs: 13维观察向量
                - robot_joint_angles (6): 0-5
                - ee_position (3): 6-8
                - ee_velocity (3): 9-11
                - ball_position (3): 12-14
                - ball_velocity (3): 15-17
                - ball_visible (1): 18
        """
        self.ax.cla()
        
        # 设置坐标轴范围和标签
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # 绘制末端执行器
        ee_pos = obs[6:9]
        self.ax.scatter(*ee_pos, c='r', marker='o', s=100, label='End Effector')
        
        # 如果球可见，绘制球
        if obs[12] > 0.5:
            ball_pos = obs[9:12]
            self.ax.scatter(*ball_pos, c='b', marker='o', s=100, label='Ball')
            
            # 绘制速度向量
            ee_vel = obs[6:9]
            ball_vel = obs[9:12]
            
            # 缩放速度向量以便可视化
            scale = 0.5
            self.ax.quiver(*ee_pos, *ee_vel, color='r', length=scale, normalize=True)
            self.ax.quiver(*ball_pos, *ball_vel, color='b', length=scale, normalize=True)
        
        self.ax.legend()
        plt.pause(0.01)
    
    def close(self):
        plt.close()

def visualize_episode(env, policy, device="cuda"):
    """可视化一个完整的回合
    
    Args:
        env: 机器人环境
        policy: 训练好的策略
        device: 运行设备
    """
    visualizer = RobotArmVisualizer()
    obs, _ = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        # 可视化当前状态
        visualizer.update_plot(obs)
        
        # 获取动作
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = policy(obs_tensor, deterministic=True)
        action = action.squeeze(0).cpu().numpy()
        
        # 执行动作
        obs, reward, done, truncated, info = env.step(action)
        
        # 如果成功抓住球，显示成功信息
        if info.get('success', False):
            print("Successfully caught the ball!")
            break
    
    visualizer.close()

if __name__ == "__main__":
    # 测试可视化
    import torch
    from env.environment import RobotArmEnv
    from sac_optimized.sac import SAC
    
    # 创建环境
    env = RobotArmEnv()
    
    # 加载训练好的模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sac = SAC(lambda: env, env.observation_space.shape, env.action_space.shape, device=device)
    sac.load("path_to_your_model.pth")
    
    # 可视化一个回合
    visualize_episode(env, sac.actor, device)
