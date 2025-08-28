import numpy as np
from flask import Flask, Response, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from mpl_toolkits.mplot3d import Axes3D
import threading
import queue
from datetime import datetime
import logging

app = Flask(__name__)

# 全局状态队列
state_queue = queue.Queue(maxsize=1)
latest_image = None
latest_stats = {
    'episode': 0,
    'step': 0,
    'reward': 0.0,
    'success': False,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Robot Arm Training Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        .stat-box {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .success { color: green; }
        .failure { color: red; }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .timestamp {
            color: #666;
            text-align: right;
            font-size: 0.9em;
        }
    </style>
    <script>
        function updateImage() {
            const img = document.getElementById('robot-view');
            const timestamp = new Date().getTime();
            img.src = '/plot?' + timestamp;
        }

        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('episode').textContent = data.episode;
                    document.getElementById('step').textContent = data.step;
                    document.getElementById('reward').textContent = data.reward.toFixed(2);
                    document.getElementById('success').textContent = data.success ? 'Success!' : 'In Progress';
                    document.getElementById('success').className = data.success ? 'success' : 'failure';
                    document.getElementById('timestamp').textContent = data.timestamp;
                });
        }

        // 定期更新
        setInterval(updateImage, 100);
        setInterval(updateStats, 100);
    </script>
</head>
<body>
    <div class="container">
        <h1>Robot Arm Training Visualization</h1>
        <div class="stats">
            <div class="stat-box">
                <h3>Episode</h3>
                <p id="episode">0</p>
            </div>
            <div class="stat-box">
                <h3>Step</h3>
                <p id="step">0</p>
            </div>
            <div class="stat-box">
                <h3>Reward</h3>
                <p id="reward">0.00</p>
            </div>
            <div class="stat-box">
                <h3>Status</h3>
                <p id="success" class="failure">In Progress</p>
            </div>
        </div>
        <img id="robot-view" src="/plot" alt="Robot Arm Visualization">
        <p class="timestamp">Last Updated: <span id="timestamp">-</span></p>
    </div>
</body>
</html>
'''

class WebVisualizer:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        plt.style.use('dark_background')  # 使用深色主题
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 存储历史轨迹
        self.ee_history = []  # 末端执行器轨迹
        self.ball_history = []  # 球体轨迹
        self.max_history = 50  # 轨迹长度
        
        # 设置坐标轴范围和标签
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # 设置视角
        self.ax.view_init(elev=20, azim=45)
        
        # 禁用 Flask 日志输出
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # 设置 Flask 应用
        self.app = app
        self.app.logger.disabled = True
        
        self.server_thread = None
        
    def start(self):
        """启动Flask服务器在后台线程"""
        self.server_thread = threading.Thread(target=lambda: self.app.run(host=self.host, port=self.port))
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"Visualization server started at http://{self.host}:{self.port}")
        
    def update(self, observation, info=None):
        """更新可视化状态
        
        Args:
            observation: 13维观察向量
                - robot_joint_angles (6): 0-5
                - ee_position (3): 6-8
                - ee_velocity (3): 9-11
                - ball_position (3): 12-14
                - ball_velocity (3): 15-17
                - ball_visible (1): 18
            info: 环境信息字典
        """
        # 更新统计信息
        global latest_stats
        latest_stats = {
            'episode': info.get('episode', 0) if info else 0,
            'step': info.get('step', 0) if info else 0,
            'reward': info.get('reward', 0.0) if info else 0.0,
            'success': info.get('success', False) if info else False,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # 清除当前图像
            self.ax.cla()
            
            # 提取位置信息
            ee_pos = observation[6:9]
            ball_pos = observation[12:15]
            ball_visible = observation[18] > 0.5
            
            # 更新历史轨迹
            self.ee_history.append(ee_pos)
            if len(self.ee_history) > self.max_history:
                self.ee_history.pop(0)
            
            if ball_visible:
                self.ball_history.append(ball_pos)
                if len(self.ball_history) > self.max_history:
                    self.ball_history.pop(0)
            
            # 绘制末端执行器轨迹
            if self.ee_history:
                ee_history = np.array(self.ee_history)
                self.ax.plot(ee_history[:,0], ee_history[:,1], ee_history[:,2], 
                           'r--', alpha=0.5, label='EE Trajectory')
            
            # 绘制球体轨迹
            if self.ball_history:
                ball_history = np.array(self.ball_history)
                self.ax.plot(ball_history[:,0], ball_history[:,1], ball_history[:,2], 
                           'b--', alpha=0.5, label='Ball Trajectory')
            
            # 绘制当前位置
            self.ax.scatter(*ee_pos, color='red', s=100, label='End Effector')
            if ball_visible:
                self.ax.scatter(*ball_pos, color='blue', s=100, label='Ball')
                
                # 绘制速度向量
                ee_vel = observation[9:12]
                ball_vel = observation[15:18]
                scale = 0.2  # 速度向量缩放因子
                
                self.ax.quiver(*ee_pos, *ee_vel, color='red', length=scale, normalize=True)
                self.ax.quiver(*ball_pos, *ball_vel, color='blue', length=scale, normalize=True)
            
            # 设置视图
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            self.ax.set_zlim([0, 2])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.legend()
            
            # 如果成功抓住球，添加文本提示
            if info and info.get('success', False):
                self.ax.text2D(0.05, 0.95, "Caught!", transform=self.ax.transAxes,
                             color='green', fontsize=15)
            
            # 更新图像
            buf = BytesIO()
            self.fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            
            # 更新最新图像
            global latest_image
            latest_image = buf.getvalue()
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            
    def close(self):
        """关闭可视化器"""
        plt.close(self.fig)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/plot')
def plot():
    if latest_image is None:
        # 返回空白图像
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center')
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype='image/png')
    return Response(latest_image, mimetype='image/png')

@app.route('/stats')
def stats():
    return latest_stats

if __name__ == '__main__':
    # 测试代码
    import numpy as np
    import time
    
    visualizer = WebVisualizer()
    visualizer.start()
    
    # 生成测试数据
    for i in range(100):
        # 模拟机械臂位置
        observation = np.random.rand(19)
        
        # 更新可视化
        visualizer.update(
            observation=observation,
            info={
                'episode': 1,
                'step': i,
                'reward': np.random.rand() * 10,
                'success': i > 90
            }
        )
        time.sleep(0.1)
    
    input("Press Enter to exit...")
