from flask import Flask, render_template, jsonify
from threading import Thread
import numpy as np
import torch
import time
from env.environment import RobotArmEnv
from sac_optimized.sac_eval import SAC_Eval

class WebVisualizer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.running = False
        self.simulation_started = False  # 新增：标记是否开始模拟
        self.robot_positions = None
        self.ball_position = None
        self._setup_routes()
        self.server = None
        
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Robot Arm Visualization</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
                    .container { max-width: 800px; margin: 0 auto; }
                    #plot { width: 100%; height: 600px; }
                    .controls {
                        margin: 20px 0;
                        text-align: center;
                    }
                    button {
                        padding: 10px 20px;
                        font-size: 16px;
                        cursor: pointer;
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        transition: background-color 0.3s;
                    }
                    button:hover {
                        background-color: #45a049;
                    }
                    button:disabled {
                        background-color: #cccccc;
                        cursor: not-allowed;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Robot Arm Ball Catching Visualization</h1>
                    <div class="controls">
                        <button id="startBtn" onclick="startSimulation()">Start Simulation</button>
                    </div>
                    <div id="plot"></div>
                </div>
                <script>
                    let layout = {
                        scene: {
                            aspectmode: "cube",
                            xaxis: { range: [-2, 2] },
                            yaxis: { range: [-2, 2] },
                            zaxis: { range: [0, 2] }
                        },
                        margin: { l: 0, r: 0, b: 0, t: 0 }
                    };
                    
                    let config = {
                        responsive: true,
                        displayModeBar: false
                    };
                    
                    // 初始化空的3D图
                    Plotly.newPlot('plot', [{
                        type: 'scatter3d',
                        mode: 'lines+markers',
                        name: 'Robot Arm',
                        x: [],
                        y: [],
                        z: [],
                        line: { color: '#1f77b4', width: 6 },
                        marker: { size: 4 }
                    }, {
                        type: 'scatter3d',
                        mode: 'markers',
                        name: 'Ball',
                        x: [],
                        y: [],
                        z: [],
                        marker: { size: 8, color: '#ff7f0e' }
                    }], layout, config);
                    
                    function startSimulation() {
                        document.getElementById('startBtn').disabled = true;
                        fetch('/start')
                            .then(response => response.json())
                            .then(data => {
                                if (data.status === 'started') {
                                    console.log('Simulation started');
                                    document.getElementById('startBtn').textContent = 'Simulation Running';
                                }
                            });
                    }
                    
                    function updatePlot() {
                        fetch('/data')
                            .then(response => response.json())
                            .then(data => {
                                if (data.robot_positions && data.ball_position) {
                                    let update = {
                                        x: [[...data.robot_positions.x], [data.ball_position.x]],
                                        y: [[...data.robot_positions.y], [data.ball_position.y]],
                                        z: [[...data.robot_positions.z], [data.ball_position.z]]
                                    };
                                    Plotly.update('plot', update);
                                }
                            });
                    }
                    
                    // 定期更新图表
                    setInterval(updatePlot, 50);
                </script>
            </body>
            </html>
            """
        
        @self.app.route('/data')
        def get_data():
            if self.robot_positions is None or self.ball_position is None:
                return jsonify({})
            return jsonify({
                'robot_positions': self.robot_positions,
                'ball_position': self.ball_position
            })
        
        @self.app.route('/start')
        def start_simulation():
            self.simulation_started = True
            return jsonify({'status': 'started'})
    
    def start(self):
        """启动Flask服务器"""
        def run_server():
            self.app.run(host=self.host, port=self.port)
        
        self.server = Thread(target=run_server)
        self.server.daemon = True
        self.server.start()
    
    def update(self, robot_positions=None, ball_position=None):
        """更新可视化数据"""
        if robot_positions is not None:
            robot_positions = np.array(robot_positions)
        if ball_position is not None:
            ball_position = np.array(ball_position)
            
        self.robot_positions = {
            'x': robot_positions[:, 0].tolist() if robot_positions is not None else None,
            'y': robot_positions[:, 1].tolist() if robot_positions is not None else None,
            'z': robot_positions[:, 2].tolist() if robot_positions is not None else None,
        }
        self.ball_position = {
            'x': ball_position[0].tolist() if ball_position is not None else None,
            'y': ball_position[1].tolist() if ball_position is not None else None,
            'z': ball_position[2].tolist() if ball_position is not None else None,
        }
    
    def close(self):
        """关闭可视化器"""
        # Flask没有内置的优雅关闭方法
        # 我们只需确保线程是守护线程即可
        pass

def load_model(checkpoint_path):
    """加载模型"""
    # 创建环境和代理
    env_config = {
        'max_steps': 200,           # 每个回合的最大步数
        'catch_radius': 0.05,       # 抓取半径（5厘米）
        'perception_radius': 3.0,   # 感知半径（3米）
        'dt': 0.02,                # 时间步长（20毫秒）
        
        # 奖励设置
        'reward_catch': 100.0,      # 成功抓住的奖励
        'reward_distance': -0.5,    # 距离奖励系数（减小）
        'reward_action': -0.05,     # 动作惩罚系数（减小）
        'reward_angle': -0.5,       # 角度奖励系数（减小）
        
        # 抓取判定
        'catch': {
            'radius': 0.05,         # 抓取半径（5厘米）
            'angle_tolerance': 0.3,  # 允许的最大角度偏差（约17度）
            'min_angle': 1.4,       # 最小垂直角度（约80度）
            'max_angle': 1.7,       # 最大垂直角度（约100度）
        },
        
        # 噪声设置（模拟现实世界的不确定性）
        'noise': {
            'action': 0.02,         # 动作噪声（2%的动作范围）
            'observation': 0.005,    # 观测噪声（5毫米）
            'ball_init': 0.01,      # 球初始状态噪声（1厘米）
            'ball_vel': 0.05        # 球速度噪声（5%的速度）
        },
        
        # 球的参数（模拟现实物理）
        'ball': {
            'g': 9.81,              # 重力加速度（米/秒²）
            'pos_range': {          # 初始位置范围（米）
                'x': [-1.0, -0.8],  # 距离机器人基座0.8-1米
                'y': [-0.3, 0.3],   # 左右30厘米范围
                'z': [1.0, 1.2]     # 高度1-1.2米
            },
            'vel_range': {          # 初始速度范围（米/秒）
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
    
    env = RobotArmEnv(env_config)
    print(f"Environment created with action space: {env.action_space.shape}")
    
    agent = SAC_Eval(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_space_shape=env.action_space.shape
    )
    
    # 加载checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        if isinstance(checkpoint, dict):
            agent.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            if 'episode' in checkpoint:
                print(f"Checkpoint from episode {checkpoint['episode']}")
            if 'avg_reward' in checkpoint:
                print(f"Average reward: {checkpoint['avg_reward']:.2f}")
        else:
            print("Warning: Checkpoint format not recognized, using untrained model")
    except Exception as e:
        print(f"Warning: Failed to load checkpoint ({str(e)}), using untrained model")
    
    return env, agent

def visualize_model(checkpoint_path=None, num_episodes=5, host='0.0.0.0', port=5000):
    """加载训练好的模型并可视化其表现"""
    # 加载模型
    env, agent = load_model(checkpoint_path)
    
    # 创建可视化器并启动服务器
    vis = WebVisualizer(host=host, port=port)
    vis.start()
    print(f"\nVisualization server started at http://{host}:{port}")
    print("Waiting for user to start simulation...")
    
    try:
        # 等待用户点击开始按钮
        while not vis.simulation_started:
            time.sleep(0.1)
        
        print("\nSimulation started!")
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            print(f"\nStarting episode {episode + 1}")
            step = 0
            
            while not done:
                # 选择动作
                state = torch.FloatTensor(obs).unsqueeze(0)
                try:
                    action = agent.select_action(state)
                    print(f"Step {step}: Action shape: {action.shape}, Action: {action}")
                except Exception as e:
                    print(f"Error selecting action: {str(e)}")
                    break
                
                # 执行动作
                try:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    
                    print(f"Step {step}: Reward: {reward:.2f}, Total: {episode_reward:.2f}")
                except Exception as e:
                    print(f"Error executing action: {str(e)}")
                    break
                
                # 更新可视化
                try:
                    vis.update(
                        robot_positions=env.get_robot_positions(),
                        ball_position=env.get_ball_position()
                    )
                except Exception as e:
                    print(f"Error updating visualization: {str(e)}")
                
                obs = next_obs
                step += 1
                time.sleep(0.02)  # 限制更新频率
                
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    finally:
        vis.close()
        print("Server stopped")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize trained model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    args = parser.parse_args()
    
    visualize_model(args.checkpoint, args.episodes, args.host, args.port)
