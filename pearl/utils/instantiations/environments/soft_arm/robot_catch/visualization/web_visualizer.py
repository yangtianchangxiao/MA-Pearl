from flask import Flask, render_template, jsonify
import os
import sys
import numpy as np
import torch

# 添加父目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from env.environment import RobotArmEnv
from env.env_factory import EnvFactory
from config.config_manager import ConfigManager
from agent.agent_factory import AgentFactory
import json

app = Flask(__name__)

class WebVisualizer:
    def __init__(self, model_path, config_path='config/default.yaml'):
        """初始化可视化器"""
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.env_config = self.config_manager.get_env_config()
        self.agent_config = self.config_manager.get_agent_config()
        
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
        
        # 初始化状态
        self.obs = None
        self.done = True
        
    def reset(self):
        """重置环境"""
        self.obs, _ = self.env.reset()
        self.done = False
        return self._get_state()
    
    def step(self):
        """执行一步"""
        if self.done:
            return self._get_state()
            
        action = self.agent.select_action(self.obs)
        self.obs, reward, self.done, _, info = self.env.step(action)
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        robot_positions = self.env.get_robot_positions().tolist()
        ball_position = self.env.get_ball_position().tolist()
        return {
            'robot_positions': robot_positions,
            'ball_position': ball_position,
            'done': self.done
        }

# 创建可视化器实例
visualizer = None

@app.route('/')
def index():
    """返回主页"""
    return render_template('index.html')

@app.route('/reset')
def reset():
    """重置环境"""
    global visualizer
    state = visualizer.reset()
    return jsonify(state)

@app.route('/step')
def step():
    """执行一步"""
    global visualizer
    state = visualizer.step()
    return jsonify(state)

def create_template_dir():
    """创建模板目录和文件"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # 创建index.html
    index_path = os.path.join(template_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Robot Arm Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body { margin: 0; }
        canvas { display: block; }
        #controls {
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="controls">
        <button onclick="reset()">Reset</button>
        <button onclick="toggleAutoStep()">Start/Stop</button>
    </div>
    <script>
        // Three.js setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Camera position
        camera.position.set(2, 2, 2);
        camera.lookAt(0, 0, 0);

        // Lights
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(1, 1, 1);
        scene.add(light);
        scene.add(new THREE.AmbientLight(0x404040));

        // Robot arm segments
        const robotSegments = [];
        const material = new THREE.MeshPhongMaterial({color: 0x0088ff});
        
        // Ball
        const ballGeometry = new THREE.SphereGeometry(0.05);
        const ballMaterial = new THREE.MeshPhongMaterial({color: 0xff0000});
        const ball = new THREE.Mesh(ballGeometry, ballMaterial);
        scene.add(ball);

        // Grid
        const grid = new THREE.GridHelper(4, 40);
        scene.add(grid);

        let autoStep = false;
        let lastStepTime = 0;
        const stepInterval = 50; // ms

        function reset() {
            fetch('/reset')
                .then(response => response.json())
                .then(updateScene);
        }

        function step() {
            fetch('/step')
                .then(response => response.json())
                .then(updateScene);
        }

        function toggleAutoStep() {
            autoStep = !autoStep;
            if (autoStep) {
                lastStepTime = performance.now();
                animate();
            }
        }

        function updateScene(state) {
            // Update robot segments
            while (robotSegments.length > 0) {
                const segment = robotSegments.pop();
                scene.remove(segment);
            }

            for (let i = 0; i < state.robot_positions.length - 1; i++) {
                const start = new THREE.Vector3(...state.robot_positions[i]);
                const end = new THREE.Vector3(...state.robot_positions[i + 1]);
                
                const direction = end.clone().sub(start);
                const length = direction.length();
                
                const geometry = new THREE.CylinderGeometry(0.02, 0.02, length);
                const segment = new THREE.Mesh(geometry, material);
                
                // Position and rotate segment
                segment.position.copy(start.clone().add(end).multiplyScalar(0.5));
                segment.quaternion.setFromUnitVectors(
                    new THREE.Vector3(0, 1, 0),
                    direction.normalize()
                );
                
                scene.add(segment);
                robotSegments.push(segment);
            }

            // Update ball position
            ball.position.set(...state.ball_position);

            if (state.done) {
                autoStep = false;
            }
        }

        function animate(currentTime) {
            if (autoStep) {
                if (currentTime - lastStepTime >= stepInterval) {
                    step();
                    lastStepTime = currentTime;
                }
                requestAnimationFrame(animate);
            }
            renderer.render(scene, camera);
        }

        // Initial reset
        reset();
        renderer.setAnimationLoop(() => {
            renderer.render(scene, camera);
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
        """)

def main():
    global visualizer
    # 设置模型路径
    model_path = os.path.join('checkpoints', 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return
    
    # 创建可视化器
    visualizer = WebVisualizer(model_path)
    
    # 创建模板目录和文件
    create_template_dir()
    
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main() 