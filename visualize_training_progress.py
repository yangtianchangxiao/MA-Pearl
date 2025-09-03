#!/usr/bin/env python3
"""
实时训练进度可视化工具
监控tmux会话中的训练进度并生成可视化图表
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import subprocess
import os

class TrainingVisualizer:
    def __init__(self, session_name="complex_kinematics_original", save_dir="training_plots"):
        self.session_name = session_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储训练数据
        self.episodes = []
        self.success_rates = []
        self.avg_rewards = []
        self.avg_distances = []
        self.speeds = []
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
    def capture_tmux_output(self):
        """捕获tmux会话的最新输出"""
        try:
            cmd = f"tmux capture-pane -t {self.session_name} -p"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            print(f"Error capturing tmux output: {e}")
            return ""
    
    def parse_training_line(self, line):
        """解析训练输出行"""
        # Episode  416 | 成功率:  37.0% | 平均奖励:   -52.2 | 平均距离: 0.339m | 速度: 83.2 eps/h | 🎓学习中
        pattern = r'Episode\s+(\d+)\s+\|\s+成功率:\s+(\d+\.\d+)%\s+\|\s+平均奖励:\s+(-?\d+\.\d+)\s+\|\s+平均距离:\s+(\d+\.\d+)m\s+\|\s+速度:\s+(\d+\.\d+)\s+eps/h'
        match = re.search(pattern, line)
        
        if match:
            episode = int(match.group(1))
            success_rate = float(match.group(2))
            avg_reward = float(match.group(3))
            avg_distance = float(match.group(4))
            speed = float(match.group(5))
            
            return episode, success_rate, avg_reward, avg_distance, speed
        return None
    
    def update_data(self):
        """更新训练数据"""
        output = self.capture_tmux_output()
        lines = output.strip().split('\n')
        
        # 解析最新的训练行
        new_entries = []
        for line in lines[-50:]:  # 只看最后50行
            parsed = self.parse_training_line(line)
            if parsed:
                episode, success_rate, avg_reward, avg_distance, speed = parsed
                
                # 只添加新的episode数据
                if not self.episodes or episode > max(self.episodes):
                    new_entries.append((episode, success_rate, avg_reward, avg_distance, speed))
        
        # 添加新数据
        for episode, success_rate, avg_reward, avg_distance, speed in new_entries:
            self.episodes.append(episode)
            self.success_rates.append(success_rate)
            self.avg_rewards.append(avg_reward)
            self.avg_distances.append(avg_distance)
            self.speeds.append(speed)
            
        return len(new_entries) > 0
    
    def create_plots(self):
        """创建训练进度图表"""
        if len(self.episodes) < 2:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'🚀 固定4节复杂运动学训练进度 (Episode {max(self.episodes)})', fontsize=16)
        
        # 1. 成功率
        ax1.plot(self.episodes, self.success_rates, 'b-', linewidth=2, label='成功率')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('成功率 (%)')
        ax1.set_title('🎯 成功率变化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 添加成功率趋势
        if len(self.episodes) >= 10:
            z = np.polyfit(self.episodes[-20:], self.success_rates[-20:], 1)
            p = np.poly1d(z)
            ax1.plot(self.episodes[-20:], p(self.episodes[-20:]), "r--", alpha=0.8, label=f'趋势: {z[0]:.3f}%/ep')
            ax1.legend()
        
        # 2. 平均奖励
        ax2.plot(self.episodes, self.avg_rewards, 'g-', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('平均奖励')
        ax2.set_title('📈 平均奖励变化')
        ax2.grid(True, alpha=0.3)
        
        # 3. 平均距离
        ax3.plot(self.episodes, self.avg_distances, 'r-', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('平均距离 (m)')
        ax3.set_title('📏 平均距离变化')
        ax3.grid(True, alpha=0.3)
        
        # 添加目标阈值线
        if self.avg_distances:
            ax3.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='目标阈值 (0.15m)')
            ax3.legend()
        
        # 4. 训练速度
        ax4.plot(self.episodes, self.speeds, 'm-', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('训练速度 (eps/h)')
        ax4.set_title('⚡ 训练速度')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/training_progress_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        # 也保存最新版本
        latest_filename = f"{self.save_dir}/latest_progress.png"
        plt.savefig(latest_filename, dpi=300, bbox_inches='tight')
        
        return filename, latest_filename
    
    def print_summary(self):
        """打印训练摘要"""
        if not self.episodes:
            return
            
        current_ep = max(self.episodes)
        current_success = self.success_rates[-1]
        current_reward = self.avg_rewards[-1]
        current_distance = self.avg_distances[-1]
        current_speed = self.speeds[-1]
        
        # 计算改善趋势
        if len(self.episodes) >= 20:
            recent_success = np.mean(self.success_rates[-10:])
            early_success = np.mean(self.success_rates[-20:-10]) if len(self.success_rates) >= 20 else self.success_rates[0]
            improvement = recent_success - early_success
            
            print(f"\n🎯 训练摘要 (Episode {current_ep}):")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📊 当前状态:")
            print(f"   成功率: {current_success:.1f}%")
            print(f"   平均奖励: {current_reward:.1f}")
            print(f"   平均距离: {current_distance:.3f}m")
            print(f"   训练速度: {current_speed:.1f} eps/h")
            print(f"")
            print(f"📈 最近趋势 (最近10 vs 前10个episodes):")
            print(f"   成功率改善: {improvement:+.1f}%")
            
            # 性能评估
            if current_success >= 40:
                print(f"🏆 评估: 优秀! 固定4节效果显著")
            elif current_success >= 30:
                print(f"🎉 评估: 良好! 学习效果明显") 
            elif current_success >= 20:
                print(f"📈 评估: 改善中, 继续训练")
            else:
                print(f"🔄 评估: 探索阶段, 需要更多训练")

def main():
    print("🎯 启动训练进度实时可视化")
    print("=" * 50)
    print("监控会话: complex_kinematics_original")
    print("按 Ctrl+C 停止监控")
    print()
    
    visualizer = TrainingVisualizer()
    
    try:
        while True:
            # 更新数据
            has_new_data = visualizer.update_data()
            
            if has_new_data:
                # 创建图表
                try:
                    saved_files = visualizer.create_plots()
                    print(f"📊 图表已更新: {saved_files[1]}")
                    
                    # 打印摘要
                    visualizer.print_summary()
                    
                except Exception as e:
                    print(f"图表生成错误: {e}")
            
            # 等待30秒后再次检查
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 停止监控")
        print("最终图表已保存在 training_plots/ 目录")

if __name__ == "__main__":
    main()