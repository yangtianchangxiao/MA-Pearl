#!/usr/bin/env python3
"""
Live Training Progress Monitor
Monitors training checkpoints and displays real-time progress without interrupting training
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import json
import time
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import seaborn as sns

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LiveTrainingMonitor:
    """Real-time training progress monitor"""
    
    def __init__(self, results_dirs, update_interval=5.0):
        """
        Initialize the monitor
        
        Args:
            results_dirs: List of result directories to monitor
            update_interval: Update interval in seconds
        """
        self.results_dirs = [Path(d) for d in results_dirs]
        self.update_interval = update_interval
        
        # Data storage
        self.training_data = {}
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Initialize data structure for each directory
        for i, results_dir in enumerate(self.results_dirs):
            self.training_data[results_dir.name] = {
                'episodes': [],
                'success_rates': [],
                'avg_rewards': [],
                'avg_distances': [],
                'timestamps': [],
                'last_modified': 0,
                'color': self.colors[i % len(self.colors)],
                'config': {}
            }
        
        print(f"🔍 Live Training Monitor Initialized")
        print(f"   Monitoring directories: {[d.name for d in self.results_dirs]}")
        print(f"   Update interval: {update_interval}s")
        
    def read_checkpoint_data(self, results_dir):
        """Read data from checkpoint file"""
        checkpoint_path = results_dir / "best_checkpoint.pt"
        config_path = results_dir / "config.json"
        
        data = self.training_data[results_dir.name]
        
        try:
            # Check if checkpoint file has been modified
            if not checkpoint_path.exists():
                return False
                
            current_modified = checkpoint_path.stat().st_mtime
            if current_modified <= data['last_modified']:
                return False  # No new data
                
            data['last_modified'] = current_modified
            
            # Load checkpoint (PyTorch 2.6 compatibility)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Load config if available
            if config_path.exists() and not data['config']:
                with open(config_path, 'r') as f:
                    data['config'] = json.load(f)
            
            # Extract training metrics
            episode = checkpoint.get('episode', 0)
            success_rate = checkpoint.get('success_rate', 0.0)
            avg_reward = checkpoint.get('avg_reward', 0.0)
            avg_distance = checkpoint.get('avg_distance', 0.0)
            
            # Update data
            if not data['episodes'] or episode != data['episodes'][-1]:
                data['episodes'].append(episode)
                data['success_rates'].append(success_rate)
                data['avg_rewards'].append(avg_reward)
                data['avg_distances'].append(avg_distance)
                data['timestamps'].append(datetime.now())
                
                # Keep only last 1000 data points for performance
                if len(data['episodes']) > 1000:
                    for key in ['episodes', 'success_rates', 'avg_rewards', 'avg_distances', 'timestamps']:
                        data[key] = data[key][-1000:]
                
                return True
                
        except Exception as e:
            print(f"⚠️  Error reading {checkpoint_path}: {e}")
            
        return False
    
    def update_plots(self, frame):
        """Update all plots with latest data"""
        # Clear all subplots
        for ax in self.axes.flat:
            ax.clear()
        
        # Read latest data from all directories
        any_updates = False
        for results_dir in self.results_dirs:
            if self.read_checkpoint_data(results_dir):
                any_updates = True
        
        # Plot 1: Success Rate Over Time
        ax1 = self.axes[0, 0]
        for dir_name, data in self.training_data.items():
            if data['episodes']:
                ax1.plot(data['episodes'], data['success_rates'], 
                        label=f"{dir_name} (Latest: {data['success_rates'][-1]:.1f}%)",
                        color=data['color'], linewidth=2, marker='o', markersize=3)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('🎯 Success Rate Progress')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot 2: Average Reward Over Time
        ax2 = self.axes[0, 1]
        for dir_name, data in self.training_data.items():
            if data['episodes']:
                ax2.plot(data['episodes'], data['avg_rewards'], 
                        label=f"{dir_name} (Latest: {data['avg_rewards'][-1]:.1f})",
                        color=data['color'], linewidth=2, marker='s', markersize=3)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('📈 Average Reward Progress')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Distance to Goal
        ax3 = self.axes[1, 0]
        for dir_name, data in self.training_data.items():
            if data['episodes'] and data['config']:
                threshold = data['config'].get('goal_threshold', 0.1)
                ax3.axhline(y=threshold, color=data['color'], linestyle='--', alpha=0.5,
                           label=f"{dir_name} threshold ({threshold}m)")
                ax3.plot(data['episodes'], data['avg_distances'], 
                        label=f"{dir_name} distance (Latest: {data['avg_distances'][-1]:.3f}m)",
                        color=data['color'], linewidth=2, marker='^', markersize=3)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Distance (m)')
        ax3.set_title('📏 Distance to Goal Progress')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Summary & Statistics
        ax4 = self.axes[1, 1]
        ax4.axis('off')
        
        summary_text = "🚀 Live Training Summary\\n\\n"
        current_time = datetime.now().strftime("%H:%M:%S")
        summary_text += f"Last Update: {current_time}\\n\\n"
        
        for dir_name, data in self.training_data.items():
            if data['episodes']:
                config = data['config']
                latest_episode = data['episodes'][-1]
                latest_success = data['success_rates'][-1]
                latest_reward = data['avg_rewards'][-1]
                latest_distance = data['avg_distances'][-1]
                
                # Calculate training speed
                if len(data['timestamps']) >= 2:
                    time_diff = (data['timestamps'][-1] - data['timestamps'][0]).total_seconds()
                    episode_diff = data['episodes'][-1] - data['episodes'][0]
                    speed = (episode_diff / time_diff) * 3600 if time_diff > 0 else 0
                else:
                    speed = 0
                
                # Configuration info
                dof_range = config.get('dof_range', 'Unknown')
                threshold = config.get('goal_threshold', 'Unknown')
                network_type = "Large Net" if config.get('hidden_dim', 128) >= 256 else "Light Net"
                
                summary_text += f"📊 {dir_name}:\\n"
                summary_text += f"  Episode: {latest_episode}\\n"
                summary_text += f"  Success: {latest_success:.1f}%\\n"
                summary_text += f"  Reward: {latest_reward:.1f}\\n"
                summary_text += f"  Distance: {latest_distance:.4f}m\\n"
                summary_text += f"  Speed: {speed:.1f} ep/h\\n"
                summary_text += f"  DOF Range: {dof_range}\\n"
                summary_text += f"  Threshold: {threshold}m\\n"
                summary_text += f"  Network: {network_type}\\n\\n"
        
        # Add performance comparison
        if len(self.training_data) > 1:
            success_rates = [(name, data['success_rates'][-1] if data['success_rates'] else 0) 
                           for name, data in self.training_data.items()]
            success_rates.sort(key=lambda x: x[1], reverse=True)
            
            summary_text += "🏆 Performance Ranking:\\n"
            for i, (name, rate) in enumerate(success_rates):
                emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "📍"
                summary_text += f"  {emoji} {name}: {rate:.1f}%\\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Update main title with current best performance
        if self.training_data:
            best_success = max(data['success_rates'][-1] if data['success_rates'] else 0 
                             for data in self.training_data.values())
            self.fig.suptitle(f'🔥 Live Training Monitor - Best Success Rate: {best_success:.1f}%', 
                            fontsize=16, fontweight='bold')
        
        # Tight layout
        plt.tight_layout()
        
        return any_updates
    
    def start_monitoring(self):
        """Start the live monitoring"""
        print("🚀 Starting live training monitor...")
        
        # Create figure and subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 10))
        self.fig.suptitle('🔥 Live Training Monitor', fontsize=16, fontweight='bold')
        
        # Create animation
        self.anim = animation.FuncAnimation(
            self.fig, self.update_plots, interval=int(self.update_interval * 1000),
            blit=False, cache_frame_data=False
        )
        
        print("✅ Monitor started! Close the window to stop monitoring.")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Live Training Progress Monitor')
    parser.add_argument('--dirs', nargs='+', 
                      default=['complex_005_results', 'complex_010_results', 'complex_kinematics_gnn_results'],
                      help='Result directories to monitor')
    parser.add_argument('--interval', type=float, default=5.0,
                      help='Update interval in seconds')
    
    args = parser.parse_args()
    
    # Create and start monitor
    monitor = LiveTrainingMonitor(
        results_dirs=args.dirs,
        update_interval=args.interval
    )
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()