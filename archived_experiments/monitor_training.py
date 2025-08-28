#!/usr/bin/env python3
"""
Monitor training progress by reading saved metrics.
"""

import json
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

def monitor_training():
    """Monitor training progress from saved metrics."""
    directories = [
        "./training_results_3dof",
        "./training_results_4dof", 
        "./training_results_5dof"
    ]
    
    print("📊 Training Progress Monitor")
    print("=" * 50)
    
    while True:
        for dof, directory in zip([3, 4, 5], directories):
            metrics_file = Path(directory) / f"metrics_{dof}dof.json"
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    if metrics['episodes']:
                        latest_episode = metrics['episodes'][-1]
                        latest_success = metrics['success_rate'][-1]
                        latest_reward = metrics['rewards'][-1]
                        buffer_size = metrics['buffer_size'][-1]
                        
                        print(f"{dof}DOF - Episode {latest_episode}: "
                              f"Success {latest_success:.1f}%, "
                              f"Reward {latest_reward:.3f}, "
                              f"Buffer {buffer_size}")
                
                except Exception as e:
                    print(f"{dof}DOF - Error reading metrics: {e}")
            else:
                print(f"{dof}DOF - Not started yet")
        
        print("-" * 50)
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")