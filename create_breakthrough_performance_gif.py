#!/usr/bin/env python3
"""
Breakthrough Performance GIF Creator
Creates animated GIF showing the 45% success rate model in action
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from pathlib import Path
import json

from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from optimized_graph_her_wrapper import OptimizedGraphHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer


class BreakthroughPerformanceGIF:
    """Create animated GIF of breakthrough model performance"""
    
    def __init__(self, checkpoint_path: str = 'complex_kinematics_gnn_results/best_checkpoint.pt'):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = 'cpu'
        
        print("🎬 Breakthrough Performance GIF Creator")
        print(f"   Model: {checkpoint_path}")
        
        # Load the breakthrough model
        self._load_model()
        
        # Animation data
        self.trajectory_data = []
        self.current_episode = 0
        
    def _load_model(self):
        """Load the breakthrough model"""
        print("📦 Loading breakthrough model...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        self.config = checkpoint.get('config', {
            'dof_range': (4, 4),
            'goal_threshold': 0.15,
            'max_episode_steps': 200
        })
        
        # Create environment  
        self.env = OptimizedGraphHERWrapper(
            dof_range=self.config['dof_range'],
            segment_length_range=self.config.get('segment_length_range', (0.1, 0.35)),
            goal_threshold=self.config['goal_threshold'],
            max_steps=self.config['max_episode_steps']
        )
        
        # Create actor network
        action_dim = max(self.config['dof_range']) * 2
        
        self.actor_network = UltraLightGNNActor(
            action_dim=action_dim,
            dof_range=self.config['dof_range'],
            hidden_dim=self.config.get('hidden_dim', 128),
            num_gnn_layers=self.config.get('num_gnn_layers', 2)
        ).to(self.device)
        
        # Create SAC learner
        state_dim = self.env.observation_space.shape[0]
        learner = ContinuousSoftActorCritic(
            action_space=self.env.action_space,
            state_dim=state_dim,
            actor_network_instance=self.actor_network,
            critic_hidden_dims=self.config.get('critic_hidden_dims', [512, 512]),
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4
        )
        
        # Create agent
        replay_buffer = create_variable_soft_arm_her_buffer(capacity=1000)
        self.agent = PearlAgent(
            policy_learner=learner,
            replay_buffer=replay_buffer
        )
        
        self.agent._action_space = self.env.action_space
        
        # Load weights with compatibility
        agent_state = checkpoint['agent_state']
        if 'policy_learner._actor.node_encoder.weight' in agent_state:
            self.agent.load_state_dict(agent_state)
        else:
            self.agent.policy_learner.load_state_dict(agent_state)
        
        self.agent.policy_learner._actor.eval()
        
        # Model info
        self.success_rate = checkpoint.get('success_rate', 45.0)
        self.episode_count = checkpoint.get('episode', 293)
        
        print(f"✅ Breakthrough model loaded!")
        print(f"   Success Rate: {self.success_rate:.1f}%")
        print(f"   Training Episodes: {self.episode_count}")
        
    def collect_episode_data(self, exploit=True):
        """Collect trajectory data from one episode"""
        print(f"\\n🎮 Collecting Episode {self.current_episode + 1} Data...")
        
        obs, action_space = self.env.reset()
        self.agent.reset(obs, action_space)
        
        # Get environment info
        current_dof = self.env.env.current_n_segments * 2
        segment_lengths = self.env.env.segment_lengths[:self.env.env.current_n_segments]
        target_pos = self.env.env.goal_position.copy()
        
        print(f"🔧 Configuration: {current_dof}DOF ({self.env.env.current_n_segments} segments)")
        print(f"   Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # Collect trajectory
        trajectory = []
        step = 0
        
        while step < 100:  # Shorter for GIF
            # Get action
            with torch.no_grad():
                action = self.agent.act(exploit=exploit)
            
            # Execute
            result = self.env.step(action)
            
            # Record position  
            current_pos = self.env.env._forward_kinematics()
            trajectory.append(current_pos.copy())
            
            obs = result.observation
            step += 1
            
            if result.terminated or result.truncated:
                success = result.terminated.item()
                break
        else:
            success = False
            
        final_distance = np.linalg.norm(trajectory[-1] - target_pos)
        
        print(f"📊 Result: {'✅ Success' if success else '❌ Failed'} | Distance: {final_distance:.3f}m")
        
        episode_data = {
            'trajectory': np.array(trajectory),
            'target': target_pos,
            'segment_lengths': segment_lengths,
            'dof': current_dof,
            'success': success,
            'final_distance': final_distance,
            'steps': len(trajectory)
        }
        
        self.current_episode += 1
        return episode_data
    
    def create_animated_gif(self, n_episodes=3, save_path="breakthrough_performance.gif"):
        """Create animated GIF showing multiple episode trajectories"""
        print(f"\\n🎥 Creating animated GIF with {n_episodes} episodes...")
        
        # Collect episode data
        episodes_data = []
        for i in range(n_episodes):
            episode_data = self.collect_episode_data(exploit=True)
            episodes_data.append(episode_data)
        
        # Setup figure
        fig = plt.figure(figsize=(16, 10))
        
        # Create subplots layout
        ax_3d = fig.add_subplot(221, projection='3d')
        ax_xy = fig.add_subplot(222)
        ax_xz = fig.add_subplot(223)
        ax_info = fig.add_subplot(224)
        
        fig.suptitle(f'🚀 BREAKTHROUGH MODEL: {self.success_rate:.1f}% Success Rate (Episode {self.episode_count})', 
                    fontsize=16, fontweight='bold')
        
        def animate(frame):
            # Clear all axes
            ax_3d.clear()
            ax_xy.clear()
            ax_xz.clear()
            ax_info.clear()
            
            # Calculate which episode and step
            total_steps = sum(len(ep['trajectory']) for ep in episodes_data)
            episode_idx = 0
            step_in_episode = frame % max(len(ep['trajectory']) for ep in episodes_data)
            
            # Find current episode
            cumulative_steps = 0
            for i, ep_data in enumerate(episodes_data):
                if frame < cumulative_steps + len(ep_data['trajectory']):
                    episode_idx = i
                    step_in_episode = frame - cumulative_steps
                    break
                cumulative_steps += len(ep_data['trajectory'])
            else:
                # Cycle back to first episode
                episode_idx = 0
                step_in_episode = frame % len(episodes_data[0]['trajectory'])
            
            current_ep = episodes_data[episode_idx]
            trajectory = current_ep['trajectory']
            target = current_ep['target']
            
            current_step = min(step_in_episode, len(trajectory) - 1)
            
            # Plot 3D trajectory
            if current_step > 0:
                ax_3d.plot(trajectory[:current_step+1, 0], 
                          trajectory[:current_step+1, 1], 
                          trajectory[:current_step+1, 2], 
                          'b-', linewidth=2, alpha=0.7, label='Trajectory')
            
            # Current position
            curr_pos = trajectory[current_step]
            ax_3d.scatter([curr_pos[0]], [curr_pos[1]], [curr_pos[2]], 
                         c='blue', s=100, label='End-Effector')
            
            # Target
            ax_3d.scatter([target[0]], [target[1]], [target[2]], 
                         c='red', s=200, marker='*', label='Target')
            
            # Success threshold sphere
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_sphere = target[0] + self.config['goal_threshold'] * np.cos(u) * np.sin(v)
            y_sphere = target[1] + self.config['goal_threshold'] * np.sin(u) * np.sin(v)
            z_sphere = target[2] + self.config['goal_threshold'] * np.cos(v)
            ax_3d.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='red')
            
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title(f'3D Trajectory - Episode {episode_idx+1}')
            ax_3d.legend()
            
            # Set consistent 3D limits
            all_points = np.vstack([trajectory, target.reshape(1, -1)])
            center = np.mean(all_points, axis=0)
            range_val = 0.5
            ax_3d.set_xlim(center[0]-range_val, center[0]+range_val)
            ax_3d.set_ylim(center[1]-range_val, center[1]+range_val)
            ax_3d.set_zlim(max(0, center[2]-range_val), center[2]+range_val)
            
            # XY projection
            if current_step > 0:
                ax_xy.plot(trajectory[:current_step+1, 0], trajectory[:current_step+1, 1], 
                          'b-', linewidth=2, alpha=0.7)
            ax_xy.scatter([curr_pos[0]], [curr_pos[1]], c='blue', s=100)
            ax_xy.scatter([target[0]], [target[1]], c='red', s=200, marker='*')
            
            # Threshold circle
            circle = Circle((target[0], target[1]), self.config['goal_threshold'], 
                           fill=False, color='red', linestyle='--', alpha=0.7)
            ax_xy.add_patch(circle)
            
            ax_xy.set_xlabel('X (m)')
            ax_xy.set_ylabel('Y (m)')
            ax_xy.set_title('XY Projection')
            ax_xy.grid(True, alpha=0.3)
            ax_xy.axis('equal')
            
            # XZ projection
            if current_step > 0:
                ax_xz.plot(trajectory[:current_step+1, 0], trajectory[:current_step+1, 2], 
                          'b-', linewidth=2, alpha=0.7)
            ax_xz.scatter([curr_pos[0]], [curr_pos[2]], c='blue', s=100)
            ax_xz.scatter([target[0]], [target[2]], c='red', s=200, marker='*')
            
            # Threshold circle
            circle_xz = Circle((target[0], target[2]), self.config['goal_threshold'], 
                              fill=False, color='red', linestyle='--', alpha=0.7)
            ax_xz.add_patch(circle_xz)
            
            ax_xz.set_xlabel('X (m)')
            ax_xz.set_ylabel('Z (m)')
            ax_xz.set_title('XZ Projection')
            ax_xz.grid(True, alpha=0.3)
            ax_xz.axis('equal')
            
            # Info panel
            ax_info.axis('off')
            
            # Current distance
            current_distance = np.linalg.norm(curr_pos - target)
            
            info_text = f"""🚀 BREAKTHROUGH MODEL DEMO
            
Episode {episode_idx+1}/{n_episodes} | Step {current_step+1}/{len(trajectory)}
            
🏆 Training Results:
  Success Rate: {self.success_rate:.1f}%
  Training Episodes: {self.episode_count}
  DOF Configuration: {current_ep['dof']} (Fixed 4 segments)
  
🎯 Current Episode:
  Current Distance: {current_distance:.4f}m
  Target Threshold: {self.config['goal_threshold']:.3f}m
  Episode Result: {'✅ Success' if current_ep['success'] else '❌ Failed'}
  Final Distance: {current_ep['final_distance']:.4f}m
  
🔧 Configuration:
  Segments: {len(current_ep['segment_lengths'])}
  Lengths: {[f'{l:.3f}' for l in current_ep['segment_lengths']]}
  
🎬 Animation: Frame {frame+1}
            """
            
            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
            
            return []
        
        # Create animation
        total_frames = max(len(ep['trajectory']) for ep in episodes_data) * n_episodes
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                     interval=200, blit=False, repeat=True)
        
        # Save as GIF
        print(f"💾 Saving GIF: {save_path}")
        anim.save(save_path, writer='pillow', fps=5, dpi=100)
        
        print(f"✅ Animated GIF created: {save_path}")
        return save_path


def main():
    """Create breakthrough performance GIF"""
    print("🎥 Breakthrough Performance GIF Creator")
    print("=" * 50)
    
    try:
        creator = BreakthroughPerformanceGIF()
        gif_path = creator.create_animated_gif(n_episodes=3)
        
        print(f"\\n🎉 Success! Animated GIF created:")
        print(f"   File: {gif_path}")
        print(f"   Shows: 45% success rate model in action")
        print(f"   Episodes: 3 different test cases")
        
    except Exception as e:
        print(f"❌ Error creating GIF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()