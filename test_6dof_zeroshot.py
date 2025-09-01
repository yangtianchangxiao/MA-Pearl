#!/usr/bin/env python3
"""
6节Zero-shot测试 - 关键Ad Hoc泛化验证

训练范围: 2-5节 (4-10 DOF)  
测试目标: 6节 (12 DOF) - 完全未见过的配置
"""

import torch
import numpy as np
import random
from optimized_graph_environment import OptimizedGraphSoftArmEnvironment
from optimized_graph_her_wrapper import OptimizedGraphHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import ContinuousSoftActorCritic
from pearl.pearl_agent import PearlAgent
from pearl.replay_buffers.sequential_decision_making.fifo_off_policy_replay_buffer import FIFOOffPolicyReplayBuffer

class ZeroShot6DOFTester:
    """6节Zero-shot测试器"""
    
    def __init__(self, checkpoint_path="random_dof_gnn_results/best_checkpoint.pt"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cpu')  # 用CPU测试
        
        print("🔬 6节Zero-shot测试器")
        print("=" * 50)
        print(f"训练范围: 2-5节 (4-10 DOF)")
        print(f"测试目标: 6节 (12 DOF)")
        print(f"验证类型: Zero-shot泛化")
        print(f"Checkpoint: {checkpoint_path}")
        
        self._load_model()
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"\n📦 加载训练模型...")
        
        # 加载checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        print(f"   训练Episodes: {checkpoint['episode']}")
        print(f"   训练成功率: {checkpoint['success_rate']:.1%}")
        print(f"   训练DOF范围: {config['dof_range']}")
        
        # 创建环境 - 注意这里用6节测试！
        self.test_env = OptimizedGraphHERWrapper(
            dof_range=(6, 6),  # 强制6节测试
            goal_threshold=config['goal_threshold'],
            max_steps=config['max_episode_steps']
        )
        
        # 创建Actor网络
        action_space = self.test_env.action_space
        self.actor_network = UltraLightGNNActor(
            action_dim=action_space.shape[0],
            dof_range=(2, 5),  # 保持训练时的配置
            hidden_dim=config['hidden_dim'],
            num_gnn_layers=config['num_gnn_layers']
        ).to(self.device)
        
        # 创建SAC
        sac = ContinuousSoftActorCritic(
            actor_network_instance=self.actor_network,
            critic_hidden_dims=config['critic_hidden_dims'],
            action_space=action_space
        )
        
        # 创建Buffer (用于SAC)
        buffer = FIFOOffPolicyReplayBuffer(capacity=1000)
        
        # 创建Agent
        self.agent = PearlAgent(
            policy_learner=sac,
            replay_buffer=buffer,
            device=self.device
        )
        
        # 加载权重
        self.agent.load_state_dict(checkpoint['agent_state'])
        
        print(f"✅ 模型加载成功 - 准备6节测试")
    
    def test_single_episode(self, episode_id=0, max_steps=200):
        """测试单个6节episode"""
        print(f"\n🎮 Episode {episode_id+1} - 6节Zero-shot测试")
        print("-" * 40)
        
        # Reset环境
        obs, action_space = self.test_env.reset(seed=episode_id)
        self.agent.reset(obs, action_space)
        
        # 获取环境信息
        current_dof = self.test_env.env.current_n_segments * 2
        segment_lengths = self.test_env.env.segment_lengths
        goal_pos = self.test_env.env.goal_position
        
        print(f"配置信息:")
        print(f"   实际节数: {self.test_env.env.current_n_segments} (目标:6)")
        print(f"   实际DOF: {current_dof} (目标:12)")
        print(f"   Segment长度: {segment_lengths}")
        print(f"   目标位置: {goal_pos}")
        
        if self.test_env.env.current_n_segments != 6:
            print(f"⚠️  环境没有正确设置为6节，实际是{self.test_env.env.current_n_segments}节")
            return False, 0, 0, float('inf')
        
        # Episode循环
        step_count = 0
        total_reward = 0
        final_distance = 0
        
        for step in range(max_steps):
            # Agent行动
            action = self.agent.act(exploit=True)  # 测试时exploit
            
            # 环境step
            action_result = self.test_env.step(action)
            self.agent.observe(action_result)
            
            step_count += 1
            total_reward += action_result.reward.item()
            
            # 计算当前距离
            current_pos = self.test_env.env._forward_kinematics()
            distance = np.linalg.norm(current_pos - goal_pos)
            final_distance = distance
            
            if step % 20 == 0:
                print(f"   Step {step:3d}: reward={action_result.reward.item():6.1f}, distance={distance:.3f}m")
            
            # 检查终止条件
            if action_result.terminated or action_result.truncated:
                success = action_result.terminated.item()
                break
        else:
            success = False
        
        # 结果
        print(f"\n📊 Episode结果:")
        print(f"   成功: {'✅ 是' if success else '❌ 否'}")
        print(f"   步数: {step_count}")
        print(f"   总奖励: {total_reward:.1f}")
        print(f"   最终距离: {final_distance:.3f}m")
        print(f"   阈值: 0.15m")
        
        return success, step_count, total_reward, final_distance
    
    def batch_test(self, n_episodes=20):
        """批量6节测试"""
        print(f"\n🧪 批量6节Zero-shot测试")
        print(f"测试Episodes: {n_episodes}")
        print("=" * 60)
        
        results = []
        success_count = 0
        
        for i in range(n_episodes):
            success, steps, reward, distance = self.test_single_episode(i)
            results.append({
                'success': success,
                'steps': steps, 
                'reward': reward,
                'distance': distance
            })
            
            if success:
                success_count += 1
        
        # 统计分析
        success_rate = success_count / n_episodes
        avg_steps = np.mean([r['steps'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        avg_distance = np.mean([r['distance'] for r in results])
        
        print(f"\n📈 6节Zero-shot结果统计:")
        print("=" * 60)
        print(f"成功率: {success_rate:.1%} ({success_count}/{n_episodes})")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"平均奖励: {avg_reward:.1f}")
        print(f"平均距离: {avg_distance:.3f}m")
        print(f"目标阈值: 0.15m")
        
        # 对比分析
        print(f"\n🎯 泛化性能分析:")
        if success_rate > 0.6:
            print(f"✅ 优秀泛化: 6节成功率{success_rate:.1%} (训练84%)")
        elif success_rate > 0.3:
            print(f"⚠️  中等泛化: 6节成功率{success_rate:.1%} (有一定泛化能力)")
        else:
            print(f"❌ 泛化失败: 6节成功率{success_rate:.1%} (需要改进)")
        
        return results, success_rate


def main():
    """主测试函数"""
    print("🚀 6节Zero-shot泛化测试")
    print("这是Ad Hoc能力的关键验证！")
    print()
    
    # 创建测试器
    tester = ZeroShot6DOFTester()
    
    # 先测试几个单独episodes
    print("\n🎯 详细单episode测试:")
    for i in range(3):
        success, steps, reward, distance = tester.test_single_episode(i)
        if not success and distance > 0.5:
            print(f"   Episode {i+1}: 可能存在大泛化gap")
    
    # 批量统计测试
    print("\n📊 批量统计测试:")
    results, success_rate = tester.batch_test(n_episodes=20)
    
    # 结论
    print(f"\n🎉 6节Zero-shot测试完成!")
    print(f"关键发现: 6节(12DOF)成功率 = {success_rate:.1%}")
    if success_rate < 0.3:
        print(f"💡 建议: 需要Learning+Game方案改进泛化能力")
    else:
        print(f"🎯 已有不错的Ad Hoc泛化基础")


if __name__ == "__main__":
    main()