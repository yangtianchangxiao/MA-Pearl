#!/usr/bin/env python3
"""Debug training loop step by step"""

import torch
from complex_kinematics_her_wrapper import ComplexKinematicsHERWrapper
from lightweight_gnn_actor import UltraLightGNNActor
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import ContinuousSoftActorCritic
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.utils.instantiations.environments.variable_soft_arm_her_factory import create_variable_soft_arm_her_buffer
from pearl.utils.instantiations.spaces.box import BoxSpace

print("🔍 Debugging training loop...")

# Create environment
env = ComplexKinematicsHERWrapper(
    dof_range=(2, 5),
    segment_length_range=(0.1, 0.35),
    goal_threshold=0.05,
    max_steps=200
)

# Create GNN actor
actor_network = UltraLightGNNActor(
    action_dim=10,
    dof_range=(2, 5),
    hidden_dim=128,
    num_gnn_layers=2
).cuda()

print("✅ Actor created")

# Action representation
action_rep_module = IdentityActionRepresentationModule(
    max_number_actions=10,
    representation_dim=10
)

# Create SAC
sac = ContinuousSoftActorCritic(
    state_dim=env.observation_space.shape[0],
    action_space=env.action_space,
    actor_network_instance=actor_network,
    critic_hidden_dims=[512, 512],
    action_representation_module=action_rep_module,
    training_rounds=25,
    batch_size=256
)

print("✅ SAC created")

# Create HER buffer
replay_buffer = create_variable_soft_arm_her_buffer(
    capacity=200000,
    joint_dim=10,
    spatial_dim=3,
    n_segments=5,
    threshold=0.05,
    include_lengths_in_obs=False
)

# Create Agent
agent = PearlAgent(
    policy_learner=sac,
    replay_buffer=replay_buffer
)

agent._action_space = env.action_space

print("✅ Agent created")

# Test single episode step by step
try:
    print("\n🧪 Testing single episode...")
    obs, action_space = env.reset()
    print(f"Reset successful: obs shape {obs.shape}")
    
    agent.reset(obs, action_space)
    print("Agent reset successful")
    
    episode_reward = 0
    step_count = 0
    
    for step in range(200):
        print(f"Step {step+1}...", end="")
        
        # Agent action
        action = agent.act(exploit=False)
        print(f"action shape {action.shape}...", end="")
        
        # Environment step
        action_result = env.step(action)
        print(f"env step done...", end="")
        
        # Agent observe
        agent.observe(action_result)
        print(f"agent observe done")
        
        episode_reward += action_result.reward.item()
        step_count += 1
        
        if action_result.terminated or action_result.truncated:
            print(f"Episode terminated at step {step_count}")
            print(f"Terminated: {action_result.terminated}, Truncated: {action_result.truncated}")
            break
    
    print(f"\n✅ Episode completed: {step_count} steps, reward: {episode_reward}")
    
except Exception as e:
    print(f"\n❌ Error at step {step_count}: {e}")
    import traceback
    traceback.print_exc()