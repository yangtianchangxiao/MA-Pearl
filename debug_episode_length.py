#!/usr/bin/env python3
"""Debug episode length issue"""

from complex_kinematics_environment import ComplexKinematicsSoftArmEnvironment
import numpy as np

env = ComplexKinematicsSoftArmEnvironment(goal_threshold=0.05, max_steps=200)

obs, info = env.reset()
print(f"Initial distance: {info['distance']:.3f}m")
step_count = 0

for i in range(200):
    action = np.random.randn(env.action_space.shape[0]) * 0.01  # Small random actions
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    
    if terminated:
        print(f"SUCCESS at step {step_count}! Distance: {info['distance']:.3f}m")
        break
    if truncated:
        print(f"TRUNCATED at step {step_count}. Final distance: {info['distance']:.3f}m")
        break
    
    if step_count % 50 == 0:
        print(f"Step {step_count}: distance={info['distance']:.3f}m, reward={reward:.1f}")

print(f"Episode ended after {step_count} steps")
print(f"Expected ~200 steps, got {step_count} steps")

if step_count < 150:
    print("⚠️  Episode too short - something wrong!")
else:
    print("✅ Episode length normal")