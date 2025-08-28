#!/usr/bin/env python3
"""
Quick test of training setup before full training.
"""

import torch
from train_arm import ArmTrainer

def test_training():
    """Test training setup with a few episodes."""
    print("🧪 Testing training setup...")
    
    # Test with 3DOF for a few episodes
    trainer = ArmTrainer(
        dof=3,
        device="cuda:0",
        save_dir="./test_results"
    )
    
    print("\n🚀 Running quick test training (10 episodes)...")
    trainer.train(episodes=10, eval_every=5)
    
    print("\n🔍 Running quick evaluation (5 episodes)...")
    eval_results = trainer.evaluate(episodes=5)
    
    print(f"\n✅ Test completed!")
    print(f"Setup is working correctly.")

if __name__ == "__main__":
    test_training()