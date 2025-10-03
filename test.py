#!/usr/bin/env python3
"""Test script for the Sawyer Kitchen environment"""

import sys
import os
import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_environment():
    """Test the Sawyer Kitchen environment"""



    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make('KitchenMinimalEnv-v0',render_mode="rgb_array", width=2560, height=1920)
    print("Environment created successfully!")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    print("Testing reset...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    img = [env.render()]
    plt.imsave(f"test.png", img[-1])

    

    
    return True

if __name__ == "__main__":
    success = test_environment()
    if success:
        print("Environment test completed successfully!")
    else:
        print("Environment test failed!")
        sys.exit(1)
