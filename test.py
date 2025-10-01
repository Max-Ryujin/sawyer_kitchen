#!/usr/bin/env python3
"""Test script for the Sawyer Kitchen environment"""

import sys
import os
import numpy as np
import cv2
import env as sawyer_kitchen
import matplotlib.pyplot as plt

def test_environment():
    """Test the Sawyer Kitchen environment"""

    try:
        env = sawyer_kitchen.KitchenMinimalEnv()
        print("Environment created successfully!")
        
        print(f"Action space: {env.nu}")
        print(f"Observation space: {env._get_observation()}")
        
        # Test reset
        print("Testing reset...")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        img = [env.render(mode='rgb_array')]

        # test each action dimension
        for dim in range(env.nu.shape[0]):
            for i in range(200):
                print(f"Testing action dimension {dim}...")
                action = np.zeros(env.nu.shape)
                action[dim] = np.random.uniform(-1, 1)
                obs, reward, done, info = env.step(action)
                img.append(env.render(mode='rgb_array'))
                #save image into a folder
                if i % 100 == 0:
                    print(f"Step {i}, Action: {action}, Reward: {reward}, Done: {done}")
                    plt.imsave(f"test_images/step_{dim}_{i}.png", img[-1])
        # create video mp4 from img
        video = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (img[0].shape[1], img[0].shape[0]))
        for i in range(len(img)):
            video.write(img[i])
        video.release()


        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_environment()
    if success:
        print("Environment test completed successfully!")
    else:
        print("Environment test failed!")
        sys.exit(1)
