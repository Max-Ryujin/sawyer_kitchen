import os
import time
import imageio
import numpy as np
import gymnasium as gym


def random_action_test(save_path: str, steps: int = 250):
    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make('KitchenMinimalEnv-v0',render_mode="rgb_array", width=2560, height=1920)
    obs, info = env.reset()

    frames = []
    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        if term:
            break

    env.close()

    if frames:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 12))
        print(f"Saved video to: {save_path}")
    else:
        print("No frames collected.")


def handcrafted_kettle_policy(obs: np.ndarray, nu: int) -> np.ndarray:
    #TODO

    return action = env.action_space.sample()


def collect_handcrafted_episode(save_path: str, steps: int = 500):
    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make('KitchenMinimalEnv-v0',render_mode="rgb_array", width=2560, height=1920)
    actual_env = env.env.env
    obs, info = env.reset()
    nu = actual_env.nu

    frames = []
    for t in range(steps):
        action = handcrafted_kettle_policy(obs, nu)
        obs, reward, term, trunc, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        if term:
            break

    env.close()

    if frames:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 12))
        print(f"Saved handcrafted-policy video to: {save_path}")
    else:
        print("No frames collected from handcrafted policy.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "policy"], default="random")
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()

    if args.mode == "random":
        random_action_test(args.out, steps=args.steps)
    else:
        collect_handcrafted_episode(args.out, steps=args.steps)
