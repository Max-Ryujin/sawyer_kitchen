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
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=2560, height=1920
    )
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
    # TODO
    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=2560, height=1920
    )
    action = env.action_space.sample()
    return action


def manuel_control(save_path: str = None):
    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=2560, height=1920
    )
    obs, info = env.reset()

    import matplotlib.pyplot as plt

    nu = env.unwrapped.nu
    action = np.zeros(nu, dtype=float)

    key_map = {
        "q": (0, 1.0),
        "a": (0, -1.0),
        "w": (1, 1.0),
        "s": (1, -1.0),
        "e": (2, 1.0),
        "d": (2, -1.0),
        "r": (3, 1.0),
        "f": (3, -1.0),
        "t": (4, 1.0),
        "g": (4, -1.0),
        "y": (5, 1.0),
        "h": (5, -1.0),
    }

    fig, ax = plt.subplots()
    plt.ion()
    img = None

    def on_key(event):
        k = event.key
        if k in key_map:
            idx, val = key_map[k]
            if idx < action.shape[0]:
                action[idx] = val

    def on_key_release(event):
        k = event.key
        if k in key_map:
            idx, val = key_map[k]
            if idx < action.shape[0]:
                action[idx] = 0.0

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    running = True
    while plt.fignum_exists(fig.number):
        obs, reward, term, trunc, info = env.step(action)
        print(action)
        if term:
            break

    env.close()
    plt.close(fig)


def collect_handcrafted_episode(save_path: str, steps: int = 500):
    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=2560, height=1920
    )
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
    parser.add_argument(
        "--mode", choices=["random", "policy", "manual"], default="random"
    )
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "random":
        random_action_test(args.out, steps=args.steps)
    elif args.mode == "manual":
        manuel_control()
    else:
        collect_handcrafted_episode(args.out, steps=args.steps)
