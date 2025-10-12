import os
import time
import imageio
import numpy as np
import gymnasium as gym
import mujoco as mj


def random_action_test(save_path: str, steps: int = 250):
    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=2560, height=1920
    )
    obs, info = env.reset(options={"randomise_cup_position": True})

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


def manual_control(save_path: str = None):
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


def test_policy(env, obs) -> np.ndarray:
    import utils

    model, data = env.unwrapped.model, env.unwrapped.data

    if env._automaton_state == "move_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.2])

        target_quat = [0.707, 0.0, 0.0, 0.707]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + delta_q

        # add two values for gripper control (use -1)
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))

        action += utils.make_gripper_action(env, close=True)
        if np.linalg.norm(target_pos - utils.get_effector_pos(env)) < 0.04:
            env._automaton_state = "move_down"
            print("Switching to move_down state")
        return action[:9]
    elif env._automaton_state == "move_down":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([0.0, 0.0, -0.2])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model, data, site_name="grip_site", target_pos=target_pos
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        if np.linalg.norm(target_pos - utils.get_effector_pos(env)) < 0.01:
            env._automaton_state = "close_gripper"
            print("Switching to close_gripper state")
        return action[:9]
    elif env._automaton_state == "close_gripper":
        action = utils.make_gripper_action(env, close=True)


def collect_policy_episode(save_path="tmp/policy.mp4", steps=800):
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    obs, _ = env.reset()
    frames = []
    env._automaton_state = "move_above"
    for t in range(steps):
        action = test_policy(env, obs)
        obs, _, term, trunc, _ = env.step(action)
        frames.append(env.render())
        if term or trunc:
            break

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 12))
    print(f"Saved test-policy video to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["random", "policy", "manual"], default="random"
    )
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    if args.mode == "random":
        random_action_test(args.out, steps=args.steps)
    elif args.mode == "manual":
        manual_control()
    else:
        collect_policy_episode(steps=args.steps)
