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


def pick_and_place_policy(env, obs, target_pos=None) -> np.ndarray:
    """State machine: approach → descend → close → lift → move → open."""
    import utils
    if not hasattr(env, "_pz_state"):
        env._pz_state = "approach"
        env._step_counter = 0


    ee_pos = utils.get_effector_pos(env.unwrapped)


    if target_pos is None:
        cup_pos = utils.get_object_pos(env, ('cup_freejoint1', 'cup1'))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.1])

    model, data = env.unwrapped.model, env.unwrapped.data

    def solve_ik(pos):
        ik_res = utils.qpos_from_site_pose(model, data, "right_ee_attachment", target_pos=pos)
        return ik_res.qpos if ik_res.success else None

    s = env._pz_state

    if s == "approach":
        cup = utils.get_object_pos(env)
        above = cup + np.array([0.0, 0.0, 0.12])
        q_ik = solve_ik(above)
        a = utils.make_joint_pd_action(env, q_ik[:7] if q_ik is not None else ee_pos)
        env._step_counter += 1
        if np.linalg.norm(above - ee_pos) < 0.03 or env._step_counter > 120:
            env._pz_state, env._step_counter = "descend", 0
        return a

    if s == "descend":
        cup = utils.get_object_pos(env)
        grasp = cup + np.array([0.0, 0.0, 0.02])
        q_ik = solve_ik(grasp)
        a = utils.make_joint_pd_action(env, q_ik[:7] if q_ik is not None else ee_pos, kp=10.0)
        env._step_counter += 1
        if np.linalg.norm(grasp - ee_pos) < 0.02 or env._step_counter > 80:
            env._pz_state, env._step_counter = "close", 0
        return a

    if s == "close":
        env._pz_state, env._step_counter = "lift", 0
        return utils.make_gripper_action(env, close=True)

    if s == "lift":
        above = target_pos + np.array([0.0, 0.0, 0.2])
        q_ik = solve_ik(above)
        a = utils.make_joint_pd_action(env, q_ik[:7] if q_ik is not None else ee_pos)
        if np.linalg.norm(above - ee_pos) < 0.05:
            env._pz_state = "move"
        return a

    if s == "move":
        q_ik = solve_ik(target_pos)
        a = utils.make_joint_pd_action(env, q_ik[:7] if q_ik is not None else ee_pos)
        if np.linalg.norm(target_pos - ee_pos) < 0.04:
            env._pz_state = "open"
        return a

    if s == "open":
        env._pz_state = "done"
        return utils.make_gripper_action(env, close=False)

    return np.zeros(env.nu, dtype=np.float32)



def collect_handcrafted_episode(save_path="tmp/kitchen_policy_run.mp4", steps=200):
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make("KitchenMinimalEnv-v0", render_mode="rgb_array", width=2560, height=1920)
    obs, _ = env.reset()
    frames = []
    for _ in range(steps):
        action = pick_and_place_policy(env, obs)
        print(action)
        print("Obs", obs)

        obs, _, term, trunc, _ = env.step(action)
        frames.append(env.render())
        if term or trunc:
            break
    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 12))
    print(f"Saved handcrafted-policy video to {save_path}")

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
        manual_control()
    else:
        collect_handcrafted_episode(steps=args.steps)
