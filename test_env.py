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


def test_policy(env, obs) -> np.ndarray:
    import utils

    model, data = env.unwrapped.model, env.unwrapped.data

    if env._automaton_state == "move_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.25])

        target_quat = [0.69636424, -0.12278780, 0.12278780, 0.69636424]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 2.5 * delta_q

        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=False, open_val=-1, close_val=1)

        ee_pos = utils.get_effector_pos(env)

        xy_dist = np.linalg.norm(target_pos[:2] - ee_pos[:2])

        # Transition condition: good xy position and stable (slow movement)
        if (
            xy_dist < 0.0065
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.000006
        ):
            env._automaton_state = "move_down"
            print("Switching to move_down state")
        return action[:9]
    elif env._automaton_state == "move_down":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.04])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model, data, site_name="grip_site", target_pos=target_pos, reg_strength=1e-4
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.5 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=False, open_val=-1, close_val=1)
        if np.linalg.norm(target_pos - utils.get_effector_pos(env)) < 0.015:
            env._automaton_state = "close_gripper"
            print("Switching to close_gripper state")
        return action[:9]
    elif env._automaton_state == "close_gripper":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.04])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model, data, site_name="grip_site", target_pos=target_pos, reg_strength=1e-4
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.5 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=True, open_val=-1, close_val=1)
        gripper_joint_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "rc_close"),
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "lc_close"),
        ]

        forces = np.array([data.qfrc_constraint[i] for i in gripper_joint_ids])
        if np.linalg.norm(forces) > 15.0:
            env._automaton_state = "lift_up"
            print("Switching to lift_up state")
        return action[:9]
    elif env._automaton_state == "lift_up":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([0.0, -0.05, 0.35])
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            reg_strength=1e-4,
            target_quat=target_quat,
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.5 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=True, open_val=-1, close_val=1)
        ee_pos = utils.get_effector_pos(env)
        xy_dist = np.linalg.norm(target_pos[:2] - ee_pos[:2])
        if (
            xy_dist < 0.03
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.00004
        ):
            env._automaton_state = "pour"
            print("Switching to pour state")
        return action[:9]
    elif env._automaton_state == "pour":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([0.0, -0.05, 0.31])
        target_quat = [0.35355341, -0.61237242, 0.61237242, 0.35355341]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            reg_strength=1e-4,
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.5 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=True, open_val=-1, close_val=1)
        return action[:9]


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
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
    print(f"Saved test-policy video to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "policy"], default="random")
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()

    if args.mode == "random":
        random_action_test(args.out, steps=args.steps)
    else:
        collect_policy_episode(steps=args.steps)
