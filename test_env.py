import os
import sys
import json
from xml.parsers.expat import model
import imageio
import pickle
from tqdm import trange
import numpy as np
import gymnasium as gym
import jax
import mujoco as mj
import kitchen_utils as utils
from collections import defaultdict

EXAMPLE_GOAL_OBS = [
    9.73506629e-01,
    -2.93381691e-01,
    1.76090479e00,
    -1.10562456e00,
    1.11492622e00,
    -2.39137149e00,
    -1.46721113e00,
    1.34919360e-02,
    1.36773931e-02,
    2.45778156e-05,
    2.95590127e-07,
    2.45777410e-05,
    2.95589217e-07,
    2.45777410e-05,
    2.95589217e-07,
    2.45777410e-05,
    2.95589217e-07,
    2.16196258e-05,
    5.08073663e-06,
    7.20666153e-17,
    -6.44129189e-03,
    8.16410258e-17,
    -9.65133494e-17,
    -1.69006109e-01,
    -8.86719226e-06,
    1.61922598e00,
    1.00000000e00,
    -2.75028993e-11,
    -1.67092130e-05,
    -1.64597213e-06,
    -4.53902006e-01,
    -8.44409347e-01,
    1.58895946e00,
    9.99999940e-01,
    -8.86493581e-05,
    7.74951332e-05,
    -2.13907711e-04,
    -4.59555119e-01,
    -9.47338104e-01,
    1.78176177e00,
    5.88065982e-01,
    -8.08163822e-01,
    -1.02991927e-02,
    -3.07170041e-02,
    -4.39218938e-01,
    -8.57732475e-01,
    1.61266494e00,
    8.25388312e-01,
    -2.58950561e-01,
    1.80888623e-01,
    4.67929512e-01,
    -4.54208374e-01,
    -8.31465244e-01,
    1.61260152e00,
    1.71368793e-01,
    -6.88356459e-01,
    1.82999000e-01,
    6.80668414e-01,
    -4.42167342e-01,
    -8.45450461e-01,
    1.61271513e00,
    7.79130816e-01,
    -1.07870616e-01,
    6.16813004e-01,
    2.93402858e-02,
    -4.68076557e-01,
    -8.46153975e-01,
    1.65958333e00,
    -3.89271319e-01,
    -4.51916039e-01,
    -7.81671286e-01,
    1.82290137e-01,
    -4.60491419e-01,
    -8.36491287e-01,
    1.61673999e00,
    -3.07923466e-01,
    3.79064977e-01,
    7.41564155e-01,
    4.59973335e-01,
    -4.64173794e-01,
    -8.47201169e-01,
    1.61207044e00,
    -5.17658710e-01,
    -1.00626186e-01,
    -7.97066987e-01,
    2.94258505e-01,
    -4.69314069e-01,
    -8.36662710e-01,
    1.61271691e00,
    -7.01713979e-01,
    -5.89282930e-01,
    4.00073618e-01,
    1.68592017e-02,
    -5.13630807e-01,
    -5.65243602e-01,
    1.60586631e00,
    -6.16780341e-01,
    5.90872824e-01,
    3.58913004e-01,
    -3.76341254e-01,
    -5.03991961e-01,
    -5.99111199e-01,
    1.60586631e00,
    2.33071029e-01,
    8.39006007e-01,
    8.11235905e-02,
    -4.84938920e-01,
    -4.38755840e-01,
    -8.35725009e-01,
    1.61270916e00,
    5.05710006e-01,
    -5.46767488e-02,
    -4.45556760e-01,
    7.36713648e-01,
    -3.80809139e-03,
    5.25044417e-03,
    -1.10222045e-02,
    4.57993709e-03,
    2.53680218e-02,
    -4.82810140e-02,
    -4.65425104e-02,
    1.49135594e-04,
    -2.51445977e-04,
    -2.73009386e-16,
    1.51177424e-14,
    4.45399074e-17,
    -2.46834125e-15,
    -7.95988889e-17,
    2.94382574e-15,
    -2.95917019e-18,
    1.58764669e-15,
    -1.41808569e-17,
    2.46714837e-16,
    3.45010780e-17,
    0.00000000e00,
    4.12477285e-17,
    -4.80008740e-17,
    -1.88886015e-17,
    1.54123404e-18,
    -3.20179064e-14,
    2.10496919e-17,
    -9.18727764e-15,
    -2.41072144e-20,
    5.68723539e-04,
    -6.45796390e-05,
    1.20262953e-03,
    -1.83793548e-02,
    -2.09042653e-02,
    -3.30917374e-03,
    1.20622790e-04,
    -2.73444643e-03,
    6.32358016e-03,
    -3.89508456e-02,
    -4.35665883e-02,
    5.02851233e-02,
    -4.25276794e-02,
    5.69648873e-05,
    4.02764790e-03,
    -5.38549137e00,
    -3.42795324e00,
    -4.74702454e00,
    3.15522403e-01,
    2.65099108e-01,
    -5.53156026e-02,
    -1.27784693e00,
    -2.57492046e01,
    6.66805878e01,
    3.05667538e-02,
    -2.27050837e-02,
    1.35716854e-03,
    5.22538841e-01,
    4.74977684e00,
    5.22130299e00,
    1.77464727e-02,
    2.63910741e-01,
    -1.26469529e00,
    2.06071205e01,
    -4.49254684e01,
    -2.31568222e01,
    -8.47106725e-02,
    2.60553956e-01,
    -2.24403724e-01,
    5.88417892e01,
    1.69806728e01,
    9.09137192e01,
    -6.94471821e-02,
    -2.93772578e-01,
    -2.71356367e-02,
    -2.28012161e01,
    1.53139343e01,
    5.01560326e01,
    2.34853849e-03,
    1.66286505e-03,
    -1.00350611e-04,
    -1.71747237e-01,
    2.13417798e-01,
    -5.93292825e-02,
    -6.98776692e-02,
    3.54777217e-01,
    3.07554580e-14,
    -4.14107323e01,
    -8.81085575e-01,
    5.21953125e01,
    -7.29810447e-02,
    3.12638819e-01,
    3.07601235e-14,
    -3.06552238e01,
    -8.25846100e00,
    4.98997536e01,
    -4.10050480e-03,
    1.08119726e-04,
    1.52804307e-03,
    -6.43244505e-01,
    6.28983676e-02,
    4.69095320e-01,
]


def random_action_test(save_path: str, steps: int = 250):
    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=2560, height=1920
    )
    obs, info = env.reset(options={"randomise_cup_position": True, "minimal": True})

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
        imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
        print(f"Saved video to: {save_path}")
    else:
        print("No frames collected.")


def pour_policy(env, obs) -> np.ndarray:

    model, data = env.unwrapped.model, env.unwrapped.data
    if env._automaton_state == "move_left":

        target_pos = np.array([-0.6, -1.0, 2.2])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
        )

        # Default to last valid qpos if IK fails
        delta_q_scaled = 2.7 * delta_q
        if np.linalg.norm(delta_q_scaled) > 4:
            delta_q_scaled = delta_q
        q_target = data.qpos[:7] + delta_q_scaled

        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=False, open_val=-1, close_val=1)
        if np.linalg.norm(target_pos - utils.get_effector_pos(env)) < 0.02:
            env._automaton_state = "move_above"
            print("Switching to move_above state")
        return action[:9]
    elif env._automaton_state == "move_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.01, 0.0, 0.27])

        target_quat = [0.69636424, -0.12278780, 0.12278780, 0.69636424]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            reg_strength=3e-2,
        )

        # Default to last valid qpos if IK fails
        delta_q_scaled = 2.7 * delta_q
        if np.linalg.norm(delta_q_scaled) > 2:
            delta_q_scaled = delta_q
        if np.linalg.norm(delta_q_scaled) < 0.08:
            delta_q_scaled = 10 * delta_q
        q_target = data.qpos[:7] + delta_q_scaled

        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=False, open_val=-1, close_val=1)

        ee_pos = utils.get_effector_pos(env)

        xy_dist = np.linalg.norm(target_pos[:2] - ee_pos[:2])

        # Transition condition: good xy position and stable (slow movement)
        if (
            xy_dist < 0.0075
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.00001
        ):
            env._automaton_state = "move_down"
            print("Switching to move_down state")
        return action[:9]
    elif env._automaton_state == "move_down":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.01, 0.0, 0.04])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model, data, site_name="grip_site", target_pos=target_pos, reg_strength=1e-4
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.75 * delta_q

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
            env._automaton_state = "lift_above"
            print("Switching to lift_above state")
        return action[:9]
    elif env._automaton_state == "lift_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([0.0, -0.05, 0.34])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            reg_strength=1e-4,
        )
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
        target_pos = cup_pos + np.array([-0.005, -0.0145, 0.285])
        target_quat = [0.29883626, -0.64085637, 0.64085637, 0.29883626]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            rot_weight=0.8,
            # reg_strength=1e-4,
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.5 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=True, open_val=-1, close_val=1)
        return action[:9]


def pour_policy_v2(env, obs) -> np.ndarray:

    model, data = env.unwrapped.model, env.unwrapped.data

    def at_target(target_pos: np.ndarray, tol=0.04) -> bool:
        ee_pos = utils.get_effector_pos(env)
        return np.linalg.norm(target_pos - ee_pos) < tol

    def make_action(q_target: np.ndarray, close: bool) -> np.ndarray:
        action = np.pad(q_target[:7], (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=close, open_val=-1, close_val=1)
        return action[:9]

    state = env._automaton_state

    # ───────────────────────────
    # Move above cup
    # ───────────────────────────
    if state == "move_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.015, 0.0, 0.3])
        target_quat = [0.69636424, -0.12278780, 0.12278780, 0.69636424]
        env._state_counter += 1

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )
        if (
            at_target(target_pos, tol=0.08)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.001
        ) or env._state_counter > 200:
            env._automaton_state = "move_towards"
            env._state_counter = 0
            env._above_position = target_pos
            print("→ move_towards")

        return make_action(q_target, close=False)

    # ───────────────────────────
    # Move towards cup
    # ───────────────────────────
    if state == "move_towards":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.015, 0.0, 0.15])
        target_quat = [0.64085639, -0.29883623, 0.29883623, 0.64085639]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )
        if at_target(target_pos, tol=0.07) or env._state_counter > 200:
            env._automaton_state = "move_down"
            env._state_counter = 0
            print("→ move_down")

        alpha = 0.1
        q_current = data.qpos[:7].copy()
        q_smooth = q_current + alpha * (q_target[:7] - q_current)
        return make_action(q_smooth, close=False)

    # ───────────────────────────
    # Move down to grasp
    # ───────────────────────────
    elif state == "move_down":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.01, 0.0, 0.075])
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )
        if (
            np.abs(target_pos[2] - utils.get_effector_pos(env)[2]) < 0.006
            and np.abs(target_pos[1] - utils.get_effector_pos(env)[1]) < 0.005
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.001
        ) or env._state_counter > 160:
            env._automaton_state = "close_gripper"
            env._state_counter = 0
            print("→ close_gripper")
        return make_action(q_target, close=False)

    # ───────────────────────────
    # Close gripper
    # ───────────────────────────
    elif state == "close_gripper":

        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.01, 0.0, 0.075])
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )

        action = make_action(q_target, close=True)

        if env._state_counter > 100:
            env._state_counter = 0
            env._automaton_state = "move_towards"

        # detect grip by constraint forces
        gripper_joint_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "rc_close"),
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "lc_close"),
        ]
        forces = np.array([data.qfrc_constraint[i] for i in gripper_joint_ids])
        if (
            np.linalg.norm(forces) > 5.0
            and forces.all() > 0
            and at_target(target_pos, tol=0.05)
        ):
            env._automaton_state = "go_up"
            env._state_counter = 0
            print("→ go up")

        return action

    # ───────────────────────────
    # Move up above cup
    # ───────────────────────────
    elif state == "go_up":
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=env._above_position,
            target_quat=target_quat,
            inplace=False,
        )
        if (
            at_target(env._above_position, tol=0.1)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.02
        ):
            env._automaton_state = "lift_above"
            print("→ lift above")

        return make_action(q_target, close=True)
    # ───────────────────────────
    # Lift the cup up
    # ───────────────────────────
    elif state == "lift_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.4])
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )

        # check distance between grip site and gripped cup
        if (
            np.linalg.norm(
                utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
                - utils.get_effector_pos(env)
            )
            > 0.9
        ):
            print("Lost grip on cup, moving back to move_above")
            env._automaton_state = "move_above"
            env._state_counter = 0
            return make_action(q_target, close=True)

        if (
            # check xy positions only
            np.linalg.norm(target_pos[:2] - utils.get_effector_pos(env)[:2]) < 0.05
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.02
        ):
            env._automaton_state = "lift_lower"
            print("→ lift_lower")

        return make_action(q_target, close=True)

    # ───────────────────────────
    # Lower cup slightly
    # ───────────────────────────
    elif state == "lift_lower":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.29])
        target_quat = [0.57922797, -0.40557978, 0.40557978, 0.57922797]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )

        # check distance between grip site and gripped cup
        if (
            np.linalg.norm(
                utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
                - utils.get_effector_pos(env)
            )
            > 0.9
        ):
            print("Lost grip on cup, moving back to move_above")
            env._automaton_state = "move_above"
            env._state_counter = 0
            return make_action(q_target, close=False)

        if (
            at_target(target_pos, tol=0.03)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.01
        ) or env._state_counter > 180:
            env._automaton_state = "tilt_halfway"
            env._state_counter = 0
            print("→ tilt_halfway")

        alpha = 0.5
        q_current = data.qpos[:7].copy()
        q_smooth = q_current + alpha * (q_target[:7] - q_current)
        if np.linalg.norm(q_smooth - q_current) < 0.05:
            q_smooth = q_target[:7]

        return make_action(q_smooth, close=True)

    # ───────────────────────────
    # Tilt halfway
    # ───────────────────────────
    elif state == "tilt_halfway":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([-0.005, -0.02, 0.28])
        target_quat = [0.45451949, -0.54167521, 0.54167521, 0.45451949]

        cup1_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        ee_pos = utils.get_effector_pos(env)
        offset = ee_pos - cup1_pos
        target_pos[0] -= offset[0]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )

        if (
            at_target(target_pos, tol=0.02)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.02
            and np.abs(target_pos[0] - ee_pos[0]) < 0.002
        ) or env._state_counter > 180:
            env._automaton_state = "start_pouring"
            env._state_counter = 0
            print("→ start pouring")

        alpha = 0.5
        q_current = data.qpos[:7].copy()
        q_smooth = q_current + alpha * (q_target[:7] - q_current)
        if np.linalg.norm(q_smooth - q_current) < 0.05:
            q_smooth = q_target[:7]

        return make_action(q_smooth, close=True)

    elif state == "start_pouring":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([-0.01, -0.026, 0.22])
        target_quat = [0.40557981, -0.57922795, 0.57922795, 0.40557981]

        cup1_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        ee_pos = utils.get_effector_pos(env)
        offset = ee_pos - cup1_pos
        target_pos[0] -= offset[0]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )

        if (
            at_target(target_pos, tol=0.02)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.02
            and np.abs(target_pos[0] - ee_pos[0]) < 0.002
        ) or env._state_counter > 80:
            env._automaton_state = "pour"
            env._state_counter = 0
            print("→ pour")

        alpha = 0.5
        q_current = data.qpos[:7].copy()
        q_smooth = q_current + alpha * (q_target[:7] - q_current)
        if np.linalg.norm(q_smooth - q_current) < 0.01:
            q_smooth = q_target[:7]

        return make_action(q_smooth, close=True)

    # ───────────────────────────
    # Final pour
    # ───────────────────────────
    elif state == "pour":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([-0.02, -0.028, 0.24])
        target_quat = [0.12278783, -0.69636423, 0.69636423, 0.12278783]

        cup1_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        ee_pos = utils.get_effector_pos(env)
        offset = ee_pos - cup1_pos

        target_pos[0] -= offset[0]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            rot_weight=0.8,
            # reg_strength=1e-4,
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.5 * delta_q

        return make_action(q_target, close=True)


def collect_policy_episode(
    save_path="tmp/policy.mp4", steps=1000, noise=False, random_action=False
):
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    obs, _ = env.reset(options={"randomise_cup_position": True, "minimal": True})
    frames = []
    # env._automaton_state = "move_left"
    env._automaton_state = "move_above"
    env._state_counter = 0
    for t in range(steps):
        action = pour_policy_v2(env, obs)
        if random_action:
            if np.random.rand() < 0.02:
                action = env.action_space.sample()
        if noise:
            action = action + np.random.normal(0, 0.01, action.shape)
        obs, _, term, trunc, _ = env.step(action)
        frames.append(env.render())
        if term or trunc:
            print(f"Episode finished after {t+1} steps.")
            print(obs)
            break

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
    print(f"Saved test-policy video to {save_path}")


def collect_crl_episode(
    save_path="tmp/crl_policy.mp4", steps=800, checkpoint_path=None
):
    """Run a single episode using a trained CRL agent."""
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for CRL mode")

    # Add OGBench implementations to path
    THIS_DIR = os.path.dirname(__file__)
    OG_IMPLS = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "impls"))
    sys.path.insert(0, OG_IMPLS)
    from agents.crl import CRLAgent, get_config
    from utils.flax_utils import restore_agent

    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    obs, _ = env.reset(options={"randomise_cup_position": True, "minimal": True})
    frames = []

    cfg = get_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["alpha"] = 0.03

    agent_tmp = CRLAgent.create(
        seed=0, ex_observations=obs, ex_actions=env.action_space.sample(), config=cfg
    )

    agent = restore_agent(agent_tmp, checkpoint_path, 4000)
    print(f"Loaded checkpoint from {checkpoint_path}")

    for t in range(steps):
        obs_arr = np.asarray(obs)
        goal_arr = env.unwrapped.create_goal_state(current_state=obs_arr)

        action = agent.sample_actions(
            observations=obs_arr,
            goals=goal_arr,
            temperature=0.5,
            seed=jax.random.PRNGKey(0),
        )

        obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
        frames.append(env.render())
        if term or trunc:
            print(f"Episode finished after {t+1} steps.")
            break

    env.close()
    for i, f in enumerate(frames):
        if f is None:
            print(f"Frame {i} is None")
        elif f.shape != frames[0].shape:
            print(f"Frame {i} has different shape: {f.shape}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
    print(f"Saved CRL agent video to {save_path}")


def collect_policy_dataset(
    save_root: str = "tmp/policy_dataset",
    episodes: int = 100,
    max_steps: int = 3600,
    width: int = 1280,
    height: int = 960,
    minimal_observations: bool = False,
):
    """Run the policy multiple times and save successful (terminated==True)
    trajectories to per-episode JSON files. Each step stores obs, action,
    terminated boolean and a relative path to the rendered image for that step.

    Images for each episode are saved in an `images/` folder next to the JSON.
    """
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=width, height=height
    )
    dataset = defaultdict(list)
    os.makedirs(save_root, exist_ok=True)

    success_count = 0
    failure_counts = defaultdict(int)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = episodes
    num_val_episodes = episodes // 10

    debug_data = defaultdict(list)
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        obs, _ = env.reset(options={"randomise_cup_position": True, "minimal": True})
        env._automaton_state = "move_above"
        env._state_counter = 0
        ep_dir = os.path.join(save_root, f"episode_{ep_idx:03d}")
        images_dir = os.path.join(ep_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        traj = []
        episode_terminated = False
        for t in range(max_steps):
            action = pour_policy_v2(env, obs)
            obs_to_store = env.unwrapped._get_observation(minimal=True)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if minimal_observations:
                dataset["observations"].append(obs_to_store)
            else:
                dataset["observations"].append(obs)
            dataset["actions"].append(action)
            dataset["terminals"].append(done)
            dataset["qpos"].append(env.unwrapped.data.qpos.copy())
            dataset["qvel"].append(env.unwrapped.data.qvel.copy())

            obs = obs_next

            if done:
                episode_terminated = True
                total_steps += t
                if ep_idx < num_train_episodes:
                    total_train_steps += t
                # save last correct dataset entries for check
                for k in dataset.keys():
                    debug_data[k].append(dataset[k][-1])
                break

            elif t == max_steps - 1:
                if env._automaton_state == "pour":
                    # print where water is
                    Goal, Start = env.unwrapped.get_particles_in_cups()
                    print(f"Goal position at max_steps: {Goal}")
                    print(f"Start position at max_steps: {Start}")
                    # render final frame
                    final_frame = env.render()
                    final_image_path = os.path.join(images_dir, f"step_{t:03d}.png")
                    imageio.imwrite(final_image_path, final_frame)
                    print(f"Saved final frame to {final_image_path}")
                # episode reached time limit without termination; not saved
                print(
                    f"Episode {ep_idx} reached max_steps ({max_steps}) without termination; not saved."
                )
                for k in dataset.keys():
                    dataset[k] = dataset[k][:-max_steps]  # remove this episode's data

                    assert np.array_equal(
                        debug_data[k][-1], dataset[k][-1]
                    ), f"Data mismatch in key {k} at episode {ep_idx}, step {t}"
                break

        # record stats for this episode
        if episode_terminated:
            success_count += 1
        else:
            # record the automaton state reached at the end of the episode
            final_state = getattr(env, "_automaton_state", None)
            failure_counts[str(final_state)] += 1

    env.close()

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    train_path = os.path.join(save_root, "train_dataset.npz")
    val_path = os.path.join(save_root, "val_dataset.npz")
    for k, v in dataset.items():
        if "observations" in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == "terminals":
            dtype = bool
        elif k == "button_states":
            dtype = np.int64
        else:
            dtype = np.float32
        train_dataset[k] = np.array(v[:total_train_steps], dtype=dtype)
        val_dataset[k] = np.array(v[total_train_steps:], dtype=dtype)

    for path, dataset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dataset)

    # write stats summary
    stats = {
        "total_episodes": episodes,
        "successful_episodes": success_count,
        "success_rate": float(success_count) / float(episodes) if episodes > 0 else 0.0,
        "failure_counts": dict(failure_counts),
    }
    stats_path = os.path.join(save_root, "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved dataset to {save_root}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["random", "policy", "dataset", "crl"], default="random"
    )
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=1300)
    parser.add_argument(
        "--minimal", action="store_true", help="Use minimal observations"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect when using dataset mode",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to CRL agent checkpoint file for crl mode",
    )
    args = parser.parse_args()

    if args.mode == "random":
        random_action_test(args.out, steps=args.steps)
    elif args.mode == "policy":
        collect_policy_episode(steps=args.steps)
    elif args.mode == "dataset":
        # Use --out as a directory for the dataset
        save_root = args.out
        collect_policy_dataset(
            save_root=save_root,
            episodes=args.episodes,
            max_steps=args.steps,
            minimal_observations=args.minimal,
        )
    elif args.mode == "crl":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when using --mode=crl")
        collect_crl_episode(
            save_path=args.out, steps=args.steps, checkpoint_path=args.checkpoint
        )
