import os
import json
import imageio
import numpy as np
from collections import defaultdict
import gymnasium as gym
import jax
import mujoco as mj
import kitchen_utils as utils
from joblib import Parallel, delayed


def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def slow_down_position(ee_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Move halfway from current position to target to slow down movement."""
    return ee_pos + 0.5 * (target_pos - ee_pos)


def rotate_quat_around_z(base_quat, angle_rad):
    w = np.cos(angle_rad / 2)
    z = np.sin(angle_rad / 2)
    q_rot = np.array([w, 0.0, 0.0, z])
    return multiply_quaternions(q_rot, base_quat)


def slow_down_quaternion(
    current_quat: np.ndarray, target_quat: np.ndarray, factor: float = 0.5
) -> np.ndarray:
    """Interpolate between current and target quaternion to slow down rotation.
    Not need in this branch.
    """
    # Normalize both quaternions
    current_quat = current_quat / np.linalg.norm(current_quat)
    target_quat = target_quat / np.linalg.norm(target_quat)

    # Calculate dot product
    dot = np.dot(current_quat, target_quat)

    # If dot product is negative, negate one quaternion to take shorter path
    if dot < 0.0:
        target_quat = -target_quat
        dot = -dot

    # Clip dot product to prevent arccos from returning NaN
    dot = np.clip(dot, -1.0, 1.0)

    # If quaternions are very close, return the target
    if dot > 0.99:
        return target_quat

    # Calculate angle between quaternions
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # Calculate interpolation weights
    theta = theta_0 * factor
    sin_theta = np.sin(theta)

    # Check for sin_theta_0 being close to zero to avoid division by zero
    if np.abs(sin_theta_0) < 1e-6:
        return current_quat

    # Perform spherical linear interpolation
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    # Interpolate
    result = (s0 * current_quat) + (s1 * target_quat)
    return result / np.linalg.norm(result)


class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.155, sigma=0.0055):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = np.copy(self.mu)


def make_task_space_action(
    target_pos: np.ndarray, env, gripper_val: float, speed=0.1
) -> np.ndarray:
    """
    Build 8D task-space action [x, y, z, gripper]
    with normalized xyz to [-1, 1] and gripper to [0, 1].

    Args:
        target_pos: 3D world position in workspace bounds
        gripper_val: scalar in [0, 1] where 0=closed, 1=open

    Returns:
        4D action array
    """
    ee_pos = utils.get_effector_pos(env)
    delta = target_pos - ee_pos

    [x, y, z] = ee_pos + (delta * speed)

    # Clamp gripper to [0, 1]
    gripper = np.clip(float(gripper_val), 0.0, 1.0)

    action = np.array(
        [x, y, z, gripper],
        dtype=np.float32,
    )
    return action


def moving_policy(env, obs, cup_number) -> np.ndarray:
    model, data = env.unwrapped.model, env.unwrapped.data

    def at_target(target_pos: np.ndarray, tol=0.04) -> bool:
        ee_pos = utils.get_effector_pos(env)
        return np.linalg.norm(target_pos - ee_pos) < tol

    state = env._automaton_state

    if state == "move_above":
        cup_pos = utils.get_object_pos(
            env, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
        target_pos = cup_pos + np.array([-0.051, 0.031, 0.3])
        env._state_counter += 1

        if (
            at_target(target_pos, tol=0.02)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.001
        ) or env._state_counter > 110:
            env._automaton_state = "move_towards"
            env._state_counter = 0
            env._above_position = target_pos
            print("→ move_towards")

        action = make_task_space_action(target_pos, env, gripper_val=0.0, speed=0.5)
        action[:3] += env._noise_generator.sample()
        return action

    if state == "move_towards":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(
            env, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
        target_pos = cup_pos + np.array([-0.03, 0.03, 0.15])
        if at_target(target_pos, tol=0.02) or env._state_counter > 110:
            env._automaton_state = "move_down"
            env._state_counter = 0
            print("→ move_down")

        action = make_task_space_action(target_pos, env, gripper_val=0.0, speed=0.5)
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "move_down":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(
            env, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
        target_pos = cup_pos + np.array([-0.025, 0.03, 0.077])
        if (
            np.abs(target_pos[2] - utils.get_effector_pos(env)[2]) < 0.006
            and np.abs(target_pos[1] - utils.get_effector_pos(env)[1]) < 0.005
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.001
        ) or env._state_counter > 80:
            env._automaton_state = "close_gripper"
            env._state_counter = 0
            print("→ close_gripper")
        action = make_task_space_action(target_pos, env, gripper_val=0.0, speed=0.5)
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "close_gripper":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(
            env, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
        target_pos = cup_pos + np.array([-0.02, 0.03, 0.075])

        gripper_joint_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "rc_close"),
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "lc_close"),
        ]
        forces = np.array([data.qfrc_constraint[i] for i in gripper_joint_ids])
        if (
            np.linalg.norm(forces) > 5.0
            and forces.all() > 0
            and at_target(target_pos, tol=0.05)
        ) or env._state_counter > 70:
            env._automaton_state = "go_up"
            env._state_counter = 0
            print("→ go up")

        action = make_task_space_action(target_pos, env, gripper_val=1.0, speed=0.5)
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "go_up":
        if (
            at_target(env._above_position, tol=0.1)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.2
        ):
            env._automaton_state = "move_cup"
            other_cup_id = 1 - cup_number
            other_cup_pos = utils.get_object_pos(
                env, (f"cup_freejoint{other_cup_id}", f"cup{other_cup_id}")
            )
            while True:
                # randomise xy position
                # Fixed value to collect a good example for evaluation
                env._cup_destination = np.array(
                    [
                        -0.66,
                        -1,
                        1.71,
                    ]
                )
                if np.linalg.norm(env._cup_destination - other_cup_pos) > 0.1:
                    break
            print("→ move_cup")

        return make_task_space_action(
            env._above_position, env, gripper_val=1.0, speed=0.9
        )

    elif state == "move_cup":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.15  # move above place position

        if (
            at_target(target_pos, tol=0.09)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.002
        ):
            env._automaton_state = "place_cup"
            print("→ place_cup")

        action = make_task_space_action(target_pos, env, gripper_val=1.0, speed=0.7)
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "place_cup":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.01
        if at_target(target_pos, tol=0.04):
            env._automaton_state = "open_gripper"
            print("→ open_gripper")

        action = make_task_space_action(target_pos, env, gripper_val=1.0, speed=0.9)
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "open_gripper":
        target_pos = env._cup_destination.copy()
        env._state_counter += 1
        if env._state_counter > 20:
            env._automaton_state = "move_up_after_release"
        action = make_task_space_action(target_pos, env, gripper_val=0.0, speed=0.7)
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "move_up_after_release":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.35  # move up
        if at_target(target_pos, tol=0.4):
            env._automaton_state = "done"
        action = make_task_space_action(target_pos, env, gripper_val=0.0, speed=0.9)
        action[:3] += env._noise_generator.sample()
        return action


def pour_policy_v2(env, obs) -> np.ndarray:

    model, data = env.unwrapped.model, env.unwrapped.data

    def at_target(target_pos: np.ndarray, tol=0.04) -> bool:
        ee_pos = utils.get_effector_pos(env)
        return np.linalg.norm(target_pos - ee_pos) < tol

    state = env._automaton_state

    if not hasattr(env, "_quat_offset") or env._quat_offset is None:
        env._quat_offset = np.random.uniform(-0.3, 0.3)

    # Move above cup
    if state == "move_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.015, 0.0, 0.3])
        target_quat = [0.69636424, -0.12278780, 0.12278780, 0.69636424]
        env._state_counter += 1

        if (
            at_target(target_pos, tol=0.075)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.01
        ):
            env._automaton_state = "move_towards"
            env._state_counter = 0
            env._above_position = target_pos
            env._quat_offset = np.random.uniform(-0.3, 0.3)
            print("→ move_towards")
        if env._state_counter > 100:
            env._state_counter = 0
            env._quat_offset = np.random.uniform(-0.3, 0.3)
            env._automaton_state = "move_towards"
            env._above_position = target_pos

        action = make_task_space_action(target_pos, env, target_quat, gripper_val=0.0)
        action[:3] += env._noise_generator.sample()
        return action

    # Move towards cup
    if state == "move_towards":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.015, 0.0, 0.15])
        target_quat = [0.64085639, -0.29883623, 0.29883623, 0.64085639]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)
        if at_target(target_pos, tol=0.075):
            env._automaton_state = "move_down"
            env._state_counter = 0
            print("→ move_down")
        if env._state_counter > 100:
            env._state_counter = 0
            env._quat_offset = np.random.uniform(-0.3, 0.3)
            env._automaton_state = "move_down"
            print("→ move_down")

        action = make_task_space_action(target_pos, env, target_quat, gripper_val=0.0)
        action[:3] += env._noise_generator.sample()
        return action

    # Move down to grasp
    elif state == "move_down":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.01, 0.0, 0.08])
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)
        if (
            np.abs(target_pos[2] - utils.get_effector_pos(env)[2]) < 0.0065
            and np.abs(target_pos[1] - utils.get_effector_pos(env)[1]) < 0.0064
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.0025
        ) or env._state_counter > 90:
            env._automaton_state = "close_gripper"
            env._state_counter = 0
            print("→ close_gripper")
        action = make_task_space_action(target_pos, env, target_quat, gripper_val=0.0)
        action[:3] += env._noise_generator.sample()
        return action

    # Close gripper
    elif state == "close_gripper":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.01, 0.0, 0.076])
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)

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

        action = make_task_space_action(target_pos, env, target_quat, gripper_val=1.0)
        action[:3] += env._noise_generator.sample()
        return action

    # Move up above cup
    elif state == "go_up":
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)
        if (
            at_target(env._above_position, tol=0.1)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.02
        ):
            env._automaton_state = "lift_above"
            env._quat_offset = np.random.uniform(-0.3, 0.3)
            print("→ lift above")

        action = make_task_space_action(
            env._above_position, target_quat, gripper_val=1.0
        )
        action[:3] += env._noise_generator.sample()
        return action

    # Lift the cup up
    elif state == "lift_above":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.4])
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)

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
            action = make_task_space_action(
                target_pos, env, target_quat, gripper_val=1.0
            )
            action[:3] += env._noise_generator.sample()
            return action

        if (
            # check xy positions only
            (
                np.linalg.norm(target_pos[:2] - utils.get_effector_pos(env)[:2]) < 0.05
                and np.linalg.norm(
                    data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
                )
                < 0.02
            )
            or env._state_counter > 100
        ):
            env._state_counter = 0
            env._automaton_state = "lift_lower"
            print("→ lift_lower")

        action = make_task_space_action(target_pos, env, target_quat, gripper_val=1.0)
        action[:3] += env._noise_generator.sample()
        return action

    # Lower cup slightly
    elif state == "lift_lower":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.29])
        target_quat = [0.57922797, -0.40557978, 0.40557978, 0.57922797]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)

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
            action = make_task_space_action(
                target_pos, env, target_quat, gripper_val=1.0
            )
            action[:3] += env._noise_generator.sample()
            return action

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

        # Slow down movement by moving halfway to the target
        ee_pos = utils.get_effector_pos(env)
        actual_pos = slow_down_position(ee_pos, target_pos)

        # Get current quaternion of the end effector
        grip_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")
        rotation_matrix = data.site_xmat[grip_site_id].reshape(3, 3)
        current_quat = np.empty(4)
        mj.mju_mat2Quat(current_quat, rotation_matrix.flatten())
        actual_quat = slow_down_quaternion(current_quat, target_quat, 0.5)

        action = make_task_space_action(actual_pos, actual_quat, gripper_val=1.0)
        action[:3] += env._noise_generator.sample()
        return action

    # Tilt halfway
    elif state == "tilt_halfway":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([-0.005, -0.02, 0.28])
        target_quat = [0.45451949, -0.54167521, 0.54167521, 0.45451949]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)
        # Align using the `cup_top` site on cup1 so the cup_top ends up above cup0.
        cup0_pos = cup_pos
        sid = utils._try_name_lookup(env, "cup1:cup_top", "site")
        if sid == -1:
            sid = utils._try_name_lookup(env, "cup_top", "site")
        if sid != -1:
            cup1_top_pos = env.unwrapped.data.site_xpos[sid].copy()
            ee_pos = utils.get_effector_pos(env)
            rel = cup1_top_pos - ee_pos
            desired_top = np.array([cup0_pos[0], cup0_pos[1], cup1_top_pos[2]])
            target_pos = desired_top - rel
        else:
            cup1_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
            ee_pos = utils.get_effector_pos(env)
            offset = ee_pos - cup1_pos
            target_pos[0] -= offset[0]

        ee_pos = utils.get_effector_pos(env)
        if (
            at_target(target_pos, tol=0.024)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.02
            and np.abs(target_pos[0] - ee_pos[0]) < 0.002
        ) or env._state_counter > 100:
            env._automaton_state = "start_pouring"
            env._state_counter = 0
            print("→ start pouring")

        # Slow down movement by moving halfway to the target
        actual_pos = slow_down_position(ee_pos, target_pos)

        # Get current quaternion of the end effector
        grip_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")
        rotation_matrix = data.site_xmat[grip_site_id].reshape(3, 3)
        current_quat = np.empty(4)
        mj.mju_mat2Quat(current_quat, rotation_matrix.flatten())
        actual_quat = slow_down_quaternion(current_quat, target_quat, 0.5)

        action = make_task_space_action(actual_pos, actual_quat, gripper_val=1.0)
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "start_pouring":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([-0.008, -0.02, 0.22])
        target_quat = [0.40557981, -0.57922795, 0.57922795, 0.40557981]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)
        # Align using cup1's `cup_top` site so cup_top will be above cup0.
        cup0_pos = cup_pos
        sid = utils._try_name_lookup(env, "cup1:cup_top", "site")
        if sid == -1:
            sid = utils._try_name_lookup(env, "cup_top", "site")
        if sid != -1:
            cup1_top_pos = env.unwrapped.data.site_xpos[sid].copy()
            ee_pos = utils.get_effector_pos(env)
            rel = cup1_top_pos - ee_pos
            desired_top = np.array([cup0_pos[0], cup0_pos[1], cup1_top_pos[2]])
            target_pos = desired_top - rel
        else:
            cup1_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
            ee_pos = utils.get_effector_pos(env)
            offset = ee_pos - cup1_pos
            target_pos[0] -= offset[0]

        ee_pos = utils.get_effector_pos(env)
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

        # Slow down movement by moving halfway to the target
        actual_pos = slow_down_position(ee_pos, target_pos)

        # Get current quaternion of the end effector
        grip_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")
        rotation_matrix = data.site_xmat[grip_site_id].reshape(3, 3)
        current_quat = np.empty(4)
        mj.mju_mat2Quat(current_quat, rotation_matrix.flatten())
        actual_quat = slow_down_quaternion(current_quat, target_quat, 0.5)

        action = make_task_space_action(actual_pos, actual_quat, gripper_val=1.0)
        action[:3] += env._noise_generator.sample()
        return action

    # Final pour
    elif state == "pour":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([-0.01, -0.02, 0.21])
        target_quat = [0.12278783, -0.69636423, 0.69636423, 0.12278783]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)
        # Compute target so that the `cup_top` site of cup1 will end up over cup0.
        cup0_pos = cup_pos
        sid = utils._try_name_lookup(env, "cup1:cup_top", "site")
        if sid == -1:
            sid = utils._try_name_lookup(env, "cup_top", "site")
        if sid != -1:
            cup1_top_pos = env.unwrapped.data.site_xpos[sid].copy()
            ee_pos = utils.get_effector_pos(env)
            rel = cup1_top_pos - ee_pos
            desired_top = np.array([cup0_pos[0], cup0_pos[1], cup1_top_pos[2]])
            target_pos = desired_top - rel
        else:
            cup1_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
            ee_pos = utils.get_effector_pos(env)
            offset = ee_pos - cup1_pos
            target_pos[0] -= offset[0]

        # Slow down movement by moving halfway to the target
        actual_pos = slow_down_position(ee_pos, target_pos)

        # Get current quaternion of the end effector
        grip_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")
        rotation_matrix = data.site_xmat[grip_site_id].reshape(3, 3)
        current_quat = np.empty(4)
        mj.mju_mat2Quat(current_quat, rotation_matrix.flatten())
        actual_quat = slow_down_quaternion(current_quat, target_quat, 0.5)

        action = make_task_space_action(actual_pos, actual_quat, gripper_val=1.0)
        action[:3] += env._noise_generator.sample()
        return action


def collect_policy_episode(
    save_path="tmp/policy.mp4",
    steps=1000,
    noise=True,
    random_action=False,
    policy_type="pouring",
):
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=640, height=480
    )
    obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
    frames = []
    env._automaton_state = "move_above"
    env._state_counter = 0
    cup = np.random.choice(np.array([0, 1]))
    for t in range(steps):
        if policy_type == "moving":
            action = moving_policy(env, obs, cup_number=cup)
            if env._automaton_state == "done":
                env._automaton_state = "move_above"
                cup = np.random.choice(np.array([0, 1]))
        else:
            action = pour_policy_v2(env, obs)

        if random_action:
            if np.random.rand() < 0.02:
                action = env.action_space.sample()
        if noise:
            action = action + np.random.normal(0, 0.01, action.shape)
        obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
        frames.append(env.render())
        if term or trunc:
            print(f"Episode finished after {t+1} steps.")
            break

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
    print(f"Saved test-policy video to {save_path}")


def run_single_episode(
    seed,
    max_steps,
    width,
    height,
    noise,
    pixel_observations,
    random_action,
    minimal_observations,
    save_failed_episodes,
    pouring_prob,
):
    """
    Worker function for parallel data collection.
    Runs one episode and returns the trajectory data.
    """
    # Re-import inside worker to ensure clean state
    import gymnasium as gym
    import numpy as np
    from collections import defaultdict

    # Register the environment inside the worker process
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")

    if pixel_observations:
        env = gym.make(
            "KitchenMinimalEnv-v0",
            render_mode="rgb_array",
            width=width,
            height=height,
            ob_type="pixels",
        )
    else:
        env = gym.make(
            "KitchenMinimalEnv-v0", render_mode="rgb_array", width=width, height=height
        )

    # Seed the environment
    obs, _ = env.reset(
        seed=seed, options={"randomise_cup_position": True, "minimal": True}
    )

    env._noise_generator = OUNoise(size=3)

    # Initialize state tracking variables locally
    env._automaton_state = "move_above"
    env._state_counter = 0

    episode_data = defaultdict(list)
    success = False
    failure_reason = "max_steps"

    move_operations = np.random.randint(0, 3)
    if move_operations == 0:
        perform_pouring = True
    else:
        perform_pouring = np.random.rand() < pouring_prob

    moves_completed = 0
    policy_mode = "moving" if move_operations > 0 else "pouring"
    # For testing:
    policy_mode = "moving"  # For testing moving only
    done2 = False
    cup = np.random.choice(np.array([0, 1]))

    steps_run = 0

    # --- State timing instrumentation ---
    last_state = getattr(env, "_automaton_state", None)
    state_step_counter = 0
    state_stats_local = defaultdict(lambda: {"total_steps": 0, "count": 0})
    # ------------------------------------

    for t in range(max_steps):
        action = None
        # --- POLICY LOGIC ---
        if policy_mode == "moving":
            action = moving_policy(env, obs, cup_number=cup)
            if env._automaton_state == "done":
                moves_completed += 1
                if moves_completed == move_operations:
                    if perform_pouring:
                        #   policy_mode = "pouring"  (For testing)
                        # THis is just for moving only
                        done2 = True
                    else:
                        done2 = True
                else:
                    cup = np.random.choice(np.array([0, 1]))
                env._automaton_state = "move_above"
        elif policy_mode == "pouring":
            action = pour_policy_v2(env, obs)
        # --------------------

        if noise:
            action = action + np.random.normal(0, 0.01, action.shape)

        episode_data["qpos"].append(env.unwrapped.data.qpos.copy())
        episode_data["qvel"].append(env.unwrapped.data.qvel.copy())

        obs_to_store = env.unwrapped._get_observation(minimal=True)
        obs_next, reward, terminated, trunc, info = env.unwrapped.step(
            action, minimal=True
        )
        done = terminated or trunc or done2

        if minimal_observations:
            episode_data["observations"].append(obs_to_store)
        else:
            episode_data["observations"].append(obs)

        episode_data["actions"].append(action)
        episode_data["terminals"].append(done)

        obs = obs_next
        steps_run += 1

        # --- State timing update ---
        state_step_counter += 1
        current_state = getattr(env, "_automaton_state", None)
        if current_state != last_state:
            # print how many steps last_state took
            print(f"State '{last_state}' took {state_step_counter} steps")
            # record stats
            state_stats_local[last_state]["total_steps"] += state_step_counter
            state_stats_local[last_state]["count"] += 1
            state_step_counter = 0
            last_state = current_state
        # ----------------------------

        if done:
            # finalize last state's counter
            if state_step_counter > 0 and last_state is not None:
                print(f"State '{last_state}' took {state_step_counter} steps")
                state_stats_local[last_state]["total_steps"] += state_step_counter
                state_stats_local[last_state]["count"] += 1
                state_step_counter = 0
            success = True
            failure_reason = None
            env._noise_generator.reset()
            break

        if t == max_steps - 1:
            failure_reason = getattr(env, "_automaton_state", "unknown")
            # finalize last state's counter even on timeout
            if state_step_counter > 0 and last_state is not None:
                print(f"State '{last_state}' took {state_step_counter} steps (end)")
                state_stats_local[last_state]["total_steps"] += state_step_counter
                state_stats_local[last_state]["count"] += 1
                state_step_counter = 0
                env._noise_generator.reset()

    env.close()

    # Return structure: (success, failure_reason, data_dict, steps_count, state_stats)
    # Convert defaultdict to plain dict for serialization
    state_stats_local = dict(state_stats_local)

    if success or save_failed_episodes:
        if not success and save_failed_episodes:
            # Mark last terminal as True if forcing save
            episode_data["terminals"][-1] = True
        return (success, failure_reason, episode_data, steps_run, state_stats_local)
    else:
        return (success, failure_reason, None, steps_run, state_stats_local)


def collect_moving_policy_dataset(
    save_root: str = "tmp/policy_dataset",
    episodes: int = 100,
    max_steps: int = 1900,
    width: int = 320,
    height: int = 240,
    noise: bool = True,
    pixel_observations: bool = False,
    random_action: bool = False,
    minimal_observations: bool = True,
    save_failed_episodes: bool = False,
    pouring_prob: float = 0.9,
):
    """
    Main entry point for parallel dataset collection.

    Orchestrates the parallel execution of `run_single_episode`, aggregates
    the data, splits it into Train/Val sets, and saves the results to .npz files.

    Args:
        save_root: Directory to save output files.
        episodes: Number of episodes.
        max_steps: Maximum steps per episode.
        width: Render width.
        height: Render height.
        noise: Add action noise.
        pixel_observations: Capture pixel inputs.
        random_action: Inject random actions (never actually used).
        minimal_observations: Use low-dim observations. (This is old and is always true. Full observation was never used)
        save_failed_episodes: Keep data from episodes that do not reach goal.
        pouring_prob: Probability of executing a pour after moving cups. This is to control the data distribution.
    """
    os.makedirs(save_root, exist_ok=True)

    num_val_attempts = max(1, episodes // 10)
    total_episodes_to_run = episodes + num_val_attempts

    print(f"Starting parallel collection of {total_episodes_to_run} episodes...")
    if pixel_observations:
        print("Ensure 'export MUJOCO_GL=egl' is set for GPU rendering.")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_episode)(
            seed=i,
            max_steps=max_steps,
            width=width,
            height=height,
            noise=noise,
            pixel_observations=pixel_observations,
            random_action=random_action,
            minimal_observations=minimal_observations,
            save_failed_episodes=save_failed_episodes,
            pouring_prob=pouring_prob,
        )
        for i in range(total_episodes_to_run)
    )

    # --- Aggregation Logic ---
    dataset = defaultdict(list)
    success_count = 0
    failure_counts = defaultdict(int)

    # Filter only valid results (non-None)
    valid_results = [r for r in results if r[2] is not None]
    num_valid = len(valid_results)

    num_val = max(1, int(num_valid * 0.1))
    num_train = num_valid - num_val

    print(
        f"Collected {num_valid} valid episodes. Splitting: {num_train} Train, {num_val} Val."
    )

    total_train_steps = 0

    # Aggregate data

    global_state_stats = defaultdict(lambda: {"total_steps": 0, "count": 0})

    for i, res in enumerate(valid_results):
        is_success, fail_reason, data, steps, state_stats = res
        if is_success:
            success_count += 1
        else:
            failure_counts[str(fail_reason)] += 1

        # accumulate per-state stats
        for st, vals in state_stats.items():
            # vals is {'total_steps':..., 'count':...}
            global_state_stats[st]["total_steps"] += vals.get("total_steps", 0)
            global_state_stats[st]["count"] += vals.get("count", 0)

        # Calculate split index based on step count
        if i < num_train:
            total_train_steps += steps

        for k, v in data.items():
            dataset[k].extend(v)

    train_dataset = {}
    val_dataset = {}
    train_path = os.path.join(save_root, "train_dataset.npz")
    val_path = os.path.join(save_root, "val_dataset.npz")

    actual_total_len = len(dataset["actions"])
    split_idx = min(total_train_steps, actual_total_len)

    for k, v in dataset.items():
        arr = np.array(v)

        # Optimize types
        if (
            "observations" in k
            and arr.dtype == np.float64
            and arr.max() > 1.0
            and pixel_observations
        ):
            arr = arr.astype(np.uint8)
        elif k == "terminals":
            arr = arr.astype(bool)
        elif arr.dtype == np.float64:
            arr = arr.astype(np.float32)

        train_dataset[k] = arr[:split_idx]
        val_dataset[k] = arr[split_idx:]

    for path, dset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dset)

    total_steps_all = sum([res[3] for res in results if res[2] is not None])
    avg_steps_per_valid_episode = (
        float(total_steps_all) / num_valid if num_valid > 0 else 0.0
    )
    avg_steps_per_attempted_episode = (
        float(total_steps_all) / total_episodes_to_run
        if total_episodes_to_run > 0
        else 0.0
    )

    state_timing_summary = {}
    for st, vals in global_state_stats.items():
        cnt = vals["count"]
        tot = vals["total_steps"]
        avg = float(tot) / cnt if cnt > 0 else 0.0
        state_timing_summary[st] = {"avg_steps": avg, "total_steps": tot, "count": cnt}

    stats = {
        "total_episodes_attempted": total_episodes_to_run,
        "valid_episodes_collected": num_valid,
        "successful_episodes": success_count,
        "success_rate": (
            float(success_count) / total_episodes_to_run
            if total_episodes_to_run > 0
            else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "split": {"train": num_train, "val": num_val},
        "state_timings": state_timing_summary,
        "avg_steps_per_valid_episode": avg_steps_per_valid_episode,
        "avg_steps_per_attempted_episode": avg_steps_per_attempted_episode,
    }
    stats_path = os.path.join(save_root, "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved dataset to {save_root}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["policy", "dataset"], default="policy")
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument(
        "--save_failed_episodes",
        action="store_true",
        help="When collecting dataset, save all episodes including failed ones",
    )
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
        "--pixel_observations",
        action="store_true",
        help="Use pixel observations when collecting dataset",
    )
    parser.add_argument(
        "--pouring_prob",
        type=float,
        default=0.9,
        help="Probability of pouring at the end of moving sequence in dataset mode",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="pouring",
        choices=["pouring", "moving"],
        help="Policy type to run in policy mode",
    )
    args = parser.parse_args()

    if args.mode == "policy":
        collect_policy_episode(steps=args.steps, policy_type=args.policy_type)
    elif args.mode == "dataset":
        collect_moving_policy_dataset(
            save_root=args.out,
            episodes=args.episodes,
            max_steps=args.steps,
            width=320,
            height=240,
            noise=True,
            pixel_observations=args.pixel_observations,
            minimal_observations=args.minimal,
            save_failed_episodes=args.save_failed_episodes,
            pouring_prob=args.pouring_prob,
        )
