"""Policy functions for the kitchen environment."""

import numpy as np
import mujoco as mj
import kitchen_utils as utils


def multiply_quaternions(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def rotate_quat_around_z(base_quat, angle_rad):
    """Rotate a quaternion around the z-axis by angle_rad."""
    w = np.cos(angle_rad / 2)
    z = np.sin(angle_rad / 2)
    q_rot = np.array([w, 0.0, 0.0, z])
    return multiply_quaternions(q_rot, base_quat)


def slow_down_position(ee_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Move halfway from current position to target to slow down movement."""
    return ee_pos + 0.5 * (target_pos - ee_pos)


def slow_down_quaternion(
    current_quat: np.ndarray, target_quat: np.ndarray, factor: float = 0.5
) -> np.ndarray:
    """Interpolate between current and target quaternion to slow down rotation."""
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


def make_task_space_action(
    target_pos: np.ndarray, gripper_val: float, rot: float
) -> np.ndarray:
    """
    Build 5D task-space action [x, y, z, gripper, rot]
    with xyz and gripper normalized to [-1, 1].

    Args:
        target_pos: 3D world position in workspace bounds
        gripper_val: scalar in [0, 1] where 0=closed, 1=open
        rot: scalar in [-1, 1] representing rotation

    Returns:
        5D action array normalized to [-1, 1]
    """

    bounds_x = np.array([-1.5, 0.0])
    bounds_y = np.array([-2.5, 0.0])
    bounds_z = np.array([1.5, 3.0])

    # Normalize xyz from workspace bounds to [-1, 1]
    x_norm = 2.0 * (target_pos[0] - bounds_x[0]) / (bounds_x[1] - bounds_x[0]) - 1.0
    y_norm = 2.0 * (target_pos[1] - bounds_y[0]) / (bounds_y[1] - bounds_y[0]) - 1.0
    z_norm = 2.0 * (target_pos[2] - bounds_z[0]) / (bounds_z[1] - bounds_z[0]) - 1.0

    # Clamp to [-1, 1] to be safe
    x_norm = np.clip(x_norm, -1.0, 1.0)
    y_norm = np.clip(y_norm, -1.0, 1.0)
    z_norm = np.clip(z_norm, -1.0, 1.0)

    # Convert gripper from [0, 1] to [-1, 1]
    gripper = np.clip(float(gripper_val), 0.0, 1.0)
    gripper = 2.0 * gripper - 1.0

    action = np.array(
        [x_norm, y_norm, z_norm, gripper, rot],
        dtype=np.float32,
    )
    return action


def moving_policy(env, obs, cup_number) -> np.ndarray:
    """Policy for moving a cup from one position to another."""
    model, data = env.unwrapped.model, env.unwrapped.data

    def at_target(target_pos: np.ndarray, tol=0.04) -> bool:
        ee_pos = utils.get_effector_pos(env)
        return np.linalg.norm(target_pos - ee_pos) < tol

    # Initialize rotation parameters at episode start
    if not hasattr(env, "_policy_rot") or env._policy_rot is None:
        env._policy_rot = 0.3  # np.random.uniform(0.0, 1.0)

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

        action = make_task_space_action(
            target_pos, gripper_val=0.0, rot=env._policy_rot
        )
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

        action = make_task_space_action(
            target_pos, gripper_val=0.0, rot=env._policy_rot
        )
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
        action = make_task_space_action(
            target_pos, gripper_val=0.0, rot=env._policy_rot
        )
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

        action = make_task_space_action(
            target_pos, gripper_val=1.0, rot=env._policy_rot
        )
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
                env._cup_destination = np.array(
                    [
                        np.random.uniform(-1.0, -0.5),
                        np.random.uniform(-1.2, -0.38),
                        1.7,
                    ]
                )
                if np.linalg.norm(env._cup_destination - other_cup_pos) > 0.11:
                    break
            print("→ move_cup")

        return make_task_space_action(
            env._above_position, gripper_val=1.0, rot=env._policy_rot
        )

    elif state == "move_cup":
        # Change rot_z when entering move_cup state
        if env._state_counter == 0:
            env._policy_rot_z = np.random.uniform(0.0, 1.0)
            env._state_counter += 1
        else:
            env._state_counter += 1

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
            env._state_counter = 0
            print("→ place_cup")

        action = make_task_space_action(
            target_pos, gripper_val=1.0, rot=env._policy_rot
        )
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "place_cup":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.01
        if at_target(target_pos, tol=0.04):
            env._automaton_state = "open_gripper"
            print("→ open_gripper")

        action = make_task_space_action(
            target_pos, gripper_val=1.0, rot=env._policy_rot
        )
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "open_gripper":
        target_pos = env._cup_destination.copy()
        env._state_counter += 1
        if env._state_counter > 20:
            env._automaton_state = "move_up_after_release"
        action = make_task_space_action(
            target_pos, gripper_val=0.0, rot=env._policy_rot
        )
        action[:3] += env._noise_generator.sample()
        return action

    elif state == "move_up_after_release":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.35  # move up
        if at_target(target_pos, tol=0.4):
            env._automaton_state = "done"
        action = make_task_space_action(
            target_pos, gripper_val=0.0, rot=env._policy_rot
        )
        action[:3] += env._noise_generator.sample()
        return action


def pour_policy_v2(env, obs) -> np.ndarray:
    """Policy for pouring water from one cup to another."""
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

        action = make_task_space_action(target_pos, gripper_val=0.0)
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

        action = make_task_space_action(target_pos, gripper_val=0.0)
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
        action = make_task_space_action(target_pos, gripper_val=0.0)
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

        action = make_task_space_action(target_pos, gripper_val=1.0)
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

        action = make_task_space_action(env._above_position, gripper_val=1.0)
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
            action = make_task_space_action(target_pos, gripper_val=1.0)
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

        action = make_task_space_action(target_pos, gripper_val=1.0)
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
            action = make_task_space_action(target_pos, gripper_val=1.0)
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

        action = make_task_space_action(actual_pos, gripper_val=1.0)
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

        action = make_task_space_action(actual_pos, env, gripper_val=1.0)
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

        action = make_task_space_action(actual_pos, env, gripper_val=1.0)
        action[:3] += env._noise_generator.sample()
        return action

    # Final pour
    elif state == "pour":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([-0.01, -0.02, 0.21])
        target_quat = [0.12278783, -0.69636423, 0.69636423, 0.12278783]
        target_quat = rotate_quat_around_z(target_quat, env._quat_offset)
        action = make_task_space_action(target_pos, gripper_val=1.0)
        action[:3] += env._noise_generator.sample()
        return action
