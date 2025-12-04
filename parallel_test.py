import os
import sys
import json
from xml.parsers.expat import model
import imageio
import pickle
from tqdm import trange
import numpy as np
from collections import defaultdict
import gymnasium as gym
import jax
import mujoco as mj
import kitchen_utils as utils
from collections import defaultdict
from joblib import Parallel, delayed


MOVING_GOAL_OBS = [
    2.159642457962036,
    -0.10995837301015854,
    1.5188195705413818,
    -1.471531629562378,
    1.076535701751709,
    -1.3156180381774902,
    -0.697909951210022,
    0.009322389028966427,
    0.009459411725401878,
    0.07799883186817169,
    0.2097632884979248,
    0.24370022118091583,
    -0.2910488545894623,
    0.14127174019813538,
    0.1319103240966797,
    0.25795871019363403,
    -0.05900793522596359,
    -0.059946853667497635,
    -0.5267668962478638,
    -0.9562153816223145,
    1.5801501274108887,
    0.9984312057495117,
    -0.03856143355369568,
    0.03994974493980408,
    -0.007226120680570602,
    -0.8000472187995911,
    -1.1000237464904785,
    1.5889480113983154,
    0.9999995231628418,
    8.086483285296708e-05,
    -0.0009184352238662541,
    0.00028437477885745466,
    -0.8119636178016663,
    -1.1021802425384521,
    1.6126853227615356,
    0.7789384722709656,
    0.6046792268753052,
    -0.013668391853570938,
    -0.16562329232692719,
    -0.7986109256744385,
    -1.1152387857437134,
    1.6127070188522339,
    0.5592268705368042,
    0.5389202833175659,
    -0.627842366695404,
    -0.05142202600836754,
    -0.7848583459854126,
    -1.101852297782898,
    1.6127344369888306,
    0.5282660126686096,
    0.4182727038860321,
    0.7361798286437988,
    0.0634213536977768,
    -0.8054980039596558,
    -1.0851140022277832,
    1.6127002239227295,
    0.4111011028289795,
    -0.5848034620285034,
    -0.6529611349105835,
    0.25028496980667114,
    -0.8140132427215576,
    -1.091016411781311,
    1.6126837730407715,
    0.9921442270278931,
    -0.10417235642671585,
    0.0673166811466217,
    0.016321102157235146,
    -0.8085049986839294,
    -1.1137375831604004,
    1.61268949508667,
    -0.38063672184944153,
    0.7345278263092041,
    -0.39539316296577454,
    -0.39905986189842224,
    -0.7901497483253479,
    -1.0851787328720093,
    1.6127279996871948,
    -0.14813897013664246,
    -0.9648413062095642,
    0.21688757836818695,
    -0.00979399774223566,
    -0.7937706112861633,
    -1.1064852476119995,
    1.6127175092697144,
    -0.03492381051182747,
    0.7362642288208008,
    0.599088191986084,
    -0.3127117156982422,
    -0.7922301292419434,
    -1.0949842929840088,
    1.612722396850586,
    0.7833766937255859,
    -0.35952121019363403,
    0.5008659958839417,
    -0.0787319466471672,
    -0.7862828373908997,
    -1.1147964000701904,
    1.6127294301986694,
    0.9410353899002075,
    0.23408016562461853,
    0.24400077760219574,
    0.011068851687014103,
    0.016990942880511284,
    0.018744247034192085,
    0.08126810193061829,
    1.2619215250015259,
    -1.1199562549591064,
    0.007560721132904291,
    8.360292122233659e-05,
    -0.0003339182585477829,
    0.0005906213191337883,
    -0.023027963936328888,
    -0.010386912152171135,
    0.0009183932561427355,
]


def moving_policy(env, obs, cup_number) -> np.ndarray:
    model, data = env.unwrapped.model, env.unwrapped.data

    def at_target(target_pos: np.ndarray, tol=0.04) -> bool:
        ee_pos = utils.get_effector_pos(env)
        return np.linalg.norm(target_pos - ee_pos) < tol

    def make_action(q_target: np.ndarray, close: bool) -> np.ndarray:
        model = env.unwrapped.model
        nu = env.unwrapped.nu
        ctrl_range = model.actuator_ctrlrange[:nu]

        low = ctrl_range[:, 0]
        high = ctrl_range[:, 1]

        arm = q_target[:7]

        arm_norm = 2.0 * (arm - low[:7]) / (high[:7] - low[:7]) - 1.0

        grip = utils.make_gripper_action(env, close=close, open_val=-1.0, close_val=1.0)
        grip_norm = grip[:nu]

        action = np.zeros(nu, dtype=np.float32)
        action[:7] = arm_norm
        action += grip_norm

        return action[:9]

    state = env._automaton_state

    if state == "move_above":
        cup_pos = utils.get_object_pos(
            env, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
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

    if state == "move_towards":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(
            env, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
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

    elif state == "move_down":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(
            env, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
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

    elif state == "close_gripper":
        env._state_counter += 1
        cup_pos = utils.get_object_pos(
            env, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
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

        return action

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
                        np.random.uniform(-0.93, -0.45),
                        np.random.uniform(-1.1, -0.4),
                        1.71,
                    ]
                )
                if np.linalg.norm(env._cup_destination - other_cup_pos) > 0.1:
                    break
            print("→ move_cup")

        return make_action(q_target, close=True)

    elif state == "move_cup":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.15  # move above place position
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
            at_target(target_pos, tol=0.09)
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.002
        ):
            env._automaton_state = "place_cup"
            print("→ place_cup")

        return make_action(q_target, close=True)

    elif state == "place_cup":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.02
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )
        if at_target(target_pos, tol=0.04):
            env._automaton_state = "open_gripper"
            print("→ open_gripper")

        return make_action(q_target, close=True)

    elif state == "open_gripper":
        target_pos = env._cup_destination.copy()
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )

        action = make_action(q_target, close=False)
        env._state_counter += 1
        if env._state_counter > 20:
            env._automaton_state = "move_up_after_release"
        return action

    elif state == "move_up_after_release":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.35  # move up
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )

        action = make_action(q_target, close=False)
        if at_target(target_pos, tol=0.6):
            env._automaton_state = "done"
        return action


def pour_policy_v2(env, obs) -> np.ndarray:

    model, data = env.unwrapped.model, env.unwrapped.data

    def at_target(target_pos: np.ndarray, tol=0.04) -> bool:
        ee_pos = utils.get_effector_pos(env)
        return np.linalg.norm(target_pos - ee_pos) < tol

    def make_action(q_target: np.ndarray, close: bool) -> np.ndarray:

        model = env.unwrapped.model
        nu = env.unwrapped.nu
        ctrl_range = model.actuator_ctrlrange[:nu]

        low = ctrl_range[:, 0]
        high = ctrl_range[:, 1]

        arm = q_target[:7]

        arm_norm = 2.0 * (arm - low[:7]) / (high[:7] - low[:7]) - 1.0

        grip = utils.make_gripper_action(env, close=close, open_val=-1.0, close_val=1.0)
        grip_norm = grip[:nu]

        action = np.zeros(nu, dtype=np.float32)
        action[:7] = arm_norm
        action += grip_norm

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
    pouring_prob
):
    """
    Worker function that creates its own environment and runs one episode.
    """
    # Re-import inside worker to ensure clean state
    import gymnasium as gym
    import numpy as np
    
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
    obs, _ = env.reset(seed=seed, options={"randomise_cup_position": True, "minimal": True})
    
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
    done2 = False
    cup = np.random.choice(np.array([0, 1]))
    
    steps_run = 0

    for t in range(max_steps):
        action = None
        # --- POLICY LOGIC ---
        if policy_mode == "moving":
            # Note: moving_policy is defined in global scope, which is fine for joblib
            action = moving_policy(env, obs, cup_number=cup)
            if env._automaton_state == "done":
                moves_completed += 1
                if moves_completed == move_operations:
                    if perform_pouring:
                        policy_mode = "pouring"
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
        if random_action:
            if np.random.rand() < 0.01:
                action = env.action_space.sample()
                
        obs_to_store = env.unwrapped._get_observation(minimal=True)
        obs_next, reward, terminated, truncated, info = env.unwrapped.step(
            action, minimal=True
        )
        done = terminated or truncated or done2

        if minimal_observations:
            episode_data["observations"].append(obs_to_store)
        else:
            episode_data["observations"].append(obs)
            
        episode_data["actions"].append(action)
        episode_data["terminals"].append(done)
        episode_data["qpos"].append(env.unwrapped.data.qpos.copy())
        episode_data["qvel"].append(env.unwrapped.data.qvel.copy())

        obs = obs_next
        steps_run += 1

        if done:
            success = True
            failure_reason = None
            break
            
        if t == max_steps - 1:
            failure_reason = getattr(env, "_automaton_state", "unknown")

    env.close()
    
    # Return structure: (success, failure_reason, data_dict, steps_count)
    if success or save_failed_episodes:
        if not success and save_failed_episodes:
            # Mark last terminal as True if forcing save
            episode_data["terminals"][-1] = True
        return (success, failure_reason, episode_data, steps_run)
    else:
        return (success, failure_reason, None, steps_run)

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
    os.makedirs(save_root, exist_ok=True)
    
    total_episodes_to_run = episodes + (episodes // 10)
    
    print(f"Starting parallel collection of {total_episodes_to_run} episodes...")
    print(f"To use GPU for rendering, ensure 'export MUJOCO_GL=egl' is set.")


    results = Parallel(n_jobs=20, verbose=10)(
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
            pouring_prob=pouring_prob
        ) 
        for i in range(total_episodes_to_run)
    )

    # --- Aggregation Logic ---
    dataset = defaultdict(list)
    success_count = 0
    failure_counts = defaultdict(int)
    total_steps = 0
    total_train_steps = 0
    
    # Filter out None results (failed episodes that weren't saved)
    valid_results = [r for r in results if r[2] is not None]
    
    # Count stats
    for res in results:
        is_success, fail_reason, _, steps = res
        if is_success:
            success_count += 1
        else:
            failure_counts[str(fail_reason)] += 1

    # Flatten the list of lists into the main dataset
    # We need to be careful to maintain the 'episode' structure if required, 
    # but usually standard RL datasets are just concatenated arrays.
    
    # If you need to distinguish episodes, you might need to add an 'timeouts' or 'episode_terminals' key.
    # Here we perform simple concatenation as per your original script.
    
    num_train_episodes = episodes
    train_episode_count = 0
    
    print("Aggregating data...")
    for i, (_, _, data, steps) in enumerate(valid_results):
        if data is None: continue
        
        total_steps += steps
        is_train = train_episode_count < num_train_episodes
        
        if is_train:
            total_train_steps += steps
            train_episode_count += 1

        for k, v in data.items():
            dataset[k].extend(v)

    # --- Saving Logic (same as original) ---
    train_dataset = {}
    val_dataset = {}
    train_path = os.path.join(save_root, "train_dataset.npz")
    val_path = os.path.join(save_root, "val_dataset.npz")

    actual_total_len = len(dataset["actions"])
    # Determine split index based on total steps calculated above
    split_idx = min(total_train_steps, actual_total_len)

    for k, v in dataset.items():
        # Convert list to numpy array
        arr = np.array(v)
        
        if "observations" in k and arr.dtype == np.float64 and arr.max() > 1.0 and pixel_observations:
             arr = arr.astype(np.uint8) # Optimization for pixels
        elif k == "terminals":
            arr = arr.astype(bool)
        elif arr.dtype == np.float64:
            arr = arr.astype(np.float32)

        train_dataset[k] = arr[:split_idx]
        val_dataset[k] = arr[split_idx:]

    for path, dset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dset)

    stats = {
        "total_episodes_attempted": total_episodes_to_run,
        "successful_episodes": success_count,
        "success_rate": float(success_count) / total_episodes_to_run if total_episodes_to_run > 0 else 0.0,
        "failure_counts": dict(failure_counts),
        "saved_failed_episodes": save_failed_episodes,
    }
    stats_path = os.path.join(save_root, "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved dataset to {save_root}")

def collect_policy_dataset(
    save_root: str = "tmp/policy_dataset",
    episodes: int = 100,
    max_steps: int = 1100,
    width: int = 320,
    height: int = 240,
    noise: bool = True,
    pixel_observations: bool = False,
    random_action: bool = False,
    minimal_observations: bool = True,
    save_failed_episodes: bool = False,
):
    """Run the policy multiple times and save trajectories.

    Args:
        save_failed_episodes: If True, saves all episodes. If False, only saves
                              successful (terminated==True) episodes.
    """
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

        episode_terminated = False

        steps_in_current_episode = 0

        for t in range(max_steps):
            action = pour_policy_v2(env, obs)
            if noise:
                action = action + np.random.normal(0, 0.02, action.shape)
            if random_action:
                if np.random.rand() < 0.01:
                    action = env.action_space.sample()
            obs_to_store = env.unwrapped._get_observation(minimal=True)
            obs_next, reward, terminated, truncated, info = env.unwrapped.step(
                action, minimal=True
            )
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
            steps_in_current_episode += 1

            if done:
                episode_terminated = True
                total_steps += steps_in_current_episode
                if ep_idx < num_train_episodes:
                    total_train_steps += steps_in_current_episode

                # Save last correct dataset entries for debugging consistency
                for k in dataset.keys():
                    debug_data[k].append(dataset[k][-1])
                break

            elif t == max_steps - 1:
                if env._automaton_state == "pour":
                    Goal, Start = env.unwrapped.get_particles_in_cups()
                    print(f"Goal position at max_steps: {Goal}")
                    print(f"Start position at max_steps: {Start}")

                if save_failed_episodes:
                    print(
                        f"Episode {ep_idx} failed but saved due to save_failed_episodes=True."
                    )
                    dataset["terminals"][-1] = True
                    total_steps += steps_in_current_episode
                    if ep_idx < num_train_episodes:
                        total_train_steps += steps_in_current_episode

                    for k in dataset.keys():
                        debug_data[k].append(dataset[k][-1])
                else:
                    print(
                        f"Episode {ep_idx} reached max_steps ({max_steps}) without termination; not saved."
                    )
                    for k in dataset.keys():
                        dataset[k] = dataset[k][:-max_steps]

                        if len(dataset[k]) > 0:
                            assert np.array_equal(
                                debug_data[k][-1], dataset[k][-1]
                            ), f"Data mismatch in key {k} at episode {ep_idx}, step {t}"
                break

        if episode_terminated:
            success_count += 1
        else:
            final_state = getattr(env, "_automaton_state", None)
            failure_counts[str(final_state)] += 1

    env.close()

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    train_path = os.path.join(save_root, "train_dataset.npz")
    val_path = os.path.join(save_root, "val_dataset.npz")

    actual_total_len = len(dataset["actions"])
    split_idx = min(total_train_steps, actual_total_len)

    for k, v in dataset.items():
        if "observations" in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == "terminals":
            dtype = bool
        elif k == "button_states":
            dtype = np.int64
        else:
            dtype = np.float32

        train_dataset[k] = np.array(v[:split_idx], dtype=dtype)
        val_dataset[k] = np.array(v[split_idx:], dtype=dtype)

    for path, dset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dset)

    stats = {
        "total_episodes_attempted": episodes + (episodes // 10),
        "successful_episodes": success_count,
        "success_rate": (
            float(success_count) / float(episodes + (episodes // 10))
            if episodes > 0
            else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "saved_failed_episodes": save_failed_episodes,
    }
    stats_path = os.path.join(save_root, "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved dataset to {save_root}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["policy", "dataset", "crl"], default="policy"
    )
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=1900)
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
        "--checkpoint",
        type=str,
        help="Path to CRL agent checkpoint file for crl mode",
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
        # Use --out as a directory for the dataset
        save_root = args.out
        if args.pixel_observations:

            collect_moving_policy_dataset(
                save_root=save_root,
                episodes=args.episodes,
                max_steps=args.steps,
                minimal_observations=args.minimal,
                save_failed_episodes=args.save_failed_episodes,
                pixel_observations=True,
                pouring_prob=args.pouring_prob,
            )
        else:
            collect_moving_policy_dataset(
                save_root=save_root,
                episodes=args.episodes,
                max_steps=args.steps,
                minimal_observations=args.minimal,
                save_failed_episodes=args.save_failed_episodes,
                pouring_prob=args.pouring_prob,
            )
    elif args.mode == "crl":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when using --mode=crl")
        collect_crl_episode(
            save_path=args.out, steps=args.steps, checkpoint_path=args.checkpoint
        )
