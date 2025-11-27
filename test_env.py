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

EXAMPLE_GOAL_OBS = [
    1.0328614711761475,
    -0.30165818333625793,
    2.068722724914551,
    -1.1879557371139526,
    1.174445629119873,
    -2.5215463638305664,
    -1.2182813882827759,
    0.01867654360830784,
    0.018937133252620697,
    -0.032288286834955215,
    0.0011035713832825422,
    0.08603318780660629,
    0.007812343537807465,
    0.003646769095212221,
    -0.17719876766204834,
    0.048163507133722305,
    0.00011716206063283607,
    -0.00012100646563339978,
    -0.5274343490600586,
    -0.7672080993652344,
    1.5888869762420654,
    0.9999953508377075,
    -0.00210916087962687,
    -0.00039728923002257943,
    -0.0021729320287704468,
    -0.5275718569755554,
    -0.878852128982544,
    1.7291043996810913,
    0.5998542308807373,
    -0.797810971736908,
    -0.018524667248129845,
    -0.05770083889365196,
    -0.5224093794822693,
    -0.751502275466919,
    1.6127880811691284,
    0.3430387079715729,
    0.36703041195869446,
    0.845798909664154,
    -0.17954768240451813,
    -0.5118367671966553,
    -0.7653200030326843,
    1.6126577854156494,
    0.5904989242553711,
    0.1474355161190033,
    0.043138161301612854,
    -0.7922833561897278,
    -0.5415235161781311,
    -0.770224392414093,
    1.6487640142440796,
    -0.35912182927131653,
    -0.007813514210283756,
    0.9263462424278259,
    -0.11337151378393173,
    -0.5367369651794434,
    -0.8497213125228882,
    1.7034391164779663,
    -0.41515588760375977,
    -0.5325943827629089,
    0.07069452852010727,
    0.7341601252555847,
    -0.542583703994751,
    -0.7583289742469788,
    1.612152099609375,
    -0.2574683725833893,
    -0.3054730296134949,
    -0.7503960132598877,
    0.5265948176383972,
    -0.5267834663391113,
    -0.7943599820137024,
    1.6693847179412842,
    -0.8120731115341187,
    -0.2888404130935669,
    -0.038021281361579895,
    -0.505631148815155,
    -0.5166738629341125,
    -0.7741424441337585,
    1.6629692316055298,
    0.44979962706565857,
    -0.32265597581863403,
    -0.5842329263687134,
    0.5935025811195374,
    -0.5313515663146973,
    -0.7558802366256714,
    1.6114898920059204,
    -0.7609355449676514,
    -0.042618799954652786,
    -0.6440088748931885,
    0.0664323940873146,
    -0.5260841846466064,
    -0.7984523773193359,
    1.6823945045471191,
    -0.6458081603050232,
    0.0942404717206955,
    -0.5352796912193298,
    0.5362147092819214,
    -0.5410668849945068,
    -0.7524141669273376,
    1.62000572681427,
    0.5761745572090149,
    -0.6101201772689819,
    0.33950120210647583,
    -0.4248708188533783,
    5.899002189835301e-06,
    0.00033008470200002193,
    0.004041054751724005,
    0.03244360163807869,
    -9.257549208996352e-06,
    0.005597819108515978,
    -0.009618513286113739,
    -0.02250034362077713,
    0.039265070110559464,
    -0.24392347037792206,
    -0.017604783177375793,
    -0.020544299855828285,
    0.008919095620512962,
    -0.008270672522485256,
    -0.002901132218539715,
    -1.1456631422042847,
    1.5197761058807373,
    -0.5358951091766357,
    0.013860710896551609,
    -0.0078430762514472,
    0.0052484869956970215,
    -3.0350613594055176,
    0.39663293957710266,
    -0.6428488492965698,
    -0.12563277781009674,
    0.2601436674594879,
    -0.626965343952179,
    44.535118103027344,
    -18.44878578186035,
    31.34378433227539,
    -0.004172059241682291,
    0.05475807562470436,
    0.007099295500665903,
    2.093017578125,
    -9.692337036132812,
    12.771499633789062,
    0.04179612919688225,
    -0.041525475680828094,
    -0.029600298032164574,
    -1.9690190553665161,
    7.921738147735596,
    -6.925976753234863,
    -0.03854032978415489,
    -0.061898790299892426,
    0.013281450606882572,
    -2.2246103286743164,
    -11.242135047912598,
    3.7153263092041016,
    0.18993382155895233,
    0.38652703166007996,
    -0.33690378069877625,
    62.28213119506836,
    24.436439514160156,
    49.95360565185547,
    -0.08947636932134628,
    0.13717946410179138,
    0.07206504791975021,
    -4.010060787200928,
    -23.729175567626953,
    -26.242204666137695,
    0.033539023250341415,
    0.25332507491111755,
    -0.41937538981437683,
    8.287856101989746,
    -25.798303604125977,
    -50.894325256347656,
    0.1195303425192833,
    0.476348876953125,
    -0.2660529315471649,
    -83.33857727050781,
    -56.04866409301758,
    31.012163162231445,
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
        cup_pos = utils.get_object_pos(env, (f"cup_freejoint{cup_number}", f"cup{cup_number}"))
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
        cup_pos = utils.get_object_pos(env, (f"cup_freejoint{cup_number}", f"cup{cup_number}"))
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
        cup_pos = utils.get_object_pos(env, (f"cup_freejoint{cup_number}", f"cup{cup_number}"))
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
        cup_pos = utils.get_object_pos(env, (f"cup_freejoint{cup_number}", f"cup{cup_number}"))
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
        ):
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
            < 0.02
        ):
            env._automaton_state = "move_cup"
            other_cup_id = 1 - cup_number
            other_cup_pos = utils.get_object_pos(env, (f"cup_freejoint{other_cup_id}", f"cup{other_cup_id}"))
            while True:
                env._cup_destination = np.array([-0.8, -1.1, 1.6]) + np.random.uniform(-0.1, 0.1, 3)
                if np.linalg.norm(env._cup_destination - other_cup_pos) > 0.2:
                    break
            print("→ move_cup")

        return make_action(q_target, close=True)

    elif state == "move_cup":
        target_pos = env._cup_destination
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )

        if at_target(target_pos, tol=0.05):
            env._automaton_state = "place_cup"
            print("→ place_cup")

        return make_action(q_target, close=True)

    elif state == "place_cup":
        target_pos = env._cup_destination.copy()
        target_pos[2] -= 0.15 # move down
        target_quat = [0.61237244, -0.35355338, 0.35355338, 0.61237244]

        q_target = utils.ik_solve_dm(
            model,
            data,
            "grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            inplace=False,
        )
        if at_target(target_pos, tol=0.05):
            env._automaton_state = "open_gripper"
            print("→ open_gripper")

        return make_action(q_target, close=True)

    elif state == "open_gripper":
        target_pos = env._cup_destination.copy()
        target_pos[2] -= 0.15 # keep it down
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
        env._state_counter +=1
        if env._state_counter > 50:
            env._automaton_state = "move_up_after_release"
        return action

    elif state == "move_up_after_release":
        target_pos = env._cup_destination.copy()
        target_pos[2] += 0.15 # move up
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
        if at_target(target_pos, tol=0.05):
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
    save_path="tmp/policy.mp4", steps=1000, noise=True, random_action=False
):
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=320, height=240
    )
    obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
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
            action = action + np.random.normal(0, 0.02, action.shape)
        obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
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

    OG_IMPLS_BASE = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench"))
    sys.path.insert(0, OG_IMPLS_BASE)
    from ogbench import load_dataset

    def normalize(x, mean, std, eps=1e-5):
        return (x - mean) / (std + eps)

    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
    frames = []

    train_path = os.path.join(
        "/u/maximilian.kannen/work/new_kitchen/tmp/normalised_noise",
        "train_dataset.npz",
    )

    train_dataset_raw = load_dataset(train_path, compact_dataset=True)

    obs_data = train_dataset_raw["observations"]
    obs_mean = np.mean(obs_data, axis=0)
    obs_std = np.std(obs_data, axis=0)

    cfg = get_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["alpha"] = 0.03

    agent_tmp = CRLAgent.create(
        seed=0, ex_observations=obs, ex_actions=env.action_space.sample(), config=cfg
    )

    agent = restore_agent(agent_tmp, checkpoint_path, 60000)
    print(f"Loaded checkpoint from {checkpoint_path}")
    obs_arr = np.asarray(obs)
    goal_arr = env.unwrapped.create_goal_state(
        current_state=obs_arr, minimal=True, fixed_goal=True
    )
    normalized_goal = normalize(goal_arr, obs_mean, obs_std)
    for t in range(steps):
        normalized_obs = normalize(obs_arr, obs_mean, obs_std)

        action = agent.sample_actions(
            observations=normalized_obs,
            goals=normalized_goal,
            temperature=0.0,
            seed=jax.random.PRNGKey(0),
        )
        action = np.clip(action, -1.0, 1.0)
        obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
        obs_arr = np.asarray(obs)
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
    max_steps: int = 1600,
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
        ep_dir = os.path.join(save_root, f"episode_{ep_idx:03d}")
        images_dir = os.path.join(ep_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

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
                    final_frame = env.render()
                    final_image_path = os.path.join(images_dir, f"step_{t:03d}.png")
                    imageio.imwrite(final_image_path, final_frame)
                    print(f"Saved final frame to {final_image_path}")

                if save_failed_episodes:
                    print(
                        f"Episode {ep_idx} failed but saved due to save_failed_episodes=True."
                    )

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


def collect_moving_policy_dataset(
    save_root: str = "tmp/policy_dataset",
    episodes: int = 100,
    max_steps: int = 1600,
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
        ep_dir = os.path.join(save_root, f"episode_{ep_idx:03d}")
        images_dir = os.path.join(ep_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        episode_terminated = False

        steps_in_current_episode = 0

        move_operations = np.random.randint(0, 3)
        moves_completed = 0
        policy_mode = "moving" if move_operations > 0 else "pouring"

        for t in range(max_steps):
            action = None
            if policy_mode == "moving":
                action = moving_policy(env, obs, 1)
                if env._automaton_state == "done":
                    moves_completed += 1
                    if moves_completed == move_operations:
                        policy_mode = "pouring"
                    env._automaton_state = "move_above"
            elif policy_mode == "pouring":
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
                    final_frame = env.render()
                    final_image_path = os.path.join(images_dir, f"step_{t:03d}.png")
                    imageio.imwrite(final_image_path, final_frame)
                    print(f"Saved final frame to {final_image_path}")

                if save_failed_episodes:
                    print(
                        f"Episode {ep_idx} failed but saved due to save_failed_episodes=True."
                    )

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
        "--mode", choices=["random", "policy", "dataset", "crl"], default="random"
    )
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=1300)
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
    args = parser.parse_args()

    if args.mode == "random":
        random_action_test(args.out, steps=args.steps)
    elif args.mode == "policy":
        collect_policy_episode(steps=args.steps)
    elif args.mode == "dataset":
        # Use --out as a directory for the dataset
        save_root = args.out
        if args.pixel_observations:

            collect_policy_dataset(
                save_root=save_root,
                episodes=args.episodes,
                max_steps=args.steps,
                minimal_observations=args.minimal,
                save_failed_episodes=args.save_failed_episodes,
                pixel_observations=True,
            )
        else:
            collect_policy_dataset(
                save_root=save_root,
                episodes=args.episodes,
                max_steps=args.steps,
                minimal_observations=args.minimal,
                save_failed_episodes=args.save_failed_episodes,
            )
    elif args.mode == "crl":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when using --mode=crl")
        collect_crl_episode(
            save_path=args.out, steps=args.steps, checkpoint_path=args.checkpoint
        )
