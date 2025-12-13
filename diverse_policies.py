import numpy as np
import mujoco as mj
import kitchen_utils as utils


# --- Math Helpers ---
def get_quaternion_from_euler(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def rotate_quat_around_z(base_quat, angle_rad):
    w = np.cos(angle_rad / 2)
    z = np.sin(angle_rad / 2)
    q_rot = np.array([w, 0.0, 0.0, z])
    return multiply_quaternions(q_rot, base_quat)


class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.005):  # Reduced sigma slightly
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


# --- Geometric Checks ---
def is_cup_grasped(env, cup_id: int, tol=0.06) -> bool:
    """
    Robustly check if the cup is actually inside the gripper.
    """
    cup_pos = utils.get_object_pos(env, (f"cup_freejoint{cup_id}", f"cup{cup_id}"))
    cup_pos += np.array([0.0, 0.0, 0.004])

    l_id = utils._try_name_lookup(env, "left_finger_tip", "site")
    r_id = utils._try_name_lookup(env, "right_finger_tip", "site")

    data = env.unwrapped.data

    if l_id != -1 and r_id != -1:
        l_pos = data.site_xpos[l_id]
        r_pos = data.site_xpos[r_id]
    else:
        l_geom = utils._try_name_lookup(env, "leftclaw_it0", "geom")
        r_geom = utils._try_name_lookup(env, "rightclaw_it", "geom")
        if l_geom != -1 and r_geom != -1:
            l_pos = data.geom_xpos[l_geom]
            r_pos = data.geom_xpos[r_geom]
        else:
            ee_pos = utils.get_effector_pos(env)
            return np.linalg.norm(ee_pos - cup_pos) < 0.06

    finger_vec = r_pos - l_pos
    finger_len = np.linalg.norm(finger_vec)

    if finger_len < 1e-4:
        return False

    finger_dir = finger_vec / finger_len
    cup_vec = cup_pos - l_pos

    proj = np.dot(cup_vec, finger_dir)
    perp_dist = np.linalg.norm(cup_vec - proj * finger_dir)

    # Relaxed check: just ensure cup is roughly between fingers
    is_between = -0.02 < proj < (finger_len + 0.02)
    return is_between and (perp_dist < tol)


# --- Main Policy Class ---


class DiversePolicy:
    def __init__(self, mode="moving", noise_scale=0.005):
        self.mode = mode
        self.noise_generator = OUNoise(size=3, sigma=noise_scale)
        self.base_grasp_quat = np.array(
            [0.61237244, -0.35355338, 0.35355338, 0.61237244]
        )
        self.reset_internal()

    def reset_internal(self):
        self.state = "init"
        self.sub_step = 0
        self.target_pos_cache = None
        self.target_quat_cache = None
        self.cup_id = 0
        self.trajectory_t = 0.0
        self.trajectory_start = None
        self.trajectory_end = None
        self.control_point = None

        # Reduced yaw to +/- 25 degrees to be safer for pouring alignment
        self.grasp_yaw = np.random.uniform(-np.pi / 7.0, np.pi / 7.0)
        self.grasp_height_offset = np.random.uniform(0.0, 0.003)
        self.noise_generator.reset()

    def _update_trajectory(self, start_pos, end_pos):
        self.trajectory_start = start_pos
        self.trajectory_end = end_pos
        self.trajectory_t = 0.0
        midpoint = (start_pos + end_pos) / 2
        offset = np.random.uniform(-0.1, 0.1, size=3)
        offset[2] = np.abs(offset[2]) + 0.12
        self.control_point = midpoint + offset

    def _get_bezier_point(self, t):
        p0 = self.trajectory_start
        p1 = self.control_point
        p2 = self.trajectory_end
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

    def _make_action(self, env, q_target, close_gripper):
        model = env.unwrapped.model
        nu = env.unwrapped.nu
        ctrl_range = model.actuator_ctrlrange[:nu]
        low, high = ctrl_range[:, 0], ctrl_range[:, 1]
        arm = q_target[:7]
        arm_norm = 2.0 * (arm - low[:7]) / (high[:7] - low[:7]) - 1.0
        grip = utils.make_gripper_action(
            env, close=close_gripper, open_val=-1.0, close_val=1.0
        )
        action = np.zeros(nu, dtype=np.float32)
        action[:7] = arm_norm
        action += grip[:nu]
        return action[:9]

    def _at_target(self, env, target, tol=0.04):
        ee_pos = utils.get_effector_pos(env)
        return np.linalg.norm(target - ee_pos) < tol

    def get_action(self, env, obs, cup_id=None) -> np.ndarray:
        env._automaton_state = self.state

        if self.state == "init":
            self.state = "approach_pre"
            self.sub_step = 0
            self.cup_id = cup_id if cup_id is not None else np.random.choice([0, 1])

        model, data = env.unwrapped.model, env.unwrapped.data
        ee_pos = utils.get_effector_pos(env)

        # --- STATE MACHINE ---

        if self.state == "approach_pre":
            cup_pos = utils.get_object_pos(
                env, (f"cup_freejoint{self.cup_id}", f"cup{self.cup_id}")
            )
            radius = 0.05
            dx = radius * np.cos(self.grasp_yaw)
            dy = radius * np.sin(self.grasp_yaw)
            target_pos = cup_pos + np.array([-0.015 - dx, -dy, 0.25])
            target_quat = rotate_quat_around_z(self.base_grasp_quat, -self.grasp_yaw)

            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=target_pos,
                target_quat=target_quat,
                inplace=False,
            )
            if (
                self._at_target(env, target_pos, tol=0.1)
                and np.linalg.norm(data.qvel[:7]) < 0.8
            ):
                self.state = "approach_descend"
                self.sub_step = 0
                self.target_quat_cache = target_quat
            return self._make_action(env, q_target, close_gripper=False)

        elif self.state == "approach_descend":
            self.sub_step += 1
            cup_pos = utils.get_object_pos(
                env, (f"cup_freejoint{self.cup_id}", f"cup{self.cup_id}")
            )

            dx = 0.01 * np.cos(self.grasp_yaw)
            dy = 0.01 * np.sin(self.grasp_yaw)

            # Go slightly deeper (0.07 instead of 0.075) to ensure grip
            target_pos = cup_pos + np.array(
                [-dx, -dy, 0.070 + self.grasp_height_offset]
            )

            target_pos += self.noise_generator.sample()

            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=target_pos,
                target_quat=self.target_quat_cache,
                inplace=False,
            )

            dz = abs(ee_pos[2] - target_pos[2])
            dist_xy = np.linalg.norm(ee_pos[:2] - target_pos[:2])

            # Check convergence
            if (dz < 0.01 and dist_xy < 0.02) or self.sub_step > 120:
                self.state = "grasp"
                self.sub_step = 0

            return self._make_action(env, q_target, close_gripper=False)

        elif self.state == "grasp":
            self.sub_step += 1
            cup_pos = utils.get_object_pos(
                env, (f"cup_freejoint{self.cup_id}", f"cup{self.cup_id}")
            )
            dx = 0.01 * np.cos(self.grasp_yaw)
            dy = 0.01 * np.sin(self.grasp_yaw)
            target_pos = cup_pos + np.array(
                [-dx, -dy, 0.070 + self.grasp_height_offset]
            )

            # NO NOISE HERE: We want a stable grip
            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=target_pos,
                target_quat=self.target_quat_cache,
                inplace=False,
            )
            action = self._make_action(env, q_target, close_gripper=True)

            if self.sub_step > 30:  # Wait for fingers to close
                grasped = is_cup_grasped(env, self.cup_id, tol=0.05)  # Relaxed check
                if grasped or self.sub_step > 60:
                    if not grasped:
                        # If we haven't grasped it by step 60, we likely missed.
                        # But we proceed anyway to avoid infinite loops, though stats will show failure later.
                        pass
                    self.state = "lift"
                    self.sub_step = 0
                    lift_target = ee_pos + np.array([0, 0, 0.25])
                    self._update_trajectory(ee_pos, lift_target)

            return action

        elif self.state == "lift":
            # FASTER LIFT: increment 0.05 instead of 0.03
            self.trajectory_t = min(1.0, self.trajectory_t + 0.05)
            target_pos = self._get_bezier_point(self.trajectory_t)

            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=target_pos,
                target_quat=self.target_quat_cache,
                inplace=False,
            )

            # Loose tolerance (0.1)
            if self.trajectory_t >= 1.0 and self._at_target(env, target_pos, 0.1):
                if self.mode == "moving":
                    self.state = "move_target_setup"
                else:
                    self.state = "pour_setup"
                self.sub_step = 0

            return self._make_action(env, q_target, close_gripper=True)

        # --- MOVING BRANCH ---
        elif self.state == "move_target_setup":
            dest = np.array(
                [
                    np.random.uniform(-0.9, -0.5),
                    np.random.uniform(-1.0, -0.5),
                    1.71 + 0.15,
                ]
            )
            self._update_trajectory(ee_pos, dest)
            self.state = "move_target"
            return self._make_action(env, data.qpos, close_gripper=True)

        elif self.state == "move_target":
            # FASTER MOVE: increment 0.03
            self.trajectory_t = min(1.0, self.trajectory_t + 0.03)
            target_pos = self._get_bezier_point(self.trajectory_t)
            target_pos += self.noise_generator.sample()

            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=target_pos,
                target_quat=self.target_quat_cache,
                inplace=False,
            )

            # Loose tolerance (0.08)
            if self.trajectory_t >= 1.0 and self._at_target(env, target_pos, 0.08):
                self.state = "place"
                self.sub_step = 0

            return self._make_action(env, q_target, close_gripper=True)

        elif self.state == "place":
            target_pos = self.trajectory_end.copy()
            target_pos[2] -= 0.14

            # NO NOISE during placement
            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=target_pos,
                target_quat=self.target_quat_cache,
                inplace=False,
            )

            if self._at_target(env, target_pos, 0.03):
                self.state = "release"

            return self._make_action(env, q_target, close_gripper=True)

        elif self.state == "release":
            self.sub_step += 1
            target_pos = self.trajectory_end.copy()
            target_pos[2] -= 0.14
            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=target_pos,
                target_quat=self.target_quat_cache,
                inplace=False,
            )

            if self.sub_step > 20:  # Faster release
                self.state = "done"

            return self._make_action(env, q_target, close_gripper=False)

        # --- POURING BRANCH ---
        elif self.state == "pour_setup":
            other_cup_id = 1 - self.cup_id
            target_cup_pos = utils.get_object_pos(
                env, (f"cup_freejoint{other_cup_id}", f"cup{other_cup_id}")
            )
            pour_pos = target_cup_pos + np.array([0, 0, 0.3])

            self._update_trajectory(ee_pos, pour_pos)
            self.state = "pour_approach"
            return self._make_action(env, data.qpos, close_gripper=True)

        elif self.state == "pour_approach":
            # FASTER APPROACH: increment 0.04
            self.trajectory_t = min(1.0, self.trajectory_t + 0.04)
            target_pos = self._get_bezier_point(self.trajectory_t)

            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=target_pos,
                target_quat=self.target_quat_cache,
                inplace=False,
            )

            # Loose tolerance (0.08)
            if self.trajectory_t >= 1.0 and self._at_target(env, target_pos, 0.08):
                self.state = "pour_tilt"
                self.sub_step = 0

            return self._make_action(env, q_target, close_gripper=True)

        elif self.state == "pour_tilt":
            self.sub_step += 1
            other_cup_id = 1 - self.cup_id
            target_cup_pos = utils.get_object_pos(
                env, (f"cup_freejoint{other_cup_id}", f"cup{other_cup_id}")
            )

            target_quat = np.array([0.12278783, -0.69636423, 0.69636423, 0.12278783])
            pour_target = target_cup_pos + np.array([-0.02, -0.02, 0.24])

            q_target = utils.ik_solve_dm(
                model,
                data,
                "grip_site",
                target_pos=pour_target,
                target_quat=target_quat,
                inplace=False,
            )

            # Use stronger alpha for faster tilt smoothing
            alpha = 0.08
            q_current = data.qpos[:7]
            q_smooth = q_current + alpha * (q_target[:7] - q_current)

            # Reduced duration from 150 to 110
            if self.sub_step > 110:
                self.state = "done"

            return self._make_action(env, q_smooth, close_gripper=True)

        elif self.state == "done":
            return np.zeros(env.unwrapped.nu)

        return np.zeros(env.unwrapped.nu)
