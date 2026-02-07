"""
This file is nearly identical to the kitchen environment from env.py
The only changes I made include:
- adding the goal position to the observation for SAC training
- letting the env handle random initial positions of the cups and goals for online training.
"""

import os
from typing import Optional, Tuple, Dict

import kitchen_utils as utils
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco as mj
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from kitchen_utils import ik_solve_dm


"""
=== QPOS / QVEL OVERVIEW ===
joint right_j0             | HINGE     | qpos[0], qvel[0]
joint right_j1             | HINGE     | qpos[1], qvel[1]
joint right_j2             | HINGE     | qpos[2], qvel[2]
joint right_j3             | HINGE     | qpos[3], qvel[3]
joint right_j4             | HINGE     | qpos[4], qvel[4]
joint right_j5             | HINGE     | qpos[5], qvel[5]
joint right_j6             | HINGE     | qpos[6], qvel[6]
joint rc_close             | SLIDE     | qpos[7], qvel[7]
joint lc_close             | SLIDE     | qpos[8], qvel[8]
joint knob_Joint_1         | HINGE     | qpos[9], qvel[9]
joint burner_Joint_1       | SLIDE     | qpos[10], qvel[10]
joint knob_Joint_2         | HINGE     | qpos[11], qvel[11]
joint burner_Joint_2       | SLIDE     | qpos[12], qvel[12]
joint knob_Joint_3         | HINGE     | qpos[13], qvel[13]
joint burner_Joint_3       | SLIDE     | qpos[14], qvel[14]
joint knob_Joint_4         | HINGE     | qpos[15], qvel[15]
joint burner_Joint_4       | SLIDE     | qpos[16], qvel[16]
joint lightswitch_joint    | HINGE     | qpos[17], qvel[17]
joint light_joint          | SLIDE     | qpos[18], qvel[18]
joint slidedoor_joint      | SLIDE     | qpos[19], qvel[19]
joint leftdoorhinge        | HINGE     | qpos[20], qvel[20]
joint rightdoorhinge       | HINGE     | qpos[21], qvel[21]
joint microjoint           | HINGE     | qpos[22], qvel[22]
joint kettle_freejoint     | FREE      | qpos[23:30] (x,y,z,quat wxyz), qvel[23:29] (lin+ang)
joint cup_freejoint0       | FREE      | qpos[30:37] (x,y,z,quat wxyz), qvel[29:35] (lin+ang)
joint cup_freejoint1       | FREE      | qpos[37:44] (x,y,z,quat wxyz), qvel[35:41] (lin+ang)
joint water_balls_freejoint00 | FREE      | qpos[44:51] (x,y,z,quat wxyz), qvel[41:47] (lin+ang)
joint water_balls_freejoint01 | FREE      | qpos[51:58] (x,y,z,quat wxyz), qvel[47:53] (lin+ang)
joint water_balls_freejoint02 | FREE      | qpos[58:65] (x,y,z,quat wxyz), qvel[53:59] (lin+ang)
joint water_balls_freejoint03 | FREE      | qpos[65:72] (x,y,z,quat wxyz), qvel[59:65] (lin+ang)
joint water_balls_freejoint04 | FREE      | qpos[72:79] (x,y,z,quat wxyz), qvel[65:71] (lin+ang)
joint water_balls_freejoint05 | FREE      | qpos[79:86] (x,y,z,quat wxyz), qvel[71:77] (lin+ang)
joint water_balls_freejoint06 | FREE      | qpos[86:93] (x,y,z,quat wxyz), qvel[77:83] (lin+ang)
joint water_balls_freejoint07 | FREE      | qpos[93:100] (x,y,z,quat wxyz), qvel[83:89] (lin+ang)
joint water_balls_freejoint08 | FREE      | qpos[100:107] (x,y,z,quat wxyz), qvel[89:95] (lin+ang)
joint water_balls_freejoint09 | FREE      | qpos[107:114] (x,y,z,quat wxyz), qvel[95:101] (lin+ang)
=== END OVERVIEW ===
"""

INIT_QPOS = np.array(
    [
        0.0,
        -1.76848573e00,
        1.84390296e00,
        -2.47685760e00,
        2.60252026e-01,
        7.12533105e-01,
        1.59515394e00,
        0.05,
        0.05,
        -2.66279850e-04,
        -5.18043486e-05,
        3.12877220e-05,
        -4.51199853e-05,
        -3.90842156e-06,
        -4.22629655e-05,
        6.28065475e-05,
        4.04984708e-05,
        4.62730939e-04,
        -2.26906415e-04,
        -4.65501369e-04,
        -6.44129196e-03,
        -1.77048263e-03,
        1.08009684e-03,
        -0.169,
        0,
        1.61944683e00,
        1.00618764e00,
        4.06395120e-03,
        -6.62095997e-03,
        0,
        -0.55,
        -0.55,
        1.6,
        1.0,
        0.0,
        0.0,
        0.0,
        -0.9,
        -0.9,
        1.6,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
)


MOVING_GOAL_OBS = [
    -1.78375408e-01,
    5.75012863e-01,
    -7.96347141e-01,
    -2.74259359e-01,
    2.43455157e-01,
    -1.68792322e-01,
    5.58651030e-01,
    -8.83071125e-01,
    -6.66102543e-02,
    1.20024584e-01,
    -8.81345928e-01,
    -5.75924758e-03,
    -9.57763579e-04,
    2.72275358e-02,
    3.21387888e-05,
    7.24034762e-05,
    3.66702094e-03,
]


POUR_GOAL_OBS = [
    2.0044804e-01,
    3.2168922e-01,
    -6.9168013e-01,
    1.8018240e-01,
    -5.6926471e-01,
    8.0050749e-01,
    5.1573012e-02,
    1.0000000e00,
    2.0052075e-01,
    3.6266208e-01,
    -8.8145667e-01,
    1.7388980e-01,
    2.8165346e-01,
    -6.8698835e-01,
    1.6937990e-03,
    -3.2546069e-04,
    5.0627277e-03,
    -7.4681111e-02,
    3.2490779e-02,
    7.9458199e-02,
    2.1463673e-01,
    3.6153507e-01,
    -8.3332032e-01,
    2.5687990e-01,
    3.5999426e-01,
    -8.6380005e-01,
    2.2144477e-01,
    3.6889440e-01,
    -8.5082227e-01,
    2.1647239e-01,
    3.7160516e-01,
    -8.1120014e-01,
    2.1982503e-01,
    3.5674363e-01,
    -7.8351241e-01,
    1.8411095e-01,
    3.1489402e-01,
    -7.2258836e-01,
    1.7752345e-01,
    3.6067042e-01,
    -8.4941882e-01,
    2.0348501e-01,
    3.4856954e-01,
    -7.5486499e-01,
    1.9583218e-01,
    3.3897918e-01,
    -7.3316163e-01,
    2.1659207e-01,
    3.4698945e-01,
    -7.5649911e-01,
]

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "kitchen", "kitchen.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.8,
    "azimuth": 350.0,
    "elevation": -35.0,
    "lookat": np.array([-0.65, -0.8, 1.75]),
}

# DEFAULT_CAMERA_CONFIG = {
#     "distance": 4.6,
#     "azimuth": 70.0,
#     "elevation": -35.0,
#     "lookat": np.array([-0.2, 0.5, 2.0]),
# }


class KitchenSACOnlineEnv(MujocoEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        model_path: str = MODEL_XML_PATH,
        render_mode: str = "rgb_array",
        ob_type: str = "states",
        randomise_cup_position: bool = False,
        minimal: bool = True,
        physics_timestep: float = 0.001,
        control_timestep: float = 0.004,
        max_episode_steps: int = 1500,
        **kwargs,
    ):
        # load model and data
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        self.max_episode_steps = max_episode_steps
        self._episode_steps = 0

        # Determine sizes
        self.nq = self.model.nq  # number of generalized coordinates
        self.nv = self.model.nv  # number of generalized velocities
        self.nu = self.model.nu  # number of actuators (action dim)

        self.set_timesteps(
            physics_timestep=float(physics_timestep),
            control_timestep=float(control_timestep),
        )

        self.goal_pos: Optional[np.ndarray] = None
        self.active_cup_id: Optional[int] = np.random.choice([0, 1])
        self._prev_cup_goal_dist = None

        # Set observation mode (either 'states' or 'pixels') and default render size
        assert ob_type in ("states", "pixels"), "ob_type must be 'states' or 'pixels'"
        self._ob_type = ob_type

        self._width = 320
        self._height = 240

        # Get actuator control ranges for proper scaling
        if (
            hasattr(self.model, "actuator_ctrlrange")
            and self.model.actuator_ctrlrange.size
        ):
            self.ctrl_range = np.array(self.model.actuator_ctrlrange).reshape(
                self.nu, 2
            )

        # Workspace bounds for denormalization: x: [-1.5, 0], y: [-2.5, 0], z: [1.5, 3]
        self.workspace_bounds = {
            "x": np.array([-1.5, 0.0]),
            "y": np.array([-2.5, 0.0]),
            "z": np.array([1.5, 3.0]),
        }

        # Helper method to normalize position to [-1, 1] using workspace bounds
        self._normalize_position = self._make_position_normalizer()

        if self._ob_type == "pixels":

            obs_dim = (self._height, self._width, 3)
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=obs_dim,
                dtype=np.uint8,
            )
        else:
            dummy_obs = self._get_observation(minimal=minimal)
            obs_dim = dummy_obs.shape[0]
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

        super().__init__(
            model_path=model_path,
            frame_skip=60,
            observation_space=self.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            **kwargs,
        )

        # Action: joints + gripper (8 dimensions: 7 arm joints + 1 gripper command)
        # 7 arm joints + 1 gripper command
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )
        self.arm_delta_scale = 0.5
        self.gripper_delta_scale = 0.15

        self.init_qpos = self.data.qpos
        self.init_qvel = self.data.qvel

        # initial qpos
        self.init_qpos[: INIT_QPOS.shape[0]] = self.get_random_robot_qpos()

        # Try to detect water particle geoms by type/rgba
        geom_type = np.asarray(self.model.geom_type).reshape(-1)
        ngeom = int(self.model.ngeom)
        geom_rgba = np.asarray(self.model.geom_rgba).reshape(ngeom, 4)
        sphere_type = mj.mjtGeom.mjGEOM_SPHERE
        target_rgba = np.array([0.2, 0.45, 0.95, 0.8])

        mask = geom_type == sphere_type
        mask &= np.all(np.isclose(geom_rgba, target_rgba, atol=1e-3), axis=1)
        water_geom_ids = np.nonzero(mask)[0]

        self._water_geom_ids = water_geom_ids.astype(int)
        self.num_water_particles = int(self._water_geom_ids.size)

        # create array for the position of the water particles (will be filled at runtime)
        self.water_particle_positions = np.zeros(
            (self.num_water_particles, 3), dtype=np.float64
        )

        if randomise_cup_position:
            self.randomise_cup_position()
        else:
            self._update_water_particle_positions()

        self._render_context = None

        # Reset to initial state
        self.reset(
            seed=None,
            options={
                "randomise_cup_position": randomise_cup_position,
                "minimal": minimal,
            },
        )

    def _make_position_normalizer(self):
        """Create a function that normalizes 3D positions using workspace bounds."""
        bounds_x = self.workspace_bounds["x"]
        bounds_y = self.workspace_bounds["y"]
        bounds_z = self.workspace_bounds["z"]

        def normalize_position(pos_3d):
            """Normalize a 3D position to [-1, 1] range using workspace bounds."""
            pos = np.asarray(pos_3d, dtype=np.float32)
            # Map from workspace bounds to [-1, 1]
            normalized = np.array(
                [
                    2.0 * (pos[0] - bounds_x[0]) / (bounds_x[1] - bounds_x[0]) - 1.0,
                    2.0 * (pos[1] - bounds_y[0]) / (bounds_y[1] - bounds_y[0]) - 1.0,
                    2.0 * (pos[2] - bounds_z[0]) / (bounds_z[1] - bounds_z[0]) - 1.0,
                ],
                dtype=np.float32,
            )
            return np.clip(normalized, -1.0, 1.0)

        return normalize_position

    def _action_rotations_to_quaternion(self, rot: float) -> np.ndarray:
        """
        Convert normalized rotation parameter to a quaternion.

        This defines a 180-degree arc around the X-axis starting from 'sideways'.

        Args:
            rot: [-1, 1] scalar.
                 -1.0 = Sideways (Opposite side 180 deg)
                 0.0 = Top-down (Vertical)
                 1.0 = Sideways (Parallel to table)

        Returns:
            Quaternion [w, x, y, z]
        """
        # 1. Base quaternion (Sideways) [0.707, 0, 0, -0.707]
        q_sideways = np.array([0.70710678, 0.0, 0.0, -0.70710678], dtype=np.float32)

        # Map rot from [-1, 1] to angle in [0, π]
        angle = (rot + 1.0) * 0.5 * np.pi

        # q_rot = [cos(angle/2), sin(angle/2), 0, 0]
        half_angle = angle / 2.0
        sin_a = np.sin(half_angle)
        cos_a = np.cos(half_angle)

        # q_x_rot   = [cos_a, sin_a, 0, 0]
        # q_sideways = [w_s,   0,     0, z_s]

        w_s = q_sideways[0]
        z_s = q_sideways[3]

        new_w = cos_a * w_s
        new_x = sin_a * w_s
        new_y = -sin_a * z_s
        new_z = cos_a * z_s

        return np.array([new_w, new_x, new_y, new_z], dtype=np.float32)

    def _quaternion_to_rotation_params(self, quat: np.ndarray) -> float:
        """
        Inverse of _action_rotations_to_quaternion.
        Analytically projects a quaternion onto the specific X-rotation arc defined above.

        Args:
            quat: array-like quaternion [w, x, y, z]

        Returns:
            rot: float in [-1, 1]
        """
        q = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm == 0:
            return 0.0
        q = q / norm

        # q_side = [0.707, 0, 0, -0.707] -> inverse = [0.707, 0, 0, 0.707]
        q_side_inv = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)

        # Calculate relative rotation: q_rel = q_current * q_base_inverse
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = q_side_inv

        # Hamilton product q * q_inv
        rel_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        rel_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2

        # Extract angle from quaternion: θ = 2 * arctan2(x_component, w_component)
        # For a quaternion [cos(θ/2), sin(θ/2), 0, 0] representing rotation around X-axis
        angle = 2.0 * np.arctan2(np.abs(rel_x), rel_w)

        # Map angle from [0, pi] to [-1, 1]
        rot = angle / np.pi * 2.0 - 1.0

        return float(np.clip(rot, -1.0, 1.0))

    def _get_task_space_obs(self):
        """Get current task-space representation.

        Returns 5D array: [x, y, z, gripper, rot]
        """
        grip_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "grip_site")

        if grip_site_id == -1:
            ee_pos = np.array([0.0, 0.0, 0.0])
            gripper_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            ee_pos = self.data.site_xpos[grip_site_id].copy()
            mat = self.data.site_xmat[grip_site_id].reshape(9)
            gripper_quat = np.empty(4)
            mj.mju_mat2Quat(gripper_quat, mat)

        pos_norm = self._normalize_position(ee_pos)

        gripper_pos = (self.data.qpos[7] + self.data.qpos[8]) / 2.0
        # Normalize gripper from [0, 0.015] to [-1, 1]
        gripper_norm = np.clip(2.0 * (gripper_pos / 0.015) - 1.0, -1.0, 1.0)

        # Get the single rotation parameter
        rot = self._quaternion_to_rotation_params(gripper_quat)

        # Concatenate: 3 pos + 1 gripper + 1 rot = 5 dims
        task_space_obs = np.concatenate([pos_norm, [gripper_norm, rot]]).astype(
            np.float32
        )

        return task_space_obs

    def get_random_robot_qpos(self):
        """Sample a random robot qpos within joint limits."""
        INIT_QPOS = np.array(
            [
                np.random.uniform(0.6, 1.4),
                np.random.uniform(-0.8, -0.2),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.2, 0.2),
                0,
                0,
                -2.66279850e-04,
                -5.18043486e-05,
                3.12877220e-05,
                -4.51199853e-05,
                -3.90842156e-06,
                -4.22629655e-05,
                6.28065475e-05,
                4.04984708e-05,
                4.62730939e-04,
                -2.26906415e-04,
                -4.65501369e-04,
                -6.44129196e-03,
                -1.77048263e-03,
                1.08009684e-03,
                -0.169,
                0,
                1.61944683e00,
                1.00618764e00,
                4.06395120e-03,
                -6.62095997e-03,
                0,
                -0.6,
                -0.8,
                1.6,
                1.0,
                0.0,
                0.0,
                0.0,
                -0.8,
                -1.1,
                1.6,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        return INIT_QPOS

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to a starting state.

        Args:
            seed: RNG seed.
            options: Dictionary containing 'randomise_cup_position' (bool)
                     and 'minimal' (bool) flags.

        Returns:
            Tuple[np.ndarray, Dict]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)

        self._episode_steps = 0

        randomise_cup_position = (
            options.get("randomise_cup_position", False) if options else False
        )
        minimal = True

        # Reset simulation state
        if self.model.nv:
            self.data.qvel[:] = np.zeros(self.nv)

        self.data.qpos[: INIT_QPOS.shape[0]] = self.get_random_robot_qpos()
        self.set_state(self.data.qpos, self.data.qvel)

        self.active_cup_id = np.random.choice([0, 1])
        self.goal_pos = self.sample_goal_position()

        mj.mj_forward(self.model, self.data)

        #  give water particles some initial random velocity
        for j in range(self.model.njnt):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if "water_balls_freejoint" in name:
                start = self.model.jnt_dofadr[j]
                self.data.qvel[start : start + 2] = np.random.uniform(-0.01, 0.01, 2)
                self.data.qvel[start + 2] = np.random.uniform(-0.1, -0.15)

        if randomise_cup_position:
            self.randomise_cup_position()
        else:
            self._reset_water_in_cups()

        self._prev_ee_cup_dist = None
        self._prev_cup_goal_dist = None

        cup_pos = self.data.qpos[30 + self.active_cup_id * 7 : 33 + self.active_cup_id * 7]

        grip_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "grip_site")
        ee_pos = self.data.site_xpos[grip_site_id]

        self._prev_ee_cup_dist = np.linalg.norm(ee_pos - cup_pos)
        self._prev_cup_goal_dist = np.linalg.norm(cup_pos[:2] - self.goal_pos[:2])


        obs = self._get_observation(minimal=minimal)
        return obs, {}

    def _reset_water_in_cups(self):
        """Helper to reset water particles into the cups."""
        cup_joint_ids = []
        for j in range(int(self.model.njnt)):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if name and "cup_freejoint" in name:
                cup_joint_ids.append(int(j))

        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)

        # Use the second cup for water initialization
        cup_jid_for_water = cup_joint_ids[1]
        cup_qpos_addr = int(self.model.jnt_qposadr[cup_jid_for_water])
        cup_pos = np.copy(self.data.qpos[cup_qpos_addr : cup_qpos_addr + 3])

        water_joint_ids = [
            j
            for j in range(int(self.model.njnt))
            if "water_balls_freejoint"
            in mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
        ]

        for i, jid in enumerate(water_joint_ids):
            qpos_addr = int(self.model.jnt_qposadr[jid])
            x = cup_pos[0] + self.np_random.uniform(-0.01, 0.01)
            y = cup_pos[1] + self.np_random.uniform(-0.01, 0.01)
            z = cup_pos[2] + 0.02 + i * 0.02 + self.np_random.uniform(0.01, 0.015)

            qpos[qpos_addr : qpos_addr + 3] = np.array([x, y, z])

            if qpos_addr + 7 <= qpos.shape[0]:
                qpos[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])

            if jid < int(self.model.njnt):
                vel_addr = int(self.model.jnt_dofadr[jid])
                qvel[vel_addr : vel_addr + 6] = 0.0
                qvel[vel_addr + 2] = self.np_random.uniform(-0.2, -0.15)

        self.set_state(qpos, qvel)
        mj.mj_forward(self.model, self.data)
        self._update_water_particle_positions()

    def randomise_cup_position(self):
        """Randomize the positions of the cups and place water particles accordingly."""
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)

        cup_joint_ids = []
        for j in range(int(self.model.njnt)):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if name and "cup_freejoint" in name:
                cup_joint_ids.append(int(j))

        random_poisition = [
            [-0.1, -0.2],
            [0.0, 0.1],
            [0.1, -0.5],
        ]

        # Randomize cup positions
        for jid in cup_joint_ids:
            # choose one random position
            pos_xy = random_poisition[self.np_random.integers(0, len(random_poisition))]
            qpos_addr = int(self.model.jnt_qposadr[jid])
            pos = np.copy(qpos[qpos_addr : qpos_addr + 3])
            # pos[0] += self.np_random.uniform(-0.15, 0.15)
            # pos[1] += self.np_random.uniform(-0.42, 0.25)
            pos[0] += pos_xy[0]
            pos[1] += pos_xy[1]
            qpos[qpos_addr : qpos_addr + 3] = pos

        # Apply full state so MuJoCo updates positions
        self.set_state(qpos, qvel)
        self._reset_water_in_cups()

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        obs = self.compute_observation(minimal=True)

        return obs

    def set_timesteps(self, physics_timestep: float, control_timestep: float) -> None:
        """Set the physics and control timesteps for the environment.

        The physics timestep will be assigned to the MjModel during compilation. The control timestep is used to
        determine the number of physics steps to take per control step. (Taken from ogbench env)
        """
        # Check timesteps divisible.
        n_steps = control_timestep / physics_timestep
        rounded_n_steps = int(round(n_steps))
        if abs(n_steps - rounded_n_steps) > 1e-6:
            raise ValueError(
                f"Control timestep {control_timestep} should be an integer multiple of "
                f"physics timestep {physics_timestep}."
            )

        self._physics_timestep = physics_timestep
        self._control_timestep = control_timestep
        self._n_steps = rounded_n_steps
        self.model.opt.timestep = self._physics_timestep

    def get_particles_in_cups(self) -> Tuple[int, int]:
        """
        Track how many water particles are in each cup, accounting for cup rotation.
        Used for goal check.

        Returns:
            Tuple[int, int]: Number of particles in both cups (cup0, cup1)
        """
        cup_suffixes = ["0", "1"]
        particles_in_cups = [0, 0]
        particles = self.water_particle_positions[: self.num_water_particles]

        for cup_idx, suffix in enumerate(cup_suffixes):
            geom_names = {
                "right": f"right_wall_cup{suffix}",
                "left": f"left_wall_cup{suffix}",
                "front": f"front_wall_cup{suffix}",
                "back": f"back_wall_cup{suffix}",
                "bottom": f"bottom_cup{suffix}",
            }

            geom_ids = {}
            for key, name in geom_names.items():
                gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
                if gid == -1:
                    print(f"Warning: Could not find geom '{name}' in the model.")
                    continue
                geom_ids[key] = int(gid)

            parent_id = self.model.geom_bodyid[geom_ids["bottom"]]

            body_xmat = self.data.xmat[parent_id].reshape(3, 3)
            body_xpos = self.data.xpos[parent_id]

            particles_local = np.zeros_like(particles)
            for i, p in enumerate(particles):
                particles_local[i] = body_xmat.T @ (p - body_xpos)

            ngeom = int(self.model.ngeom)
            geom_pos = np.asarray(self.model.geom_pos).reshape(ngeom, 3)
            geom_size = np.asarray(self.model.geom_size).reshape(ngeom, 3)

            bottom_pos = geom_pos[geom_ids["bottom"]]
            bottom_size = geom_size[geom_ids["bottom"]]
            right_size = geom_size[geom_ids["right"]]

            x_bound = bottom_size[0]
            y_bound = bottom_size[1]
            z_min = bottom_pos[2] + bottom_size[2]
            z_max = bottom_pos[2] + right_size[2] * 2

            for p_local in particles_local:
                x, y, z = p_local
                if (
                    -x_bound <= x <= x_bound
                    and -y_bound <= y <= y_bound
                    and z_min <= z <= z_max
                ):
                    particles_in_cups[cup_idx] += 1

        return tuple(particles_in_cups)

    def step(
        self,
        action: np.ndarray,
        minimal=True,
        goal=None,
    ):

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._episode_steps += 1

        # Split action
        arm_delta = action[:7]          # [-1, 1]
        gripper_delta = action[7]       # [-1, 1]

        # Current joint positions
        qpos = self.data.qpos.copy()

        # Arm joints: assume indices [0:7]
        qpos[:7] += self.arm_delta_scale * arm_delta

        # Clip to joint limits
        if self.model.jnt_limited is not None:
            for j in range(7):
                if self.model.jnt_limited[j]:
                    jmin, jmax = self.model.jnt_range[j]
                    qpos[j] = np.clip(qpos[j], jmin, jmax)

        # Gripper joints (indices 7,8)
        # Use symmetric control
        grip = (qpos[7] + qpos[8]) * 0.5
        grip += self.gripper_delta_scale * gripper_delta
        grip = np.clip(grip, 0.0, 1.0)

        qpos[7] = grip
        qpos[8] = grip

        # Apply control targets
        self.data.ctrl[:9] = qpos[:9]

        # Step physics
        mj.mj_step(self.model, self.data, nstep=self._n_steps)

        # Update water particles
        self._update_water_particle_positions()

        # Observation
        obs = self.compute_observation(minimal=minimal)

        # Reward
        reward = self._compute_reward(obs, action)

        # Termination

        terminated = self.check_moving_success()

        truncated = self._episode_steps >= self.max_episode_steps

        info = {}

        return obs, float(reward), bool(terminated), bool(truncated), info

    def compute_observation(self, minimal=False):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()

        return self._get_observation(minimal=minimal)

    def _get_observation(self, minimal=True) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        robot_qpos = qpos[:9]
        robot_qvel = qvel[:9]

        goal_pos = self.goal_pos
        if goal_pos is None:
            goal_pos = self.sample_goal_position()

        grip_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "grip_site")
        ee_pos = self.data.site_xpos[grip_site_id].copy()

        # Cup positions
        cup0_pos = qpos[30:33]
        cup1_pos = qpos[37:40]

        goal_pos = self.goal_pos

        # Relative vectors (key for learning)
        ee_to_cup0 = cup0_pos - ee_pos
        ee_to_cup1 = cup1_pos - ee_pos
        cup0_to_goal = goal_pos - cup0_pos
        cup1_to_goal = goal_pos - cup1_pos

        # Active cup encoding
        active_cup = np.array(
            [1.0, 0.0] if self.active_cup_id == 0 else [0.0, 1.0],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                ee_to_cup0,
                ee_to_cup1,
                cup0_to_goal,
                cup1_to_goal,
                active_cup,
            ],
            dtype=np.float32,
        )

        return obs

    def _get_obs(self):  # not used
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        return obs


    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        # Based on metaworld ^reward function
        cup_idx = 30 + self.active_cup_id * 7
        cup_pos = self.data.qpos[cup_idx : cup_idx + 3]
        cup_quat = self.data.qpos[cup_idx + 3 : cup_idx + 7]
        
        grip_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "grip_site")
        ee_pos = self.data.site_xpos[grip_site_id]
        
        # Get gripper state 
        gripper_action = action[-1] 

        ee_cup_dist = np.linalg.norm(ee_pos - cup_pos)
        cup_goal_dist = np.linalg.norm(cup_pos - self.goal_pos)


        # Returns 1.0 if dist is 0, falls to 0.0 as dist increases.
        reach_reward = 1.0 / (1.0 + 10.0 * ee_cup_dist**2) 

        # We want gripper CLOSED when NEAR cup, but OPEN when FAR (to approach).
        caging_reward = 0.0
        if ee_cup_dist < 0.05:
            # If near, reward closing.
            caging_reward = max(gripper_action, 0) 
        else:
            # If far, reward opening
            caging_reward = max(-gripper_action, 0) * 0.1

        in_place_reward = 1.0 / (1.0 + 5.0 * cup_goal_dist**2)
        

        reward = reach_reward + caging_reward
        
        # only reward with grasping
        is_grasped = (ee_cup_dist < 0.03) and (gripper_action > 0.2)
        
        if is_grasped:
            reward += 5.0 * in_place_reward
            
            # If grasped and lifted off table (assuming table is at z=0.0)
            if cup_pos[2] > 1.65: 
                reward += 2.0
        
        # Sparse success bonus
        if cup_goal_dist < 0.05:
            reward += 1.0

        # Orientation Penalty
        w, x, y, z = cup_quat
        z_align = 1.0 - 2.0 * (x * x + y * y)
        if z_align < 0.7:
            reward -= 1.0

        return float(reward)

    def _is_terminated(self, obs: np.ndarray) -> bool:
        # change condition to make dataset generation faster
        return True if self.get_particles_in_cups()[0] >= 5 else False

    def close(self):
        self._render_context = None

    def get_pixel_observation(self):
        frame = self.render()

        if isinstance(frame, np.ndarray):
            return frame.astype(np.uint8)
        return np.asarray(frame, dtype=np.uint8)

    def _update_water_particle_positions(self) -> None:
        """Read current world-space positions for detected water particle geoms into
        self.water_particle_positions. Uses data.geom_xpos (ngeom x 3)."""
        ngeom = int(self.model.ngeom)
        geom_xpos = np.asarray(self.data.geom_xpos).reshape(ngeom, 3)
        for i, gid in enumerate(self._water_geom_ids):
            self.water_particle_positions[i, :] = geom_xpos[int(gid)]

    def get_pouring_goal_state(self) -> np.ndarray:
        return POUR_GOAL_OBS

    def create_moving_goal_state(
        self,
    ) -> np.ndarray:
        return MOVING_GOAL_OBS

    def sample_goal_position(self) -> np.ndarray:
        cup_number = self.active_cup_id
        other_cup_id = 1 - cup_number
        other_cup_pos = utils.get_object_pos(
            self, (f"cup_freejoint{other_cup_id}", f"cup{other_cup_id}")
        )
        active_cup_pos = utils.get_object_pos(
            self, (f"cup_freejoint{cup_number}", f"cup{cup_number}")
        )
        while True:
            # randomise xy position
            candidate = np.array(
                [
                    np.random.uniform(-1.0, -0.5),
                    np.random.uniform(-1.2, -0.38),
                    1.7,
                ]
            )
            if np.linalg.norm(candidate - other_cup_pos) > 0.11 and np.linalg.norm(candidate - active_cup_pos) > 0.13:
                self.goal_pos = candidate
                break
        # testing
        self.goal_pos = np.array([-0.75, -1.15, 1.7])
        return self.goal_pos

    def check_moving_success(
        self, pos_tol: float = 0.05, rot_tol: float = 0.9
    ) -> bool:
        """
        Checks if the task is successful based on the cup position and orientation.
        Assumes goal_state is a minimal observation.

        Args:
            pos_tol: Euclidean distance tolerance for position.
            rot_tol: Tolerance for upright orientation (1.0 = perfect, 0.0 = 90 deg tilt).
        """
        cup = self.active_cup_id
        if cup == 1:
            pos = self.data.qpos[37:40]
            quat = self.data.qpos[40:44]
        else:
            pos = self.data.qpos[30:33]
            quat = self.data.qpos[33:37]

        target = self.goal_pos

        dist0 = np.linalg.norm(pos - target)
        pos_ok = dist0 < pos_tol

        w, x, y, z = quat
        z_align = 1.0 - 2.0 * (x * x + y * y)
        rot_ok = z_align > (1.0 - rot_tol)

        return bool(pos_ok and rot_ok)