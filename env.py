"""
Minimal gymnasium environment template for modern mujoco + gymnasium.

Install:
    pip install gymnasium mujoco
(or your chosen mujoco build; adapt imports if you use a different package)

Notes:
- Put your kitchen model and assets in a folder (e.g., models/kitchen/).
- Update MODEL_XML_PATH to point to the kitchen xml.
- Customize _get_observation() and _compute_reward() for your task.
"""

import os
from typing import Optional, Tuple, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco as mj
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


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
joint cup_freejoint        | FREE      | qpos[30:37] (x,y,z,quat wxyz), qvel[29:35] (lin+ang)
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


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "kitchen", "kitchen.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.1,
    "azimuth": 200.0,
    "elevation": -35.0,
    "lookat": np.array([-0.65, -0.65, 1.75]),
}

# DEFAULT_CAMERA_CONFIG = {
#     "distance": 4.6,
#     "azimuth": 70.0,
#     "elevation": -35.0,
#     "lookat": np.array([-0.2, 0.5, 2.0]),
# }


class KitchenMinimalEnv(MujocoEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        model_path: str = MODEL_XML_PATH,
        render_mode: str = "rgb_array",
        randomise_cup_position: bool = False,
        **kwargs,
    ):

        # load model and data
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        # Determine sizes
        self.nq = self.model.nq  # number of generalized coordinates
        self.nv = self.model.nv  # number of generalized velocities
        self.nu = self.model.nu  # number of actuators (action dim)

        # Get actuator control ranges for proper scaling
        if (
            hasattr(self.model, "actuator_ctrlrange")
            and self.model.actuator_ctrlrange.size
        ):
            self.ctrl_range = np.array(self.model.actuator_ctrlrange).reshape(
                self.nu, 2
            )
        else:
            self.ctrl_range = np.tile(np.array([-1.0, 1.0]), (self.nu, 1))

        # By default, we use continuous actions in [-1, 1] mapped to ctrl ranges
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # Minimal observation: concatenation of qpos and qvel
        obs_dim = self.nq + self.nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        super().__init__(
            model_path=model_path,
            frame_skip=20,
            observation_space=self.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            **kwargs,
        )

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
        self._width = 1920
        self._height = 2560

        # Reset to initial state
        self.reset(seed=None)

    def get_random_robot_qpos(self):
        """Sample a random robot qpos within joint limits."""
        INIT_QPOS = np.array(
            [
                np.random.uniform(0.9, 1.1),
                np.random.uniform(-0.7, -0.3),
                0,
                0,
                0,
                0,
                0,
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
        return INIT_QPOS

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        randomise_cup_position = (
            options.get("randomise_cup_position", False) if options else False
        )

        # Reset simulation state
        if self.model.nv:
            self.data.qvel[:] = np.zeros(self.nv)

        self.data.qpos[: INIT_QPOS.shape[0]] = self.get_random_robot_qpos()
        self.set_state(self.data.qpos, self.data.qvel)

        # mj.mj_forward(self.model, self.data)

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
            # update water particle world positions now that we ran forward
            self._update_water_particle_positions()

        obs = self._get_observation()
        info = {}
        return obs, info

    def randomise_cup_position(self):
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)

        cup_joint_ids = []
        for j in range(int(self.model.njnt)):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if name and "cup_freejoint" in name:
                cup_joint_ids.append(int(j))

        # Randomize cup positions
        for jid in cup_joint_ids:
            qpos_addr = int(self.model.jnt_qposadr[jid])
            pos = np.copy(qpos[qpos_addr : qpos_addr + 3])
            pos[0] += self.np_random.uniform(-0.1, 0.1)
            pos[1] += self.np_random.uniform(-0.3, 0.0)
            qpos[qpos_addr : qpos_addr + 3] = pos

        # Apply full state so MuJoCo updates positions
        self.set_state(qpos, qvel)

        cup_jid_for_water = cup_joint_ids[1]
        cup_qpos_addr = int(self.model.jnt_qposadr[cup_jid_for_water])
        cup_pos = np.copy(self.data.qpos[cup_qpos_addr : cup_qpos_addr + 3])

        water_joint_ids = []
        for j in range(int(self.model.njnt)):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if name and "water_balls_freejoint" in name:
                water_joint_ids.append(int(j))

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
                # freejoint has 6 dofs (3 lin, 3 ang)
                qvel[vel_addr : vel_addr + 6] = 0.0
                # give some initial downward velocity
                qvel[vel_addr + 2] = self.np_random.uniform(-0.2, -0.15)

        # Apply state and forward simulate so data.geom_xpos update
        self.set_state(qpos, qvel)
        mj.mj_forward(self.model, self.data)
        # update tracked water particle world positions now
        self._update_water_particle_positions()

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        obs = self._get_obs()

        return obs

    def goal_achieved(self) -> bool:
        """
        Check if the goal is achieved (i.e., all water particles are in cup0).
        """
        geom_names = {
            "right": "right_wall_cup0",
            "left": "left_wall_cup0",
            "front": "front_wall_cup0",
            "back": "back_wall_cup0",
            "bottom": "bottom_cup0",
        }

        # Get geom ids
        geom_ids = {}
        for key, name in geom_names.items():
            gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            if gid == -1:
                # If any geom is missing, we cannot evaluate the goal reliably.
                raise ValueError(f"Could not find geom '{name}' in the model.")
            geom_ids[key] = int(gid)

        # Read world-space positions and model half-sizes
        ngeom = int(self.model.ngeom)
        geom_xpos = np.asarray(self.data.geom_xpos).reshape(ngeom, 3)
        geom_size = np.asarray(self.model.geom_size).reshape(ngeom, 3)

        # Compute inner faces and bounds for the cup interior.
        # For axis-aligned box walls:
        # - right wall inner x = center_x - half_size_x
        # - left wall inner x  = center_x + half_size_x
        # - front wall inner y = center_y - half_size_y
        # - back wall inner y  = center_y + half_size_y
        # - bottom top z       = center_z + half_size_z
        rpos = geom_xpos[geom_ids["right"]]
        rsize = geom_size[geom_ids["right"]]
        lpos = geom_xpos[geom_ids["left"]]
        lsize = geom_size[geom_ids["left"]]
        fpos = geom_xpos[geom_ids["front"]]
        fsize = geom_size[geom_ids["front"]]
        bkpos = geom_xpos[geom_ids["back"]]
        bksize = geom_size[geom_ids["back"]]
        botpos = geom_xpos[geom_ids["bottom"]]
        botsize = geom_size[geom_ids["bottom"]]

        right_inner_x = float(rpos[0] - rsize[0])
        left_inner_x = float(lpos[0] + lsize[0])
        front_inner_y = float(fpos[1] - fsize[1])
        back_inner_y = float(bkpos[1] + bksize[1])
        bottom_top_z = float(botpos[2] + botsize[2])

        # wall top z (use one of the vertical walls)
        wall_top_z = float(rpos[2] + rsize[2])

        # Ensure min/max ordering in case of sign conventions
        x_min = min(left_inner_x, right_inner_x)
        x_max = max(left_inner_x, right_inner_x)
        y_min = min(back_inner_y, front_inner_y)
        y_max = max(back_inner_y, front_inner_y)
        z_min = bottom_top_z
        z_max = wall_top_z

        # Check every tracked water particle position
        for p in self.water_particle_positions[: self.num_water_particles]:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            inside_x = x_min <= x <= x_max
            inside_y = y_min <= y <= y_max
            inside_z = z_min <= z <= z_max
            if not (inside_x and inside_y and inside_z):
                return False

        return True

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.nu)

        # set controls
        self.data.ctrl[: self.nu] = action

        # Step the physics forward.
        mj.mj_step(self.model, self.data)

        # update water particle world positions after stepping
        self._update_water_particle_positions()

        # Build observation
        obs = self._get_observation()
        reward = self._compute_reward(obs, action)
        terminated = self._is_terminated(obs)
        truncated = False
        info = {}

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _get_observation(self) -> np.ndarray:
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        return obs

    def _get_obs(self):
        # Gather simulated observation
        # TODO
        #  robot_qpos, robot_qvel = robot_get_obs(
        #     self.model, self.data, self.model_names.joint_names
        # )
        # Simulate observation noise
        # robot_qpos += (
        # self.robot_noise_ratio
        # * self.robot_pos_noise_amp[:9]
        # * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qpos.shape)
        # )
        # robot_qvel += (
        # self.robot_noise_ratio
        # * self.robot_vel_noise_amp[:9]
        # * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qvel.shape)
        # )

        # self._last_robot_qpos = robot_qpos

        # return np.concatenate((robot_qpos.copy(), robot_qvel.copy()))
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        return obs

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        return 1.0 if self.goal_achieved() else 0.0

    def _is_terminated(self, obs: np.ndarray) -> bool:
        return self.goal_achieved()

    def close(self):
        self._render_context = None

    def _update_water_particle_positions(self) -> None:
        """Read current world-space positions for detected water particle geoms into
        self.water_particle_positions. Uses data.geom_xpos (ngeom x 3)."""
        ngeom = int(self.model.ngeom)
        geom_xpos = np.asarray(self.data.geom_xpos).reshape(ngeom, 3)
        for i, gid in enumerate(self._water_geom_ids):
            self.water_particle_positions[i, :] = geom_xpos[int(gid)]
