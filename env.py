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

GOAL_JOINTS = [
    1.0328614711761475,
    -0.30165818333625793,
    2.068722724914551,
    -1.1879557371139526,
    1.174445629119873,
    -2.5215463638305664,
    -1.2182813882827759,
    0.01867654360830784,
    0.018937133252620697,
]

GOAL_STATE = [
    1.07545805e00,
    -3.45810235e-01,
    2.12984538e00,
    -1.11880410e00,
    1.32228053e00,
    -2.49284577e00,
    -9.52631652e-01,
    1.86824538e-02,
    1.89980175e-02,
    -5.04269898e-02,
    -1.70732401e-02,
    1.31066039e-01,
    -1.31743038e-02,
    3.54583524e-02,
    -1.72494531e-01,
    9.97730568e-02,
    1.51158994e-04,
    -1.77415219e-04,
    -5.99206924e-01,
    -7.99567699e-01,
    1.58896220e00,
    9.99796391e-01,
    8.91402306e-04,
    -6.78929791e-05,
    2.01592017e-02,
    -5.94504178e-01,
    -9.06082630e-01,
    1.72596216e00,
    6.02466166e-01,
    -7.95019090e-01,
    -1.67549141e-02,
    -6.85448870e-02,
    -6.01585090e-01,
    -8.22981358e-01,
    1.68080163e00,
    6.86983585e-01,
    -7.16947675e-01,
    -7.16830939e-02,
    9.43460166e-02,
    -5.96832573e-01,
    -7.94919491e-01,
    1.61637115e00,
    2.40278896e-02,
    -3.01388592e-01,
    -1.10549189e-01,
    -9.46766317e-01,
    -5.93187511e-01,
    -7.83918083e-01,
    1.61267149e00,
    -2.74654448e-01,
    -2.81748533e-01,
    1.48824334e-01,
    9.07212198e-01,
    -5.92452705e-01,
    -8.11384082e-01,
    1.66036189e00,
    -1.69055730e-01,
    7.95042813e-01,
    -5.18276691e-01,
    2.65925527e-01,
    -6.01209283e-01,
    -8.58185172e-01,
    1.69537389e00,
    3.89461875e-01,
    -5.64100325e-01,
    5.33016622e-01,
    -4.95987475e-01,
    -5.96812725e-01,
    -8.03222537e-01,
    1.64823198e00,
    9.07635242e-02,
    7.68257141e-01,
    -2.59039879e-01,
    5.78309000e-01,
    -6.10493064e-01,
    -7.84668565e-01,
    1.61266506e00,
    -8.65852475e-01,
    -1.72842279e-01,
    -3.19054574e-01,
    -3.44425917e-01,
    -5.84101558e-01,
    -7.88053215e-01,
    1.61259270e00,
    7.25094259e-01,
    2.09179133e-01,
    6.55796170e-01,
    2.03413237e-02,
    -6.15491152e-01,
    -8.02469015e-01,
    1.61277854e00,
    -1.81121320e-01,
    8.22195351e-01,
    5.08990586e-01,
    -1.79216087e-01,
    -6.02438390e-01,
    -8.12050045e-01,
    1.66211891e00,
    -2.57646907e-02,
    4.36448872e-01,
    1.97745025e-01,
    8.77351403e-01,
    -1.41050527e-03,
    -7.47250393e-04,
    2.47015897e-03,
    -2.67232396e-02,
    4.03264761e-02,
    2.43832730e-03,
    -1.15214223e-02,
    -3.21231522e-02,
    4.17195596e-02,
    -2.77386099e-01,
    -3.38911787e-02,
    -3.19476947e-02,
]


MOVING_GOAL_STATE = [
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

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "kitchen", "kitchen.xml")

# DEFAULT_CAMERA_CONFIG = {
#    "distance": 2.2,
#    "azimuth": 200.0,
#    "elevation": -35.0,
#    "lookat": np.array([-0.65, -0.65, 1.75]),
# }

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


class KitchenMinimalEnv(MujocoEnv):
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
        **kwargs,
    ):
        # load model and data
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        # Determine sizes
        self.nq = self.model.nq  # number of generalized coordinates
        self.nv = self.model.nv  # number of generalized velocities
        self.nu = self.model.nu  # number of actuators (action dim)

        self.set_timesteps(
            physics_timestep=float(physics_timestep),
            control_timestep=float(control_timestep),
        )

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

        self.action_range = np.tile(np.array([-1.0, 1.0]), (self.nu, 1))

        # By default, we use continuous actions in [-1, 1] mapped to ctrl ranges
        self.action_space = spaces.Box(
            low=self.action_range[:, 0].astype(np.float32),
            high=self.action_range[:, 1].astype(np.float32),
            shape=(self.nu,),
            dtype=np.float32,
        )

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

    def get_random_robot_qpos(self):
        """Sample a random robot qpos within joint limits."""
        INIT_QPOS = np.array(
            [
                np.random.uniform(0.9, 1.1),
                np.random.uniform(-0.7, -0.3),
                np.random.uniform(-0.1, 0.1),
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
        super().reset(seed=seed)

        randomise_cup_position = (
            options.get("randomise_cup_position", False) if options else False
        )
        minimal = options.get("minimal", False)

        # Reset simulation state
        if self.model.nv:
            self.data.qvel[:] = np.zeros(self.nv)

        self.data.qpos[: INIT_QPOS.shape[0]] = self.get_random_robot_qpos()
        self.set_state(self.data.qpos, self.data.qvel)

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
            cup_joint_ids = []
            for j in range(int(self.model.njnt)):
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
                if name and "cup_freejoint" in name:
                    cup_joint_ids.append(int(j))
            qpos = np.array(self.data.qpos).reshape(-1)
            qvel = np.array(self.data.qvel).reshape(-1)
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
            self._update_water_particle_positions()

        obs = self.compute_observation(minimal=minimal)
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
            pos[1] += self.np_random.uniform(-0.4, 0.2)
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
        obs = self.compute_observation(minimal=True)

        return obs

    def set_timesteps(self, physics_timestep: float, control_timestep: float) -> None:
        """Set the physics and control timesteps for the environment.

        The physics timestep will be assigned to the MjModel during compilation. The control timestep is used to
        determine the number of physics steps to take per control step.
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
        self, action: np.ndarray, minimal=True
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.nu)
        n_scaled = self.nu - 2

        ctrl_low = self.ctrl_range[:n_scaled, 0]
        ctrl_high = self.ctrl_range[:n_scaled, 1]

        # Scale normalized [-1, 1] → actuator ctrl range
        scaled_part = ctrl_low + (action[:n_scaled] + 1.0) * 0.5 * (
            ctrl_high - ctrl_low
        )

        raw_part = action[n_scaled:]
        self.data.ctrl[:n_scaled] = scaled_part
        self.data.ctrl[n_scaled : self.nu] = raw_part

        # Step the physics forward.
        mj.mj_step(self.model, self.data, nstep=self._n_steps)

        # update water particle world positions after stepping
        self._update_water_particle_positions()

        # Build observation
        obs = self.compute_observation(minimal=minimal)
        reward = self._compute_reward(obs, action)
        Goal, Start = self.get_particles_in_cups()
        terminated = True if Goal >= 6 else False
        truncated = terminated
        info = {}

        return obs, float(reward), bool(terminated), bool(truncated), info

    def compute_observation(self, minimal=False):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()

        return self._get_observation(minimal=minimal)

    def _get_observation(self, minimal=False) -> np.ndarray:
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)
        obs = np.concatenate([qpos, qvel]).astype(np.float32)

        if minimal:
            # Return only robot joint qpos and qvel (first 8 qpos/qvel),
            # gripper state, water particle positions and velocities and cup positions and velocities
            # qpos and qvel 0 to 8 and from qpos from 30 to end and qvel from 29 to end
            obs = np.concatenate([qpos[:9], qvel[:9], qpos[30:], qvel[29:41]]).astype(
                np.float32
            )
        return obs

    def _get_obs(self):  # not used I think
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        return obs

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        return 1.0 if self.get_particles_in_cups()[0] == 10 else 0.0

    def _is_terminated(self, obs: np.ndarray) -> bool:
        return True if self.get_particles_in_cups()[0] >= 6 else False

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

    def create_goal_state(
        self, minimal=True, current_state=None, fixed_goal=False
    ) -> np.ndarray:
        """Takes the current state and moves the water particles into the target cup by getting their relative positions to the original cup and change it to the target cup.

        Args:
            minimal: If True, returns the minimal state representation like _get_observation
            current_state: Optional current state to base goal on. If None, uses current env state

        Returns:
            np.ndarray: Goal state observation (minimal)
        """
        # Accept either a full state (qpos+qvel) or a minimal observation
        qpos_full = np.zeros(self.nq, dtype=np.float64)
        qvel_full = np.zeros(self.nv, dtype=np.float64)
        if fixed_goal:
            current_state = GOAL_STATE
        if current_state is None:
            # use current full simulator state
            qpos_local = np.array(self.data.qpos).reshape(-1)
            qvel_local = np.array(self.data.qvel).reshape(-1)
            qpos_full = qpos_local.copy()
            qvel_full = qvel_local.copy()
        else:
            state = np.asarray(current_state).astype(np.float64).copy()
            full_len = int(self.nq + self.nv)
            minimal_len = 18 + (self.nq - 30) + 12

            if state.size == full_len:
                qpos_full = state[: self.nq].copy()
                qvel_full = state[self.nq : self.nq + self.nv].copy()
            elif state.size == minimal_len:
                robot_qpos9 = state[0:9].copy()
                robot_qvel9 = state[9:18].copy()
                qpos_tail_len = self.nq - 30
                qvel_tail_len = 12
                qpos_tail = state[18 : 18 + qpos_tail_len].copy()
                qvel_tail = state[
                    18 + qpos_tail_len : 18 + qpos_tail_len + qvel_tail_len
                ].copy()

                qpos_full = np.zeros(self.nq, dtype=np.float64)
                qvel_full = np.zeros(self.nv, dtype=np.float64)

                qpos_full[:9] = robot_qpos9
                qvel_full[:9] = robot_qvel9
                qpos_full[30:] = qpos_tail
                qvel_full[29:41] = qvel_tail
        if not fixed_goal:
            qpos_full[:9] = GOAL_JOINTS
            qvel_full[:9] = 0

        state_full = np.concatenate([qpos_full, qvel_full])

        source_cup_pos = state_full[37:40]
        target_cup_pos = state_full[30:33]
        cup_offset = target_cup_pos - source_cup_pos

        num_particles = 10
        for i in range(num_particles):
            particle_pos_start = 44 + (i * 7)
            state_full[particle_pos_start : particle_pos_start + 3] += cup_offset

        if minimal:
            qpos_out = state_full[: self.nq]
            qvel_out = state_full[self.nq : self.nq + self.nv]
            return np.concatenate(
                [qpos_out[:9], qvel_out[:9], qpos_out[30:], qvel_out[29:41]]
            ).astype(np.float32)

        return state_full.astype(np.float32)

    def create_moving_goal_state(
        self, minimal=True, current_state=None, fixed_goal=False
    ) -> np.ndarray:

        # Accept either a full state (qpos+qvel) or a minimal observation
        qpos_full = np.zeros(self.nq, dtype=np.float64)
        qvel_full = np.zeros(self.nv, dtype=np.float64)
        if fixed_goal:
            current_state = MOVING_GOAL_STATE
        if current_state is None:
            # use current full simulator state
            qpos_local = np.array(self.data.qpos).reshape(-1)
            qvel_local = np.array(self.data.qvel).reshape(-1)
            qpos_full = qpos_local.copy()
            qvel_full = qvel_local.copy()
        else:
            state = np.asarray(current_state).astype(np.float64).copy()
            full_len = int(self.nq + self.nv)
            minimal_len = 18 + (self.nq - 30) + 12
            print("state size:", state.size)
            print("full len:", full_len)
            print("minimal len:", minimal_len)
            if state.size == full_len:
                qpos_full = state[: self.nq].copy()
                qvel_full = state[self.nq : self.nq + self.nv].copy()
            elif state.size == minimal_len:
                robot_qpos9 = state[0:9].copy()
                robot_qvel9 = state[9:18].copy()
                qpos_tail_len = self.nq - 30
                qvel_tail_len = 12
                qpos_tail = state[18 : 18 + qpos_tail_len].copy()
                qvel_tail = state[
                    18 + qpos_tail_len : 18 + qpos_tail_len + qvel_tail_len
                ].copy()

                qpos_full = np.zeros(self.nq, dtype=np.float64)
                qvel_full = np.zeros(self.nv, dtype=np.float64)

                qpos_full[:9] = robot_qpos9
                qvel_full[:9] = robot_qvel9
                qpos_full[30:] = qpos_tail
                qvel_full[29:41] = qvel_tail
        if not fixed_goal:
            qpos_full[:9] = GOAL_JOINTS
            qvel_full[:9] = 0

        state_full = np.concatenate([qpos_full, qvel_full])

        source_cup_pos = state_full[37:40]
        target_cup_pos = state_full[30:33]
        cup_offset = target_cup_pos - source_cup_pos

        num_particles = 10
        for i in range(num_particles):
            particle_pos_start = 44 + (i * 7)
            state_full[particle_pos_start : particle_pos_start + 3] += cup_offset

        if minimal:
            qpos_out = state_full[: self.nq]
            qvel_out = state_full[self.nq : self.nq + self.nv]
            return np.concatenate(
                [qpos_out[:9], qvel_out[:9], qpos_out[30:], qvel_out[29:41]]
            ).astype(np.float32)

        return state_full.astype(np.float32)

    def check_moving_success(
        self, goal_state: np.ndarray, pos_tol: float = 0.05, rot_tol: float = 0.8
    ) -> bool:
        """
        Checks if the task is successful based on the cup position and orientation.
        Assumes goal_state is a minimal observation.

        Args:
            goal_state: The goal observation (minimal format).
            pos_tol: Euclidean distance tolerance for position.
            rot_tol: Tolerance for upright orientation (1.0 = perfect, 0.0 = 90 deg tilt).
        """
        curr_pos = self.data.qpos[30:33]
        curr_quat = self.data.qpos[33:37]

        target_pos = goal_state[18:21]

        dist = np.linalg.norm(curr_pos - target_pos)
        pos_ok = dist < pos_tol

        w, x, y, z = curr_quat
        z_align = 1.0 - 2.0 * (x * x + y * y)
        rot_ok = z_align > (1.0 - rot_tol)

        return bool(pos_ok and rot_ok)
