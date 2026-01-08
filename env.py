import os
from typing import Optional, Tuple, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco as mj
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from kitchen_utils import ik_solve_dm


"""
=== OLD QPOS / QVEL OVERVIEW ===
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



"""
=== NEW QPOS / QVEL OVERVIEW FOR THE NEW ROBOT ===
joint shoulder_pan_joint       | HINGE     | qpos[0], qvel[0]
joint shoulder_lift_joint      | HINGE     | qpos[1], qvel[1]
joint elbow_joint              | HINGE     | qpos[2], qvel[2]
joint wrist_1_joint            | HINGE     | qpos[3], qvel[3]
joint wrist_2_joint            | HINGE     | qpos[4], qvel[4]
joint wrist_3_joint            | HINGE     | qpos[5], qvel[5]
joint robotiq_right_driver_joint | HINGE     | qpos[6], qvel[6]
joint robotiq_right_coupler_joint | HINGE     | qpos[7], qvel[7]
joint robotiq_right_spring_link_joint | HINGE     | qpos[8], qvel[8]
joint robotiq_right_follower_joint | HINGE     | qpos[9], qvel[9]
joint robotiq_left_driver_joint | HINGE     | qpos[10], qvel[10]
joint robotiq_left_coupler_joint | HINGE     | qpos[11], qvel[11]
joint robotiq_left_spring_link_joint | HINGE     | qpos[12], qvel[12]
joint robotiq_left_follower_joint | HINGE     | qpos[13], qvel[13]
joint knob_Joint_1             | HINGE     | qpos[14], qvel[14]
joint burner_Joint_1           | SLIDE     | qpos[15], qvel[15]
joint knob_Joint_2             | HINGE     | qpos[16], qvel[16]
joint burner_Joint_2           | SLIDE     | qpos[17], qvel[17]
joint knob_Joint_3             | HINGE     | qpos[18], qvel[18]
joint burner_Joint_3           | SLIDE     | qpos[19], qvel[19]
joint knob_Joint_4             | HINGE     | qpos[20], qvel[20]
joint burner_Joint_4           | SLIDE     | qpos[21], qvel[21]
joint lightswitch_joint        | HINGE     | qpos[22], qvel[22]
joint light_joint              | SLIDE     | qpos[23], qvel[23]
joint slidedoor_joint          | SLIDE     | qpos[24], qvel[24]
joint leftdoorhinge            | HINGE     | qpos[25], qvel[25]
joint rightdoorhinge           | HINGE     | qpos[26], qvel[26]
joint microjoint               | HINGE     | qpos[27], qvel[27]
joint kettle_freejoint         | FREE      | qpos[28:35] (x,y,z,quat wxyz), qvel[28:34] (lin+ang)
joint cup_freejoint0           | FREE      | qpos[35:42] (x,y,z,quat wxyz), qvel[34:40] (lin+ang)
joint cup_freejoint1           | FREE      | qpos[42:49] (x,y,z,quat wxyz), qvel[40:46] (lin+ang)
joint water_balls_freejoint00  | FREE      | qpos[49:56] (x,y,z,quat wxyz), qvel[46:52] (lin+ang)
joint water_balls_freejoint01  | FREE      | qpos[56:63] (x,y,z,quat wxyz), qvel[52:58] (lin+ang)
joint water_balls_freejoint02  | FREE      | qpos[63:70] (x,y,z,quat wxyz), qvel[58:64] (lin+ang)
joint water_balls_freejoint03  | FREE      | qpos[70:77] (x,y,z,quat wxyz), qvel[64:70] (lin+ang)
joint water_balls_freejoint04  | FREE      | qpos[77:84] (x,y,z,quat wxyz), qvel[70:76] (lin+ang)
joint water_balls_freejoint05  | FREE      | qpos[84:91] (x,y,z,quat wxyz), qvel[76:82] (lin+ang)
joint water_balls_freejoint06  | FREE      | qpos[91:98] (x,y,z,quat wxyz), qvel[82:88] (lin+ang)
joint water_balls_freejoint07  | FREE      | qpos[98:105] (x,y,z,quat wxyz), qvel[88:94] (lin+ang)
joint water_balls_freejoint08  | FREE      | qpos[105:112] (x,y,z,quat wxyz), qvel[94:100] (lin+ang)
joint water_balls_freejoint09  | FREE      | qpos[112:119] (x,y,z,quat wxyz), qvel[100:106] (lin+ang)
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

MOVING_GOAL_OBS = [
    2.2986150e-01,
    6.3572401e-01,
    -7.8073329e-01,
    6.5986031e-01,
    -4.1356409e-01,
    3.8507444e-01,
    4.9524418e-01,
    2.7653375e-01,
    2.5921717e-01,
    6.1863756e-01,
    -8.7163925e-01,
    -6.6676617e-02,
    1.1999092e-01,
    -8.8139915e-01,
    3.3063307e-02,
    -1.1448758e-02,
    1.5769099e-01,
    2.4746694e-03,
    -2.5997448e-04,
    -9.4263116e-03,
    -4.7571659e-02,
    1.1017485e-01,
    -8.4973305e-01,
    -4.7629356e-02,
    1.2431745e-01,
    -8.4973431e-01,
    -8.3629690e-02,
    1.3201304e-01,
    -8.4969598e-01,
    -8.7012611e-02,
    1.2408896e-01,
    -8.4969169e-01,
    -7.3797069e-02,
    1.2533340e-01,
    -8.4970599e-01,
    -6.3867569e-02,
    1.3067284e-01,
    -8.4971732e-01,
    -6.0883600e-02,
    1.1068640e-01,
    -8.4971875e-01,
    -5.0748665e-02,
    1.3212614e-01,
    -8.4973162e-01,
    -6.4534426e-02,
    1.1855602e-01,
    -8.4971541e-01,
    -8.1557512e-02,
    1.1674376e-01,
    -8.4969682e-01,
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


class KitchenMinimalEnv2(MujocoEnv):
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

        # Task-space action: [x, y, z, qx, qy, qz, qw, gripper]
        # All normalized to [-1, 1]: xyz position, quaternion (4D), and gripper [0, 1]
        # Workspace bounds for denormalization: x: [-1.5, 0], y: [-2.5, 0], z: [1.5, 3]
        self.workspace_bounds = {
            "x": np.array([-1.5, 0.0]),
            "y": np.array([-2.5, 0.0]),
            "z": np.array([1.5, 3.0]),
        }

        # Helper method to normalize position to [-1, 1] using workspace bounds
        self._normalize_position = self._make_position_normalizer()

        self.action_space = spaces.Box(
            low=np.array(
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32
            ),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(8,),
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

    def _make_position_normalizer(self):
        """Create a function that normalizes 3D positions using workspace bounds."""
        bounds_x = self.workspace_bounds["x"]
        bounds_y = self.workspace_bounds["y"]
        bounds_z = self.workspace_bounds["z"]

        def normalize_position(pos_3d):
            """Normalize a 3D position to [-1, 1] range using workspace bounds."""
            pos = np.asarray(pos_3d, dtype=np.float32)
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

    def _get_task_space_obs(self):
        """Get current task-space representation as 8D action-like observation.

        Returns 8D array: [x_norm, y_norm, z_norm, qx, qy, qz, qw, gripper]
        where positions are normalized to [-1, 1] and gripper is in [0, 1].
        """
        # Get current end-effector position and orientation from grip_site
        grip_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "grip_site")
        if grip_site_id == -1:
            # Fallback: use last 7 joint forward kinematics
            ee_pos = np.array([0.0, 0.0, 0.0])
            ee_quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            ee_pos = self.data.site_xpos[grip_site_id].copy()
            ee_quat = self.data.site_xmat[grip_site_id].reshape(3, 3)
            # Convert rotation matrix to quaternion (wxyz)
            ee_quat = self._rot_matrix_to_quat(ee_quat)

        # Normalize position using workspace bounds
        pos_norm = self._normalize_position(ee_pos)

        # Gripper state: average of two gripper joint positions, scaled from [0, 0.015] to [0, 1]
        gripper_pos = (self.data.qpos[7] + self.data.qpos[8]) / 2.0
        gripper_norm = np.clip(gripper_pos / 0.015, 0.0, 1.0)

        task_space_obs = np.concatenate([pos_norm, ee_quat, [gripper_norm]]).astype(
            np.float32
        )

        return task_space_obs

    def _rot_matrix_to_quat(self, rot_mat):
        """Convert 3x3 rotation matrix to quaternion (wxyz format)."""
        # Compute quaternion from rotation matrix using Shepperd's method
        trace = np.trace(rot_mat)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
            y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
            z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
        elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
            w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            x = 0.25 * s
            y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
            w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            y = 0.25 * s
            z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
            w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z], dtype=np.float32)

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
            pos[0] += self.np_random.uniform(-0.15, 0.15)
            pos[1] += self.np_random.uniform(-0.42, 0.25)
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
        action = np.asarray(action, dtype=np.float32).reshape(8)

        # Parse task-space action: [x, y, z, qx, qy, qz, qw, gripper]
        # Note: xyz are normalized to [-1, 1], denormalize using workspace bounds
        action_xyz_norm = action[:3]
        target_quat = action[3:7]
        gripper_val = action[7]

        # Denormalize xyz from [-1, 1] to workspace bounds
        bounds_x = self.workspace_bounds["x"]
        bounds_y = self.workspace_bounds["y"]
        bounds_z = self.workspace_bounds["z"]

        target_pos = np.array(
            [
                bounds_x[0]
                + (action_xyz_norm[0] + 1.0) * 0.5 * (bounds_x[1] - bounds_x[0]),
                bounds_y[0]
                + (action_xyz_norm[1] + 1.0) * 0.5 * (bounds_y[1] - bounds_y[0]),
                bounds_z[0]
                + (action_xyz_norm[2] + 1.0) * 0.5 * (bounds_z[1] - bounds_z[0]),
            ]
        )

        # Solve IK to get target joint positions (6 arm joints)
        joint_indices = np.arange(6)  # 6 arm joints
        target_qpos = ik_solve_dm(
            self.model,
            self.data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            joint_indices=joint_indices,
            inplace=False,
        )

        self.data.ctrl[:6] = target_qpos[:6]

        gripper_ctrl = np.clip(gripper_val, 0.0, 1.0) * 255.0
        self.data.ctrl[6] = gripper_ctrl

        # Step the physics forward.
        mj.mj_step(self.model, self.data, nstep=self._n_steps)

        # update water particle world positions after stepping
        self._update_water_particle_positions()

        # Build observation
        obs = self.compute_observation(minimal=minimal)
        reward = self._compute_reward(obs, action)
        Goal, Start = self.get_particles_in_cups()
        terminated = True if Goal >= 5 else False
        truncated = terminated
        info = {}

        return obs, float(reward), bool(terminated), bool(truncated), info

    def compute_observation(self, minimal=False):
        if self._ob_type == "pixels":
            return self.get_pixel_observation()

        return self._get_observation(minimal=minimal)

    def _get_observation(self, minimal=True) -> np.ndarray:
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)
        obs = np.concatenate([qpos, qvel]).astype(np.float32)

        if minimal:
            # plus normalized cup/water particle positions and their velocities
            task_space_obs = self._get_task_space_obs()  # 8D: xyz_norm + quat + gripper

            # Normalize cup positions using workspace bounds
            cup0_pos_norm = self._normalize_position(qpos[35:38])
            cup1_pos_norm = self._normalize_position(qpos[42:45])

            # Cup velocities (not normalized, use as-is)
            cup0_vel = qvel[34:37]
            cup1_vel = qvel[40:43]

            # Water particle positions and velocities (qpos[44:] and qvel[41:])
            # Normalize water particle positions
            water_qpos_norm_list = []
            num_particles = 10
            for i in range(num_particles):
                water_pos_idx = 49 + (i * 7)
                water_pos = qpos[water_pos_idx : water_pos_idx + 3]
                water_pos_norm = self._normalize_position(water_pos)
                water_qpos_norm_list.append(water_pos_norm)
            water_qpos_norm = np.concatenate(water_qpos_norm_list)

            obs = np.concatenate(
                [
                    task_space_obs,  # 8D
                    cup0_pos_norm,  # 3D (normalized)
                    cup1_pos_norm,  # 3D (normalized)
                    cup0_vel,  # 3D
                    cup1_vel,  # 3D
                    water_qpos_norm,  # num_particles * 3 (normalized)
                ]
            ).astype(np.float32)
        return obs

    def _get_obs(self):  # not used I think
        qpos = np.array(self.data.qpos).reshape(-1)
        qvel = np.array(self.data.qvel).reshape(-1)
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        return obs

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        return 1.0 if self.get_particles_in_cups()[0] == 10 else 0.0

    def _is_terminated(self, obs: np.ndarray) -> bool:
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

    def check_moving_success(
        self, goal_state: np.ndarray, pos_tol: float = 0.03, rot_tol: float = 0.9
    ) -> bool:
        """
        Checks if the task is successful based on the cup position and orientation.
        Assumes goal_state is a minimal observation.

        Args:
            goal_state: The goal observation (minimal format).
            pos_tol: Euclidean distance tolerance for position.
            rot_tol: Tolerance for upright orientation (1.0 = perfect, 0.0 = 90 deg tilt).
        """
        curr_pos = self.data.qpos[35:38]
        curr_quat = self.data.qpos[38:42]

        # In the new minimal observation layout the target cup position is at
        # indices 8:11 (task_space_obs 0:8, cup0_pos 8:11, cup1_pos 11:14, ...)
        target_pos = goal_state[8:11]

        dist = np.linalg.norm(curr_pos - target_pos)
        pos_ok = dist < pos_tol

        w, x, y, z = curr_quat
        z_align = 1.0 - 2.0 * (x * x + y * y)
        rot_ok = z_align > (1.0 - rot_tol)

        return bool(pos_ok and rot_ok)
