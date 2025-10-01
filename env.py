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


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "kitchen", "kitchen.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.6,
    "azimuth": 70.0,
    "elevation": -35.0,
    "lookat": np.array([-0.2, 0.5, 2.0]),
}


class KitchenMinimalEnv(MujocoEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(self, model_path: str = MODEL_XML_PATH, render_mode: str = "rgb_array", **kwargs):

        # load model and data
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        # Determine sizes
        self.nq = self.model.nq     # number of generalized coordinates
        self.nv = self.model.nv     # number of generalized velocities
        self.nu = self.model.nu     # number of actuators (action dim)

        # By default, we use continuous actions in [-1, 1] mapped to ctrl ranges
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # Minimal observation: concatenation of qpos and qvel
        obs_dim = self.nq + self.nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        super().__init__(model_path=model_path, frame_skip=40, observation_space=self.observation_space, default_camera_config=DEFAULT_CAMERA_CONFIG, render_mode=render_mode, **kwargs)

        self.init_qpos = self.data.qpos
        self.init_qvel = self.data.qvel

        self._render_context = None
        self._width = 1920
        self._height = 2560


        # Reset to initial state
        self.reset(seed=None)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Reset simulation state
        if self.model.nq:
            self.data.qpos[:] = self.model.key_qpos[0] if self.model.key_qpos.size else np.zeros(self.nq)
        if self.model.nv:
            self.data.qvel[:] = np.zeros(self.nv)


        mj.mj_forward(self.model, self.data)

        obs = self._get_observation()
        info = {}
        return obs, info
    

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        obs = self._get_obs()

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Clip action and map into data.ctrl. The interpretation of ctrl depends on the actuator.
        action = np.asarray(action, dtype=np.float32).reshape(self.nu)
        action = np.clip(action, -1.0, 1.0)


        if hasattr(self.model, "actuator_ctrlrange") and self.model.actuator_ctrlrange.size:
            # actuator_ctrlrange has shape (nu, 2)
            ctrl_range = np.array(self.model.actuator_ctrlrange).reshape(self.nu, 2)
            # map action from [-1,1] -> [min,max]
            data_ctrl = ((action + 1.0) / 2.0) * (ctrl_range[:, 1] - ctrl_range[:, 0]) + ctrl_range[:, 0]
        else:
            data_ctrl = action

        # set controls
        self.data.ctrl[: self.nu] = data_ctrl

        # Step the physics forward.
        mj.mj_step(self.model, self.data)

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
        #TODO 
        #  robot_qpos, robot_qvel = robot_get_obs(
        #     self.model, self.data, self.model_names.joint_names
        #)
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
        return 0.0

    def _is_terminated(self, obs: np.ndarray) -> bool:
        return False


    def close(self):
        self._render_context = None