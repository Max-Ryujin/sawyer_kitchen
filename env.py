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

import mujoco as mj  # modern mujoco python bindings

# Path to the MJCF model. Update to the copied kitchen xml path.
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "models", "kitchen", "kitchen.xml")


class KitchenMinimalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, model_path: str = MODEL_XML_PATH):
        super().__init__()

        # load model and data
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        # Determine sizes
        self.nq = self.model.nq     # number of generalized coordinates
        self.nv = self.model.nv     # number of generalized velocities
        self.nu = self.model.nu     # number of actuators (action dim)

        # Basic action & observation spaces (customize as needed)
        # By default, we use continuous actions in [-1, 1] mapped to ctrl ranges
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # Minimal observation: concatenation of qpos and qvel
        # You will likely want to include sensor outputs or site positions as well.
        obs_dim = self.nq + self.nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Rendering helper (we'll lazily create offscreen render context)
        self._render_context = None
        self._width = 800
        self._height = 600

        # Reset to initial state
        self.reset(seed=None)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        # Optionally set rng/seed handling (gymnasium required pattern)
        super().reset(seed=seed)

        # Set qpos and qvel to model defaults (model.key_qpos may have defaults)
        # This uses the model's default positions/velocities; customize as needed.
        if self.model.nq:
            self.data.qpos[:] = self.model.key_qpos[0] if self.model.key_qpos.size else np.zeros(self.nq)
        if self.model.nv:
            self.data.qvel[:] = np.zeros(self.nv)

        # Forward the model to compute derived quantities
        mj.mj_forward(self.model, self.data)

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Clip action and map into data.ctrl. The interpretation of ctrl depends on the actuator.
        action = np.asarray(action, dtype=np.float32).reshape(self.nu)
        action = np.clip(action, -1.0, 1.0)

        # If you want to map [-1,1] to actuator control ranges:
        # If model.actuator_ctrlrange exists, each actuator may have a min/max
        # We implement a safe mapping that respects ctrlrange when available.
        if hasattr(self.model, "actuator_ctrlrange") and self.model.actuator_ctrlrange.size:
            # actuator_ctrlrange has shape (nu, 2)
            ctrl_range = np.array(self.model.actuator_ctrlrange).reshape(self.nu, 2)
            # map action from [-1,1] -> [min,max]
            data_ctrl = ((action + 1.0) / 2.0) * (ctrl_range[:, 1] - ctrl_range[:, 0]) + ctrl_range[:, 0]
        else:
            # fallback: pass action directly (useful for torque actuators)
            data_ctrl = action

        # set controls
        self.data.ctrl[: self.nu] = data_ctrl

        # Step the physics forward. Use mj_step for a single step; many controllers step multiple times per env step.
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

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        return 0.0

    def _is_terminated(self, obs: np.ndarray) -> bool:
        return False

    def render(self, mode: str = "human"):
        if mode == "rgb_array":

            img = mj.render(self.model, self.data, width=self._width, height=self._height, camera_id=0)
            return img
        elif mode == "human":
            # Launch interactive viewer (if available). This is a simple blocking viewer.
            # Some mujoco builds provide a viewer module; this is a minimal pattern:
            try:
                import mujoco.viewer as viewer
                # viewer.launch pass model path or model/data as supported:
                viewer.launch(self.model)  # adapt if your mujoco viewer requires (model, data)
            except Exception:
                img = self.render(mode="rgb_array")
                try:
                    import matplotlib.pyplot as plt
                    plt.imshow(img)
                    plt.axis("off")
                    plt.show()
                except Exception:
                    pass
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def close(self):
        self._render_context = None