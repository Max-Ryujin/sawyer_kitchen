"""
A dataset loader that converts the JSON trajectory files produced by test_env.py
into batches compatible with OGBench's CRL agent.
"""

import os
import json
import numpy as np


def flatten_obs(obs):
    """Convert observation dict into flat array."""
    if isinstance(obs, dict):
        # Just concatenate all values in the observation dict
        return np.concatenate([np.array(v).ravel() for v in obs.values()]).astype(
            np.float32
        )
    return np.array(obs).ravel().astype(np.float32)


class CRLDataset:
    def __init__(self, dataset_dir):
        assert os.path.isdir(dataset_dir), f"dataset_dir not found: {dataset_dir}"
        self.dataset_dir = dataset_dir

        # Load all_episodes.json from the dataset directory
        all_episodes_path = os.path.join(dataset_dir, "all_episodes.json")
        if not os.path.exists(all_episodes_path):
            raise FileNotFoundError(f"all_episodes.json not found in {dataset_dir}")

        with open(all_episodes_path, "r") as f:
            data = json.load(f)

        # Split episodes into train and validation (last 20% for validation)
        episodes = data["episodes"]
        split_idx = int(len(episodes) * 0.8)  # 80-20 split
        train_episodes = episodes[:split_idx]
        val_episodes = episodes[split_idx:]

        print(
            f"Found {len(train_episodes)} training episodes and {len(val_episodes)} validation episodes in {dataset_dir}"
        )

        self.trajs = []  # train trajectories (raw lists)
        self.val_trajs = []  # validation trajectories (raw lists)

        # Convert episodes to trajectories format
        def _process_episode(episode_data):
            # Minimal required keys for an episode
            required_keys = {
                "episode_index",
                "trajectory",
            }
            if not all(k in episode_data for k in required_keys):
                raise ValueError(
                    f"Episode missing required keys. Found {episode_data.keys()}, needed {required_keys}"
                )

            trajectory = episode_data["trajectory"]
            if not trajectory:
                raise ValueError("Empty trajectory")

            # Extract data - only obs and action are required in each step
            obs = []
            actions = []
            infos = []
            for i, step in enumerate(trajectory):
                # These keys must be present in every step
                if "obs" not in step or "action" not in step:
                    raise ValueError(f"Step {i} missing obs or action")
                obs.append(step["obs"])
                actions.append(step["action"])
                # Collect any state info that's available
                infos.append({})
                if "qpos" in step:
                    infos[-1]["qpos"] = step["qpos"]
                if "qvel" in step:
                    infos[-1]["qvel"] = step["qvel"]

            # For CRL, mark the last step as done regardless of termination
            dones = [False] * (len(trajectory) - 1) + [True]

            # Verify data is in expected format
            for i, (o, a) in enumerate(zip(obs, actions)):
                if not isinstance(o, (list, dict)):
                    raise ValueError(
                        f"Step {i}: obs must be list or dict, got {type(o)}"
                    )
                if not isinstance(a, list):
                    raise ValueError(f"Step {i}: action must be list, got {type(a)}")

            return {
                "observations": obs,
                "actions": actions,
                "dones": dones,
                "infos": infos,
            }

        # Process all episodes
        self.trajs.extend(_process_episode(ep) for ep in train_episodes)
        self.val_trajs.extend(_process_episode(ep) for ep in val_episodes)

        assert self.trajs or self.val_trajs, "No trajectories loaded (empty dataset?)"

        # precompute flattened observations per trajectory for train and val
        self._flat_trajs = []  # list of dicts with numpy arrays (train)
        self._val_flat_trajs = []  # validation

        def _precompute(tr_list, out_flat):
            for idx, tr in enumerate(tr_list):
                try:
                    # Verify required keys
                    if not all(
                        k in tr for k in ["observations", "actions", "dones", "infos"]
                    ):
                        raise ValueError(f"Trajectory {idx} missing required keys")

                    # Convert observations to flat arrays with validation
                    try:
                        flat_obs = np.stack(
                            [flatten_obs(o) for o in tr["observations"]]
                        ).astype(np.float32)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to flatten observations in trajectory {idx}: {e}"
                        )

                    # Convert actions with validation
                    try:
                        actions = np.stack(
                            [
                                np.array(a, dtype=np.float32).ravel()
                                for a in tr["actions"]
                            ]
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert actions in trajectory {idx}: {e}"
                        )

                    if len(actions) != len(flat_obs):
                        raise ValueError(
                            f"Trajectory {idx}: mismatched lengths - obs: {len(flat_obs)}, actions: {len(actions)}"
                        )

                    dones = np.array(tr["dones"], dtype=np.bool_)
                    if len(dones) != len(flat_obs):
                        raise ValueError(
                            f"Trajectory {idx}: mismatched lengths - obs: {len(flat_obs)}, dones: {len(dones)}"
                        )

                    # Infos should already be validated by _process_episode
                    infos = tr["infos"]
                    if len(infos) != len(flat_obs):
                        raise ValueError(
                            f"Trajectory {idx}: mismatched lengths - obs: {len(flat_obs)}, infos: {len(infos)}"
                        )

                    out_flat.append(
                        {
                            "observations": flat_obs,
                            "actions": actions,
                            "dones": dones,
                            "infos": infos,
                        }
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to precompute trajectory {idx}: {e}"
                    ) from e

        _precompute(self.trajs, self._flat_trajs)
        _precompute(self.val_trajs, self._val_flat_trajs)

        # infer dims from any available trajectory
        src = None
        if self._flat_trajs:
            src = self._flat_trajs[0]
        elif self._val_flat_trajs:
            src = self._val_flat_trajs[0]
        assert src is not None, "Unable to infer obs/action dims (no trajectories)"

        self.obs_dim = src["observations"].shape[1]
        self.act_dim = src["actions"].shape[1]
        print(
            f"Loaded {len(self._flat_trajs)} training trajectories and {len(self._val_flat_trajs)} validation trajectories. obs_dim={self.obs_dim}, act_dim={self.act_dim}"
        )

    def example(self):
        # return example observation batch and action batch (one element each)
        ex_obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        ex_act = np.zeros((1, self.act_dim), dtype=np.float32)
        return ex_obs, ex_act

    def sample_batch(self, batch_size=256):
        """Sample a batch. For each sample pick a trajectory and a timestep t, then a future t'>t
        and use obs[t] as observation, action[t] as action, and obs[t'] as both actor_goal and value_goal.
        """
        obs_batch = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.act_dim), dtype=np.float32)
        actor_goal = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        value_goal = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        dones_batch = np.zeros((batch_size,), dtype=np.bool_)

        for i in range(batch_size):
            # pick a trajectory with length >= 2
            tries = 0
            while True:
                tr = self._flat_trajs[np.random.randint(len(self._flat_trajs))]
                T = tr["observations"].shape[0]
                if T >= 2:
                    break
                tries += 1
                assert tries < 100, "Unable to find trajectory with length >= 2"

            t = np.random.randint(0, T - 1)  # leave at least one future step
            t_future = np.random.randint(t + 1, T)

            obs_batch[i] = tr["observations"][t]
            act_batch[i] = tr["actions"][t]
            actor_goal[i] = tr["observations"][t_future]
            value_goal[i] = tr["observations"][t_future]
            dones_batch[i] = bool(tr["dones"][t])

        batch = {
            "observations": obs_batch,
            "actions": act_batch,
            "actor_goals": actor_goal,
            "value_goals": value_goal,
            "dones": dones_batch,
        }
        return batch

    def has_validation(self):
        """Return True if validation trajectories were loaded."""
        return bool(self._val_flat_trajs)

    def sample_val_batch(self, batch_size=256):
        """Sample a batch from the validation trajectories. Same format as sample_batch.

        Raises AssertionError if no validation data was loaded.
        """
        assert self._val_flat_trajs, "No validation trajectories available"

        obs_batch = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.act_dim), dtype=np.float32)
        actor_goal = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        value_goal = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        dones_batch = np.zeros((batch_size,), dtype=np.bool_)

        for i in range(batch_size):
            # pick a validation trajectory with length >= 2
            tries = 0
            while True:
                tr = self._val_flat_trajs[np.random.randint(len(self._val_flat_trajs))]
                T = tr["observations"].shape[0]
                if T >= 2:
                    break
                tries += 1
                assert (
                    tries < 100
                ), "Unable to find validation trajectory with length >= 2"

            t = np.random.randint(0, T - 1)
            t_future = np.random.randint(t + 1, T)

            obs_batch[i] = tr["observations"][t]
            act_batch[i] = tr["actions"][t]
            actor_goal[i] = tr["observations"][t_future]
            value_goal[i] = tr["observations"][t_future]
            dones_batch[i] = bool(tr["dones"][t])

        batch = {
            "observations": obs_batch,
            "actions": act_batch,
            "actor_goals": actor_goal,
            "value_goals": value_goal,
            "dones": dones_batch,
        }
        return batch
