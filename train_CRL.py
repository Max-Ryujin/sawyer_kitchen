"""
Minimal training script that wires OGBench's CRL agent to the local kitchen trajectories.
"""

import os
import sys
import argparse
import multiprocessing as mp
from typing import Dict, Optional, Tuple, Any

# Environment configuration
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import numpy as np
import gymnasium as gym
import imageio
import wandb

# --- Path Setup ---
THIS_DIR = os.path.dirname(__file__)
OG_IMPLS = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench", "impls"))
OG_IMPLS_BASE = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench"))
sys.path.insert(0, OG_IMPLS)
sys.path.insert(0, OG_IMPLS_BASE)

# --- Agent Imports ---
from agents.crl import CRLAgent, get_config as get_crl_config
from agents.qrl import QRLAgent, get_config as get_qrl_config
from agents.tmd import TMDAgent, get_config as get_tmd_config
from agents.gciql import GCIQLAgent, get_config as get_gciql_config
from agents.gcivl import GCIVLAgent, get_config as get_gcivl_config
from agents.hiql import HIQLAgent, get_config as get_hiql_config

# --- Utils ---
from utils.flax_utils import save_agent
from utils.datasets import GCDataset, Dataset, HGCDataset
from ogbench import load_dataset


def normalize(x, mean, std, eps=1e-5):
    """Normalize all dimensions element-wise."""
    return (x - mean) / (std + eps)


def normalize_observations_selective(x, obs_mean, obs_std, vel_indices, eps=1e-5):
    """
    Selectively normalize only velocity dimensions.

    Positions and water particles are already normalized by env.py using fixed bounds.
    Only velocities (unbounded) need dataset-based normalization.

    Handles both single observations (1D) and batches (2D).

    Args:
        x: observation array (1D or 2D batch, or list)
        obs_mean: mean per dimension
        obs_std: std per dimension
        vel_indices: indices of velocity dimensions to normalize
        eps: small constant for numerical stability
    """
    x_norm = np.asarray(x, dtype=np.float32).copy()
    # Use ... (Ellipsis) to handle both 1D and 2D arrays
    # For 1D: x[..., vel_indices] = x[vel_indices]
    # For 2D: x[..., vel_indices] = x[:, vel_indices]
    x_norm[..., vel_indices] = (x_norm[..., vel_indices] - obs_mean[vel_indices]) / (
        obs_std[vel_indices] + eps
    )
    return x_norm.astype(np.float32)


def evaluate_agent(
    agent,
    obs_mean,
    obs_std,
    val_dataset,
    num_episodes=5,
    steps=1500,
    video=False,
    save_file_prefix=None,
    env=None,
    vel_idx=None,
):
    if vel_idx is None:
        vel_idx = np.arange(7, 10)

    if env is None:
        env = gym.make(
            "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
        )

    fixed_success_count = 0
    fixed_frames = []
    fixed_success_frames_list = []

    # for i in range(num_episodes):
    #     obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
    #     raw_obs = np.asarray(obs)

    #     # Get pouring goal from environment (positions already normalized, velocities are raw)
    #     goal_arr = env.unwrapped.get_pouring_goal_state()
    #     # normalized_goal = normalize_observations_selective(
    #         # goal_arr, obs_mean, obs_std, vel_idx
    #     # )

    #     current_frames = []
    #     is_success = False

    #     for t in range(steps):
    #         # normalized_obs = normalize_observations_selective(
    #         #     raw_obs, obs_mean, obs_std, vel_idx
    #         # )
    #         action = agent.sample_actions(
    #             observations=raw_obs[None],
    #             goals=goal_arr[None],
    #             temperature=0.0,
    #             seed=jax.random.PRNGKey(i * 10000 + t),
    #         )
    #         # Flatten action back to [Dim]
    #         action = np.array(action).flatten()
    #         action = np.clip(action, -1, 1)
    #         obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
    #         raw_obs = np.asarray(obs)

    #         if video:
    #             current_frames.append(env.render())

    #         if term or trunc:
    #             fixed_success_count += 1
    #             is_success = True
    #             break

    #     if video:
    #         if i == 0:
    #             fixed_frames = current_frames
    #         if is_success:
    #             fixed_success_frames_list.append(current_frames)

    # if video and save_file_prefix:
    #     imageio.mimwrite(
    #         f"{save_file_prefix}_pour.mp4",
    #         fixed_frames,
    #         fps=env.metadata.get("render_fps", 24),
    #     )
    #     # Save all successful attempts
    #     for idx, frames in enumerate(fixed_success_frames_list):
    #         imageio.mimwrite(
    #             f"{save_file_prefix}_pour_success_{idx}.mp4",
    #             frames,
    #             fps=env.metadata.get("render_fps", 24),
    #         )

    moving_success_count = 0
    moving_frames = []
    moving_success_frames_list = []

    for i in range(num_episodes):
        obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
        raw_obs = np.asarray(obs)

        # Create moving goal state (positions already normalized, velocities are raw)
        goal_arr = env.unwrapped.create_moving_goal_state()
        normalized_goal = normalize_observations_selective(
            goal_arr, obs_mean, obs_std, vel_idx
        )
        current_frames = []
        is_success = False

        for t in range(steps):
            normalized_obs = normalize_observations_selective(
                raw_obs, obs_mean, obs_std, vel_idx
            )
            print(
                "obs, goal shapes: ",
                normalized_obs[None].shape,
                normalized_goal[None].shape,
            )
            action = agent.sample_actions(
                observations=normalized_obs[None],
                goals=normalized_goal[None],
                temperature=0.0,
                seed=jax.random.PRNGKey(i * 10000 + t),
            )
            # Flatten action back to [Dim]
            action = np.array(action).flatten()
            action = np.clip(action, -1, 1)
            obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
            raw_obs = np.asarray(obs)

            if video:
                current_frames.append(env.render())

            if env.unwrapped.check_moving_success(goal_arr):
                moving_success_count += 1
                is_success = True
                break

        if video:
            if i == 0:
                moving_frames = current_frames
            if is_success:
                moving_success_frames_list.append(current_frames)

    if video and save_file_prefix:
        imageio.mimwrite(
            f"{save_file_prefix}_moving.mp4",
            moving_frames,
            fps=env.metadata.get("render_fps", 24),
        )
        # Save all successful attempts
        for idx, frames in enumerate(moving_success_frames_list):
            imageio.mimwrite(
                f"{save_file_prefix}_moving_success_{idx}.mp4",
                frames,
                fps=env.metadata.get("render_fps", 24),
            )

    moving_success_rate = moving_success_count / num_episodes

    fixed_success_rate = fixed_success_count / num_episodes

    terminals = val_dataset["terminals"].flatten().astype(bool)
    if "timeouts" in val_dataset:
        terminals = terminals | val_dataset["timeouts"].flatten().astype(bool)

    episode_ends = np.where(terminals)[0]
    episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))

    valid_indices = [
        i for i in range(len(episode_starts)) if episode_starts[i] < episode_ends[i]
    ]

    rand_success_count = 0
    val_test_frames = None

    for i in range(2 * num_episodes):
        ep_idx = valid_indices[i]

        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]

        qpos = val_dataset["qpos"][start_idx]
        qvel = val_dataset["qvel"][start_idx]
        # Goal from dataset is already in normalized format (from training preprocessing)
        goal_arr = val_dataset["observations"][end_idx]
        normalized_goal = normalize_observations_selective(
            goal_arr, obs_mean, obs_std, vel_idx
        )
        obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
        env.unwrapped.set_state(qpos, qvel)
        if hasattr(env.unwrapped, "sim"):
            env.unwrapped.sim.forward()
        obs = env.unwrapped._get_observation(minimal=True)
        raw_obs = np.asarray(obs)

        current_frames = []
        is_success = False

        for t in range(steps):

            normalized_obs = normalize_observations_selective(
                raw_obs, obs_mean, obs_std, vel_idx
            )

            action = agent.sample_actions(
                observations=normalized_obs[None],
                goals=normalized_goal[None],
                temperature=0.0,
                seed=jax.random.PRNGKey(i * 10000 + t),
            )
            # Flatten action back to [Dim]
            action = np.array(action).flatten()
            action = np.clip(action, -1, 1)

            obs, _, term, trunc, _ = env.unwrapped.step(
                action, minimal=True, goal=goal_arr
            )
            raw_obs = np.asarray(obs)

            if term or trunc:
                rand_success_count += 1
                is_success = True
                break

            if video:
                current_frames.append(env.render())

        # Save video from first validation test
        if video and (i == 0 or i == 1) and save_file_prefix:
            val_test_frames = current_frames

        # Save video only if successful
        if video and is_success and save_file_prefix:
            save_path = f"{save_file_prefix}_val_ep{ep_idx}_success.mp4"
            imageio.mimwrite(
                save_path, current_frames, fps=env.metadata.get("render_fps", 24)
            )

    # Save one validation test video
    if video and val_test_frames and save_file_prefix:
        imageio.mimwrite(
            f"{save_file_prefix}_val_test.mp4",
            val_test_frames,
            fps=env.metadata.get("render_fps", 24),
        )

    rand_success_rate = rand_success_count / (2 * num_episodes)

    return {
        "pouring_success_rate": fixed_success_rate,
        "moving_success_rate": moving_success_rate,
        "validation_success_rate": rand_success_rate,
    }


def main(args):
    if args.agent_type == "CRL":
        cfg = get_crl_config()
    elif args.agent_type == "QRL":
        cfg = get_qrl_config()
    elif args.agent_type == "TMD":
        cfg = get_tmd_config()
    elif args.agent_type == "GCIQL":
        cfg = get_gciql_config()
    elif args.agent_type == "GCIVL":
        cfg = get_gcivl_config()
    elif args.agent_type == "HIQL":
        cfg = get_hiql_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["batch_size"] = args.batch_size
    cfg["alpha"] = args.alpha
    if args.awr:
        cfg["actor_loss"] = "awr"
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    print("Initializing validation environment...")
    val_env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    print("Training config:", cfg)
    train_path = os.path.join(args.dataset_dir, "train_dataset.npz")
    val_path = os.path.join(args.dataset_dir, "val_dataset.npz")

    train_dataset_raw = load_dataset(train_path, compact_dataset=True)
    val_dataset_raw = load_dataset(val_path, compact_dataset=True, add_info=True)
    # Normalize observations: only normalize velocity components.
    # Positions are already normalized by env.py using fixed workspace bounds.
    # Velocities are unbounded, so we normalize using dataset statistics.
    obs_data = train_dataset_raw["observations"]

    # Velocity indices in minimal observation: 10-14
    vel_idx = np.arange(7, 10)

    obs_mean = np.zeros(obs_data.shape[1], dtype=np.float32)
    print(obs_mean.shape)
    obs_std = np.ones(obs_data.shape[1], dtype=np.float32)
    # Only compute statistics for velocity dimensions
    obs_mean[vel_idx] = np.mean(obs_data[:, vel_idx], axis=0)
    obs_std[vel_idx] = np.std(obs_data[:, vel_idx], axis=0)
    obs_std[obs_std < 1e-3] = 1.0
    print("Observation mean:", obs_mean)
    print("Observation std:", obs_std)

    train_dataset_norm = dict(train_dataset_raw)
    train_dataset_norm["observations"] = normalize_observations_selective(
        train_dataset_raw["observations"], obs_mean, obs_std, vel_idx
    )

    val_dataset_norm = dict(val_dataset_raw)
    val_dataset_norm["observations"] = normalize_observations_selective(
        val_dataset_raw["observations"], obs_mean, obs_std, vel_idx
    )

    base_train = Dataset.create(**train_dataset_norm)
    if args.agent_type == "HIQL":
        train_dataset = HGCDataset(base_train, cfg)
    else:
        train_dataset = GCDataset(base_train, cfg)

    base_val = Dataset.create(**val_dataset_norm)
    if args.agent_type == "HIQL":
        val_dataset = HGCDataset(base_val, cfg)
    else:
        val_dataset = GCDataset(base_val, cfg)

    example_batch = train_dataset.sample(1)

    if args.agent_type == "CRL":

        agent = CRLAgent.create(
            seed=3141,
            ex_observations=example_batch["observations"],
            ex_actions=example_batch["actions"],
            config=cfg,
        )
    elif args.agent_type == "QRL":

        agent = QRLAgent.create(
            seed=3141,
            ex_observations=example_batch["observations"],
            ex_actions=example_batch["actions"],
            config=cfg,
        )
    elif args.agent_type == "TMD":

        agent = TMDAgent.create(
            seed=3141,
            ex_observations=example_batch["observations"],
            ex_actions=example_batch["actions"],
            config=cfg,
        )
    elif args.agent_type == "GCIQL":

        agent = GCIQLAgent.create(
            seed=3141,
            ex_observations=example_batch["observations"],
            ex_actions=example_batch["actions"],
            config=cfg,
        )
    elif args.agent_type == "HIQL":
        #        cfg["subgoal_steps"] = 10
        agent = HIQLAgent.create(
            seed=3141,
            ex_observations=example_batch["observations"],
            ex_actions=example_batch["actions"],
            config=cfg,
        )
    elif args.agent_type == "GCIVL":

        agent = GCIVLAgent.create(
            seed=3141,
            ex_observations=example_batch["observations"],
            ex_actions=example_batch["actions"],
            config=cfg,
        )

    _wandb_run = None

    wandb_cfg = dict(cfg)
    wandb_cfg.update(
        {
            "dataset_dir": args.dataset_dir,
            "example_obs_shape": getattr(example_batch["observations"], "shape", None),
            "example_act_shape": getattr(example_batch["actions"], "shape", None),
        }
    )
    _wandb_run = wandb.init(
        project=args.wandb_project or None,
        name=args.wandb_name or None,
        config=wandb_cfg,
    )
    save_dir = os.path.join(
        os.path.dirname(args.dataset_dir), "checkpoints", args.wandb_name
    )

    steps = args.steps
    print_every = max(1, steps // 10)

    def _to_scalar(v):
        try:
            if hasattr(v, "item"):
                return float(v.item())
            return float(np.array(v).mean())
        except Exception:
            return None

    os.makedirs(save_dir, exist_ok=True)

    for step in range(1, steps + 1):
        batch = train_dataset.sample(cfg["batch_size"])

        if step % 10 == 0:
            val_batch = val_dataset.sample(cfg["batch_size"])
        else:
            val_batch = None

        agent, info = agent.update(batch)

        if step % 10 == 0 and val_batch is not None:
            val_loss, val_info = agent.total_loss(val_batch, agent.network.params)
            for k in sorted(val_info.keys()):
                v = val_info[k]
                vv = _to_scalar(v)
                print(f"  val_{k}: {vv}")
            info.update({"val_" + k: v for k, v in val_info.items()})

        if step % print_every == 0:
            print(f"Step {step}/{steps} — info keys: {list(info.keys())}")
            for k in sorted(info.keys()):
                v = info[k]
                if hasattr(v, "item"):
                    vv = float(v.item())
                else:
                    vv = float(np.array(v).mean())
                print(f"  {k}: {vv}")

            save_file_prefix = os.path.join(save_dir, f"eval_step_{step}")
            eval_metrics = evaluate_agent(
                agent,
                obs_mean,
                obs_std,
                val_dataset=val_dataset_raw,
                num_episodes=5,
                video=True,
                save_file_prefix=save_file_prefix,
                env=val_env,
                vel_idx=vel_idx,
            )
            info["eval/fixed_success_rate"] = eval_metrics["pouring_success_rate"]
            info["eval/moving_success_rate"] = eval_metrics["moving_success_rate"]
            info["eval/validation_success_rate"] = eval_metrics[
                "validation_success_rate"
            ]

        if _wandb_run is not None:
            log_dict = {}
            for k, v in info.items():
                vv = _to_scalar(v)
                if vv is not None:
                    log_dict[k] = vv
            if log_dict:
                wandb.log(log_dict, step=step)

    print("Training finished, saving checkpoint...")

    save_agent(agent, save_dir, step)
    print(f"Saved CRL agent checkpoint to {save_dir}")

    if _wandb_run is not None:
        wandb.save(save_dir)
        print("Saved agent checkpoint to wandb")
        wandb.finish()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=str,
    )
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable logging to Weights & Biases (wandb)",
    )
    p.add_argument(
        "--wandb-project", type=str, default="kitchen", help="wandb project name"
    )
    p.add_argument("--batch-size", type=int, default=1024, help="training batch size")
    p.add_argument(
        "--alpha", type=float, default=0.3, help="Alpha parameter for the agent"
    )
    p.add_argument("--wandb-name", type=str, default=None, help="wandb run name")
    p.add_argument(
        "--agent-type", type=str, default="CRL", help="Type of agent to train"
    )
    p.add_argument("--awr", action="store_true", help="Use AWR for actor loss")
    args = p.parse_args()
    print("Args:", args)
    main(args)
