"""
Minimal training script for SAC agent on the kitchen environment.
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
from agents.sac import SACAgent, get_config as get_sac_config

# --- Utils ---
from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

# --- Local Imports ---
from SACEnv import KitchenSACOnlineEnv


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def evaluate_agent(
    agent,
    num_episodes=5,
    steps=1500,
    video=False,
    save_file_prefix=None,
    env=None,
):

    if env is None:
        env = gym.make(
            "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
        )

    fixed_success_count = 0
    fixed_frames = []
    fixed_success_frames_list = []

    moving_success_count = 0
    moving_frames = []
    moving_success_frames_list = []

    for i in range(num_episodes):
        obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})

        current_frames = []
        is_success = False

        for t in range(steps):

            action = agent.sample_actions(
                observations=obs[None],
                temperature=0.0,
                seed=jax.random.PRNGKey(i * 10000 + t),
            )

            # Flatten action back to [Dim]
            action = np.array(action).flatten()
            action = np.clip(action, -1, 1)
            obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)

            if video:
                current_frames.append(env.render())

            if env.unwrapped.check_moving_success_without_goal():
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

    rand_success_count = 0
    val_test_frames = None

    for i in range(2 * num_episodes):

        obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
        if hasattr(env.unwrapped, "sim"):
            env.unwrapped.sim.forward()
        obs = env.unwrapped._get_observation(minimal=True)

        current_frames = []
        is_success = False

        for t in range(steps):

            action = agent.sample_actions(
                observations=obs[None],
                temperature=0.0,
                seed=jax.random.PRNGKey(i * 10000 + t),
            )

            # Flatten action back to [Dim]
            action = np.array(action).flatten()
            action = np.clip(action, -1, 1)

            obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)

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
            save_path = f"{save_file_prefix}_val_ep{i}_success.mp4"
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
        "moving_success_rate": moving_success_rate,
        "validation_success_rate": rand_success_rate,
    }


def main(args):
    """Main training loop for SAC agent."""
    cfg = get_sac_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["batch_size"] = args.batch_size

    print("Training config:", cfg)

    # Initialize environments
    print("Initializing training environment...")
    env = KitchenSACOnlineEnv(render_mode="rgb_array")

    print("Initializing evaluation environment...")
    eval_env = KitchenSACOnlineEnv(render_mode="rgb_array")

    # Example transition for replay buffer initialization
    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=int(1e6))

    # Initialize agent
    print("Creating SAC agent...")
    np.random.seed(args.seed)
    agent = SACAgent.create(
        seed=args.seed,
        ex_observations=example_transition["observations"],
        ex_actions=example_transition["actions"],
        config=cfg,
    )

    # Set up wandb
    _wandb_run = None
    wandb_cfg = dict(cfg)
    wandb_cfg.update(
        {
            "seed": args.seed,
            "example_obs_shape": getattr(
                example_transition["observations"], "shape", None
            ),
            "example_act_shape": getattr(example_transition["actions"], "shape", None),
        }
    )
    _wandb_run = wandb.init(
        project=args.wandb_project or None,
        name=args.wandb_name or None,
        config=wandb_cfg,
    )

    save_dir = (
        os.path.join(args.save_dir, args.wandb_name)
        if args.wandb_name
        else args.save_dir
    )
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    expl_rng = jax.random.PRNGKey(args.seed)
    ob, _ = env.reset()

    print(f"Starting training for {args.train_steps} steps...")

    def _to_scalar(v):
        try:
            return float(v)
        except Exception:
            return v

    for step in range(1, args.train_steps + 1):
        # Sample action
        if step < args.seed_steps:
            action = env.action_space.sample()
        else:
            expl_rng, key = jax.random.split(expl_rng)
            action = agent.sample_actions(
                observations=ob[None], seed=key, temperature=0.0
            )
            action = np.array(action).flatten()

        # Step environment
        action = np.clip(action, -1.0, 1.0)
        next_ob, reward, terminated, truncated, info = env.step(action)

        replay_buffer.add_transition(
            dict(
                observations=ob,
                actions=action,
                rewards=reward,
                masks=float(not terminated),
                next_observations=next_ob,
            )
        )
        ob = next_ob

        if terminated or truncated:
            expl_metrics = {
                f"exploration/{k}": np.mean(v) for k, v in flatten(info).items()
            }
            ob, _ = env.reset()

        if replay_buffer.size < args.seed_steps:
            continue

        batch = replay_buffer.sample(cfg["batch_size"])
        agent, update_info = agent.update(batch)

        if step % 10 == 0 and _wandb_run is not None:
            log_dict = {f"training/{k}": _to_scalar(v) for k, v in update_info.items()}
            wandb.log(log_dict, step=step)

        if step % args.save_interval == 0:
            print(f"Saving agent at step {step}...")
            save_agent(agent, save_dir, step)

        if step % max(1, args.train_steps // 10) == 0:
            print(f"Step {step}/{args.train_steps}")
            eval_metrics = evaluate_agent(
                agent,
                num_episodes=5,
                video=True,
                save_file_prefix=os.path.join(save_dir, f"eval_step_{step}"),
                env=eval_env,
            )
            update_info["eval/moving_success_rate"] = eval_metrics[
                "moving_success_rate"
            ]
            update_info["eval/validation_success_rate"] = eval_metrics[
                "validation_success_rate"
            ]

        if _wandb_run is not None:
            log_dict = {}
            for k, v in update_info.items():
                vv = _to_scalar(v)
                if vv is not None:
                    log_dict[k] = vv
            if log_dict:
                wandb.log(log_dict, step=step)

    print("Training finished, saving final checkpoint...")
    save_agent(agent, save_dir, args.train_steps)

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
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--train-steps", type=int, default=100000, help="Number of training steps"
    )
    p.add_argument(
        "--seed-steps", type=int, default=1000, help="Number of seed exploration steps"
    )
    p.add_argument(
        "--save-interval", type=int, default=10000, help="Save checkpoint interval"
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="tmp/sac_checkpoints",
        help="Directory to save checkpoints",
    )
    p.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    p.add_argument(
        "--wandb-project", type=str, default="kitchen", help="wandb project name"
    )
    p.add_argument("--wandb-name", type=str, default=None, help="wandb run name")

    args = p.parse_args()
    print("Args:", args)
    main(args)
