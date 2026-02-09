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
    """
    Simplified evaluation loop for Online SAC.
    """
    success_count = 0
    total_returns = []
    video_saved = False

    for i in range(num_episodes):
        obs, _ = env.reset()

        # Handle custom sim forwarding if required by specific env wrapper
        if hasattr(env.unwrapped, "sim"):
            env.unwrapped.sim.forward()
            # Re-fetch obs after forward if necessary, depends on env impl
            # obs = env.unwrapped._get_observation()

        current_frames = []
        episode_return = 0.0
        is_success = False

        for t in range(steps):
            # Deterministic action for evaluation (temperature=0.0)
            action = agent.sample_actions(
                observations=obs[None],
                temperature=0.0,
                seed=jax.random.PRNGKey(i * 10000 + t),
            )

            # Flatten action back to [Dim]
            action = np.array(action).flatten()
            action = np.clip(action, -1, 1)

            # Note: keeping minimal=True as per original script, assuming env requires it
            try:
                obs, reward, term, trunc, info = env.unwrapped.step(
                    action, minimal=True
                )
            except TypeError:
                # Fallback if env doesn't accept kwargs in step
                obs, reward, term, trunc, info = env.step(action)

            episode_return += reward

            if video and not video_saved:
                current_frames.append(env.render())

            # Check success condition (env specific, usually term=True or info['success'])
            if term or trunc:
                if term or reward > 6.0:
                    is_success = True
                break

        total_returns.append(episode_return)

        if is_success:
            success_count += 1

        # Save video logic:
        # Save the first successful episode we find.
        # If we reach the last episode and haven't saved a success yet, save that one just to see what's happening.
        if video and save_file_prefix and not video_saved:
            if is_success or (i == num_episodes - 1):
                suffix = "success" if is_success else "fail"
                save_path = f"{save_file_prefix}_{suffix}.mp4"
                imageio.mimwrite(
                    save_path, current_frames, fps=env.metadata.get("render_fps", 24)
                )
                video_saved = True

    success_rate = success_count / num_episodes
    mean_return = np.mean(total_returns)

    return {
        "success_rate": success_rate,
        "mean_return": mean_return,
    }


def main(args):
    """Main training loop for SAC agent."""
    cfg = get_sac_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["batch_size"] = args.batch_size

    # taken from ogbench
    # value_hidden_dims="(1024, 1024, 1024)" --agent.layer_norm=True --agent.min_q=False
    cfg["value_hidden_dims"] = (1024, 1024, 1024)
    cfg["layer_norm"] = True
    # cfg["min_q"] = False
    print("Training config:", cfg)

    # Initialize environments
    print("Initializing training environment...")
    env = KitchenSACOnlineEnv(render_mode="rgb_array", randomise_cup_position=True)

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

    # Buffer for video frames of the current episode
    current_episode_frames = []
    episode_idx = 0

    # Render first frame
    current_episode_frames.append(env.render())

    for step in range(1, args.train_steps + 1):
        # Sample action
        if step < args.seed_steps:
            action = env.action_space.sample()
        else:
            expl_rng, key = jax.random.split(expl_rng)
            action = agent.sample_actions(
                observations=ob[None], seed=key, temperature=1.0
            )
            action = np.array(action).flatten()

        # Step environment
        action = np.clip(action, -1.0, 1.0)
        next_ob, reward, terminated, truncated, info = env.step(action)

        # Render frame for video buffer
        current_episode_frames.append(env.render())

        # log reward in wandb
        if _wandb_run is not None:
            wandb.log({"reward": reward}, step=step)

        done = terminated or truncated
        mask = 0.0 if done else 1.0

        replay_buffer.add_transition(
            dict(
                observations=ob,
                actions=action,
                rewards=reward,
                masks=mask,
                next_observations=next_ob,
            )
        )
        ob = next_ob

        if done:
            episode_idx += 1
            expl_metrics = {
                f"exploration/{k}": np.mean(v) for k, v in flatten(info).items()
            }

            if reward > 5.0:
                video_filename = f"train_ep_{episode_idx}_success_step_{step}.mp4"
                video_path = os.path.join(save_dir, video_filename)

                imageio.mimwrite(
                    video_path,
                    current_episode_frames,
                    fps=env.metadata.get("render_fps", 24),
                )

            # Clear frames for next episode
            current_episode_frames = []

            ob, _ = env.reset()
            # Render first frame of new episode
            current_episode_frames.append(env.render())

        if replay_buffer.size < args.seed_steps:
            continue

        if step % 2 == 0:  # Ogbench does every 4
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
                agent=agent,
                env=eval_env,
                num_episodes=10,
                video=True,
                save_file_prefix=os.path.join(save_dir, f"eval_step_{step}"),
            )

            update_info["eval/success_rate"] = eval_metrics["success_rate"]
            update_info["eval/mean_return"] = eval_metrics["mean_return"]

            print(f"Eval Success Rate: {eval_metrics['success_rate']:.2f}")

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
        "--seed-steps", type=int, default=2000, help="Number of seed exploration steps"
    )
    p.add_argument(
        "--save-interval", type=int, default=100000, help="Save checkpoint interval"
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
