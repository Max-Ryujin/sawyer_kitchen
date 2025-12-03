"""
Minimal training script that wires OGBench's CRL agent to the local kitchen trajectories.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import argparse
import pickle
import jax
import numpy as np
import gymnasium as gym
import imageio
import multiprocessing as mp

THIS_DIR = os.path.dirname(__file__)
OG_IMPLS = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench", "impls"))
OG_IMPLS_BASE = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench"))
sys.path.insert(0, OG_IMPLS)
sys.path.insert(0, OG_IMPLS_BASE)

from agents.crl import CRLAgent, get_config
from agents.qrl import QRLAgent
from agents.qrl import get_config as get_qrl_config
from utils.flax_utils import save_agent
from utils.datasets import GCDataset, Dataset
from ogbench import load_dataset
import wandb


def normalize(x, mean, std, eps=1e-5):
    return (x - mean) / (std + eps)


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
):
    if env is None:
        env = gym.make(
            "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
        )

    fixed_success_count = 0
    fixed_frames = []

    for i in range(num_episodes):
        obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
        raw_obs = np.asarray(obs)

        goal_arr = env.unwrapped.create_goal_state(
            current_state=raw_obs, minimal=True, fixed_goal=True
        )
        normalized_goal = normalize(goal_arr, obs_mean, obs_std)

        current_frames = []
        is_success = False

        for t in range(steps):
            normalized_obs = normalize(raw_obs, obs_mean, obs_std)
            action = agent.sample_actions(
                observations=normalized_obs,
                goals=normalized_goal,
                temperature=0.0,
                seed=jax.random.PRNGKey(0),
            )
            action = np.clip(action, -1, 1)
            obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
            raw_obs = np.asarray(obs)

            if i == 0 and video:
                current_frames.append(env.render())

            if term or trunc:
                fixed_success_count += 1
                is_success = True
                break

        if i == 0 and video:
            fixed_frames = current_frames

    if video and save_file_prefix:
        imageio.mimwrite(
            f"{save_file_prefix}_fixed.mp4",
            fixed_frames,
            fps=env.metadata.get("render_fps", 24),
        )

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

    for i in range(num_episodes):
        ep_idx = valid_indices[i]

        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]

        qpos = val_dataset["qpos"][start_idx]
        qvel = val_dataset["qvel"][start_idx]
        goal_arr = val_dataset["observations"][end_idx]

        normalized_goal = normalize(goal_arr, obs_mean, obs_std)

        obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
        env.unwrapped.set_state(qpos, qvel)
        if hasattr(env.unwrapped, "sim"):
            env.unwrapped.sim.forward()
        obs = env.unwrapped._get_observation(minimal=True)
        raw_obs = np.asarray(obs)

        current_frames = []
        is_success = False

        for t in range(steps):
            normalized_obs = normalize(raw_obs, obs_mean, obs_std)

            action = agent.sample_actions(
                observations=normalized_obs,
                goals=normalized_goal,
                temperature=0.0,
                seed=jax.random.PRNGKey(i * 1000 + t),
            )
            action = np.clip(action, -1, 1)

            obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
            raw_obs = np.asarray(obs)

            if term or trunc:
                rand_success_count += 1
                is_success = True
                break

            if video:
                current_frames.append(env.render())

        # Save video only if successful
        if video and (is_success or i == 0) and save_file_prefix:
            save_path = f"{save_file_prefix}_val_ep{ep_idx}_success.mp4"
            imageio.mimwrite(
                save_path, current_frames, fps=env.metadata.get("render_fps", 24)
            )

    rand_success_rate = rand_success_count / num_episodes

    print(
        f"Fixed Success Rate: {fixed_success_rate:.2f} | Val Set Success Rate: {rand_success_rate:.2f}"
    )

    return {
        "fixed_success_rate": fixed_success_rate,
        "validation_success_rate": rand_success_rate,
    }


def main(args):
    if args.agent_type == "CRL":
        cfg = get_config()
    elif args.agent_type == "QRL":
        cfg = get_qrl_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["batch_size"] = args.batch_size
    cfg["alpha"] = args.alpha
    # cfg["actor_loss"] = "awr"
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    print("Initializing validation environment...")
    val_env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    print("Training config:", cfg)
    train_path = os.path.join(args.dataset_dir, "train_dataset.npz")
    val_path = os.path.join(args.dataset_dir, "val_dataset.npz")

    train_dataset_raw = load_dataset(train_path, compact_dataset=True)

    obs_data = train_dataset_raw["observations"]
    obs_mean = np.mean(obs_data, axis=0)
    obs_std = np.std(obs_data, axis=0)

    obs_std[obs_std < 1e-2] = 1.0

    train_dataset_norm = dict(train_dataset_raw)
    train_dataset_norm["observations"] = normalize(
        train_dataset_raw["observations"], obs_mean, obs_std
    )

    if "next_observations" in train_dataset_norm:
        train_dataset_norm["next_observations"] = normalize(
            train_dataset_raw["next_observations"], obs_mean, obs_std
        )

    val_dataset_raw = load_dataset(val_path, compact_dataset=True, add_info=True)
    val_dataset_norm = dict(val_dataset_raw)
    val_dataset_norm["observations"] = normalize(
        val_dataset_raw["observations"], obs_mean, obs_std
    )

    base_train = Dataset.create(**train_dataset_norm)
    train_dataset = GCDataset(base_train, cfg)

    base_val = Dataset.create(**val_dataset_norm)
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

        if step % print_every == 0:
            print(f"Step {step}/{steps} — info keys: {list(info.keys())}")
            for k in sorted(info.keys()):
                v = info[k]
                if hasattr(v, "item"):
                    vv = float(v.item())
                else:
                    vv = float(np.array(v).mean())
                print(f"  {k}: {vv}")

            if val_batch is not None:
                val_loss, val_info = agent.total_loss(val_batch, agent.network.params)
                print(f"  Validation info keys: {list(val_info.keys())}")

                for k in sorted(val_info.keys()):
                    v = val_info[k]
                    vv = _to_scalar(v)
                    print(f"  val_{k}: {vv}")
                info.update({"val_" + k: v for k, v in val_info.items()})

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
            )
            info["eval/fixed_success_rate"] = eval_metrics["fixed_success_rate"]
            info["eval/random_success_rate"] = eval_metrics["random_success_rate"]

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
    p.add_argument("--steps", type=int, default=10000)
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
    args = p.parse_args()
    print("Args:", args)
    main(args)
