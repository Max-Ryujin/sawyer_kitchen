"""
Minimal training script that wires OGBench's CRL agent to the local kitchen trajectories.
"""

import os
import sys
import argparse
import pickle
import jax
import numpy as np
import gymnasium as gym
import imageio

THIS_DIR = os.path.dirname(__file__)
OG_IMPLS = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench", "impls"))
print("Adding OGBench impls to sys.path:", OG_IMPLS)
sys.path.insert(0, OG_IMPLS)

from agents.crl import CRLAgent, get_config
from utils.flax_utils import save_agent
from crl_dataset import CRLDataset
import wandb


def evaluate_agent(agent, num_episodes=10, steps=4000, video=False, save_file=None):

    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    obs, _ = env.reset(options={"randomise_cup_position": True, "minimal": True})

    success_count = 0
    frames = []
    for i in range(num_episodes):
        obs, _ = env.reset(options={"randomise_cup_position": True, "minimal": True})

        for t in range(steps):
            obs_arr = np.asarray(obs)
            goal_arr = env.unwrapped.create_goal_state(current_state=obs_arr)

            action = agent.sample_actions(
                observations=obs_arr,
                goals=goal_arr,
                temperature=0.0,
                seed=jax.random.PRNGKey(0),
            )

            obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
            if i == 0 and video:
                frames.append(env.render())
            if term or trunc:
                # count as success
                success_count += 1
                break
        env.close()
        if video and i == 0:
            imageio.mimwrite(save_file, frames, fps=env.metadata.get("render_fps", 24))

    success_rate = success_count / num_episodes
    print(f"Evaluation over {num_episodes} episodes: Success rate = {success_rate}")
    return success_rate


def main(args):
    ds = CRLDataset(args.dataset_dir)
    ex_obs, ex_act = ds.example()

    cfg = get_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["alpha"] = 0.03

    print(
        "Creating CRL agent with config:",
        {
            k: cfg[k]
            for k in ["lr", "batch_size", "latent_dim", "actor_loss"]
            if k in cfg
        },
    )

    agent = CRLAgent.create(
        seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=cfg
    )

    _wandb_run = None

    wandb_cfg = dict(cfg)
    wandb_cfg.update(
        {
            "dataset_dir": args.dataset_dir,
            "example_obs_shape": getattr(ex_obs, "shape", None),
            "example_act_shape": getattr(ex_act, "shape", None),
        }
    )
    _wandb_run = wandb.init(
        project=args.wandb_project or None,
        name=args.wandb_name or None,
        config=wandb_cfg,
        reinit=True,
    )
    print(
        f"wandb run initialized: project={args.wandb_project}, name={args.wandb_name}"
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

    for step in range(1, steps + 1):
        batch = ds.sample_batch(batch_size=cfg["batch_size"])

        if ds.has_validation() and step % 10 == 0:
            val_batch = ds.sample_val_batch(batch_size=cfg["batch_size"])
        else:
            val_batch = None

        agent, info = agent.update(batch)

        if step % print_every == 0 or step == 1:
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
            save_file = os.path.join(save_dir, f"eval.{step}.mp4")
            success_rate = evaluate_agent(
                agent, num_episodes=10, video=True, save_file=save_file
            )
            info["eval/success_rate"] = success_rate

        if _wandb_run is not None:
            log_dict = {}
            for k, v in info.items():
                vv = _to_scalar(v)
                if vv is not None:
                    log_dict[k] = vv
            if log_dict:
                wandb.log(log_dict, step=step)

    print("Training finished, saving checkpoint...")

    os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "agent.pkl")
    # change to use the wandb name as file name

    save_agent(agent, save_dir, step)
    print(f"Saved CRL agent checkpoint to {save_dir}")

    if _wandb_run is not None:
        wandb.save(save_dir)
        print("Saved agent checkpoint to wandb")
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=str,
    )
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable logging to Weights & Biases (wandb)",
    )
    p.add_argument(
        "--wandb-project", type=str, default="kitchen", help="wandb project name"
    )
    p.add_argument("--wandb-name", type=str, default=None, help="wandb run name")
    args = p.parse_args()
    print("Args:", args)
    main(args)
