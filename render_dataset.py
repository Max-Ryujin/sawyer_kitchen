import argparse
import os
import sys
import numpy as np
import gymnasium as gym
import imageio
from tqdm import tqdm


def render_dataset_episodes(dataset_path, output_dir, num_episodes=5):
    # 1. Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        data = np.load(dataset_path, allow_pickle=True)
        # Handle cases where npz is loaded as a dict-like object
        qpos_data = data["qpos"]
        qvel_data = data["qvel"]
        terminals = data["terminals"]

        # Check for timeouts if available to correctly identify episode ends
        if "timeouts" in data:
            terminals = np.logical_or(terminals, data["timeouts"])
    except KeyError as e:
        print(
            f"Error: Dataset missing required key {e}. Needs 'qpos', 'qvel', and 'terminals'."
        )
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    episode_ends = np.where(terminals)[0]

    episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))

    count = min(num_episodes, len(episode_starts))
    print(f"Found {len(episode_starts)} episodes. Rendering the first {count}...")

    os.makedirs(output_dir, exist_ok=True)

    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")

    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )

    # 4. Render Loop
    for i in range(count):
        start_idx = episode_starts[i]
        end_idx = episode_ends[i]

        # Ensure valid indices
        if start_idx >= len(qpos_data) or end_idx >= len(qpos_data):
            break

        print(f"Rendering Episode {i+1} (Steps {start_idx} to {end_idx})...")

        frames = []
        env.reset(options={"minimal": True})

        for t in tqdm(range(start_idx, end_idx + 1), leave=False):
            qpos = qpos_data[t]
            qvel = qvel_data[t]

            # Set the simulation state directly
            env.unwrapped.set_state(qpos, qvel)

            if hasattr(env.unwrapped, "model") and hasattr(env.unwrapped, "data"):
                import mujoco as mj

                mj.mj_forward(env.unwrapped.model, env.unwrapped.data)
            elif hasattr(env.unwrapped, "sim"):
                env.unwrapped.sim.forward()

            frame = env.render()
            frames.append(frame)

        # Save video
        save_path = os.path.join(output_dir, f"episode_{i}_replay.mp4")
        fps = env.metadata.get("render_fps", 24)
        imageio.mimwrite(save_path, frames, fps=fps)
        print(f"Saved: {save_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render episodes from a recorded .npz dataset."
    )
    parser.add_argument("dataset", type=str, help="Path to the .npz dataset file")
    parser.add_argument(
        "--out", type=str, default="rendered_episodes", help="Directory to save videos"
    )
    parser.add_argument("--n", type=int, default=5, help="Number of episodes to render")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"File not found: {args.dataset}")
    else:
        render_dataset_episodes(args.dataset, args.out, args.n)
