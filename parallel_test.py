import os
import json
import imageio
import numpy as np
from collections import defaultdict
import gymnasium as gym
import jax
import mujoco as mj
import kitchen_utils as utils
from joblib import Parallel, delayed
from policies import moving_policy, pour_policy_v2


class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.155, sigma=0.0055):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def reset(self):
        self.state = np.copy(self.mu)


def collect_policy_episode(
    save_path="tmp/policy.mp4",
    steps=1000,
    noise=True,
    random_action=False,
    policy_type="pouring",
):
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=640, height=480
    )
    obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
    frames = []
    env._automaton_state = "move_above"
    env._state_counter = 0
    # cup = np.random.choice(np.array([0, 1]))
    cup = 0
    for t in range(steps):
        if policy_type == "moving":
            action = moving_policy(env, obs, cup_number=cup)
            if env._automaton_state == "done":
                env._automaton_state = "move_above"
                cup = np.random.choice(np.array([0, 1]))
        else:
            action = pour_policy_v2(env, obs)

        if random_action:
            if np.random.rand() < 0.02:
                action = env.action_space.sample()
        if noise:
            action = action + np.random.normal(0, 0.01, action.shape)
        obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
        frames.append(env.render())
        if term or trunc:
            print(f"Episode finished after {t+1} steps.")
            break

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
    print(f"Saved test-policy video to {save_path}")


def run_single_episode(
    seed,
    max_steps,
    width,
    height,
    noise,
    pixel_observations,
    minimal_observations,
    save_failed_episodes,
    pouring_prob,
):
    """
    Worker function for parallel data collection.
    Runs one episode and returns the trajectory data.
    """
    # Re-import inside worker to ensure clean state
    import gymnasium as gym
    import numpy as np
    from collections import defaultdict

    # Register the environment inside the worker process
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")

    if pixel_observations:
        env = gym.make(
            "KitchenMinimalEnv-v0",
            render_mode="rgb_array",
            width=width,
            height=height,
            ob_type="pixels",
        )
    else:
        env = gym.make(
            "KitchenMinimalEnv-v0", render_mode="rgb_array", width=width, height=height
        )

    # Seed the environment
    obs, _ = env.reset(
        seed=seed, options={"randomise_cup_position": True, "minimal": True}
    )

    env._noise_generator = OUNoise(size=3)

    # Initialize state tracking variables locally
    env._automaton_state = "move_above"
    env._state_counter = 0

    episode_data = defaultdict(list)
    success = False
    failure_reason = "max_steps"

    move_operations = np.random.randint(0, 3)
    if move_operations == 0:
        perform_pouring = True
    else:
        perform_pouring = np.random.rand() < pouring_prob

    moves_completed = 0
    policy_mode = "moving" if move_operations > 0 else "pouring"
    # For testing:
    policy_mode = "moving"  # For testing moving only
    done2 = False
    # cup = np.random.choice(np.array([0, 1]))
    # I am fixing the cup and remove the other cup from the observation data to prevent the critic from cheating
    cup = 0
    steps_run = 0

    # --- State timing instrumentation ---
    last_state = getattr(env, "_automaton_state", None)
    state_step_counter = 0
    state_stats_local = defaultdict(lambda: {"total_steps": 0, "count": 0})
    # ------------------------------------

    for t in range(max_steps):
        action = None
        # --- POLICY LOGIC ---
        if policy_mode == "moving":
            action = moving_policy(env, obs, cup_number=cup)
            if env._automaton_state == "done":
                moves_completed += 1
                if moves_completed == move_operations:
                    if perform_pouring:
                        #   policy_mode = "pouring"  (For testing)
                        # THis is just for moving only
                        done2 = True
                    else:
                        done2 = True
                else:
                    # cup = np.random.choice(np.array([0, 1]))
                    cup = 0
                env._automaton_state = "move_above"
        elif policy_mode == "pouring":
            action = pour_policy_v2(env, obs)
        # --------------------

        if noise:
            action = action + np.random.normal(0, 0.01, action.shape)

        episode_data["qpos"].append(env.unwrapped.data.qpos.copy())
        episode_data["qvel"].append(env.unwrapped.data.qvel.copy())

        obs_to_store = env.unwrapped._get_observation(minimal=True)
        obs_next, reward, terminated, trunc, info = env.unwrapped.step(
            action, minimal=True
        )
        done = terminated or trunc or done2

        if minimal_observations:
            episode_data["observations"].append(obs_to_store)
        else:
            episode_data["observations"].append(obs)

        episode_data["actions"].append(action)
        episode_data["terminals"].append(done)

        obs = obs_next
        steps_run += 1

        # --- State timing update ---
        state_step_counter += 1
        current_state = getattr(env, "_automaton_state", None)
        if current_state != last_state:
            # print how many steps last_state took
            print(f"State '{last_state}' took {state_step_counter} steps")
            # record stats
            state_stats_local[last_state]["total_steps"] += state_step_counter
            state_stats_local[last_state]["count"] += 1
            state_step_counter = 0
            last_state = current_state
        # ----------------------------

        if done:
            # finalize last state's counter
            if state_step_counter > 0 and last_state is not None:
                print(f"State '{last_state}' took {state_step_counter} steps")
                state_stats_local[last_state]["total_steps"] += state_step_counter
                state_stats_local[last_state]["count"] += 1
                state_step_counter = 0
            # Only mark as successful if terminated or trunc from environment
            if terminated or trunc or done2:
                success = True
                failure_reason = None
            env._noise_generator.reset()
            break

        if t == max_steps - 1:
            failure_reason = getattr(env, "_automaton_state", "unknown")
            # finalize last state's counter even on timeout
            if state_step_counter > 0 and last_state is not None:
                print(f"State '{last_state}' took {state_step_counter} steps (end)")
                state_stats_local[last_state]["total_steps"] += state_step_counter
                state_stats_local[last_state]["count"] += 1
                state_step_counter = 0
                env._noise_generator.reset()

    env.close()

    # Return structure: (success, failure_reason, data_dict, steps_count, state_stats)
    # Convert defaultdict to plain dict for serialization
    state_stats_local = dict(state_stats_local)

    if success or save_failed_episodes:
        if not success and save_failed_episodes:
            # Mark last terminal as True if forcing save
            episode_data["terminals"][-1] = True
        return (success, failure_reason, episode_data, steps_run, state_stats_local)
    else:
        return (success, failure_reason, None, steps_run, state_stats_local)


def collect_moving_policy_dataset(
    save_root: str = "tmp/policy_dataset",
    episodes: int = 100,
    max_steps: int = 1900,
    width: int = 320,
    height: int = 240,
    noise: bool = True,
    pixel_observations: bool = False,
    random_action: bool = False,
    minimal_observations: bool = True,
    save_failed_episodes: bool = False,
    pouring_prob: float = 0.9,
):
    """
    Main entry point for parallel dataset collection.

    Orchestrates the parallel execution of `run_single_episode`, aggregates
    the data, splits it into Train/Val sets, and saves the results to .npz files.

    Args:
        save_root: Directory to save output files.
        episodes: Number of episodes.
        max_steps: Maximum steps per episode.
        width: Render width.
        height: Render height.
        noise: Add action noise.
        pixel_observations: Capture pixel inputs.
        random_action: Inject random actions (never actually used).
        minimal_observations: Use low-dim observations. (This is old and is always true. Full observation was never used)
        save_failed_episodes: Keep data from episodes that do not reach goal.
        pouring_prob: Probability of executing a pour after moving cups. This is to control the data distribution.
    """
    os.makedirs(save_root, exist_ok=True)

    num_val_attempts = max(1, episodes // 10)
    total_episodes_to_run = episodes + num_val_attempts

    print(f"Starting parallel collection of {total_episodes_to_run} episodes...")
    if pixel_observations:
        print("Ensure 'export MUJOCO_GL=egl' is set for GPU rendering.")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_episode)(
            seed=i,
            max_steps=max_steps,
            width=width,
            height=height,
            noise=noise,
            pixel_observations=pixel_observations,
            minimal_observations=minimal_observations,
            save_failed_episodes=save_failed_episodes,
            pouring_prob=pouring_prob,
        )
        for i in range(total_episodes_to_run)
    )

    # --- Aggregation Logic ---
    dataset = defaultdict(list)
    success_count = 0
    failure_counts = defaultdict(int)

    # Aggregate failure counts from ALL results first
    for res in results:
        is_success, fail_reason, data, steps, state_stats = res
        if is_success:
            success_count += 1
        else:
            failure_counts[str(fail_reason)] += 1

    # Filter only valid results (non-None) for dataset
    valid_results = [r for r in results if r[2] is not None]
    num_valid = len(valid_results)

    num_val = max(1, int(num_valid * 0.1))
    num_train = num_valid - num_val

    print(
        f"Collected {num_valid} valid episodes. Splitting: {num_train} Train, {num_val} Val."
    )

    total_train_steps = 0

    # Aggregate data from valid results only

    global_state_stats = defaultdict(lambda: {"total_steps": 0, "count": 0})

    for i, res in enumerate(valid_results):
        is_success, fail_reason, data, steps, state_stats = res
        # Note: is_success already counted above, skip re-counting

        # accumulate per-state stats
        for st, vals in state_stats.items():
            # vals is {'total_steps':..., 'count':...}
            global_state_stats[st]["total_steps"] += vals.get("total_steps", 0)
            global_state_stats[st]["count"] += vals.get("count", 0)

        # Calculate split index based on step count
        if i < num_train:
            total_train_steps += steps

        for k, v in data.items():
            dataset[k].extend(v)

    train_dataset = {}
    val_dataset = {}
    train_path = os.path.join(save_root, "train_dataset.npz")
    val_path = os.path.join(save_root, "val_dataset.npz")

    actual_total_len = len(dataset["actions"])
    split_idx = min(total_train_steps, actual_total_len)

    for k, v in dataset.items():
        arr = np.array(v)

        # Optimize types
        if (
            "observations" in k
            and arr.dtype == np.float64
            and arr.max() > 1.0
            and pixel_observations
        ):
            arr = arr.astype(np.uint8)
        elif k == "terminals":
            arr = arr.astype(bool)
        elif arr.dtype == np.float64:
            arr = arr.astype(np.float32)

        train_dataset[k] = arr[:split_idx]
        val_dataset[k] = arr[split_idx:]

    for path, dset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dset)

    total_steps_all = sum([res[3] for res in results if res[2] is not None])
    avg_steps_per_valid_episode = (
        float(total_steps_all) / num_valid if num_valid > 0 else 0.0
    )
    avg_steps_per_attempted_episode = (
        float(total_steps_all) / total_episodes_to_run
        if total_episodes_to_run > 0
        else 0.0
    )

    state_timing_summary = {}
    for st, vals in global_state_stats.items():
        cnt = vals["count"]
        tot = vals["total_steps"]
        avg = float(tot) / cnt if cnt > 0 else 0.0
        state_timing_summary[st] = {"avg_steps": avg, "total_steps": tot, "count": cnt}

    stats = {
        "total_episodes_attempted": total_episodes_to_run,
        "valid_episodes_collected": num_valid,
        "successful_episodes": success_count,
        "success_rate": (
            float(success_count) / total_episodes_to_run
            if total_episodes_to_run > 0
            else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "split": {"train": num_train, "val": num_val},
        "state_timings": state_timing_summary,
        "avg_steps_per_valid_episode": avg_steps_per_valid_episode,
        "avg_steps_per_attempted_episode": avg_steps_per_attempted_episode,
        "failed_episodes": num_valid - success_count,
    }
    stats_path = os.path.join(save_root, "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved dataset to {save_root}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["policy", "dataset"], default="policy")
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument(
        "--save_failed_episodes",
        action="store_true",
        help="When collecting dataset, save all episodes including failed ones",
    )
    parser.add_argument(
        "--minimal", action="store_true", help="Use minimal observations"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect when using dataset mode",
    )
    parser.add_argument(
        "--pixel_observations",
        action="store_true",
        help="Use pixel observations when collecting dataset",
    )
    parser.add_argument(
        "--pouring_prob",
        type=float,
        default=0.9,
        help="Probability of pouring at the end of moving sequence in dataset mode",
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="pouring",
        choices=["pouring", "moving"],
        help="Policy type to run in policy mode",
    )
    args = parser.parse_args()

    if args.mode == "policy":
        collect_policy_episode(steps=args.steps, policy_type=args.policy_type)
    elif args.mode == "dataset":
        collect_moving_policy_dataset(
            save_root=args.out,
            episodes=args.episodes,
            max_steps=args.steps,
            width=320,
            height=240,
            noise=True,
            pixel_observations=args.pixel_observations,
            minimal_observations=args.minimal,
            save_failed_episodes=args.save_failed_episodes,
            pouring_prob=args.pouring_prob,
        )
