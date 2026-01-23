import os
import sys
import json
from xml.parsers.expat import model
import imageio
import pickle
from tqdm import trange
import numpy as np
from collections import defaultdict
import gymnasium as gym
import jax
import mujoco as mj
import kitchen_utils as utils
from policies import moving_policy, pour_policy_v2


MOVING_GOAL_OBS = [
    2.159642457962036,
    -0.10995837301015854,
    1.5188195705413818,
    -1.471531629562378,
    1.076535701751709,
    -1.3156180381774902,
    -0.697909951210022,
    0.009322389028966427,
    0.009459411725401878,
    0.07799883186817169,
    0.2097632884979248,
    0.24370022118091583,
    -0.2910488545894623,
    0.14127174019813538,
    0.1319103240966797,
    0.25795871019363403,
    -0.05900793522596359,
    -0.059946853667497635,
    -0.5267668962478638,
    -0.9562153816223145,
    1.5801501274108887,
    0.9984312057495117,
    -0.03856143355369568,
    0.03994974493980408,
    -0.007226120680570602,
    -0.8000472187995911,
    -1.1000237464904785,
    1.5889480113983154,
    0.9999995231628418,
    8.086483285296708e-05,
    -0.0009184352238662541,
    0.00028437477885745466,
    -0.8119636178016663,
    -1.1021802425384521,
    1.6126853227615356,
    0.7789384722709656,
    0.6046792268753052,
    -0.013668391853570938,
    -0.16562329232692719,
    -0.7986109256744385,
    -1.1152387857437134,
    1.6127070188522339,
    0.5592268705368042,
    0.5389202833175659,
    -0.627842366695404,
    -0.05142202600836754,
    -0.7848583459854126,
    -1.101852297782898,
    1.6127344369888306,
    0.5282660126686096,
    0.4182727038860321,
    0.7361798286437988,
    0.0634213536977768,
    -0.8054980039596558,
    -1.0851140022277832,
    1.6127002239227295,
    0.4111011028289795,
    -0.5848034620285034,
    -0.6529611349105835,
    0.25028496980667114,
    -0.8140132427215576,
    -1.091016411781311,
    1.6126837730407715,
    0.9921442270278931,
    -0.10417235642671585,
    0.0673166811466217,
    0.016321102157235146,
    -0.8085049986839294,
    -1.1137375831604004,
    1.61268949508667,
    -0.38063672184944153,
    0.7345278263092041,
    -0.39539316296577454,
    -0.39905986189842224,
    -0.7901497483253479,
    -1.0851787328720093,
    1.6127279996871948,
    -0.14813897013664246,
    -0.9648413062095642,
    0.21688757836818695,
    -0.00979399774223566,
    -0.7937706112861633,
    -1.1064852476119995,
    1.6127175092697144,
    -0.03492381051182747,
    0.7362642288208008,
    0.599088191986084,
    -0.3127117156982422,
    -0.7922301292419434,
    -1.0949842929840088,
    1.612722396850586,
    0.7833766937255859,
    -0.35952121019363403,
    0.5008659958839417,
    -0.0787319466471672,
    -0.7862828373908997,
    -1.1147964000701904,
    1.6127294301986694,
    0.9410353899002075,
    0.23408016562461853,
    0.24400077760219574,
    0.011068851687014103,
    0.016990942880511284,
    0.018744247034192085,
    0.08126810193061829,
    1.2619215250015259,
    -1.1199562549591064,
    0.007560721132904291,
    8.360292122233659e-05,
    -0.0003339182585477829,
    0.0005906213191337883,
    -0.023027963936328888,
    -0.010386912152171135,
    0.0009183932561427355,
]


class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.005):  # Reduced sigma slightly
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return x

    def reset(self):
        self.state = np.copy(self.mu)


def collect_policy_episode(
    save_path="tmp/policy.mp4",
    steps=1000,
    noise=False,
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
    noise = False
    env._state_counter = 0
    env._noise_generator = OUNoise(3)
    cup = 0
    for t in range(steps):
        if policy_type == "moving":
            action = moving_policy(env, obs, cup_number=cup)
            if env._automaton_state == "done":
                # log full qpos and qvel and obs
                print("Full qpos:", env.unwrapped.data.qpos)
                print("Full qvel:", env.unwrapped.data.qvel)
                print("Full obs: ", obs)
                # env._automaton_state = "move_above"
                # cup = np.random.choice(np.array([0, 1]))
                break
        else:
            action = pour_policy_v2(env, obs)

        obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
        frames.append(env.render())
        if term or trunc:
            print("Full qpos:", env.unwrapped.data.qpos)
            print("Full qvel:", env.unwrapped.data.qvel)
            print("Full obs: ", obs)
            print(f"Episode finished after {t+1} steps.")
            break

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
    print(f"Saved test-policy video to {save_path}")


def collect_crl_episode(
    save_path="tmp/crl_policy.mp4",
    steps=800,
    checkpoint_path=None,
    policy_type="pouring",
):
    """Run a single episode using a trained CRL agent."""
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for CRL mode")

    # Add OGBench implementations to path
    THIS_DIR = os.path.dirname(__file__)
    OG_IMPLS = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "impls"))
    sys.path.insert(0, OG_IMPLS)
    from agents.crl import CRLAgent, get_config
    from agents.qrl import QRLAgent, get_config as get_qrl_config
    from agents.tmd import TMDAgent, get_config as get_tmd_config
    from utils.flax_utils import restore_agent

    OG_IMPLS_BASE = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench"))
    sys.path.insert(0, OG_IMPLS_BASE)
    from ogbench import load_dataset

    def normalize(x, mean, std, eps=1e-5):
        return (x - mean) / (std + eps)

    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    obs, _ = env.reset(options={"randomise_cup_position": False, "minimal": True})
    frames = []

    train_path = os.path.dirname(checkpoint_path)
    two_levels_up = os.path.dirname(os.path.dirname(train_path))
    train_path = os.path.join(two_levels_up, "train_dataset.npz")

    train_dataset_raw = load_dataset(train_path, compact_dataset=True)

    obs_data = train_dataset_raw["observations"]
    obs_mean = np.mean(obs_data, axis=0)
    obs_std = np.std(obs_data, axis=0)

    cfg = get_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["alpha"] = 0.1
    cfg["actor_loss"] = "awr"

    agent_tmp = CRLAgent.create(
        seed=0, ex_observations=obs, ex_actions=env.action_space.sample(), config=cfg
    )

    agent = restore_agent(agent_tmp, checkpoint_path, 100000)
    print(f"Loaded checkpoint from {checkpoint_path}")
    obs_arr = np.asarray(obs)
    if policy_type == "pouring":
        goal_arr = env.unwrapped.create_goal_state(
            current_state=obs_arr, minimal=True, fixed_goal=True
        )
    else:
        goal_arr = env.unwrapped.create_moving_goal_state(
            current_state=obs_arr, minimal=True, fixed_goal=True
        )
    normalized_goal = normalize(goal_arr, obs_mean, obs_std)
    for t in range(steps):
        normalized_obs = normalize(obs_arr, obs_mean, obs_std)

        action = agent.sample_actions(
            observations=normalized_obs,
            goals=normalized_goal,
            temperature=0.0,
            seed=jax.random.PRNGKey(0),
        )
        action = np.clip(action, -1.0, 1.0)
        obs, _, term, trunc, _ = env.unwrapped.step(action, minimal=True)
        obs_arr = np.asarray(obs)
        frames.append(env.render())
        if policy_type == "moving":
            if env.unwrapped.check_moving_success(goal_arr):
                print(f"Moving task successful after {t+1} steps.")
                break
        else:
            if term or trunc:
                print(f"Episode finished after {t+1} steps.")
                break

    env.close()
    for i, f in enumerate(frames):
        if f is None:
            print(f"Frame {i} is None")
        elif f.shape != frames[0].shape:
            print(f"Frame {i} has different shape: {f.shape}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
    print(f"Saved CRL agent video to {save_path}")


def collect_policy_dataset(
    save_root: str = "tmp/policy_dataset",
    episodes: int = 100,
    max_steps: int = 1100,
    width: int = 320,
    height: int = 240,
    noise: bool = True,
    pixel_observations: bool = False,
    random_action: bool = False,
    minimal_observations: bool = True,
    save_failed_episodes: bool = False,
):
    """Run the policy multiple times and save trajectories.

    Args:
        save_failed_episodes: If True, saves all episodes. If False, only saves
                              successful (terminated==True) episodes.
    """
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
    dataset = defaultdict(list)
    os.makedirs(save_root, exist_ok=True)

    success_count = 0
    failure_counts = defaultdict(int)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = episodes
    num_val_episodes = episodes // 10
    env._noise_generator = OUNoise(3)
    debug_data = defaultdict(list)
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        obs, _ = env.reset(options={"randomise_cup_position": True, "minimal": True})
        env._automaton_state = "move_above"
        env._state_counter = 0
        env._noise_generator.reset()
        episode_terminated = False

        steps_in_current_episode = 0

        for t in range(max_steps):
            action = pour_policy_v2(env, obs)
            if noise:
                action = action + np.random.normal(0, 0.02, action.shape)
            if random_action:
                if np.random.rand() < 0.01:
                    action = env.action_space.sample()
            obs_to_store = env.unwrapped._get_observation(minimal=True)
            obs_next, reward, terminated, truncated, info = env.unwrapped.step(
                action, minimal=True
            )
            done = terminated or truncated

            if minimal_observations:
                dataset["observations"].append(obs_to_store)
            else:
                dataset["observations"].append(obs)
            dataset["actions"].append(action)
            dataset["terminals"].append(done)
            dataset["qpos"].append(env.unwrapped.data.qpos.copy())
            dataset["qvel"].append(env.unwrapped.data.qvel.copy())

            obs = obs_next
            steps_in_current_episode += 1

            if done:
                episode_terminated = True
                total_steps += steps_in_current_episode
                if ep_idx < num_train_episodes:
                    total_train_steps += steps_in_current_episode

                # Save last correct dataset entries for debugging consistency
                for k in dataset.keys():
                    debug_data[k].append(dataset[k][-1])
                break

            elif t == max_steps - 1:
                if env._automaton_state == "pour":
                    Goal, Start = env.unwrapped.get_particles_in_cups()
                    print(f"Goal position at max_steps: {Goal}")
                    print(f"Start position at max_steps: {Start}")

                if save_failed_episodes:
                    print(
                        f"Episode {ep_idx} failed but saved due to save_failed_episodes=True."
                    )
                    dataset["terminals"][-1] = True
                    total_steps += steps_in_current_episode
                    if ep_idx < num_train_episodes:
                        total_train_steps += steps_in_current_episode

                    for k in dataset.keys():
                        debug_data[k].append(dataset[k][-1])
                else:
                    print(
                        f"Episode {ep_idx} reached max_steps ({max_steps}) without termination; not saved."
                    )
                    for k in dataset.keys():
                        dataset[k] = dataset[k][:-max_steps]

                        if len(dataset[k]) > 0:
                            assert np.array_equal(
                                debug_data[k][-1], dataset[k][-1]
                            ), f"Data mismatch in key {k} at episode {ep_idx}, step {t}"
                break

        if episode_terminated:
            success_count += 1
        else:
            final_state = getattr(env, "_automaton_state", None)
            failure_counts[str(final_state)] += 1

    env.close()

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    train_path = os.path.join(save_root, "train_dataset.npz")
    val_path = os.path.join(save_root, "val_dataset.npz")

    actual_total_len = len(dataset["actions"])
    split_idx = min(total_train_steps, actual_total_len)

    for k, v in dataset.items():
        if "observations" in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == "terminals":
            dtype = bool
        elif k == "button_states":
            dtype = np.int64
        else:
            dtype = np.float32

        train_dataset[k] = np.array(v[:split_idx], dtype=dtype)
        val_dataset[k] = np.array(v[split_idx:], dtype=dtype)

    for path, dset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dset)

    stats = {
        "total_episodes_attempted": episodes + (episodes // 10),
        "successful_episodes": success_count,
        "success_rate": (
            float(success_count) / float(episodes + (episodes // 10))
            if episodes > 0
            else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "saved_failed_episodes": save_failed_episodes,
    }
    stats_path = os.path.join(save_root, "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved dataset to {save_root}")


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
    """Run the policy multiple times and save trajectories.

    Args:
        save_failed_episodes: If True, saves all episodes. If False, only saves
                              successful (terminated==True) episodes.
    """
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
    dataset = defaultdict(list)
    os.makedirs(save_root, exist_ok=True)

    success_count = 0
    failure_counts = defaultdict(int)
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = episodes
    num_val_episodes = episodes // 10
    env._noise_generator = OUNoise(3)
    debug_data = defaultdict(list)
    for ep_idx in trange(num_train_episodes + num_val_episodes):
        obs, _ = env.reset(options={"randomise_cup_position": True, "minimal": True})
        env._automaton_state = "move_above"
        env._state_counter = 0
        env._noise_generator.reset()
        episode_terminated = False

        steps_in_current_episode = 0

        move_operations = np.random.randint(0, 3)
        if move_operations == 0:
            perform_pouring = True
        else:
            perform_pouring = np.random.rand() < pouring_prob
        moves_completed = 0
        policy_mode = "moving" if move_operations > 0 else "pouring"
        done2 = False
        # cup = np.random.choice(np.array([0, 1]))
        0
        for t in range(max_steps):
            action = None
            if policy_mode == "moving":
                action = moving_policy(env, obs, cup_number=cup)
                if env._automaton_state == "done":
                    moves_completed += 1
                    if moves_completed == move_operations:
                        if perform_pouring:
                            policy_mode = "pouring"
                        else:
                            done2 = True
                    else:
                        cup = np.random.choice(np.array([0, 1]))
                    env._automaton_state = "move_above"
            elif policy_mode == "pouring":
                action = pour_policy_v2(env, obs)

            if noise:
                action = action + np.random.normal(0, 0.01, action.shape)
            if random_action:
                if np.random.rand() < 0.01:
                    action = env.action_space.sample()
            obs_to_store = env.unwrapped._get_observation(minimal=True)
            obs_next, reward, terminated, truncated, info = env.unwrapped.step(
                action, minimal=True
            )
            done = terminated or truncated or done2

            if minimal_observations:
                dataset["observations"].append(obs_to_store)
            else:
                dataset["observations"].append(obs)
            dataset["actions"].append(action)
            dataset["terminals"].append(done)
            dataset["qpos"].append(env.unwrapped.data.qpos.copy())
            dataset["qvel"].append(env.unwrapped.data.qvel.copy())

            obs = obs_next
            steps_in_current_episode += 1

            if done:
                episode_terminated = True
                total_steps += steps_in_current_episode
                if ep_idx < num_train_episodes:
                    total_train_steps += steps_in_current_episode

                # Save last correct dataset entries for debugging consistency
                for k in dataset.keys():
                    debug_data[k].append(dataset[k][-1])
                break

            elif t == max_steps - 1:
                if env._automaton_state == "pour":
                    Goal, Start = env.unwrapped.get_particles_in_cups()
                    print(f"Goal position at max_steps: {Goal}")
                    print(f"Start position at max_steps: {Start}")

                if save_failed_episodes:
                    print(
                        f"Episode {ep_idx} failed but saved due to save_failed_episodes=True."
                    )
                    dataset["terminals"][-1] = True
                    total_steps += steps_in_current_episode
                    if ep_idx < num_train_episodes:
                        total_train_steps += steps_in_current_episode

                    for k in dataset.keys():
                        debug_data[k].append(dataset[k][-1])
                else:
                    print(
                        f"Episode {ep_idx} reached max_steps ({max_steps}) without termination; not saved."
                    )
                    for k in dataset.keys():
                        dataset[k] = dataset[k][:-max_steps]

                        if len(dataset[k]) > 0:
                            assert np.array_equal(
                                debug_data[k][-1], dataset[k][-1]
                            ), f"Data mismatch in key {k} at episode {ep_idx}, step {t}"
                break

        if episode_terminated:
            success_count += 1
        else:
            final_state = getattr(env, "_automaton_state", None)
            failure_counts[str(final_state)] += 1

    env.close()

    # Split the dataset into training and validation sets.
    train_dataset = {}
    val_dataset = {}
    train_path = os.path.join(save_root, "train_dataset.npz")
    val_path = os.path.join(save_root, "val_dataset.npz")

    actual_total_len = len(dataset["actions"])
    split_idx = min(total_train_steps, actual_total_len)

    for k, v in dataset.items():
        if "observations" in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == "terminals":
            dtype = bool
        elif k == "button_states":
            dtype = np.int64
        else:
            dtype = np.float32

        train_dataset[k] = np.array(v[:split_idx], dtype=dtype)
        val_dataset[k] = np.array(v[split_idx:], dtype=dtype)

    for path, dset in [(train_path, train_dataset), (val_path, val_dataset)]:
        np.savez_compressed(path, **dset)

    stats = {
        "total_episodes_attempted": episodes + (episodes // 10),
        "successful_episodes": success_count,
        "success_rate": (
            float(success_count) / float(episodes + (episodes // 10))
            if episodes > 0
            else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "saved_failed_episodes": save_failed_episodes,
    }
    stats_path = os.path.join(save_root, "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved dataset to {save_root}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["policy", "dataset", "crl"], default="policy"
    )
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=1900)
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
        "--checkpoint",
        type=str,
        help="Path to CRL agent checkpoint file for crl mode",
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
        # Use --out as a directory for the dataset
        save_root = args.out
        if args.pixel_observations:

            collect_moving_policy_dataset(
                save_root=save_root,
                episodes=args.episodes,
                max_steps=args.steps,
                minimal_observations=args.minimal,
                save_failed_episodes=args.save_failed_episodes,
                pixel_observations=True,
                pouring_prob=args.pouring_prob,
            )
        else:
            collect_moving_policy_dataset(
                save_root=save_root,
                episodes=args.episodes,
                max_steps=args.steps,
                minimal_observations=args.minimal,
                save_failed_episodes=args.save_failed_episodes,
                pouring_prob=args.pouring_prob,
            )
    elif args.mode == "crl":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when using --mode=crl")
        collect_crl_episode(
            save_path=args.out, steps=args.steps, checkpoint_path=args.checkpoint
        )
