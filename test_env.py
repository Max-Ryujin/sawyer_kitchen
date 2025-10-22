import os
import json
import imageio
import numpy as np
import gymnasium as gym
import mujoco as mj
import utils
from collections import defaultdict


def random_action_test(save_path: str, steps: int = 250):
    gym.register(
        id="KitchenMinimalEnv-v0",
        entry_point="env:KitchenMinimalEnv",
    )
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=2560, height=1920
    )
    obs, info = env.reset(options={"randomise_cup_position": True})

    frames = []
    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        if term:
            break

    env.close()

    if frames:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 12))
        print(f"Saved video to: {save_path}")
    else:
        print("No frames collected.")


def combine_episode_jsons(save_root: str):
    """Combine per-episode trajectory.json files under save_root into
    a single `all_episodes.json`. Image paths are rewritten to be
    relative to save_root (e.g. `episode_000/images/step_000.png`).
    """
    combined = {
        "num_episodes": 0,
        "episodes": [],
    }

    if not os.path.isdir(save_root):
        print(f"No dataset directory found at {save_root}; skipping combine.")
        return

    for name in sorted(os.listdir(save_root)):
        ep_dir = os.path.join(save_root, name)
        traj_path = os.path.join(ep_dir, "trajectory.json")
        if os.path.isdir(ep_dir) and os.path.exists(traj_path):
            with open(traj_path, "r") as fh:
                data = json.load(fh)

            for entry in data.get("trajectory", []):
                img_rel = entry.get("image_path")
                if img_rel:
                    entry["image_path"] = os.path.normpath(os.path.join(name, img_rel))

            combined["episodes"].append(data)

    combined["num_episodes"] = len(combined["episodes"])
    all_path = os.path.join(save_root, "all_episodes.json")

    with open(all_path, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(
        f"Wrote combined dataset with {combined['num_episodes']} episodes to {all_path}"
    )


def pour_policy(env, obs) -> np.ndarray:

    model, data = env.unwrapped.model, env.unwrapped.data
    if env._automaton_state == "move_left":

        target_pos = np.array([-0.6, -1.0, 2.2])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
        )

        # Default to last valid qpos if IK fails
        delta_q_scaled = 2.7 * delta_q
        if np.linalg.norm(delta_q_scaled) > 4:
            delta_q_scaled = delta_q
        q_target = data.qpos[:7] + delta_q_scaled

        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=False, open_val=-1, close_val=1)
        if np.linalg.norm(target_pos - utils.get_effector_pos(env)) < 0.02:
            env._automaton_state = "move_above"
            print("Switching to move_above state")
        return action[:9]
    elif env._automaton_state == "move_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.01, 0.0, 0.27])

        target_quat = [0.69636424, -0.12278780, 0.12278780, 0.69636424]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            reg_strength=3e-2,
        )

        # Default to last valid qpos if IK fails
        delta_q_scaled = 2.7 * delta_q
        if np.linalg.norm(delta_q_scaled) > 2:
            delta_q_scaled = delta_q
        if np.linalg.norm(delta_q_scaled) < 0.08:
            delta_q_scaled = 10 * delta_q
        q_target = data.qpos[:7] + delta_q_scaled

        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=False, open_val=-1, close_val=1)

        ee_pos = utils.get_effector_pos(env)

        xy_dist = np.linalg.norm(target_pos[:2] - ee_pos[:2])

        # Transition condition: good xy position and stable (slow movement)
        if (
            xy_dist < 0.0075
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.00001
        ):
            env._automaton_state = "move_down"
            print("Switching to move_down state")
        return action[:9]
    elif env._automaton_state == "move_down":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([-0.01, 0.0, 0.04])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model, data, site_name="grip_site", target_pos=target_pos, reg_strength=1e-4
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.7 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=False, open_val=-1, close_val=1)
        if np.linalg.norm(target_pos - utils.get_effector_pos(env)) < 0.015:
            env._automaton_state = "close_gripper"
            print("Switching to close_gripper state")
        return action[:9]
    elif env._automaton_state == "close_gripper":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint1", "cup1"))
        target_pos = cup_pos + np.array([0.0, 0.0, 0.04])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model, data, site_name="grip_site", target_pos=target_pos, reg_strength=1e-4
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.5 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=True, open_val=-1, close_val=1)
        gripper_joint_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "rc_close"),
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "lc_close"),
        ]

        forces = np.array([data.qfrc_constraint[i] for i in gripper_joint_ids])
        if np.linalg.norm(forces) > 15.0:
            env._automaton_state = "lift_above"
            print("Switching to lift_above state")
        return action[:9]
    elif env._automaton_state == "lift_above":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([0.0, -0.05, 0.34])

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            reg_strength=1e-4,
        )
        q_target = data.qpos[:7] + 0.5 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=True, open_val=-1, close_val=1)
        ee_pos = utils.get_effector_pos(env)
        xy_dist = np.linalg.norm(target_pos[:2] - ee_pos[:2])
        if (
            xy_dist < 0.03
            and np.linalg.norm(
                data.qvel[mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "grip_site")]
            )
            < 0.00004
        ):
            env._automaton_state = "pour"
            print("Switching to pour state")
        return action[:9]
    elif env._automaton_state == "pour":
        cup_pos = utils.get_object_pos(env, ("cup_freejoint0", "cup0"))
        target_pos = cup_pos + np.array([-0.005, -0.0145, 0.285])
        target_quat = [0.29883626, -0.64085637, 0.64085637, 0.29883626]

        # Solve IK for the target position
        delta_q = utils.ik_step(
            model,
            data,
            site_name="grip_site",
            target_pos=target_pos,
            target_quat=target_quat,
            rot_weight=0.8,
            # reg_strength=1e-4,
        )

        # Default to last valid qpos if IK fails
        q_target = data.qpos[:7] + 0.5 * delta_q

        # add two values for gripper control
        action = np.pad(q_target, (0, env.unwrapped.nu - 7))
        action += utils.make_gripper_action(env, close=True, open_val=-1, close_val=1)
        return action[:9]


def collect_policy_episode(save_path="tmp/policy.mp4", steps=800):
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=1280, height=960
    )
    obs, _ = env.reset(options={"randomise_cup_position": True})
    frames = []
    env._automaton_state = "move_left"
    for t in range(steps):
        action = pour_policy(env, obs)
        obs, _, term, trunc, _ = env.step(action)
        frames.append(env.render())
        if term or trunc:
            print(f"Episode finished after {t+1} steps.")
            break

    env.close()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, frames, fps=env.metadata.get("render_fps", 24))
    print(f"Saved test-policy video to {save_path}")


def _serialize_obs(obs):
    """Convert observations to JSON-serializable structures.

    Supports numpy arrays and dicts of arrays. Falls back to string repr.
    """
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            try:
                out[k] = np.asarray(v).tolist()
            except Exception:
                out[k] = str(v)
        return out
    try:
        return np.asarray(obs).tolist()
    except Exception:
        return str(obs)


def collect_policy_dataset(
    save_root: str = "tmp/policy_dataset",
    episodes: int = 100,
    max_steps: int = 3600,
    width: int = 1280,
    height: int = 960,
):
    """Run the policy multiple times and save successful (terminated==True)
    trajectories to per-episode JSON files. Each step stores obs, action,
    terminated boolean and a relative path to the rendered image for that step.

    Images for each episode are saved in an `images/` folder next to the JSON.
    """
    gym.register(id="KitchenMinimalEnv-v0", entry_point="env:KitchenMinimalEnv")
    env = gym.make(
        "KitchenMinimalEnv-v0", render_mode="rgb_array", width=width, height=height
    )

    os.makedirs(save_root, exist_ok=True)
    attempts = episodes

    success_count = 0
    failure_counts = defaultdict(int)

    for ep in range(attempts):
        obs, _ = env.reset(options={"randomise_cup_position": True})
        env._automaton_state = "move_left"

        ep_dir = os.path.join(save_root, f"episode_{ep:03d}")
        images_dir = os.path.join(ep_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        traj = []
        episode_terminated = False
        for t in range(max_steps):
            action = pour_policy(env, obs)
            obs_next, reward, terminated, truncated, info = env.step(action)

            entry = {
                "obs": _serialize_obs(obs),
                "action": np.asarray(action).tolist(),
                "terminated": bool(terminated),
            }
            traj.append(entry)

            obs = obs_next

            if terminated:
                episode_terminated = True
                json_path = os.path.join(ep_dir, "trajectory.json")
                meta = {
                    "episode_index": ep,
                    "length": len(traj),
                    "terminated": True,
                    "final_state": env._automaton_state,
                    "trajectory": traj,
                }
                with open(json_path, "w") as fh:
                    json.dump(meta, fh, indent=2)
                print(f"Saved successful episode {ep} to {json_path}")
                break

            if t == max_steps - 1:

                if env._automaton_state == "move_above":
                    # print the gripper position
                    grip_pos = utils.get_effector_pos(env)
                    print(f"Gripper position at max_steps: {grip_pos}")
                    # print gripper speed
                    grip_speed = env.unwrapped.data.qvel[
                        mj.mj_name2id(
                            env.unwrapped.model, mj.mjtObj.mjOBJ_SITE, "grip_site"
                        )
                    ]
                    print(f"Gripper speed at max_steps: {grip_speed}")
                    # render final frame
                    final_frame = env.render()
                    final_image_path = os.path.join(images_dir, f"step_{t:03d}.png")
                    imageio.imwrite(final_image_path, final_frame)
                    print(f"Saved final frame to {final_image_path}")
                if env._automaton_state == "pour":
                    # print where water is
                    Goal, Start = env.unwrapped.get_particles_in_cups()
                    print(f"Goal position at max_steps: {Goal}")
                    print(f"Start position at max_steps: {Start}")
                    # render final frame
                    final_frame = env.render()
                    final_image_path = os.path.join(images_dir, f"step_{t:03d}.png")
                    imageio.imwrite(final_image_path, final_frame)
                    print(f"Saved final frame to {final_image_path}")
                # episode reached time limit without termination; not saved
                print(
                    f"Episode {ep} reached max_steps ({max_steps}) without termination; not saved."
                )

        # record stats for this episode
        if episode_terminated:
            success_count += 1
        else:
            # record the automaton state reached at the end of the episode
            final_state = getattr(env, "_automaton_state", None)
            failure_counts[str(final_state)] += 1

    env.close()

    # write stats summary
    stats = {
        "total_episodes": attempts,
        "successful_episodes": success_count,
        "success_rate": float(success_count) / float(attempts) if attempts > 0 else 0.0,
        "failure_counts": dict(failure_counts),
    }
    stats_path = os.path.join(save_root, "stats.json")

    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Wrote stats to {stats_path}: {stats}")

    # combine per-episode JSONs into one file for convenience
    combine_episode_jsons(save_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["random", "policy", "dataset"], default="random"
    )
    parser.add_argument("--out", default="tmp/kitchen_run.mp4")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect when using dataset mode",
    )
    args = parser.parse_args()

    if args.mode == "random":
        random_action_test(args.out, steps=args.steps)
    elif args.mode == "policy":
        collect_policy_episode(steps=args.steps)
    elif args.mode == "dataset":
        # Use --out as a directory for the dataset
        save_root = args.out
        collect_policy_dataset(
            save_root=save_root, episodes=args.episodes, max_steps=args.steps
        )
