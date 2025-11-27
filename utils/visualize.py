import numpy as np
import time
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from envs import make_humanoid_env


def visualize(model_path, n_episodes=3, deterministic=True, save_video=False,
              video_folder="./videos", video_length=1000, fps=30):
    """
    Visualize trained PPO model in real-time

    Args:
        model_path: Path to model .zip file
        n_episodes: Number of episodes to visualize (default: 3)
        deterministic: If True, use policy mean
        save_video: If True, record MP4 videos
        video_folder: Folder to save videos
        video_length: Max video duration in steps
        fps: Frames per second for video
    """

    print("\n" + "=" * 70)
    print("MODEL VISUALIZATION")
    print("=" * 70 + "\n")

    print(f"Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Episodes: {n_episodes}")
    print(f"   Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"   Video recording: {'YES' if save_video else 'NO'}")
    if save_video:
        print(f"   Video folder: {video_folder}")
        print(f"   FPS: {fps}\n")
    else:
        print()

    # Load model
    print("Loading model...")
    try:
        model = PPO.load(model_path)
        print("   [OK] Model loaded successfully\n")
    except Exception as e:
        print(f"   [ERROR] Failed to load: {e}")
        return

    # Create environment
    if save_video:
        print("Configuring video recording...")
        Path(video_folder).mkdir(parents=True, exist_ok=True)

        def make_env():
            return make_humanoid_env(render_mode="rgb_array")

        env = DummyVecEnv([make_env])

        env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix="humanoid"
        )
        print(f"   [OK] Videos will be saved to: {video_folder}\n")

    else:
        print("Creating visualization environment...")
        env = make_humanoid_env(render_mode="human")
        print("   [OK] Environment created\n")

    # Visualize episodes
    print("=" * 70)
    print(f"STARTING VISUALIZATION ({n_episodes} episodes)")
    print("=" * 70 + "\n")

    if not save_video:
        print("TIP: Watch the MuJoCo window that opened!")
        print("     (You can reposition and zoom in/out)\n")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        print(f"\n{'=' * 70}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'=' * 70}")

        if save_video:
            obs = env.reset()
        else:
            obs, info = env.reset()

        done = False
        step = 0
        episode_reward = 0

        print("   Running...", end=" ", flush=True)

        while not done:
            # Predict action
            if save_video:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                done = done[0]
                reward = reward[0]
            else:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_reward += reward
            step += 1

            # Slow down for real-time visualization (not for video recording)
            if not save_video:
                time.sleep(0.01)  # ~100 FPS (MuJoCo runs at 500Hz normally)

            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"{step}", end=".", flush=True)

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        print(f" DONE")
        print(f"   Total reward: {episode_reward:.2f}")
        print(f"   Duration: {step} steps")
        print(f"   Result: {'Success (1000 steps)' if step >= 1000 else 'Fell'}")

    # Summary
    print(f"\n{'=' * 70}")
    print("VISUALIZATION SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"   Average reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"   Average duration: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f} steps")
    print(f"   Success rate: {sum(1 for l in episode_lengths if l >= 1000)}/{n_episodes} episodes")

    if save_video:
        print(f"\n   [OK] Videos saved to: {video_folder}")
        print(f"   Format: MP4, {fps} FPS")

        video_files = list(Path(video_folder).glob("*.mp4"))
        if video_files:
            print(f"   Files created:")
            for vf in sorted(video_files)[-n_episodes:]:
                print(f"      - {vf.name}")

    print(f"\n{'=' * 70}\n")

    env.close()
    print("[OK] Visualization complete!\n")


def visualize_best_episodes(model_path, n_total_episodes=100, n_show_best=10,
                            deterministic=True, save_video=False, video_folder="./videos"):
    """
    Run multiple episodes and visualize only the best ones

    Args:
        model_path: Path to model .zip
        n_total_episodes: Total episodes to evaluate (default: 100)
        n_show_best: Number of best episodes to visualize (default: 10)
        deterministic: Use deterministic policy
        save_video: Record videos of best episodes
        video_folder: Folder to save videos
    """

    print("\n" + "=" * 70)
    print(f"BEST EPISODES VISUALIZATION (Top {n_show_best} of {n_total_episodes})")
    print("=" * 70 + "\n")

    # Load model
    print("Loading model...")
    try:
        model = PPO.load(model_path)
        print("   [OK] Model loaded\n")
    except Exception as e:
        print(f"   [ERROR] Failed to load: {e}")
        return

    # Phase 1: Evaluate all episodes without rendering
    print("=" * 70)
    print(f"PHASE 1: Evaluating {n_total_episodes} episodes (no rendering)")
    print("=" * 70 + "\n")

    env = make_humanoid_env(render_mode=None)

    episode_data = []  # Store (reward, length, seed)

    for episode in range(n_total_episodes):
        if episode % 10 == 0:
            print(f"Evaluating episode {episode}/{n_total_episodes}...", end="\r")

        obs, info = env.reset(seed=episode)
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

        episode_data.append({
            'reward': episode_reward,
            'length': step,
            'seed': episode,
            'success': step >= 1000
        })

    env.close()

    # Sort by reward (descending)
    episode_data.sort(key=lambda x: x['reward'], reverse=True)
    best_episodes = episode_data[:n_show_best]

    print(f"\nEvaluation complete!                                ")
    print(f"\nTop {n_show_best} episodes by reward:")
    for i, ep in enumerate(best_episodes, 1):
        print(f"   {i:2d}. Seed {ep['seed']:3d} | Reward: {ep['reward']:8.2f} | "
              f"Steps: {ep['length']:4d} | {'Success' if ep['success'] else 'Failed'}")

    # Phase 2: Visualize best episodes
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Visualizing top {n_show_best} episodes")
    print("=" * 70 + "\n")

    if save_video:
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        print(f"Videos will be saved to: {video_folder}\n")

    for i, ep_data in enumerate(best_episodes, 1):
        print(f"\n--- Best Episode #{i} (Seed {ep_data['seed']}) ---")

        if save_video:
            def make_env():
                return make_humanoid_env(render_mode="rgb_array")

            env = DummyVecEnv([make_env])
            env = VecVideoRecorder(
                env, video_folder,
                record_video_trigger=lambda x: x == 0,
                video_length=1000,
                name_prefix=f"best_ep_{i:02d}_seed{ep_data['seed']}"
            )
            obs = env.reset()
        else:
            env = make_humanoid_env(render_mode="human")
            obs, _ = env.reset(seed=ep_data['seed'])

        done = False
        step = 0
        total_reward = 0

        print("   Playing...", end=" ", flush=True)

        while not done:
            if save_video:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                done = done[0]
                reward = reward[0]
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                time.sleep(0.01)

            total_reward += reward
            step += 1

            if step % 100 == 0:
                print(f"{step}", end=".", flush=True)

        print(f" DONE")
        print(f"   Reward: {total_reward:.2f} | Steps: {step}")

        env.close()

        if not save_video and i < len(best_episodes):
            input("\n   Press ENTER for next episode...")

    print(f"\n{'=' * 70}")
    print("[OK] Best episodes visualization complete!")
    print("=" * 70 + "\n")


def create_gif_from_video(video_path, gif_path=None, fps=30):
    """Convert MP4 video to GIF (requires imageio and imageio-ffmpeg)"""
    try:
        import imageio

        if gif_path is None:
            gif_path = video_path.replace('.mp4', '.gif')

        print(f"Converting to GIF: {video_path} -> {gif_path}")

        reader = imageio.get_reader(video_path)
        frames = [frame for frame in reader]
        imageio.mimsave(gif_path, frames, fps=fps)

        print(f"   [OK] GIF created: {gif_path}")

    except ImportError:
        print("   [ERROR] imageio not installed. Install: pip install imageio imageio-ffmpeg")
    except Exception as e:
        print(f"   [ERROR]: {e}")


if __name__ == "__main__":
    print("Testing visualize module\n")
    print("You need a trained model to test!")
    print("   Example: python utils/visualize.py\n")

    test_model = "./configs/models/ppo_humanoid_final.zip"

    if os.path.exists(test_model):
        print(f"[OK] Model found: {test_model}\n")

        print("Choose visualization mode:")
        print("  1. Standard visualization (3 episodes)")
        print("  2. Best episodes (top 10 of 100)")
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "2":
            visualize_best_episodes(test_model, n_total_episodes=100, n_show_best=10)
        else:
            save_vid = input("Record video? (y/n): ").lower() == 'y'
            visualize(test_model, n_episodes=3, save_video=save_vid, video_folder="./videos")
    else:
        print(f"[ERROR] No model found at: {test_model}")
        print("   Train a model first with: python main.py train")
