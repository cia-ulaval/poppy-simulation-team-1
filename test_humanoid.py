import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(render=False):
    def _init():
        env = gym.make(
            "Humanoid-v5",
            render_mode="human" if render else None,
            terminate_when_unhealthy=True,
            healthy_z_range=(1.0, 2.0),
        )
        return Monitor(env)

    return _init


def run_random_baseline(n_episodes=5, render=False):
    env = gym.make(
        "Humanoid-v5",
        render_mode="human" if render else None,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    )

    episode_rewards = []
    episode_lengths = []
    per_step_rewards = []
    max_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        steps = 0
        step_rewards = []

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_rewards.append(reward)
            total += reward
            steps += 1
            done = terminated or truncated
            if render:
                time.sleep(0.01)

        episode_rewards.append(total)
        episode_lengths.append(steps)
        per_step_rewards.append(step_rewards)
        max_steps = max(max_steps, steps)
        print(f"Random: épisode {ep+1}/{n_episodes} -> reward {total:.2f}, steps {steps}")

    env.close()

    reward_matrix = np.full((n_episodes, max_steps), np.nan, dtype=np.float32)
    for i, row in enumerate(per_step_rewards):
        reward_matrix[i, : len(row)] = row

    return episode_rewards, episode_lengths, reward_matrix


def train_ppo(total_timesteps=int(1e6), log_dir="./ppo_logs", policy_kwargs=None, seed=0):
    
    os.makedirs(log_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env(render=False)])
    vec_env = VecMonitor(vec_env)

    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir, name_prefix="ppo_humanoid")

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs or dict(),
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model_path = os.path.join(log_dir, "ppo_humanoid_final.zip")
    model.save(model_path)
    vec_env.close()

    print(f"Model saved to: {model_path}")
    return model_path


def evaluate_model(model_path, n_episodes=5, render=False, deterministic=True):
    env = Monitor(gym.make(
        "Humanoid-v5",
        render_mode="human" if render else None,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    ))

    model = PPO.load(model_path)

    episode_rewards = []
    episode_lengths = []
    per_step_rewards = []
    max_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        steps = 0
        step_rewards = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            step_rewards.append(reward)
            total += reward
            steps += 1
            done = terminated or truncated
            if render:
                time.sleep(0.01)

        episode_rewards.append(total)
        episode_lengths.append(steps)
        per_step_rewards.append(step_rewards)
        max_steps = max(max_steps, steps)
        print(f"PPO: épisode {ep+1}/{n_episodes} -> reward {total:.2f}, steps {steps}")

    env.close()

    reward_matrix = np.full((n_episodes, max_steps), np.nan, dtype=np.float32)
    for i, row in enumerate(per_step_rewards):
        reward_matrix[i, : len(row)] = row

    return episode_rewards, episode_lengths, reward_matrix


def plot_comparison(random_matrix, ppo_matrix, random_rewards, ppo_rewards, out_file="comparison_rewards.png"):
    n_rows = 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)

    ax_r = axes[0, 0]
    im_r = ax_r.imshow(random_matrix, aspect="auto", interpolation="nearest")
    ax_r.set_title("Random - rewards par step (ligne = épisode)")
    ax_r.set_ylabel("Épisode")
    ax_r.set_xlabel("Step")
    fig.colorbar(im_r, ax=ax_r, orientation="vertical", pad=0.02)

    ax_p = axes[0, 1]
    im_p = ax_p.imshow(ppo_matrix, aspect="auto", interpolation="nearest")
    ax_p.set_title("PPO - rewards par step (ligne = épisode)")
    ax_p.set_ylabel("Épisode")
    ax_p.set_xlabel("Step")
    fig.colorbar(im_p, ax=ax_p, orientation="vertical", pad=0.02)

    ax_r2 = axes[1, 0]
    ax_r2.plot(np.arange(1, len(random_rewards) + 1), random_rewards, marker="o")
    ax_r2.set_title("Random - reward total par épisode")
    ax_r2.set_xlabel("Épisode")
    ax_r2.set_ylabel("Reward total")
    ax_r2.grid(True)

    ax_p2 = axes[1, 1]
    ax_p2.plot(np.arange(1, len(ppo_rewards) + 1), ppo_rewards, marker="o")
    ax_p2.set_title("PPO - reward total par épisode")
    ax_p2.set_xlabel("Épisode")
    ax_p2.set_ylabel("Reward total")
    ax_p2.grid(True)

    plt.savefig(out_file, dpi=150)
    plt.show()
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    # Paramètres
    N_EPISODES_EVAL = 8
    TRAIN = True        
    TOTAL_TIMESTEPS = int(3e15)  
    LOG_DIR = "./ppo_logs"

    print("=== Running random baseline ===")
    rand_rewards, rand_lengths, rand_matrix = run_random_baseline(n_episodes=N_EPISODES_EVAL, render=False)

    if TRAIN:
        print("=== Training PPO ===")
        model_path = train_ppo(total_timesteps=TOTAL_TIMESTEPS, log_dir=LOG_DIR)
    else:
        model_path = os.path.join(LOG_DIR, "ppo_humanoid_final.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Active TRAIN=True or place a model there.")

    print("=== Evaluating trained PPO ===")
    ppo_rewards, ppo_lengths, ppo_matrix = evaluate_model(model_path, n_episodes=N_EPISODES_EVAL, render=False, deterministic=True)

    plot_comparison(rand_matrix, ppo_matrix, rand_rewards, ppo_rewards)

    print("Done.")
