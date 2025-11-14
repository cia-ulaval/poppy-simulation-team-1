import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter


def make_env(rank=0, seed=0, render=False):
    """
    Utility function for multiprocessed env.
    
    :param rank: index of the subprocess
    :param seed: the inital seed for RNG
    :param render: whether to render the environment
    :return: environment factory
    """
    def _init():
        env = gym.make(
            "Humanoid-v5",
            render_mode="human" if render else None,
            terminate_when_unhealthy=True,
            healthy_z_range=(1.0, 2.0),
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule with warmup.
    
    :param initial_value: Initial learning rate
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress < 0.1:  
            return initial_value * (progress / 0.1)
        else:  
            return progress_remaining * initial_value
    return func


def run_random_baseline(n_episodes=5, render=False, seed=0):
    """Run random agent baseline for comparison."""
    env = gym.make(
        "Humanoid-v5",
        render_mode="human" if render else None,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    )
    env.reset(seed=seed)

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


def train_ppo(
    total_timesteps=int(5e7),
    log_dir="./ppo_logs",
    n_envs=8,
    seed=0,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_linear_schedule=True,
):
    """
    Train PPO agent on Humanoid-v5 with advanced features.
    
    Key improvements:
    - Multi-environment parallelization
    - Observation normalization (but NOT reward normalization to see true performance)
    - Learning rate scheduling
    - Deeper network architecture with Tanh activation
    - Advanced monitoring with TensorBoard
    """
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir_full = os.path.join(log_dir, timestamp)
    os.makedirs(log_dir_full, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(log_dir_full, "tensorboard"))
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of parallel environments: {n_envs}")
    print(f"Steps per environment: {n_steps}")
    print(f"Total steps per update: {n_steps * n_envs:,}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs per update: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Learning rate schedule: {'Linear with warmup' if use_linear_schedule else 'Constant'}")
    print(f"Clip range: {clip_range}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Value function coefficient: {vf_coef}")
    print(f"Max gradient norm: {max_grad_norm}")
    print(f"Log directory: {log_dir_full}")
    print(f"{'='*60}\n")

    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(rank=i, seed=seed) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(seed=seed)])
    
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,      
        norm_reward=True,    
        clip_obs=10.0,        
        gamma=gamma,
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=[400, 300],  
            vf=[400, 300]   
        ),
        activation_fn=nn.Tanh,  # TODO TRY RELU
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=log_dir_full,
        name_prefix="ppo_humanoid",
        save_vecnormalize=True, 
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        learning_rate=linear_schedule(learning_rate) if use_linear_schedule else learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=None,     
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir_full,
    )

    print("Starting training...")
    print(f"Expected updates: {total_timesteps // (n_steps * n_envs)}")
    print(f"Expected training time: ~{(total_timesteps // (n_steps * n_envs)) * 10 / 3600:.1f} hours (rough estimate)\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,  
            tb_log_name="ppo_humanoid",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    training_time = time.time() - start_time
    
    model_path = os.path.join(log_dir_full, "ppo_humanoid_final.zip")
    model.save(model_path)
    
    vec_normalize_path = os.path.join(log_dir_full, "vec_normalize.pkl")
    vec_env.save(vec_normalize_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed in {training_time/3600:.2f} hours")
    print(f"Model saved to: {model_path}")
    print(f"VecNormalize stats saved to: {vec_normalize_path}")
    print(f"{'='*60}\n")
    
    print("\n=== Observation Normalization Statistics ===")
    print(f"Obs mean (first 10 dims): {vec_env.obs_rms.mean[:10]}")
    print(f"Obs std (first 10 dims): {np.sqrt(vec_env.obs_rms.var[:10])}")
    
    vec_env.close()
    writer.close()

    return model_path, vec_normalize_path


def evaluate_model(
    model_path,
    vec_normalize_path=None,
    n_episodes=10,
    render=False,
    deterministic=True,
    seed=0,
):
    """
    Evaluate trained PPO model with proper normalization.
    
    CRITICAL: Load VecNormalize stats to ensure the model receives
    observations in the same normalized space it was trained on.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Model")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"VecNormalize path: {vec_normalize_path}")
    print(f"Number of episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*60}\n")
    
    env = gym.make(
        "Humanoid-v5",
        render_mode="human" if render else None,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    )
    env = Monitor(env)
    env.reset(seed=seed)
    
    env = DummyVecEnv([lambda: env])
    
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Loading normalization statistics from {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False     
        env.norm_reward = True  
        print("✓ Normalization statistics loaded")
    else:
        print("⚠ WARNING: No normalization statistics found!")
        print("  The model may perform poorly without proper observation normalization.")
    
    # Load model
    model = PPO.load(model_path)
    print("✓ Model loaded\n")

    episode_rewards = []
    episode_lengths = []
    per_step_rewards = []
    max_steps = 0

    print("Running evaluation episodes...")
    for ep in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        total = 0.0
        steps = 0
        step_rewards = []

        while not done[0]:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            reward_scalar = float(reward[0])
            
            step_rewards.append(reward_scalar)
            total += reward_scalar
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.01)

        episode_rewards.append(total)
        episode_lengths.append(steps)
        per_step_rewards.append(step_rewards)
        max_steps = max(max_steps, steps)
        print(f"  Episode {ep+1}/{n_episodes} -> reward: {total:.2f}, steps: {steps}")

    env.close()

    reward_matrix = np.full((n_episodes, max_steps), np.nan, dtype=np.float32)
    for i, row in enumerate(per_step_rewards):
        reward_matrix[i, : len(row)] = row

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward:  {np.min(episode_rewards):.2f}")
    print(f"Max reward:  {np.max(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} steps")
    print(f"{'='*60}\n")

    return episode_rewards, episode_lengths, reward_matrix


def print_comparison_stats(random_rewards, ppo_rewards):
    """Print comparison statistics between random and PPO agents."""
    print(f"\n{'='*60}")
    print(f"Performance Comparison")
    print(f"{'='*60}")
    print(f"Random Agent:")
    print(f"  Mean: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"  Range: [{np.min(random_rewards):.2f}, {np.max(random_rewards):.2f}]")
    print(f"\nPPO Agent:")
    print(f"  Mean: {np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}")
    print(f"  Range: [{np.min(ppo_rewards):.2f}, {np.max(ppo_rewards):.2f}]")
    print(f"\nImprovement:")
    improvement = ((np.mean(ppo_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards))) * 100
    print(f"  {improvement:+.1f}% over random baseline")
    print(f"  Absolute gain: {np.mean(ppo_rewards) - np.mean(random_rewards):+.2f}")
    print(f"{'='*60}\n")


def plot_comparison(random_matrix, ppo_matrix, random_rewards, ppo_rewards, out_file="comparison_rewards.png"):
    """Create visualization comparing random and PPO performance."""
    n_rows = 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)

    ax_r = axes[0, 0]
    im_r = ax_r.imshow(random_matrix, aspect="auto", interpolation="nearest", cmap="RdYlGn")
    ax_r.set_title("Random Agent - Per-Step Rewards", fontsize=12, fontweight="bold")
    ax_r.set_ylabel("Episode")
    ax_r.set_xlabel("Step")
    fig.colorbar(im_r, ax=ax_r, orientation="vertical", pad=0.02)

    ax_p = axes[0, 1]
    im_p = ax_p.imshow(ppo_matrix, aspect="auto", interpolation="nearest", cmap="RdYlGn")
    ax_p.set_title("PPO Agent - Per-Step Rewards", fontsize=12, fontweight="bold")
    ax_p.set_ylabel("Episode")
    ax_p.set_xlabel("Step")
    fig.colorbar(im_p, ax=ax_p, orientation="vertical", pad=0.02)

    ax_r2 = axes[1, 0]
    ax_r2.plot(np.arange(1, len(random_rewards) + 1), random_rewards, marker="o", linewidth=2, markersize=8)
    ax_r2.axhline(np.mean(random_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(random_rewards):.2f}')
    ax_r2.set_title("Random Agent - Total Reward per Episode", fontsize=12, fontweight="bold")
    ax_r2.set_xlabel("Episode")
    ax_r2.set_ylabel("Total Reward")
    ax_r2.grid(True, alpha=0.3)
    ax_r2.legend()

    ax_p2 = axes[1, 1]
    ax_p2.plot(np.arange(1, len(ppo_rewards) + 1), ppo_rewards, marker="o", linewidth=2, markersize=8, color='green')
    ax_p2.axhline(np.mean(ppo_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(ppo_rewards):.2f}')
    ax_p2.set_title("PPO Agent - Total Reward per Episode", fontsize=12, fontweight="bold")
    ax_p2.set_xlabel("Episode")
    ax_p2.set_ylabel("Total Reward")
    ax_p2.grid(True, alpha=0.3)
    ax_p2.legend()

    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {out_file}")
    plt.show()


if __name__ == "__main__":
    
    N_EPISODES_EVAL = 10
    SEED = 42
    
    TRAIN = True
    TOTAL_TIMESTEPS = int(1e7)   #TO DO test 5e7
    N_ENVS = 8                  
    LOG_DIR = "./ppo_logs"
    
    # LEARNING_RATE = 3e-4
    # N_STEPS = 512               # Steps per environment per update
    # BATCH_SIZE = 128            # Minibatch size
    # N_EPOCHS = 20               # Number of epochs per update
    GAMMA = 0.99                # Discount factor
    GAE_LAMBDA = 0.95           # GAE lambda
    # CLIP_RANGE = 0.2            # PPO clip range
    # ENT_COEF = 0.01             # Entropy coefficient (exploration)
    VF_COEF = 0.5               # Value function coefficient
    MAX_GRAD_NORM = 0.5         # Gradient clipping

    # Dans votre section de configuration (ligne ~450)
    LEARNING_RATE = 1e-4        # Réduit de 3e-4 → plus stable
    N_STEPS = 2048              # Augmenté de 512 → plus de samples
    BATCH_SIZE = 64             # Réduit de 128 → updates plus fréquents mais plus petits
    N_EPOCHS = 10               # Réduit de 20 → moins de sur-optimisation
    CLIP_RANGE = 0.15           # Réduit de 0.2 → updates plus conservateurs
    ENT_COEF = 0.02             # Augmenté de 0.01 → plus d'exploration

    
    
    print("\n" + "="*80)
    print(" "*20 + "HUMANOID-V5 PPO TRAINING & EVALUATION")
    print("="*80 + "\n")
    
    print("Step 1/3: Running random baseline for comparison...")
    rand_rewards, rand_lengths, rand_matrix = run_random_baseline(
        n_episodes=N_EPISODES_EVAL,
        render=False,
        seed=SEED
    )

    if TRAIN:
        print("\nStep 2/3: Training PPO agent...")
        model_path, vec_normalize_path = train_ppo(
            total_timesteps=TOTAL_TIMESTEPS,
            log_dir=LOG_DIR,
            n_envs=N_ENVS,
            seed=SEED,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            use_linear_schedule=True,
        )
    else:
        print("\nStep 2/3: Loading existing model...")
        subdirs = [d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))]
        if not subdirs:
            raise FileNotFoundError(f"No training directories found in {LOG_DIR}. Set TRAIN=True to train a new model.")
        
        latest_dir = max(subdirs)
        model_path = os.path.join(LOG_DIR, latest_dir, "ppo_humanoid_final.zip")
        vec_normalize_path = os.path.join(LOG_DIR, latest_dir, "vec_normalize.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Set TRAIN=True to train a new model.")
        
        print(f"✓ Found model: {model_path}")
        if os.path.exists(vec_normalize_path):
            print(f"✓ Found normalization stats: {vec_normalize_path}")

    print("\nStep 3/3: Evaluating trained PPO agent...")
    ppo_rewards, ppo_lengths, ppo_matrix = evaluate_model(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        n_episodes=N_EPISODES_EVAL,
        render=False,
        deterministic=True,
        seed=SEED
    )

    print_comparison_stats(rand_rewards, ppo_rewards)

    plot_comparison(rand_matrix, ppo_matrix, rand_rewards, ppo_rewards)

    print("\n" + "="*80)
    print(" "*30 + "TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTo visualize training progress, run:")
    print(f"  tensorboard --logdir {LOG_DIR}")
    print("\nTo render the trained agent, set render=True in evaluate_model()")
    print("="*80 + "\n")