import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
from glob import glob

from stable_baselines3 import PPO, TD3, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise
from torch.utils.tensorboard import SummaryWriter

# Try to import tensorboard for reading logs
try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard not available for reading logs. Install with: pip install tensorboard")


def make_env(rank=0, seed=0, render=False):
    """
    Utility function for multiprocessed env.
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
    """Linear learning rate schedule with warmup."""
    def func(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress < 0.1:  
            return initial_value * (progress / 0.1)
        else:  
            return progress_remaining * initial_value
    return func


def train_random_policy(total_timesteps=int(1e7), log_dir="./baseline_logs", seed=42):
    """
    'Train' a random policy on Humanoid-v5 (just collect random episodes for comparison).
    """
    print(f"\n{'='*80}")
    print(f"🎲 RUNNING RANDOM POLICY")
    print(f"{'='*80}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"{'='*80}\n")
    
    # Create log directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algo_log_dir = os.path.join(log_dir, f"random_{timestamp}")
    os.makedirs(algo_log_dir, exist_ok=True)
    
    # Create environment
    env = gym.make(
        "Humanoid-v5",
        render_mode=None,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    )
    env = Monitor(env)
    env.reset(seed=seed)
    
    # TensorBoard writer for logging
    writer = SummaryWriter(log_dir=os.path.join(algo_log_dir, "random_humanoid_1"))
    
    # Run random episodes
    start_time = time.time()
    timestep = 0
    episode = 0
    episode_rewards = []
    episode_lengths = []
    last_log_timestep = 0
    LOG_FREQUENCY = 10000  # Log every 500 timesteps
    
    print(f"🎯 Running random policy (logging every {LOG_FREQUENCY} timesteps)...\n")
    
    # Buffer to accumulate rewards between logs
    rewards_buffer = []
    
    while timestep < total_timesteps:
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done and timestep < total_timesteps:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            timestep += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        rewards_buffer.append(episode_reward)
        episode += 1
        
        # Log to TensorBoard only every LOG_FREQUENCY timesteps
        if timestep - last_log_timestep >= LOG_FREQUENCY or timestep >= total_timesteps:
            mean_reward = np.mean(rewards_buffer) if rewards_buffer else 0.0
            writer.add_scalar("rollout/ep_rew_mean", mean_reward, timestep)
            
            last_log_timestep = timestep
            rewards_buffer = []  # Clear buffer
            
            print(f"  Episode {episode} | Timestep {timestep:,}/{total_timesteps:,} | Mean Reward: {mean_reward:.2f}")
    
    writer.close()
    env.close()
    
    training_time = time.time() - start_time
    
    # Save dummy "model" info (just for consistency)
    model_info_path = os.path.join(algo_log_dir, "random_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"Random Policy\n")
        f.write(f"Total timesteps: {timestep}\n")
        f.write(f"Total episodes: {episode}\n")
        f.write(f"Mean reward: {np.mean(episode_rewards):.2f}\n")
        f.write(f"Std reward: {np.std(episode_rewards):.2f}\n")
    
    print(f"\n{'='*80}")
    print(f"✅ RANDOM POLICY COMPLETE")
    print(f"{'='*80}")
    print(f"Time: {training_time/60:.2f} minutes")
    print(f"Episodes: {episode}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"{'='*80}\n")
    
    return {
        'algo': 'RANDOM',
        'model_path': model_info_path,  # Not a real model, just for consistency
        'vec_normalize_path': None,
        'training_time': training_time,
        'log_dir': algo_log_dir,
    }


def train_algorithm(
    algo_name="PPO",
    total_timesteps=int(1e7),
    log_dir="./baseline_logs",
    n_envs=8,
    seed=42,
):
    """
    Train a specific RL algorithm on Humanoid-v5.
    
    Args:
        algo_name: One of ["PPO", "TD3", "SAC", "A2C", "RANDOM"]
        total_timesteps: Total training timesteps
        log_dir: Base directory for logs
        n_envs: Number of parallel environments
        seed: Random seed
    """
    
    # Handle random policy separately
    if algo_name == "RANDOM":
        return train_random_policy(total_timesteps=total_timesteps, log_dir=log_dir, seed=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING {algo_name.upper()}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"{'='*80}\n")
    
    # Create log directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algo_log_dir = os.path.join(log_dir, f"{algo_name.lower()}_{timestamp}")
    os.makedirs(algo_log_dir, exist_ok=True)
    
    # Create environments
    if algo_name in ["PPO", "A2C"]:
        # On-policy algorithms: use multiple envs
        if n_envs > 1:
            vec_env = SubprocVecEnv([make_env(rank=i, seed=seed) for i in range(n_envs)])
        else:
            vec_env = DummyVecEnv([make_env(seed=seed)])
    else:
        # Off-policy algorithms: typically use 1 env for training
        vec_env = DummyVecEnv([make_env(seed=seed)])
        n_envs = 1  # Override for off-policy
    
    # Normalize observations
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    
    # Network architecture
    if algo_name in ["PPO", "A2C"]:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU,
        )
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256]),
            activation_fn=nn.ReLU,
        )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // n_envs, 1000),
        save_path=algo_log_dir,
        name_prefix=f"{algo_name.lower()}_humanoid",
        save_vecnormalize=True,
    )
    
    # Eval environment
    eval_env = DummyVecEnv([make_env(seed=seed + 1000)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
        training=False,
    )
    
    # Eval callback - Fixed frequency for consistent plotting
    # Use absolute frequency regardless of n_envs to ensure consistent x-axis
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=algo_log_dir,
        log_path=algo_log_dir,
        eval_freq=5000,  # Fixed: evaluate every 5k timesteps for all algos
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    # Create model based on algorithm
    start_time = time.time()
    
    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=algo_log_dir,
            device=device,
            verbose=1,
            seed=seed,
        )
        
    elif algo_name == "TD3":
        n_actions = vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        
        model = TD3(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-3,
            buffer_size=1000000,
            learning_starts=1000,  # Reduced from 10000 to start logging earlier
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=-1,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=algo_log_dir,
            device=device,
            verbose=1,
            seed=seed,
        )
        
    elif algo_name == "SAC":
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=1000,  # Reduced from 10000 to start logging earlier
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=policy_kwargs,
            tensorboard_log=algo_log_dir,
            device=device,
            verbose=1,
            seed=seed,
        )
        
    elif algo_name == "A2C":
        model = A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=algo_log_dir,
            device=device,
            verbose=1,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Train
    print(f"\n🎯 Starting {algo_name} training...\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=f"{algo_name.lower()}_humanoid",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print(f"\n⚠️ {algo_name} training interrupted by user!")
    
    training_time = time.time() - start_time
    
    # Save final model
    model_path = os.path.join(algo_log_dir, f"{algo_name.lower()}_humanoid_final.zip")
    model.save(model_path)
    
    vec_normalize_path = os.path.join(algo_log_dir, "vec_normalize.pkl")
    vec_env.save(vec_normalize_path)
    
    print(f"\n{'='*80}")
    print(f"✅ {algo_name} TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Time: {training_time/3600:.2f} hours")
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vec_normalize_path}")
    print(f"{'='*80}\n")
    
    vec_env.close()
    eval_env.close()
    
    return {
        'algo': algo_name,
        'model_path': model_path,
        'vec_normalize_path': vec_normalize_path,
        'training_time': training_time,
        'log_dir': algo_log_dir,
    }


def read_tensorboard_logs(log_dir, algo_name):
    """
    Read TensorBoard logs to extract training rewards over time.
    Returns: (timesteps, rewards) or None if not available
    """
    if not TENSORBOARD_AVAILABLE:
        return None
    
    try:
        # Find event files
        event_files = glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
        
        if not event_files:
            print(f"⚠️  No TensorBoard event files found in {log_dir}")
            return None
        
        # Use the most recent event file
        event_file = max(event_files, key=os.path.getmtime)
        
        # Read events
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Try to find reward scalar
        scalar_tags = ea.Tags()['scalars']
        
        # Common reward tag names
        reward_tags = [
            'rollout/ep_rew_mean',
            'eval/mean_reward',
            'train/reward',
            'rollout/reward',
        ]
        
        timesteps = []
        rewards = []
        
        for tag in reward_tags:
            if tag in scalar_tags:
                events = ea.Scalars(tag)
                timesteps = [e.step for e in events]
                rewards = [e.value for e in events]
                print(f"✅ Found reward data: {tag}")
                break
        
        if not timesteps:
            print(f"⚠️  No reward data found in TensorBoard logs for {algo_name}")
            return None
        
        # ✅ ADD INITIAL POINT AT TIMESTEP 0 with first reward value
        # This ensures all curves start from the same x-axis point
        if timesteps[0] > 0:
            timesteps = [0] + timesteps
            rewards = [rewards[0]] + rewards  # Use first logged reward as initial value
            print(f"   Added initial point at timestep 0 for {algo_name}")
        
        return timesteps, rewards
    
    except Exception as e:
        print(f"⚠️  Error reading TensorBoard logs: {e}")
        return None


def evaluate_algorithm(model_path, vec_normalize_path, algo_name, n_episodes=20, seed=42):
    """
    Evaluate a trained algorithm.
    """
    print(f"\n{'='*70}")
    print(f"📊 EVALUATING {algo_name.upper()}")
    print(f"{'='*70}\n")
    
    # Create environment
    env = gym.make(
        "Humanoid-v5",
        render_mode=None,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    )
    env = Monitor(env)
    env.reset(seed=seed)
    
    # Handle random policy
    if algo_name == "RANDOM":
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"  Episode {ep+1}/{n_episodes}: Reward={total_reward:.2f}, Steps={steps}")
        
        env.close()
    else:
        # Regular RL algorithm evaluation
        env = DummyVecEnv([lambda: env])
        
        # Load normalization
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
        
        # Load model
        if algo_name == "PPO":
            model = PPO.load(model_path, env=env)
        elif algo_name == "TD3":
            model = TD3.load(model_path, env=env)
        elif algo_name == "SAC":
            model = SAC.load(model_path, env=env)
        elif algo_name == "A2C":
            model = A2C.load(model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        # Evaluate
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs = env.reset()
            done = np.array([False])
            total_reward = 0.0
            steps = 0
            
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += float(reward[0])
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"  Episode {ep+1}/{n_episodes}: Reward={total_reward:.2f}, Steps={steps}")
        
        env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n{'='*70}")
    print(f"📈 {algo_name.upper()} RESULTS")
    print(f"{'='*70}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward:  {np.min(episode_rewards):.2f}")
    print(f"Max Reward:  {np.max(episode_rewards):.2f}")
    print(f"Mean Length: {mean_length:.1f} steps")
    print(f"{'='*70}\n")
    
    return {
        'algo': algo_name,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': mean_length,
        'rewards': episode_rewards,
    }


def plot_training_curves(training_results, save_dir="./figs"):
    """
    Plot training reward curves over time for all algorithms.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("📊 CREATING TRAINING CURVES")
    print(f"{'='*80}\n")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'RANDOM': '#95a5a6',
        'PPO': '#3498db',
        'TD3': '#e74c3c',
        'SAC': '#2ecc71',
        'A2C': '#f39c12'
    }
    
    has_data = False
    
    for train_result in training_results:
        algo_name = train_result['algo']
        log_dir = train_result['log_dir']
        
        # Try to read TensorBoard logs
        data = read_tensorboard_logs(log_dir, algo_name)
        
        if data is not None:
            timesteps, rewards = data
            
            # Smooth all curves using moving average (including RANDOM now)
            window = min(10, len(rewards) // 20 + 1)
            if len(rewards) > window:
                rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
                timesteps_smooth = timesteps[:len(rewards_smooth)]
            else:
                rewards_smooth = rewards
                timesteps_smooth = timesteps
            
            # Plot
            linestyle = '--' if algo_name == 'RANDOM' else '-'
            linewidth = 1.5 if algo_name == 'RANDOM' else 2
            
            ax.plot(timesteps_smooth, rewards_smooth, 
                   label=algo_name, color=colors.get(algo_name, '#95a5a6'),
                   linewidth=linewidth, alpha=0.8, linestyle=linestyle)
            
            # Also plot raw data with transparency for all algorithms
            ax.plot(timesteps, rewards, 
                   color=colors.get(algo_name, '#95a5a6'),
                   linewidth=0.5, alpha=0.15)
            
            has_data = True
            print(f"✅ Plotted {algo_name}: {len(timesteps)} data points")
    
    if not has_data:
        print("⚠️  No training data available to plot")
        plt.close()
        return
    
    ax.set_xlabel('Timesteps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax.set_title('Training Progress - Reward vs Timesteps', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"training_curves_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Training curves saved: {save_path}")
    
    latest_path = os.path.join(save_dir, "training_curves_latest.png")
    plt.savefig(latest_path, dpi=150, bbox_inches='tight')
    print(f"✅ Latest curves saved: {latest_path}\n")
    
    plt.close()


def plot_comparison_all(results, save_dir="./figs"):
    """
    Create comparison plots for all algorithms.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    algos = [r['algo'] for r in results]
    means = [r['mean_reward'] for r in results]
    stds = [r['std_reward'] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot
    ax1 = axes[0]
    colors_list = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    colors = [colors_list[i % len(colors_list)] for i in range(len(algos))]
    
    bars = ax1.bar(algos, means, yerr=stds, capsize=10, color=colors, 
                   alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Comparison - Mean Reward', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.0f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Box plot
    ax2 = axes[1]
    rewards_data = [r['rewards'] for r in results]
    bp = ax2.boxplot(rewards_data, labels=algos, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Reward Distribution', fontsize=12, fontweight='bold')
    ax2.set_title('Algorithm Comparison - Reward Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"algorithm_comparison_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {save_path}")
    
    latest_path = os.path.join(save_dir, "algorithm_comparison_latest.png")
    plt.savefig(latest_path, dpi=150, bbox_inches='tight')
    print(f"✅ Latest plot saved: {latest_path}\n")
    
    plt.close()


def print_final_comparison(results):
    """
    Print final comparison table.
    """
    print(f"\n{'='*100}")
    print(f"{'🏆 FINAL COMPARISON - HUMANOID-V5':^100}")
    print(f"{'='*100}\n")
    
    # Sort by mean reward
    results_sorted = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    
    # Print table
    print(f"{'Rank':<6} {'Algorithm':<12} {'Mean Reward':<20} {'Min':<12} {'Max':<12} {'Mean Length':<12}")
    print(f"{'-'*100}")
    
    for i, r in enumerate(results_sorted, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{rank_emoji:<6} {r['algo']:<12} {r['mean_reward']:>8.2f} ± {r['std_reward']:<6.2f}  "
              f"{r['min_reward']:>8.2f}    {r['max_reward']:>8.2f}    {r['mean_length']:>8.1f}")
    
    print(f"{'-'*100}\n")
    
    # Performance gain
    if len(results_sorted) > 1:
        best = results_sorted[0]
        worst = results_sorted[-1]
        gain = ((best['mean_reward'] - worst['mean_reward']) / abs(worst['mean_reward'])) * 100
        
        print(f"📊 Performance Gap:")
        print(f"   Best ({best['algo']}): {best['mean_reward']:.2f}")
        print(f"   Worst ({worst['algo']}): {worst['mean_reward']:.2f}")
        print(f"   Improvement: {gain:+.1f}%\n")
    
    print(f"{'='*100}\n")


if __name__ == "__main__":
    
    # Configuration
    # ALGORITHMS = ["RANDOM", "PPO", "A2C", "SAC"]
    ALGORITHMS = ["RANDOM", "PPO", "A2C"]
    TOTAL_TIMESTEPS = int(1e7)  # 20k timesteps
    N_ENVS = 32
    SEED = 42
    LOG_DIR = "./baseline_logs"
    FIGS_DIR = "./figs"
    N_EVAL_EPISODES = 20
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)
    
    print("\n" + "="*100)
    print(" "*30 + "🤖 HUMANOID-V5 ALGORITHM COMPARISON 🤖")
    print(" "*35 + "💪 POWERED BY RTX 4090 💪")
    print("="*100)
    print(f"\nAlgorithms to test: {', '.join(ALGORITHMS)}")
    print(f"Total timesteps per algorithm: {TOTAL_TIMESTEPS:,}")
    print(f"Evaluation episodes: {N_EVAL_EPISODES}")
    print(f"\n{'='*100}\n")
    
    # Train all algorithms
    training_results = []
    
    for algo in ALGORITHMS:
        result = train_algorithm(
            algo_name=algo,
            total_timesteps=TOTAL_TIMESTEPS,
            log_dir=LOG_DIR,
            n_envs=N_ENVS,
            seed=SEED,
        )
        training_results.append(result)
        
        # Sleep a bit between trainings
        print(f"\n⏸️  Waiting 5 seconds before next algorithm...\n")
        time.sleep(5)
    
    print(f"\n{'='*100}")
    print(" "*35 + "✅ ALL TRAINING COMPLETE!")
    print(f"{'='*100}\n")
    
    # Plot training curves
    plot_training_curves(training_results, save_dir=FIGS_DIR)
    
    # Evaluate all algorithms
    print(f"\n{'='*100}")
    print(" "*30 + "🔍 STARTING EVALUATION PHASE")
    print(f"{'='*100}\n")
    
    eval_results = []
    
    for train_result in training_results:
        eval_result = evaluate_algorithm(
            model_path=train_result['model_path'],
            vec_normalize_path=train_result['vec_normalize_path'],
            algo_name=train_result['algo'],
            n_episodes=N_EVAL_EPISODES,
            seed=SEED,
        )
        eval_results.append(eval_result)
    
    # Plot comparison
    plot_comparison_all(eval_results, save_dir=FIGS_DIR)
    
    # Print final comparison
    print_final_comparison(eval_results)
    
    # Save results to file
    results_file = os.path.join(LOG_DIR, "comparison_results.txt")
    with open(results_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("HUMANOID-V5 ALGORITHM COMPARISON RESULTS\n")
        f.write("="*100 + "\n\n")
        f.write(f"Total timesteps: {TOTAL_TIMESTEPS:,}\n")
        f.write(f"Evaluation episodes: {N_EVAL_EPISODES}\n\n")
        
        results_sorted = sorted(eval_results, key=lambda x: x['mean_reward'], reverse=True)
        
        for i, r in enumerate(results_sorted, 1):
            f.write(f"{i}. {r['algo']}: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}\n")
    
    print(f"📝 Results saved to: {results_file}")
    
    print("\n" + "="*100)
    print(" "*35 + "🎉 BENCHMARK COMPLETE! 🎉")
    print("="*100)
    print(f"\n📊 View training progress:")
    print(f"   tensorboard --logdir={LOG_DIR}")
    print(f"\n📈 Plots generated:")
    print(f"   {FIGS_DIR}/training_curves_latest.png  ← REWARD OVER TIME")
    print(f"   {FIGS_DIR}/algorithm_comparison_latest.png")
    print(f"\n📝 Detailed results:")
    print(f"   {results_file}")
    print("\n" + "="*100 + "\n")