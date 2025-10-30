import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs import make_humanoid_env
from config import Config


def make_env(rank, cfg, render_mode=None):
    # Factory function for creating environments
    def _init():
        env = make_humanoid_env(render_mode=render_mode, **cfg.get_env_kwargs())
        env = Monitor(env)
        env.reset(seed=rank)
        return env
    return _init


def create_vectorized_env(n_envs, cfg, render_mode=None, use_subproc=True):
    # Create multiple parallel environments
    print(f"Creating {n_envs} parallel environments...")

    env_fns = [make_env(i, cfg, render_mode) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
        print(f"   [OK] SubprocVecEnv created (true parallelism)")
    else:
        vec_env = DummyVecEnv(env_fns)
        print(f"   [OK] DummyVecEnv created (sequential)")

    return vec_env


def create_callbacks(cfg, eval_env=None):
    # Create training callbacks
    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_freq,
        save_path=cfg.models_dir,
        name_prefix="ppo_humanoid",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    print(f"   Checkpoint every {cfg.save_freq:,} steps -> {cfg.models_dir}")

    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=cfg.models_dir,
            log_path=cfg.log_dir,
            eval_freq=cfg.eval_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)
        print(f"   Evaluation every {cfg.eval_freq:,} steps")

    return callbacks


def train(cfg, total_timesteps=None, n_envs=None, model_name="ppo_humanoid_final",
          resume_from=None, use_eval=True):

    print("\n" + "=" * 60)
    print("PPO TRAINING - HUMANOID-V5")
    print("=" * 60 + "\n")

    total_timesteps = total_timesteps or cfg.total_timesteps
    n_envs = n_envs or cfg.n_envs

    print(f"Parameters:")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Parallel envs: {n_envs}")
    print(f"   Learning rate: {cfg.learning_rate}")
    print(f"   Batch size: {cfg.batch_size}")
    print(f"   Network: {cfg.policy_layers}\n")

    # Create environments
    print("="*60)
    print("STEP 1: Creating environments")
    print("="*60 + "\n")

    train_env = create_vectorized_env(n_envs=n_envs, cfg=cfg, render_mode=None, use_subproc=True)

    eval_env = None
    if use_eval:
        print(f"\nCreating evaluation environment...")
        eval_env = DummyVecEnv([make_env(rank=9999, cfg=cfg, render_mode=None)])
        print(f"   [OK] Eval env created\n")

    # Create or load model
    print("="*60)
    print("STEP 2: Configuring PPO model")
    print("="*60 + "\n")

    if resume_from is not None:
        print(f"Loading model from: {resume_from}")
        model = PPO.load(resume_from, env=train_env, tensorboard_log=cfg.tensorboard_dir)
        print(f"   [OK] Model loaded, resuming training\n")
    else:
        print(f"Creating new PPO model...")
        ppo_kwargs = cfg.get_ppo_kwargs()
        ppo_kwargs['policy_kwargs'] = dict(net_arch=cfg.network_arch)
        model = PPO(policy="MlpPolicy", env=train_env, **ppo_kwargs)
        print(f"   [OK] Model created\n")

    # Create callbacks
    print("="*60)
    print("STEP 3: Configuring callbacks")
    print("="*60 + "\n")

    callbacks = create_callbacks(cfg=cfg, eval_env=eval_env)

    # Start training
    print("\n" + "="*60)
    print("STEP 4: STARTING TRAINING")
    print("="*60 + "\n")

    print(f"Estimated time: ~{total_timesteps / (n_envs * 1000):.0f} minutes")
    print(f"TensorBoard: tensorboard --logdir={cfg.tensorboard_dir}")
    print(f"\n{'='*60}\n")

    start_time = time.time()

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                   log_interval=cfg.log_interval, progress_bar=False)

        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[OK] TRAINING COMPLETE!")
        print(f"Total time: {training_time/60:.1f} minutes")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print(f"\n\n[ATTENTION] Training interrupted by user (Ctrl+C)")
        training_time = time.time() - start_time
        print(f"Time elapsed: {training_time/60:.1f} minutes\n")

    # Save final model
    print("="*60)
    print("STEP 5: Saving final model")
    print("="*60 + "\n")

    final_model_path = os.path.join(cfg.models_dir, model_name)
    model.save(final_model_path)
    print(f"[OK] Model saved: {final_model_path}.zip\n")

    train_env.close()
    if eval_env is not None:
        eval_env.close()

    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60 + "\n")

    print(f"Next steps:")
    print(f"   1. Evaluate: python main.py evaluate {final_model_path}.zip")
    print(f"   2. Visualize: python main.py visualize {final_model_path}.zip")
    print(f"   3. TensorBoard: tensorboard --logdir={cfg.tensorboard_dir}\n")

    return model


if __name__ == "__main__":
    print("Testing train module\n")
    print("[ATTENTION] Test mode: short training (10k steps)")
    print("   For full training, use main.py\n")

    input("Press ENTER to start test...")

    cfg = Config("configs/ppo_humanoid.yaml")
    model = train(cfg=cfg, total_timesteps=10_000, n_envs=2,
                 model_name="test_model", use_eval=False)

    print("\n[OK] Test complete! Train module works.")
