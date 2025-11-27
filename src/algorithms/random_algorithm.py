import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.core.interfaces import TrainingResult, EvaluationResult, TrainingObserver
from src.config.settings import ExperimentConfig, AlgorithmType
from src.environments.env_factory import HumanoidEnvFactory


class RandomAlgorithm:
    """
    Random policy for baseline comparison.
    
    Not a true RL algorithm, but useful to establish:
    - Minimum expected performance
    - Sanity check that learning is happening
    
    Implements the same interface as other algorithms for consistency.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        env_factory: Optional[HumanoidEnvFactory] = None,
        observers: Optional[List[TrainingObserver]] = None,
    ):
        self.config = config
        self.env_factory = env_factory or HumanoidEnvFactory(config.environment)
        self.observers = observers or []
        self._log_dir: Optional[Path] = None
    
    @property
    def name(self) -> str:
        return "RANDOM"
    
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.RANDOM
    
    @property
    def is_on_policy(self) -> bool:
        return True  # Conceptually, random is "on-policy"
    
    def _setup_log_dir(self) -> Path:
        """Create timestamped log directory."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = self.config.training.log_dir / f"random_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = log_dir
        return log_dir
    
    def train(self, total_timesteps: Optional[int] = None) -> TrainingResult:
        """
        'Train' random policy - actually just collects random episodes.
        
        This establishes a baseline for comparison with learned policies.
        """
        timesteps = total_timesteps or self.config.training.total_timesteps
        seed = self.config.training.seed
        
        print(f"\n{'='*80}")
        print(f"🎲 RUNNING RANDOM POLICY")
        print(f"{'='*80}")
        print(f"Total timesteps: {timesteps:,}")
        print(f"{'='*80}\n")
        
        self._setup_log_dir()
        
        # Create single environment
        env = self.env_factory.create_single_env(seed=seed)
        
        # TensorBoard writer
        writer = SummaryWriter(
            log_dir=str(self._log_dir / "random_humanoid_1")
        )
        
        # Run random episodes
        start_time = time.time()
        timestep = 0
        episode = 0
        episode_rewards = []
        episode_lengths = []
        last_log_timestep = 0
        log_frequency = 10_000
        rewards_buffer = []
        
        print(f"🎯 Running random policy (logging every {log_frequency} timesteps)...\n")
        
        while timestep < timesteps:
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done and timestep < timesteps:
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
            
            # Log periodically
            if timestep - last_log_timestep >= log_frequency or timestep >= timesteps:
                mean_reward = np.mean(rewards_buffer) if rewards_buffer else 0.0
                writer.add_scalar("rollout/ep_rew_mean", mean_reward, timestep)
                last_log_timestep = timestep
                rewards_buffer = []
                
                print(f"  Episode {episode} | Timestep {timestep:,}/{timesteps:,} | "
                      f"Mean Reward: {mean_reward:.2f}")
        
        writer.close()
        env.close()
        
        training_time = time.time() - start_time
        
        # Save info file
        info_path = self._log_dir / "random_info.txt"
        with open(info_path, 'w') as f:
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
        
        return TrainingResult(
            algorithm_name=self.name,
            model_path=info_path,
            vec_normalize_path=None,
            training_time_seconds=training_time,
            log_dir=self._log_dir,
            total_timesteps=timestep,
        )
    
    def evaluate(
        self,
        model_path: Path = None,
        vec_normalize_path: Path = None,
        n_episodes: int = 20,
    ) -> EvaluationResult:
        """Evaluate random policy."""
        seed = self.config.training.seed
        env = self.env_factory.create_single_env(seed=seed)
        
        episode_rewards = []
        episode_lengths = []
        
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING RANDOM POLICY")
        print(f"{'='*70}\n")
        
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
        
        result = EvaluationResult(
            algorithm_name=self.name,
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            min_reward=float(np.min(episode_rewards)),
            max_reward=float(np.max(episode_rewards)),
            mean_length=float(np.mean(episode_lengths)),
            episode_rewards=episode_rewards,
            episode_lengths=[int(l) for l in episode_lengths],
        )
        
        print(f"\n{'='*70}")
        print(f"📈 RANDOM POLICY RESULTS")
        print(f"{'='*70}")
        print(f"Mean Reward: {result.reward_summary}")
        print(f"{'='*70}\n")
        
        return result