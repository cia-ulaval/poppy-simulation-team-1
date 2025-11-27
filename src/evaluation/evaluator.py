from pathlib import Path
from typing import Optional, Union
import time

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, TD3, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.core.interfaces import EvaluationResult
from src.config.settings import EnvironmentConfig, AlgorithmType


class ModelEvaluator:
    """
    Evaluates trained RL models.
    
    Responsibilities:
    - Load trained models
    - Run evaluation episodes
    - Collect and report metrics
    
    Supports both visualization and batch evaluation modes.
    """
    
    # Map algorithm types to their classes
    _model_classes = {
        AlgorithmType.PPO: PPO,
        AlgorithmType.TD3: TD3,
        AlgorithmType.SAC: SAC,
        AlgorithmType.A2C: A2C,
    }
    
    def __init__(
        self,
        env_config: EnvironmentConfig,
        algorithm_type: AlgorithmType,
    ):
        self.env_config = env_config
        self.algorithm_type = algorithm_type
    
    def _create_base_env(
        self,
        seed: int,
        render: bool = False,
    ) -> gym.Env:
        """Create base environment for evaluation."""
        env = gym.make(
            self.env_config.env_id,
            render_mode="human" if render else None,
            terminate_when_unhealthy=self.env_config.terminate_when_unhealthy,
            healthy_z_range=self.env_config.healthy_z_range,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    
    def _load_model(
        self,
        model_path: Path,
        env: VecNormalize,
    ):
        """Load the appropriate model type."""
        if self.algorithm_type == AlgorithmType.RANDOM:
            return None
        
        model_class = self._model_classes.get(self.algorithm_type)
        if model_class is None:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")
        
        return model_class.load(str(model_path), env=env)
    
    def _auto_detect_vec_normalize(
        self,
        model_path: Path,
    ) -> Optional[Path]:
        """Auto-detect VecNormalize path from model path."""
        model_path = Path(model_path)
        model_dir = model_path.parent
        
        # Try common patterns
        patterns = [
            model_dir / "vec_normalize.pkl",
            model_path.with_suffix('.pkl').with_name(
                model_path.stem.replace('_final', '_vecnormalize') + '.pkl'
            ),
        ]
        
        # For checkpoint files like ppo_humanoid_1000000_steps.zip
        if "_steps" in model_path.stem:
            checkpoint_pattern = model_path.stem.replace(
                model_path.stem.split('_')[0] + '_humanoid',
                model_path.stem.split('_')[0] + '_humanoid_vecnormalize'
            ) + '.pkl'
            patterns.append(model_dir / checkpoint_pattern)
        
        for pattern in patterns:
            if pattern.exists():
                return pattern
        
        return None
    
    def evaluate(
        self,
        model_path: Path,
        vec_normalize_path: Optional[Path] = None,
        n_episodes: int = 20,
        seed: int = 42,
        render: bool = False,
        fps: int = 50,
    ) -> EvaluationResult:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to saved model
            vec_normalize_path: Path to VecNormalize stats (auto-detected if None)
            n_episodes: Number of episodes to evaluate
            seed: Random seed
            render: Whether to render environment
            fps: Frames per second for rendering
            
        Returns:
            EvaluationResult with metrics
        """
        algorithm_name = self.algorithm_type.name
        
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING {algorithm_name}")
        print(f"{'='*70}")
        print(f"Model: {model_path}")
        
        # Auto-detect vec_normalize if not provided
        if vec_normalize_path is None:
            vec_normalize_path = self._auto_detect_vec_normalize(model_path)
        
        print(f"VecNormalize: {vec_normalize_path or 'None'}")
        print(f"Episodes: {n_episodes}")
        print(f"Mode: {'RENDER' if render else 'HEADLESS'}")
        print(f"{'='*70}\n")
        
        # Create environment
        base_env = self._create_base_env(seed=seed, render=render)
        env = DummyVecEnv([lambda: base_env])
        
        # Load normalization stats
        vec_normalize_loaded = False
        if vec_normalize_path and vec_normalize_path.exists():
            try:
                env = VecNormalize.load(str(vec_normalize_path), env)
                env.training = False
                env.norm_reward = False
                vec_normalize_loaded = True
                print(f"✅ Loaded normalization stats")
            except Exception as e:
                print(f"⚠️  Failed to load VecNormalize: {e}")
        
        # Load model
        model = self._load_model(model_path, env)
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        sleep_time = 1.0 / fps if (render and fps > 0) else 0
        
        start_time = time.time()
        
        for ep in range(n_episodes):
            obs = env.reset()
            done = np.array([False])
            total_reward = 0.0
            steps = 0
            
            while not done[0]:
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Random policy
                    action = [env.action_space.sample()]
                
                obs, reward, done, info = env.step(action)
                total_reward += float(reward[0])
                steps += 1
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"  Episode {ep+1}/{n_episodes}: Reward={total_reward:.2f}, Steps={steps}")
        
        eval_time = time.time() - start_time
        env.close()
        
        # Compute statistics
        result = EvaluationResult(
            algorithm_name=algorithm_name,
            mean_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            min_reward=float(np.min(episode_rewards)),
            max_reward=float(np.max(episode_rewards)),
            mean_length=float(np.mean(episode_lengths)),
            episode_rewards=episode_rewards,
            episode_lengths=[int(l) for l in episode_lengths],
        )
        
        # Print results
        self._print_results(result, eval_time, vec_normalize_loaded)
        
        return result
    
    def _print_results(
        self,
        result: EvaluationResult,
        eval_time: float,
        vec_normalize_loaded: bool,
    ) -> None:
        """Print evaluation results."""
        print(f"\n{'='*70}")
        print(f"📈 {result.algorithm_name} RESULTS")
        print(f"{'='*70}")
        print(f"⏱️  Time: {eval_time:.1f}s ({eval_time/len(result.episode_rewards):.1f}s/episode)")
        print(f"\n🎯 Rewards:")
        print(f"    Mean:  {result.mean_reward:>8.2f} ± {result.std_reward:.2f}")
        print(f"    Min:   {result.min_reward:>8.2f}")
        print(f"    Max:   {result.max_reward:>8.2f}")
        print(f"\n📏 Episode Length:")
        print(f"    Mean:  {result.mean_length:>8.1f} steps")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if vec_normalize_loaded:
            if result.mean_reward > 5000:
                print(f"    🎉 EXCELLENT! The robot walks very well!")
            elif result.mean_reward > 2000:
                print(f"    ✅ GOOD! The robot has learned to walk!")
            elif result.mean_reward > 500:
                print(f"    🆗 OKAY! The robot is starting to walk!")
            elif result.mean_reward > 100:
                print(f"    ⚠️  POOR! The robot is struggling!")
            else:
                print(f"    ❌ BAD! The robot hasn't learned much!")
        else:
            print(f"    ⚠️  Hard to evaluate without normalization stats!")
        
        print(f"{'='*70}\n")


class BatchEvaluator:
    """
    Evaluates multiple models and compares them.
    
    Uses Composite pattern to evaluate multiple algorithms.
    """
    
    def __init__(self, env_config: EnvironmentConfig):
        self.env_config = env_config
    
    def evaluate_all(
        self,
        training_results: list,
        n_episodes: int = 20,
        seed: int = 42,
    ) -> list[EvaluationResult]:
        """Evaluate all trained models."""
        results = []
        
        for train_result in training_results:
            algo_type = AlgorithmType[train_result.algorithm_name]
            
            evaluator = ModelEvaluator(
                env_config=self.env_config,
                algorithm_type=algo_type,
            )
            
            result = evaluator.evaluate(
                model_path=train_result.model_path,
                vec_normalize_path=train_result.vec_normalize_path,
                n_episodes=n_episodes,
                seed=seed,
            )
            results.append(result)
        
        return results