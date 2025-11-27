import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

import numpy as np
import torch

from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from src.core.interfaces import TrainingResult, EvaluationResult, TrainingObserver
from src.config.settings import (
    ExperimentConfig,
    AlgorithmType,
    NetworkConfig,
)
from src.environments.env_factory import HumanoidEnvFactory
from src.evaluation.evaluator import ModelEvaluator


class BaseAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    
    Uses Template Method pattern:
    - train() defines the skeleton of the training algorithm
    - Subclasses override specific steps (_create_model, etc.)
    
    Also uses Strategy pattern - different algorithms are interchangeable strategies.
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
        
        self._model: Any = None
        self._train_env: Optional[VecEnv] = None
        self._eval_env: Optional[VecEnv] = None
        self._log_dir: Optional[Path] = None
        
        # Device detection
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass
    
    @property
    @abstractmethod
    def algorithm_type(self) -> AlgorithmType:
        """Return algorithm type enum."""
        pass
    
    @property
    @abstractmethod
    def is_on_policy(self) -> bool:
        """Return True if on-policy algorithm."""
        pass
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def n_envs(self) -> int:
        """Number of parallel environments to use."""
        if self.is_on_policy:
            return self.config.environment.n_envs
        return 1  # Off-policy uses single env
    
    def _get_policy_kwargs(self) -> dict:
        """Get policy kwargs for network architecture."""
        return self.config.network.to_policy_kwargs(self.is_on_policy)
    
    def _setup_log_dir(self) -> Path:
        """Create timestamped log directory."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = self.config.training.log_dir / f"{self.name.lower()}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = log_dir
        return log_dir
    
    def _setup_environments(self) -> None:
        """Setup training and evaluation environments."""
        seed = self.config.training.seed
        
        # Training env
        self._train_env = self.env_factory.create_training_env(
            n_envs=self.n_envs,
            seed=seed,
        )
        
        # Evaluation env
        self._eval_env = self.env_factory.create_eval_env(
            seed=seed + 1000,
        )
    
    def _create_callbacks(self) -> List[BaseCallback]:
        """Create training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_freq = max(
            self.config.training.checkpoint_freq // self.n_envs,
            1000
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(self._log_dir),
            name_prefix=f"{self.name.lower()}_humanoid",
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Eval callback
        if self._eval_env is not None:
            eval_callback = EvalCallback(
                self._eval_env,
                best_model_save_path=str(self._log_dir),
                log_path=str(self._log_dir),
                eval_freq=self.config.training.eval_freq,
                n_eval_episodes=self.config.training.n_eval_episodes,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        return callbacks
    
    @abstractmethod
    def _create_model(self) -> Any:
        """
        Factory method to create the specific model.
        Subclasses must implement this.
        """
        pass
    
    def _notify_training_start(self) -> None:
        """Notify observers of training start."""
        for observer in self.observers:
            observer.on_training_start(
                self.name,
                {"device": str(self.device), "n_envs": self.n_envs}
            )
    
    def _notify_training_end(self, result: TrainingResult) -> None:
        """Notify observers of training end."""
        for observer in self.observers:
            observer.on_training_end(result)
    
    def _print_training_header(self) -> None:
        """Print training start information."""
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING {self.name.upper()}")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total timesteps: {self.config.training.total_timesteps:,}")
        print(f"Parallel environments: {self.n_envs}")
        print(f"Log directory: {self._log_dir}")
        print(f"{'='*80}\n")
    
    def _print_training_footer(self, training_time: float) -> None:
        """Print training end information."""
        print(f"\n{'='*80}")
        print(f"✅ {self.name} TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Time: {training_time/3600:.2f} hours ({training_time/60:.1f} minutes)")
        print(f"{'='*80}\n")
    
    def train(self, total_timesteps: Optional[int] = None) -> TrainingResult:
        """
        Template method for training.
        Defines the skeleton of the training algorithm.
        """
        timesteps = total_timesteps or self.config.training.total_timesteps
        
        # Setup
        self._setup_log_dir()
        self._setup_environments()
        callbacks = self._create_callbacks()
        
        self._print_training_header()
        self._notify_training_start()
        
        # Create model (factory method - implemented by subclasses)
        self._model = self._create_model()
        
        # Train
        start_time = time.time()
        
        try:
            self._model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                tb_log_name=f"{self.name.lower()}_humanoid",
                progress_bar=True,
            )
        except KeyboardInterrupt:
            print(f"\n⚠️ {self.name} training interrupted by user!")
        
        training_time = time.time() - start_time
        
        # Save final model
        model_path = self._log_dir / f"{self.name.lower()}_humanoid_final.zip"
        self._model.save(str(model_path))
        
        vec_normalize_path = self._log_dir / "vec_normalize.pkl"
        if isinstance(self._train_env, VecNormalize):
            self._train_env.save(str(vec_normalize_path))
        
        self._print_training_footer(training_time)
        
        # Create result
        result = TrainingResult(
            algorithm_name=self.name,
            model_path=model_path,
            vec_normalize_path=vec_normalize_path if vec_normalize_path.exists() else None,
            training_time_seconds=training_time,
            log_dir=self._log_dir,
            total_timesteps=timesteps,
        )
        
        self._notify_training_end(result)
        
        # Cleanup
        self._cleanup()
        
        return result
    
    def _cleanup(self) -> None:
        """Cleanup environments after training."""
        if self._train_env is not None:
            self._train_env.close()
        if self._eval_env is not None:
            self._eval_env.close()
    
    def evaluate(
        self,
        model_path: Path,
        vec_normalize_path: Optional[Path] = None,
        n_episodes: int = 20,
    ) -> EvaluationResult:
        """Evaluate a trained model."""
        
        evaluator = ModelEvaluator(
            env_config=self.config.environment,
            algorithm_type=self.algorithm_type,
        )
        
        return evaluator.evaluate(
            model_path=model_path,
            vec_normalize_path=vec_normalize_path,
            n_episodes=n_episodes,
            seed=self.config.training.seed,
        )