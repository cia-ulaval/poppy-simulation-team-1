from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Optional, Dict, Any, List, Tuple, runtime_checkable

import numpy as np


@dataclass
class TrainingResult:
    """Immutable result of a training run."""
    algorithm_name: str
    model_path: Path
    vec_normalize_path: Optional[Path]
    training_time_seconds: float
    log_dir: Path
    total_timesteps: int
    
    @property
    def training_time_minutes(self) -> float:
        return self.training_time_seconds / 60
    
    @property
    def training_time_hours(self) -> float:
        return self.training_time_seconds / 3600


@dataclass
class EvaluationResult:
    """Immutable result of model evaluation."""
    algorithm_name: str
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_length: float
    episode_rewards: List[float]
    episode_lengths: List[int]
    
    @property
    def reward_summary(self) -> str:
        return f"{self.mean_reward:.2f} ± {self.std_reward:.2f}"


@runtime_checkable
class Trainable(Protocol):
    """Protocol for trainable algorithms."""
    
    def train(self, total_timesteps: int) -> TrainingResult:
        """Train the algorithm for specified timesteps."""
        ...
    
    @property
    def name(self) -> str:
        """Algorithm name."""
        ...


@runtime_checkable
class Evaluable(Protocol):
    """Protocol for evaluable models."""
    
    def evaluate(self, n_episodes: int) -> EvaluationResult:
        """Evaluate the model for n episodes."""
        ...


@runtime_checkable
class Saveable(Protocol):
    """Protocol for saveable models."""
    
    def save(self, path: Path) -> None:
        """Save model to path."""
        ...
    
    @classmethod
    def load(cls, path: Path) -> "Saveable":
        """Load model from path."""
        ...


class Algorithm(ABC):
    """
    Abstract base class for RL algorithms.
    Implements Template Method pattern for common training flow.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass
    
    @property
    @abstractmethod
    def is_on_policy(self) -> bool:
        """Return True if on-policy algorithm."""
        pass
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Factory method to create the model. Subclasses implement this."""
        pass
    
    @abstractmethod
    def train(self, total_timesteps: int) -> TrainingResult:
        """Train the algorithm."""
        pass
    
    @abstractmethod
    def evaluate(self, n_episodes: int) -> EvaluationResult:
        """Evaluate the trained model."""
        pass
    
    def _setup_environment(self):
        """Template method hook for environment setup."""
        pass
    
    def _setup_callbacks(self):
        """Template method hook for callback setup."""
        pass
    
    def _post_training(self):
        """Template method hook for post-training operations."""
        pass


class EnvironmentFactory(Protocol):
    """Protocol for environment factories."""
    
    def create_training_env(self, n_envs: int, seed: int) -> Any:
        """Create training environment(s)."""
        ...
    
    def create_eval_env(self, seed: int) -> Any:
        """Create evaluation environment."""
        ...


class TrainingObserver(Protocol):
    """
    Observer pattern for training events.
    Allows decoupled logging, metrics, etc.
    """
    
    def on_training_start(self, algorithm: str, config: Dict[str, Any]) -> None:
        """Called when training starts."""
        ...
    
    def on_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Called on each logged step."""
        ...
    
    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        """Called at end of each episode."""
        ...
    
    def on_training_end(self, result: TrainingResult) -> None:
        """Called when training ends."""
        ...


class MetricsReader(Protocol):
    """Protocol for reading training metrics."""
    
    def read_rewards(self, log_dir: Path) -> Optional[Tuple[List[int], List[float]]]:
        """Read timesteps and rewards from logs."""
        ...