from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional
from pathlib import Path

import torch.nn as nn


class AlgorithmType(Enum):
    """Supported RL algorithms."""
    RANDOM = auto()
    PPO = auto()
    TD3 = auto()
    SAC = auto()
    A2C = auto()


@dataclass(frozen=True)
class NetworkConfig:
    """Neural network architecture configuration."""
    hidden_layers: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    
    def to_policy_kwargs(self, is_on_policy: bool = True) -> Dict[str, Any]:
        """Convert to stable-baselines3 policy_kwargs format."""
        
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }
        
        activation_fn = activation_map.get(self.activation.lower(), nn.ReLU)
        
        if is_on_policy:
            return {
                "net_arch": {
                    "pi": list(self.hidden_layers),
                    "vf": list(self.hidden_layers),
                },
                "activation_fn": activation_fn,
            }
        else:
            return {
                "net_arch": {
                    "pi": list(self.hidden_layers),
                    "qf": list(self.hidden_layers),
                },
                "activation_fn": activation_fn,
            }


@dataclass(frozen=True)
class EnvironmentConfig:
    """Environment configuration."""
    env_id: str = "Humanoid-v5"
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = (1.0, 2.0)
    n_envs: int = 8
    normalize_obs: bool = True
    normalize_reward: bool = False
    clip_obs: float = 10.0
    gamma: float = 0.99


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""
    total_timesteps: int = int(1e7)
    seed: int = 42
    checkpoint_freq: int = 100_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 5
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    
    def __post_init__(self):
        # Ensure log_dir is a Path
        if isinstance(self.log_dir, str):
            object.__setattr__(self, 'log_dir', Path(self.log_dir))


@dataclass(frozen=True)
class PPOConfig:
    """PPO-specific hyperparameters."""
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class TD3Config:
    """TD3-specific hyperparameters."""
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 1_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = -1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    action_noise_sigma: float = 0.1


@dataclass(frozen=True)
class SACConfig:
    """SAC-specific hyperparameters."""
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 1_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"


@dataclass(frozen=True)
class A2CConfig:
    """A2C-specific hyperparameters."""
    learning_rate: float = 7e-4
    n_steps: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class ExperimentConfig:
    """Complete experiment configuration - mutable for building."""
    algorithms: List[AlgorithmType] = field(
        default_factory=lambda: [AlgorithmType.PPO]
    )
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    sac: SACConfig = field(default_factory=SACConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    figs_dir: Path = field(default_factory=lambda: Path("./figs"))
    n_eval_episodes_final: int = 20
    
    def get_algo_config(self, algo_type: AlgorithmType):
        """Get algorithm-specific config."""
        config_map = {
            AlgorithmType.PPO: self.ppo,
            AlgorithmType.TD3: self.td3,
            AlgorithmType.SAC: self.sac,
            AlgorithmType.A2C: self.a2c,
            AlgorithmType.RANDOM: None,
        }
        return config_map.get(algo_type)