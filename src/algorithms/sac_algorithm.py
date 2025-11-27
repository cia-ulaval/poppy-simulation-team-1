from typing import Any

from stable_baselines3 import SAC

from src.core.base_algorithm import BaseAlgorithm
from src.config.settings import AlgorithmType


class SACAlgorithm(BaseAlgorithm):
    """
    Soft Actor-Critic implementation.
    
    SAC is an off-policy algorithm that:
    - Maximizes both expected reward and entropy
    - Uses automatic temperature adjustment
    - Is sample efficient and stable
    """
    
    @property
    def name(self) -> str:
        return "SAC"
    
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.SAC
    
    @property
    def is_on_policy(self) -> bool:
        return False
    
    def _create_model(self) -> SAC:
        """Create SAC model with configured hyperparameters."""
        sac_config = self.config.sac
        
        model = SAC(
            policy="MlpPolicy",
            env=self._train_env,
            learning_rate=sac_config.learning_rate,
            buffer_size=sac_config.buffer_size,
            learning_starts=sac_config.learning_starts,
            batch_size=sac_config.batch_size,
            tau=sac_config.tau,
            gamma=sac_config.gamma,
            train_freq=sac_config.train_freq,
            gradient_steps=sac_config.gradient_steps,
            ent_coef=sac_config.ent_coef,
            policy_kwargs=self._get_policy_kwargs(),
            tensorboard_log=str(self._log_dir),
            device=self.device,
            verbose=1,
            seed=self.config.training.seed,
        )
        
        return model