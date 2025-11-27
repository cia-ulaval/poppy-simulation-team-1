from typing import Any

from stable_baselines3 import A2C

from src.core.base_algorithm import BaseAlgorithm
from src.config.settings import AlgorithmType


class A2CAlgorithm(BaseAlgorithm):
    """
    Advantage Actor-Critic implementation.
    
    A2C is an on-policy algorithm that:
    - Uses synchronous updates (vs A3C's async)
    - Is simpler than PPO but less stable
    - Works well with multiple parallel environments
    """

    @property
    def name(self) -> str:
        return "A2C"
    
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.A2C
    
    @property
    def is_on_policy(self) -> bool:
        return True
    
    def _create_model(self) -> A2C:
        """Create A2C model with configured hyperparameters."""
        a2c_config = self.config.a2c
        
        model = A2C(
            policy="MlpPolicy",
            env=self._train_env,
            learning_rate=a2c_config.learning_rate,
            n_steps=a2c_config.n_steps,
            gamma=a2c_config.gamma,
            gae_lambda=a2c_config.gae_lambda,
            ent_coef=a2c_config.ent_coef,
            vf_coef=a2c_config.vf_coef,
            max_grad_norm=a2c_config.max_grad_norm,
            policy_kwargs=self._get_policy_kwargs(),
            tensorboard_log=str(self._log_dir),
            device=self.device,
            verbose=1,
            seed=self.config.training.seed,
        )
        
        return model