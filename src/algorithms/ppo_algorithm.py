from typing import Any

from stable_baselines3 import PPO

from src.core.base_algorithm import BaseAlgorithm
from src.config.settings import AlgorithmType


class PPOAlgorithm(BaseAlgorithm):
    """
    Proximal Policy Optimization implementation.
    
    PPO is an on-policy algorithm that:
    - Uses multiple parallel environments for sample efficiency
    - Clips policy updates to prevent destructive large updates
    - Is generally robust and easy to tune
    """
    
    @property
    def name(self) -> str:
        return "PPO"
    
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.PPO
    
    @property
    def is_on_policy(self) -> bool:
        return True
    
    def _create_model(self) -> PPO:
        """Create PPO model with configured hyperparameters."""
        ppo_config = self.config.ppo
        
        model = PPO(
            policy="MlpPolicy",
            env=self._train_env,
            learning_rate=ppo_config.learning_rate,
            n_steps=ppo_config.n_steps,
            batch_size=ppo_config.batch_size,
            n_epochs=ppo_config.n_epochs,
            gamma=ppo_config.gamma,
            gae_lambda=ppo_config.gae_lambda,
            clip_range=ppo_config.clip_range,
            ent_coef=ppo_config.ent_coef,
            vf_coef=ppo_config.vf_coef,
            max_grad_norm=ppo_config.max_grad_norm,
            policy_kwargs=self._get_policy_kwargs(),
            tensorboard_log=str(self._log_dir),
            device=self.device,
            verbose=1,
            seed=self.config.training.seed,
        )
        
        return model