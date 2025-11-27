from typing import Any

import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from src.core.base_algorithm import BaseAlgorithm
from src.config.settings import AlgorithmType


class TD3Algorithm(BaseAlgorithm):
    """
    Twin Delayed DDPG implementation.
    
    TD3 is an off-policy algorithm that:
    - Uses twin Q-networks to reduce overestimation
    - Delays policy updates for stability
    - Adds noise to target policy for smoothing
    """
    
    @property
    def name(self) -> str:
        return "TD3"
    
    @property
    def algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.TD3
    
    @property
    def is_on_policy(self) -> bool:
        return False
    
    def _create_action_noise(self) -> NormalActionNoise:
        """Create action noise for exploration."""
        n_actions = self._train_env.action_space.shape[-1]
        sigma = self.config.td3.action_noise_sigma
        
        return NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=sigma * np.ones(n_actions),
        )
    
    def _create_model(self) -> TD3:
        """Create TD3 model with configured hyperparameters."""
        td3_config = self.config.td3
        
        model = TD3(
            policy="MlpPolicy",
            env=self._train_env,
            learning_rate=td3_config.learning_rate,
            buffer_size=td3_config.buffer_size,
            learning_starts=td3_config.learning_starts,
            batch_size=td3_config.batch_size,
            tau=td3_config.tau,
            gamma=td3_config.gamma,
            train_freq=td3_config.train_freq,
            gradient_steps=td3_config.gradient_steps,
            action_noise=self._create_action_noise(),
            policy_delay=td3_config.policy_delay,
            target_policy_noise=td3_config.target_policy_noise,
            target_noise_clip=td3_config.target_noise_clip,
            policy_kwargs=self._get_policy_kwargs(),
            tensorboard_log=str(self._log_dir),
            device=self.device,
            verbose=1,
            seed=self.config.training.seed,
        )
        
        return model