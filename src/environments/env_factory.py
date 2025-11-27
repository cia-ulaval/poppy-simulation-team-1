from typing import Callable, Optional

import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.config.settings import EnvironmentConfig


class HumanoidEnvFactory:
    """
    Factory for creating Humanoid-v5 environments.
    
    Encapsulates all environment creation logic including:
    - Base environment setup
    - Vectorization (Dummy or Subprocess)
    - Observation/reward normalization
    - Monitoring
    
    Benefits:
    - Single Responsibility: Only handles env creation
    - Open/Closed: Easy to extend for new env types
    - Dependency Inversion: Other code depends on factory, not gym details
    """
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
    
    def _make_env_fn(
        self,
        rank: int = 0,
        seed: int = 0,
        render: bool = False,
    ) -> Callable[[], gym.Env]:
        """
        Create a thunk (parameterless function) that creates an environment.
        Required for vectorized environments.
        """
        config = self.config
        
        def _init() -> gym.Env:
            env = gym.make(
                config.env_id,
                render_mode="human" if render else None,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                healthy_z_range=config.healthy_z_range,
            )
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        
        set_random_seed(seed)
        return _init
    
    def create_single_env(
        self,
        seed: int = 0,
        render: bool = False,
    ) -> gym.Env:
        """Create a single non-vectorized environment."""
        env = gym.make(
            self.config.env_id,
            render_mode="human" if render else None,
            terminate_when_unhealthy=self.config.terminate_when_unhealthy,
            healthy_z_range=self.config.healthy_z_range,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    
    def create_training_env(
        self,
        n_envs: int = 1,
        seed: int = 0,
        use_subprocess: bool = True,
    ) -> VecNormalize:
        """
        Create vectorized training environment with normalization.
        
        Args:
            n_envs: Number of parallel environments
            seed: Random seed
            use_subprocess: Use SubprocVecEnv for true parallelism
            
        Returns:
            VecNormalize wrapped vectorized environment
        """
        env_fns = [
            self._make_env_fn(rank=i, seed=seed)
            for i in range(n_envs)
        ]
        
        if n_envs > 1 and use_subprocess:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)
        
        # Apply normalization
        vec_env = VecNormalize(
            vec_env,
            norm_obs=self.config.normalize_obs,
            norm_reward=self.config.normalize_reward,
            clip_obs=self.config.clip_obs,
            gamma=self.config.gamma,
        )
        
        return vec_env
    
    def create_eval_env(
        self,
        seed: int = 0,
    ) -> VecNormalize:
        """
        Create evaluation environment.
        
        Evaluation env uses:
        - Single environment (no parallelism needed)
        - Same normalization settings but training=False
        """
        vec_env = DummyVecEnv([self._make_env_fn(seed=seed)])
        
        vec_env = VecNormalize(
            vec_env,
            norm_obs=self.config.normalize_obs,
            norm_reward=self.config.normalize_reward,
            clip_obs=self.config.clip_obs,
            gamma=self.config.gamma,
            training=False,  #TODO Don't update stats during eval
        )
        
        return vec_env
    
    def create_render_env(self, seed: int = 0) -> gym.Env:
        """Create environment for visualization with rendering."""
        return self.create_single_env(seed=seed, render=True)
    
    @staticmethod
    def load_normalized_env(
        vec_normalize_path: str,
        base_env: DummyVecEnv,
        training: bool = False,
    ) -> VecNormalize:
        """
        Load a VecNormalize wrapper from a saved file.
        
        Args:
            vec_normalize_path: Path to saved .pkl file
            base_env: Base vectorized environment to wrap
            training: Whether to continue updating normalization stats
        """
        env = VecNormalize.load(vec_normalize_path, base_env)
        env.training = training
        env.norm_reward = False  #TODO Usually don't normalize reward during eval
        return env