from pathlib import Path
from typing import Callable, Optional
import tempfile
import shutil

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from scripts.solidenv_modifier.humanoid_advanced import add_obstacles, duplicate_env
from src.config.settings import EnvironmentConfig, PoppyEnvironmentConfig
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv, register_poppy_env


class HumanoidEnvFactory:
    """
    Factory for creating Humanoid-v5 environments.
    
    Encapsulates all environment creation logic including:
    - Base environment setup
    - Vectorization (Dummy or Subprocess)
    - Observation/reward normalization
    - Monitoring
    - Temporary file management
    
    Benefits:
    - Single Responsibility: Only handles env creation
    - Open/Closed: Easy to extend for new env types
    - Dependency Inversion: Other code depends on factory, not gym details
    """
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self._temp_dir: Optional[str] = None
    
    @property
    def temp_dir(self) -> str:
        """Get or create the temporary directory for generated files."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="humanoid_env_")
        return self._temp_dir
    
    def cleanup(self):
        """Remove all temporary files created by this factory."""
        if self._temp_dir is not None and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.cleanup()
    
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
            if config.with_obstacles:
                env_path = duplicate_env("scripts/solidenv_modifier/humanoid.xml", output_dir=self.temp_dir)
                add_obstacles(n_obstacles=10, source=env_path)
            else:
                env_path = None
            env = gym.make(
                config.env_id,
                xml_file=str(Path(env_path).absolute()) if env_path else None,
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
            norm_reward=False,   # Always show true rewards during eval
            clip_obs=self.config.clip_obs,
            gamma=self.config.gamma,
            training=False,      # Don't update stats during eval
        )
        
        return vec_env
    
    def create_render_env(self, seed: int = 0) -> gym.Env:
        """Create environment for visualization with rendering."""
        return self.create_single_env(seed=seed, render=True)
    
    @staticmethod
    def create_poppy_training_env(
        config: PoppyEnvironmentConfig,
        n_envs: Optional[int] = None,
        seed: int = 0,
        use_subprocess: bool = True,
    ) -> VecNormalize:
        """
        Create a vectorized training environment for the Poppy Humanoid.

        Each sub-environment is a ``PoppyHumanoidEnv`` with floor domain
        randomization enabled according to ``config.domain_randomization``.

        Args:
            config: Poppy-specific environment configuration.
            n_envs: Number of parallel environments (defaults to config.n_envs).
            seed: Random seed.
            use_subprocess: Use SubprocVecEnv for true parallelism.

        Returns:
            VecNormalize-wrapped vectorized environment.
        """
        register_poppy_env()

        dr = config.domain_randomization
        n = n_envs if n_envs is not None else config.n_envs

        def _make_poppy(rank: int) -> Callable[[], gym.Env]:
            def _init() -> gym.Env:
                env = PoppyHumanoidEnv(
                    floor_noise=dr.enabled,
                    friction_range=dr.friction_range,
                    restitution_range=dr.restitution_range,
                    healthy_z_range=config.healthy_z_range,
                    terminate_when_unhealthy=config.terminate_when_unhealthy,
                    frame_skip=config.frame_skip,
                )
                env = TimeLimit(env, max_episode_steps=1000)
                env = Monitor(env)
                env.reset(seed=seed + rank)
                return env
            set_random_seed(seed)
            return _init

        env_fns = [_make_poppy(i) for i in range(n)]

        if n > 1 and use_subprocess:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)

        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            clip_obs=config.clip_obs,
            gamma=config.gamma,
        )
        return vec_env

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