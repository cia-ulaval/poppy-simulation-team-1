from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from stable_baselines3 import PPO, TD3, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from .pypot_adapter import PypotAdapter, RobotState
from .observation_mapper import ObservationMapper
from .action_mapper import ActionMapper, ControlMode
from .robot_config import RobotConfig, POPPY_HUMANOID_CONFIG, N_JOINTS

logger = logging.getLogger(__name__)

_MODEL_CLASSES = [PPO, SAC, TD3, A2C]

# orchestre tout, point d'entrée
class RobotController:

    def __init__(
        self,
        model,
        vec_normalize: Optional[VecNormalize],
        adapter: PypotAdapter,
        config: RobotConfig = POPPY_HUMANOID_CONFIG,
        control_mode: ControlMode = ControlMode.DELTA,
    ) -> None:
        self._model = model
        self._vec_normalize = vec_normalize
        self._adapter = adapter
        self._config = config

        training_mean = None
        if vec_normalize is not None:
            training_mean = np.asarray(vec_normalize.obs_rms.mean, dtype=np.float64)

        self._obs_mapper = ObservationMapper(training_obs_mean=training_mean)
        self._act_mapper = ActionMapper(config, control_mode)

        self._last_action: np.ndarray = np.zeros(N_JOINTS)
        self._step_count: int = 0
        self._running: bool = False


    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path,
        adapter: PypotAdapter,
        vec_normalize_path: Optional[Path] = None,
        config: RobotConfig = POPPY_HUMANOID_CONFIG,
        control_mode: ControlMode = ControlMode.DELTA,
    ) -> "RobotController":
        """Load a trained model + normalizer and return a ready controller."""
        model_path = Path(model_path)

        # Auto-detect vec_normalize
        if vec_normalize_path is None:
            candidate = model_path.parent / "vec_normalize.pkl"
            if candidate.exists():
                vec_normalize_path = candidate
                logger.info("Auto-detected VecNormalize at %s", vec_normalize_path)

        # Load VecNormalize
        vec_normalize: Optional[VecNormalize] = None
        if vec_normalize_path and Path(vec_normalize_path).exists():
            dummy_env = DummyVecEnv([lambda: gym.make("Humanoid-v5")])
            vec_normalize = VecNormalize.load(str(vec_normalize_path), dummy_env)
            vec_normalize.training = False
            vec_normalize.norm_reward = False
            logger.info("Loaded VecNormalize from %s", vec_normalize_path)
        else:
            logger.warning("No VecNormalize found -- running without normalisation.")

        # Try each SB3 algo until one works
        model = None
        for cls_ in _MODEL_CLASSES:
            try:
                model = cls_.load(str(model_path), env=vec_normalize)
                break
            except Exception:
                continue
        if model is None:
            raise ValueError(f"Could not load model from {model_path}")

        return cls(
            model=model,
            vec_normalize=vec_normalize,
            adapter=adapter,
            config=config,
            control_mode=control_mode,
        )

    def _step(self) -> tuple[np.ndarray, np.ndarray]:
        """One control step. Returns (obs, action)."""
        state: RobotState = self._adapter.read_state(self._last_action)

        obs = self._obs_mapper.map(state)

        if self._vec_normalize is not None:
            obs_norm = self._vec_normalize.normalize_obs(
                obs.reshape(1, -1)
            ).flatten()
        else:
            obs_norm = obs

        action, _ = self._model.predict(obs_norm.reshape(1, -1), deterministic=True) # prédiction ici important
        action = np.asarray(action).flatten()
        self._last_action = action.copy()

        current_deg = self._act_mapper.current_positions_from_state(state)
        self._adapter.send_commands(self._act_mapper.map(action, current_deg))

        self._step_count += 1
        return obs, action

    def step_once(self) -> tuple[np.ndarray, np.ndarray]:
        return self._step()

    def run(
        self,
        n_steps: Optional[int] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        """Run the control loop until n_steps, duration, or Ctrl-C."""
        dt = self._config.dt
        self._running = True
        self._step_count = 0
        start = time.monotonic()

        try:
            while self._running:
                t0 = time.monotonic()
                self._step()

                if n_steps is not None and self._step_count >= n_steps:
                    break
                if duration_s is not None and (time.monotonic() - start) >= duration_s:
                    break

                sleep = dt - (time.monotonic() - t0)
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            self._running = False
            logger.info("Ran %d steps.", self._step_count)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self.stop()
        self._adapter.close()
        if self._vec_normalize is not None:
            self._vec_normalize.close()
