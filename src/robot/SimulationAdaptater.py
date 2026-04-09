from __future__ import annotations

from time import sleep

import numpy as np
from gymnasium.wrappers import TimeLimit

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

from .ros_publisher import MotorArrayPublisher


class SimulationAdapter:
    def __init__(self) -> None:
        self._env = PoppyHumanoidEnv(floor_noise=False, render_mode="human")
        self._env = TimeLimit(self._env, max_episode_steps=1000)
        self._publisher = MotorArrayPublisher()
        self._first_reset = True
        model = self._env.unwrapped.model
        self._joint_names = [model.joint(i).name for i in range(model.njnt)]

    def reset(self) -> np.ndarray:
        self._first_reset = False
        obs, _ = self._env.reset()
        return obs

    def _get_joint_positions_rad(self) -> np.ndarray:
        obs = self._env.unwrapped._get_obs()
        return obs[5:30].copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        sleep(5)
        obs, _, terminated, truncated, _ = self._env.step(action)  # step

        joint_rad = self._get_joint_positions_rad()
        self._publisher.publish(motor_ids=self._joint_names, angles_rad=joint_rad)

        done = terminated or truncated
        if done:
            obs = self.reset()

        return obs, done

    def close(self) -> None:
        self._env.close()
        self._publisher.close()
