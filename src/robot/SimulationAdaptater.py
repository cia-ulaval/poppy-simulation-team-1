from __future__ import annotations

import numpy as np
from gymnasium.wrappers import TimeLimit

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv
from .ros_publisher import MotorArrayPublisher

# FIXME: 17 joints mais est-ce qu'on a besoin des 25 ??
JOINT_INDICES = np.array(
    [12, 10, 11, 0, 1, 2, 3, 5, 6, 7, 8, 21, 22, 24, 17, 18, 20],
    dtype=np.int64,
)
N_JOINTS = len(JOINT_INDICES)
MOTOR_IDS = np.arange(N_JOINTS, dtype=np.int32)


class SimulationAdapter:
    def __init__(self) -> None:
        self._env = PoppyHumanoidEnv(floor_noise=False, render_mode="human")
        self._env = TimeLimit(self._env, max_episode_steps=1000)
        self._publisher = MotorArrayPublisher()
        self._first_reset = True

    def reset(self) -> np.ndarray:
        self._first_reset = False
        obs, _ = self._env.reset()
        return obs

    def _get_joint_positions_rad(self) -> np.ndarray:
        obs = self._env.unwrapped._get_obs()
        return obs[5:30][JOINT_INDICES].copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        obs, _, terminated, truncated, _ = self._env.step(action) # step

        joint_rad = self._get_joint_positions_rad()
        self._publisher.publish(motor_ids=MOTOR_IDS, angles_rad=joint_rad)

        done = terminated or truncated
        if done:
            obs = self.reset()

        return obs, done

    def close(self) -> None:
        self._env.close()
        self._publisher.close()
