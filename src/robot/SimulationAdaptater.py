from __future__ import annotations

import os
from time import sleep

import numpy as np
from gymnasium.wrappers import TimeLimit

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

from .ros_publisher import MotorArrayPublisher


class SimulationAdapter:
    """Adapte l'environnement simulé pour publier vers le pont rosbridge."""

    def __init__(self) -> None:
        period_raw = os.environ.get("POPPY_CONTROL_PERIOD_S", "5.0")
        try:
            self._control_period_s = float(period_raw)
        except ValueError as exc:
            raise ValueError(
                f"POPPY_CONTROL_PERIOD_S doit être un nombre, reçu {period_raw!r}"
            ) from exc

        if self._control_period_s < 0:
            raise ValueError(
                f"POPPY_CONTROL_PERIOD_S doit être positif ou nul, "
                f"reçu {self._control_period_s}"
            )

        # Le pont tourne sans serveur d'affichage dans le conteneur.
        self._env = PoppyHumanoidEnv(floor_noise=False, render_mode=None)
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
        # _get_obs commence par z + quaternion racine, puis les 25 joints.
        return obs[5:30].copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        # Attendre la période de contrôle configurée avant d'envoyer la
        # prochaine commande. Une valeur nulle permet de désactiver l'attente.
        sleep(self._control_period_s)
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
