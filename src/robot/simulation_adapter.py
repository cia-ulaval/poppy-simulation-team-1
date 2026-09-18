"""Pont entre l'environnement simulé et le robot réel, via rosbridge.

Déroule la simulation pas à pas et publie, après chaque pas, la position
angulaire des 25 articulations vers le pont. Rien ici ne lit l'état du robot
réel : la boucle est ouverte.
"""

from __future__ import annotations

import os
from time import sleep

from gymnasium.wrappers import TimeLimit
from numpy.typing import NDArray

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv
from src.robot.ros_publisher import MotorArrayPublisher

# L'observation commence par z + quaternion racine (5 valeurs), puis les 25
# positions articulaires. Voir la docstring de PoppyHumanoidEnv.
_JOINTS_SLICE = slice(5, 30)

_MAX_EPISODE_STEPS = 1000


class SimulationAdapter:
    """Adapte l'environnement simulé pour publier vers le pont rosbridge."""

    def __init__(self) -> None:
        """Construit l'environnement et ouvre le lien vers le pont.

        Raises:
            ValueError: Si ``POPPY_CONTROL_PERIOD_S`` n'est pas un nombre
                positif ou nul.
        """
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
        self._env = TimeLimit(self._env, max_episode_steps=_MAX_EPISODE_STEPS)
        self._publisher = MotorArrayPublisher()

        # range(1, njnt) : l'indice 0 est l'articulation libre de la racine,
        # qui n'a pas de moteur. Les 25 noms restants correspondent un pour un
        # aux 25 angles renvoyés par _get_joint_positions_rad.
        model = self._env.unwrapped.model
        self._joint_names = [model.joint(i).name for i in range(1, model.njnt)]

    def reset(self) -> NDArray:
        """Réinitialise l'épisode.

        Returns:
            L'observation initiale, forme ``(63,)``.
        """
        obs, _ = self._env.reset()
        return obs

    def _get_joint_positions_rad(self) -> NDArray:
        """Retourne les positions des 25 articulations, en radians.

        Returns:
            Les positions articulaires, forme ``(25,)``.
        """
        obs = self._env.unwrapped._get_obs()
        return obs[_JOINTS_SLICE].copy()

    def step(self, action: NDArray) -> tuple[NDArray, bool]:
        """Avance d'un pas et publie les positions obtenues.

        Args:
            action: Action à appliquer, forme ``(25,)``, dans ``[-1, 1]``.

        Returns:
            L'observation suivante et un drapeau de fin d'épisode. Si
            l'épisode se termine, l'observation renvoyée est celle du nouvel
            épisode.
        """
        # Attendre la période de contrôle configurée avant d'envoyer la
        # prochaine commande. Une valeur nulle permet de désactiver l'attente.
        sleep(self._control_period_s)
        obs, _, terminated, truncated, _ = self._env.step(action)

        joint_rad = self._get_joint_positions_rad()
        self._publisher.publish(motor_ids=self._joint_names, angles_rad=joint_rad)

        done = terminated or truncated
        if done:
            obs = self.reset()

        return obs, done

    def close(self) -> None:
        """Ferme l'environnement et le lien vers le pont."""
        self._env.close()
        self._publisher.close()
