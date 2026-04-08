from __future__ import annotations

from enum import Enum, auto
from typing import Dict

import numpy as np

from .robot_config import (
    MUJOCO_ACTUATOR_ORDER,
    PYPOT_ACTUATOR_ORDER,
    MUJOCO_JOINT_LIMITS_DEG,
    ACTUATOR_TO_JOINT,
    N_JOINTS,
    RobotConfig,
    POPPY_HUMANOID_CONFIG,
)


class ControlMode(Enum):
    DELTA = auto()
    DIRECT = auto()



# traduis la sortie du modèle pr le robot:

# en sortie on a 17 nombres entre -0.4 et 0.4
# DELTA (par défaut) : action * gain -> delta clippé +-10°, ajouté à la pos courante. 
# DIRECT : on interpole vers la pos cible.
# Clippé aux limites articulaires dans tous les cas.


class ActionMapper:
    
    ACTION_MIN: float = -0.4
    ACTION_MAX: float = 0.4

    def __init__(
        self,
        config: RobotConfig = POPPY_HUMANOID_CONFIG,
        mode: ControlMode = ControlMode.DELTA,
    ) -> None:
        self._config = config
        self._mode = mode

        self._limits_lo = np.array(
            [MUJOCO_JOINT_LIMITS_DEG[j][0] for j in MUJOCO_ACTUATOR_ORDER],
            dtype=np.float64,
        )
        self._limits_hi = np.array(
            [MUJOCO_JOINT_LIMITS_DEG[j][1] for j in MUJOCO_ACTUATOR_ORDER],
            dtype=np.float64,
        )

    def map(
        self,
        action: np.ndarray,
        current_positions_deg: np.ndarray,
    ) -> Dict[str, float]:
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape != (N_JOINTS,):
            raise ValueError(f"Expected action shape ({N_JOINTS},), got {action.shape}")

        if self._mode is ControlMode.DELTA:
            delta = np.clip(
                action * self._config.action_gain,
                -self._config.max_position_delta_deg,
                self._config.max_position_delta_deg,
            )
            goals = current_positions_deg + delta
        else:
            t = (action - self.ACTION_MIN) / (self.ACTION_MAX - self.ACTION_MIN)
            goals = self._limits_lo + t * (self._limits_hi - self._limits_lo)

        goals = np.clip(goals, self._limits_lo, self._limits_hi)

        return {
            PYPOT_ACTUATOR_ORDER[i]: float(goals[i])
            for i in range(N_JOINTS)
        }

    def current_positions_from_state(self, state) -> np.ndarray:
        joint_deg = np.rad2deg(state.joint_positions)
        return joint_deg[ACTUATOR_TO_JOINT]
