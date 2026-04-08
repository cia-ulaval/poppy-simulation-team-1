from __future__ import annotations

from typing import Optional

import numpy as np

from .robot_config import OBS_DIM, OBS_SLICES, ACTUATOR_TO_JOINT, N_JOINTS
from .pypot_adapter import RobotState

# traduis l'état du Robot vers les données attendues par le modèle
# 348 dimensions d'observation
# 62 dims dispo sur le vrai robot (hauteur, quaternion, 17 pos, 17 vel, ang vel torse, last action).
# 286 dims restantes (cinert, cvel, cfrc_ext) = internes MuJoCo, pas utilisées apriori.

class ObservationMapper:
    def __init__(self, training_obs_mean: Optional[np.ndarray] = None) -> None:
        self._base = np.zeros(OBS_DIM, dtype=np.float64)
        if training_obs_mean is not None:
            for key in ("cinert", "cvel", "cfrc_ext"):
                self._base[OBS_SLICES[key]] = training_obs_mean[OBS_SLICES[key]]

    def map(self, state: RobotState) -> np.ndarray:
        obs = self._base.copy()

        obs[OBS_SLICES["root_z"]] = state.root_height
        obs[OBS_SLICES["root_quat"]] = state.root_quaternion
        obs[OBS_SLICES["joint_pos"]] = state.joint_positions

        obs[OBS_SLICES["root_linvel"]] = state.root_linear_velocity
        obs[OBS_SLICES["root_angvel"]] = state.root_angular_velocity
        obs[OBS_SLICES["joint_vel"]] = state.joint_velocities

        frc_joint_order = np.zeros(N_JOINTS)
        frc_joint_order[ACTUATOR_TO_JOINT] = state.last_action
        obs[OBS_SLICES["actuator_frc"]] = frc_joint_order

        return obs
