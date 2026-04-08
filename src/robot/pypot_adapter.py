from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .robot_config import (
    PYPOT_MOTOR_ORDER,
    RobotConfig,
    POPPY_HUMANOID_CONFIG,
)

# parle au robot (50hz)
# lis les données avec read_state 
# envoie les commandes au robot avec send_commands

@dataclass
class RobotState:

    joint_positions: np.ndarray
    joint_velocities: np.ndarray

    root_height: float = 0.9
    root_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    root_linear_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    root_angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    last_action: np.ndarray = field(
        default_factory=lambda: np.zeros(17)
    )


class PypotAdapter:

    def __init__(self, robot, config: RobotConfig = POPPY_HUMANOID_CONFIG) -> None:
        self._robot = robot
        self._config = config
        self._motor_names: List[str] = config.pypot_motor_names

    def read_state(self, last_action: Optional[np.ndarray] = None) -> RobotState:
        
        positions_rad = np.array(
            [np.deg2rad(getattr(self._robot, name).present_position)
             for name in self._motor_names],
            dtype=np.float64,
        )
        
        velocities_rad = np.array(
            [np.deg2rad(getattr(self._robot, name).present_speed)
             for name in self._motor_names],
            dtype=np.float64,
        )

        state = RobotState(
            joint_positions=positions_rad,
            joint_velocities=velocities_rad,
            last_action=last_action if last_action is not None else np.zeros(17),
        )

        if self._config.has_imu:
            self._fill_imu(state)

        return state

    def send_commands(self, goal_positions_deg: Dict[str, float]) -> None:
        for name, pos in goal_positions_deg.items():
            getattr(self._robot, name).goal_position = pos

    def close(self) -> None:
        for name in self._motor_names:
            getattr(self._robot, name).compliant = True
        self._robot.close()

    def _fill_imu(self, state: RobotState) -> None:
        try:
            imu = self._robot.imu
            euler_deg = np.asarray(imu.euler_angle, dtype=np.float64)
            roll, pitch, yaw = np.deg2rad(euler_deg)
            state.root_quaternion = _euler_to_quaternion(roll, pitch, yaw)
            state.root_angular_velocity = np.deg2rad(
                np.asarray(imu.angular_velocity, dtype=np.float64)
            )
            state.root_height = self._config.nominal_root_height_m
        except Exception:
            pass

# l'imu de poppy donne des angles d'Euler mais le modèle entraîné en 
# simulation attend des quaternions
def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles (radians) to quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)
