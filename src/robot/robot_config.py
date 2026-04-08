from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Toutes les constantes

MUJOCO_JOINT_ORDER: List[str] = [
    "abdomen_z",
    "abdomen_y",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
]

N_JOINTS: int = len(MUJOCO_JOINT_ORDER) # 17


MUJOCO_ACTUATOR_ORDER: List[str] = [
    "abdomen_y",
    "abdomen_z",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
]

ACTUATOR_TO_JOINT: np.ndarray = np.array(
    [MUJOCO_JOINT_ORDER.index(n) for n in MUJOCO_ACTUATOR_ORDER]
)


OBS_SLICES = {
    "root_z": slice(0, 1),
    "root_quat": slice(1, 5),
    "joint_pos": slice(5, 22),
    "root_linvel": slice(22, 25),
    "root_angvel": slice(25, 28),
    "joint_vel": slice(28, 45),
    "cinert": slice(45, 175),
    "cvel": slice(175, 253),
    "actuator_frc": slice(253, 270),
    "cfrc_ext": slice(270, 348),
}

OBS_DIM: int = 348

MUJOCO_JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "abdomen_z": (-45, 45),
    "abdomen_y": (-75, 30),
    "abdomen_x": (-35, 35),
    "right_hip_x": (-25, 5),
    "right_hip_z": (-60, 35),
    "right_hip_y": (-110, 20),
    "right_knee": (-160, -2),
    "left_hip_x": (-25, 5),
    "left_hip_z": (-60, 35),
    "left_hip_y": (-110, 20),
    "left_knee": (-160, -2),
    "right_shoulder1": (-85, 60),
    "right_shoulder2": (-85, 60),
    "right_elbow": (-90, 50),
    "left_shoulder1": (-60, 85),
    "left_shoulder2": (-60, 85),
    "left_elbow": (-90, 50),
}

PYPOT_TO_MUJOCO: Dict[str, str] = {
    "abs_z": "abdomen_z",
    "abs_y": "abdomen_y",
    "abs_x": "abdomen_x",
    "r_hip_x": "right_hip_x",
    "r_hip_z": "right_hip_z",
    "r_hip_y": "right_hip_y",
    "r_knee_y": "right_knee",
    "l_hip_x": "left_hip_x",
    "l_hip_z": "left_hip_z",
    "l_hip_y": "left_hip_y",
    "l_knee_y": "left_knee",
    "r_shoulder_y": "right_shoulder1",
    "r_shoulder_x": "right_shoulder2",
    "r_elbow_y": "right_elbow",
    "l_shoulder_y": "left_shoulder1",
    "l_shoulder_x": "left_shoulder2",
    "l_elbow_y": "left_elbow",
}

MUJOCO_TO_PYPOT: Dict[str, str] = {v: k for k, v in PYPOT_TO_MUJOCO.items()}

# Ordered list of pypot motor names matching MUJOCO_JOINT_ORDER
PYPOT_MOTOR_ORDER: List[str] = [MUJOCO_TO_PYPOT[j] for j in MUJOCO_JOINT_ORDER]

# Ordered list of pypot motor names matching MUJOCO_ACTUATOR_ORDER
# (use this when mapping model actions -> motor commands)
PYPOT_ACTUATOR_ORDER: List[str] = [MUJOCO_TO_PYPOT[j] for j in MUJOCO_ACTUATOR_ORDER]

NOMINAL_POSE_DEG: Dict[str, float] = {
    "abs_z": 0.0,
    "abs_y": 0.0,
    "abs_x": 0.0,
    "r_hip_x": 0.0,
    "r_hip_z": 0.0,
    "r_hip_y": -20.0,
    "r_knee_y": -30.0,
    "l_hip_x": 0.0,
    "l_hip_z": 0.0,
    "l_hip_y": -20.0,
    "l_knee_y": -30.0,
    "r_shoulder_y": 0.0,
    "r_shoulder_x": -10.0,
    "r_elbow_y": -30.0,
    "l_shoulder_y": 0.0,
    "l_shoulder_x": 10.0,
    "l_elbow_y": -30.0,
}


@dataclass(frozen=True)
class RobotConfig:
    control_freq_hz: float = 50.0
    max_position_delta_deg: float = 10.0
    action_gain: float = 30.0 # deg per unit of action
    nominal_root_height_m: float = 0.9 # Poppy Humanoid standing height
    has_imu: bool = True
    pypot_motor_names: List[str] = field(
        default_factory=lambda: list(PYPOT_MOTOR_ORDER)
    )

    @property
    def dt(self) -> float:
        """Control period in seconds."""
        return 1.0 / self.control_freq_hz

# Pre-built config for the standard Poppy Humanoid robot
POPPY_HUMANOID_CONFIG = RobotConfig()
