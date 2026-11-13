"""Conversion entre les angles de la simulation et les positions des moteurs réels.

La calibration vit dans ``configs/robot_calibration.yaml``. Pour chaque articulation :

    ticks = zero + sens * angle_rad * K

avec ``K`` = 4096 / 2π ticks/rad pour les Dynamixel MX (360° sur 4096 pas) et
1024 / radians(300) ticks/rad pour les AX (300° sur 1024 pas). ``angle_rad`` est
l'angle URDF, celui du MJCF avec ``fix_left_knee_axis`` et ``fix_arm_frames``.

Une erreur de sens ou de zéro ne lève rien : elle envoie une articulation dans la
mauvaise direction, ou la décale d'un angle constant. D'où le test
``tests/test_robot_calibration.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "robot_calibration.yaml"
)

# Pas par radian. MX : 4096 pas sur 360°. AX : 1024 pas sur 300°.
TICKS_PER_RAD = {
    "MX": 4096 / (2 * math.pi),
    "AX": 1024 / math.radians(300),
}


@dataclass(frozen=True)
class JointCalibration:
    """Calibration d'une articulation.

    Attributes:
        name: Nom de l'articulation, identique au nom du joint MJCF.
        motor_id: Identifiant Dynamixel.
        bus: ``"jambes"`` ou ``"haut"``.
        model: Modèle du moteur, par exemple ``"MX-28"`` ou ``"AX-12"``.
        sign: +1 si le moteur tourne dans le sens de la simulation, -1 sinon.
        zero: Position moteur (ticks) quand l'angle de la simulation vaut 0.
        zero_verified: Vrai si une pose observée a confirmé ``zero``.
    """

    name: str
    motor_id: int
    bus: str
    model: str
    sign: int
    zero: int
    zero_verified: bool

    @property
    def ticks_per_rad(self) -> float:
        """Pas moteur par radian, selon la famille (MX ou AX)."""
        return TICKS_PER_RAD[self.model[:2]]

    def angle_to_ticks(self, angle_rad: float) -> int:
        """Position moteur (ticks, entier) pour un angle de simulation en radians."""
        return round(self.zero + self.sign * angle_rad * self.ticks_per_rad)

    def ticks_to_angle(self, ticks: int) -> float:
        """Angle de simulation (rad) pour une position moteur en ticks."""
        return self.sign * (ticks - self.zero) / self.ticks_per_rad


def load_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> dict[str, JointCalibration]:
    """Charge la calibration des 25 articulations.

    Args:
        path: Fichier YAML de calibration.

    Returns:
        Les calibrations, indexées par nom d'articulation.

    Raises:
        ValueError: Si un sens n'est pas ±1 ou si un modèle de moteur est inconnu.
    """
    with path.open(encoding="utf-8") as fichier:
        brut = yaml.safe_load(fichier)["articulations"]

    calibrations = {}
    for nom, champs in brut.items():
        calibration = JointCalibration(
            name=nom,
            motor_id=int(champs["id"]),
            bus=str(champs["bus"]),
            model=str(champs["modele"]),
            sign=int(champs["sens"]),
            zero=int(champs["zero"]),
            zero_verified=bool(champs["zero_verifie"]),
        )
        if calibration.sign not in (-1, 1):
            raise ValueError(f"{nom} : sens {calibration.sign}, attendu +1 ou -1")
        if calibration.model[:2] not in TICKS_PER_RAD:
            raise ValueError(f"{nom} : modèle de moteur inconnu {calibration.model!r}")
        calibrations[nom] = calibration
    return calibrations
