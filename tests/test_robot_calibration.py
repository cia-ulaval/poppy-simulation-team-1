"""La calibration robot correspond bien aux articulations de la simulation.

Une calibration fausse ne lève rien : un nom mal orthographié laisse une articulation
sans conversion, un identifiant en double envoie deux angles au même moteur, une
conversion qui ne s'inverse pas décale le robot sans bruit. Ces tests ne disent pas
si les sens sont physiquement justes, seulement que le fichier est cohérent.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import pytest

from src.robot.calibration import load_calibration

_MJCF = Path(__file__).resolve().parent.parent / "assets" / "poppy_humanoid" / "poppy_humanoid.xml"


def test_couvre_exactement_les_25_articulations_du_mjcf() -> None:
    modele = mujoco.MjModel.from_xml_path(str(_MJCF))
    noms_mjcf = {modele.joint(i).name for i in range(1, modele.njnt)}  # 0 : racine libre
    assert set(load_calibration()) == noms_mjcf


def test_identifiants_uniques_et_bus_coherent() -> None:
    calibrations = load_calibration().values()
    identifiants = [c.motor_id for c in calibrations]
    assert len(identifiants) == len(set(identifiants))
    for c in calibrations:
        assert c.bus == ("jambes" if c.motor_id < 30 else "haut"), c.name


@pytest.mark.parametrize("angle", [-1.2, -0.3, 0.0, 0.4, 1.5])
def test_conversion_aller_retour(angle: float) -> None:
    for c in load_calibration().values():
        retour = c.ticks_to_angle(c.angle_to_ticks(angle))
        assert retour == pytest.approx(angle, abs=1.5 / c.ticks_per_rad), c.name


def test_sens_inverse_et_resolution_ax() -> None:
    calibrations = load_calibration()
    # r_shoulder_x est inversé : un angle positif de la simulation doit baisser les ticks.
    assert calibrations["r_shoulder_x"].angle_to_ticks(0.5) < calibrations["r_shoulder_x"].zero
    # head_z est un AX-12 : 1024 pas sur 300°, donc 30° = 102 pas environ.
    assert calibrations["head_z"].angle_to_ticks(math.radians(30)) == 512 + 102
