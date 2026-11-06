"""Symétrie gauche ↔ droite du modèle MuJoCo de Poppy.

Ces tests existent parce que le genou gauche a plié à l'envers pendant sept
mois sans que rien ne le signale : les politiques apprenaient à marcher avec,
en boitant. Chaque paire d'articulations doit tourner autour d'axes reflets
l'un de l'autre, avec des butées reflets l'une de l'autre, et chaque paire de
corps doit avoir la même masse.

Ils tournent sur le modèle **corrigé** (``fix_left_knee_axis``). Le MJCF brut
échoue sur les genoux, ce que ``test_mjcf_brut_a_le_genou_gauche_a_l_envers``
verrouille : le jour où le MJCF est corrigé à la source, il passe au rouge et
l'option devient inutile.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

# Pseudo-vecteur (axe de rotation) dans un miroir x → −x, la gauche de Poppy
# étant son +x.
_REFLECT_AXIS = np.array([1.0, -1.0, -1.0])

# Écarts tolérés, hérités du vrai robot : r_shoulder_x va de −110° à 105°,
# l_shoulder_x de −105° à 110° (config moteurs de poppy-project), et la cuisse
# droite pèse 1,3 % de plus que la gauche dans l'URDF.
_RANGE_TOLERANCE = np.radians(6.0)
_MASS_TOLERANCE = 0.02


def _partner(name: str) -> str:
    if name.startswith("r_"):
        return "l_" + name[2:]
    return name


def _pairs(env: PoppyHumanoidEnv) -> list[tuple[str, str]]:
    names = [env.model.joint(i).name for i in range(1, env.model.njnt)]
    return [(n, _partner(n)) for n in names if not n.startswith("l_")]


def _standing(env: PoppyHumanoidEnv) -> mujoco.MjData:
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.init_qpos
    mujoco.mj_kinematics(env.model, data)
    return data


def _range_mismatch(env: PoppyHumanoidEnv, right: str, left: str) -> float:
    model, data = env.model, _standing(env)
    a, b = model.joint(right), model.joint(left)
    sign = np.sign((data.xaxis[a.id] * _REFLECT_AXIS) @ data.xaxis[b.id])
    mirrored = np.sort(sign * b.range)
    return float(np.max(np.abs(mirrored - a.range)))


@pytest.fixture(scope="module")
def env() -> PoppyHumanoidEnv:
    environment = PoppyHumanoidEnv(floor_noise=False, fix_left_knee_axis=True)
    yield environment
    environment.close()


def test_axes_reflets(env: PoppyHumanoidEnv) -> None:
    """Chaque articulation tourne autour du reflet de l'axe de sa partenaire."""
    data = _standing(env)
    for right, left in _pairs(env):
        a, b = env.model.joint(right).id, env.model.joint(left).id
        cosine = (data.xaxis[a] * _REFLECT_AXIS) @ data.xaxis[b]
        assert abs(abs(cosine) - 1.0) < 0.01, f"{right} / {left} : cos = {cosine:.3f}"


def test_butees_reflets(env: PoppyHumanoidEnv) -> None:
    """Les butées d'une articulation sont le reflet de celles de sa partenaire."""
    for right, left in _pairs(env):
        mismatch = _range_mismatch(env, right, left)
        assert mismatch < _RANGE_TOLERANCE, (
            f"{right} / {left} : butées décalées de {np.degrees(mismatch):.0f}°"
        )


def test_masses_et_actionneurs_symetriques(env: PoppyHumanoidEnv) -> None:
    """Corps de même masse, actionneurs de même force, chacun sur son articulation."""
    model = env.model
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    for name in bodies:
        if name.startswith("r_"):
            right, left = model.body(name).mass[0], model.body(_partner(name)).mass[0]
            assert abs(right - left) / max(right, left) < _MASS_TOLERANCE, name
    for i in range(model.nu):
        assert model.actuator(i).name == model.joint(model.actuator_trnid[i, 0]).name
    for right, left in _pairs(env):
        np.testing.assert_allclose(model.actuator(right).ctrlrange,
                                   model.actuator(left).ctrlrange, err_msg=right)


def test_mjcf_brut_a_le_genou_gauche_a_l_envers() -> None:
    """Le MJCF d'origine garde le défaut : sans correction, les genoux divergent."""
    raw = PoppyHumanoidEnv(floor_noise=False)
    try:
        assert _range_mismatch(raw, "r_knee_y", "l_knee_y") > np.radians(100.0)
    finally:
        raw.close()
