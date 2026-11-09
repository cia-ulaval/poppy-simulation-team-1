"""La flèche de poussée arrive sur le bassin, dans le sens de la force, et ne change rien à la physique."""

from __future__ import annotations

import mujoco
import numpy as np

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv
from src.environments.push_display import PushMarker


def _env() -> PoppyHumanoidEnv:
    env = PoppyHumanoidEnv(floor_noise=False)
    env.reset(seed=0)
    return env


def test_rien_sans_poussee() -> None:
    env = _env()
    marker = PushMarker(dt=env.dt)
    marker.update(env.data)
    assert marker.markers(env.data) == []


def test_fleche_entrante_vers_le_bassin() -> None:
    env = _env()
    marker = PushMarker(dt=env.dt)
    env.apply_push(np.array([10.0, 0.0]), 0.15)
    marker.update(env.data)
    arrow, flash = marker.markers(env.data)
    assert arrow["type"] == mujoco.mjtGeom.mjGEOM_ARROW
    pelvis = env.data.xpos[1]
    axis = arrow["mat"].reshape(3, 3)[:, 2]
    # La flèche pointe dans le sens de la force (+x) ...
    np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-9)
    # ... part du côté d'où l'on pousse et s'arrête sur le bassin.
    tip = arrow["pos"] + axis * arrow["size"][2]
    assert arrow["pos"][0] < tip[0] < pelvis[0]
    np.testing.assert_allclose(tip, flash["pos"])
    assert np.linalg.norm(tip - pelvis) < 0.1


def test_reste_visible_puis_disparait() -> None:
    env = _env()
    marker = PushMarker(dt=env.dt, hold=0.05)
    env.apply_push(np.array([0.0, 5.0]), 0.01)
    marker.update(env.data)
    env.data.xfrc_applied[1, 0:3] = 0.0     # fin de la poussée
    # 0,05 s à 10 ms par pas : encore 4 images, puis plus rien.
    for _ in range(4):
        marker.update(env.data)
        assert marker.markers(env.data)
    marker.update(env.data)
    assert marker.markers(env.data) == []


def test_ne_touche_pas_la_physique() -> None:
    env = _env()
    before = env.data.xfrc_applied.copy()
    marker = PushMarker(dt=env.dt)
    env.apply_push(np.array([3.0, 4.0]), 0.15)
    pushed = env.data.xfrc_applied.copy()
    marker.update(env.data)
    marker.markers(env.data)
    np.testing.assert_array_equal(env.data.xfrc_applied, pushed)
    assert not np.array_equal(before, pushed)
