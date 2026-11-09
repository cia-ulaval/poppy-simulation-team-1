"""La flèche de poussée vient frapper le bassin, dans le sens de la force, et ne change rien à la physique."""

from __future__ import annotations

import mujoco
import numpy as np

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv
from src.environments.push_display import PushMarker


def _env() -> PoppyHumanoidEnv:
    env = PoppyHumanoidEnv(floor_noise=False)
    env.reset(seed=0)
    return env


def _by_type(markers: list[dict], geom_type: int) -> list[dict]:
    return [m for m in markers if m["type"] == geom_type]


def _tip(arrow: dict) -> np.ndarray:
    return arrow["pos"] + arrow["mat"].reshape(3, 3)[:, 2] * arrow["size"][2]


def test_rien_sans_poussee() -> None:
    env = _env()
    marker = PushMarker(dt=env.dt)
    marker.update(env.data)
    assert marker.markers(env.data) == []


def test_la_fleche_arrive_puis_frappe_le_bassin() -> None:
    env = _env()
    marker = PushMarker(dt=env.dt)
    env.apply_push(np.array([10.0, 0.0]), 0.15)
    pelvis = env.data.xpos[1]

    marker.update(env.data)
    first = marker.markers(env.data)
    (arrow,) = _by_type(first, mujoco.mjtGeom.mjGEOM_ARROW)
    # Elle pointe dans le sens de la force (+x), du côté d'où l'on pousse ...
    np.testing.assert_allclose(arrow["mat"].reshape(3, 3)[:, 2], [1.0, 0.0, 0.0], atol=1e-9)
    assert arrow["pos"][0] < _tip(arrow)[0] < pelvis[0]
    # ... et n'a pas encore touché : pas d'impact.
    assert not _by_type(first, mujoco.mjtGeom.mjGEOM_CYLINDER)
    far = pelvis[0] - _tip(arrow)[0]

    for _ in range(5):
        marker.update(env.data)
    hit = marker.markers(env.data)
    (arrow,) = _by_type(hit, mujoco.mjtGeom.mjGEOM_ARROW)
    (ring,) = _by_type(hit, mujoco.mjtGeom.mjGEOM_CYLINDER)
    # Au contact : la pointe touche le bassin, là où sont l'éclat et l'onde de choc.
    assert pelvis[0] - _tip(arrow)[0] < far
    np.testing.assert_allclose(_tip(arrow), ring["pos"])
    assert np.linalg.norm(_tip(arrow) - pelvis) < 0.1
    assert any(m.get("label") == "10 N" for m in hit)


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
