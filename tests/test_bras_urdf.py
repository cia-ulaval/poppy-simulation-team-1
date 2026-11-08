"""Bras du MJCF orientés comme dans l'URDF (``fix_arm_frames``).

Le MJCF a recopié le ``rpy="π 0 ±π/2"`` URDF des deux ``*_shoulder_motor``
dans ``euler``, que MuJoCo lit en convention intrinsèque x-y-z ; l'URDF est
extrinsèque (R = Rz·Ry·Rx). Les deux diffèrent de 180° : à angles nuls, les
bras du MJCF croisent la poitrine. Ces tests verrouillent la correction faite
au chargement, et le défaut du MJCF brut : le jour où le MJCF est corrigé à la
source, ``test_mjcf_brut_a_les_bras_tournes`` passe au rouge et l'option
devient inutile.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.environments.mirror import build_mirror_spec
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

_URDF = Path(__file__).parent.parent / "assets" / "poppy_humanoid" / "poppy_humanoid.urdf"


def _rot(axis: int, angle: float) -> np.ndarray:
    """Rotation d'``angle`` autour de l'axe ``axis`` (0 = x, 1 = y, 2 = z)."""
    c, s = np.cos(angle), np.sin(angle)
    i, j = (axis + 1) % 3, (axis + 2) % 3   # ordre cyclique : y→z, z→x, x→y
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def _frames_differing_from_urdf(model: mujoco.MjModel) -> list[str]:
    """Corps dont l'orientation fixe diffère de l'origine URDF de leur articulation."""
    differing = []
    for joint in ET.parse(_URDF).getroot().findall("joint"):
        rpy = [float(v) for v in joint.find("origin").get("rpy", "0 0 0").split()]
        child = joint.find("child").get("link")
        expected = _rot(2, rpy[2]) @ _rot(1, rpy[1]) @ _rot(0, rpy[0])
        actual = np.zeros(9)
        mujoco.mju_quat2Mat(actual, model.body(child).quat)
        cosine = (np.trace(actual.reshape(3, 3).T @ expected) - 1.0) / 2.0
        if np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) > 0.01:
            differing.append(child)
    return differing


@pytest.fixture(scope="module")
def env() -> PoppyHumanoidEnv:
    environment = PoppyHumanoidEnv(floor_noise=False, fix_left_knee_axis=True,
                                   fix_arm_frames=True)
    yield environment
    environment.close()


def test_mjcf_brut_a_les_bras_tournes() -> None:
    """Sans l'option, seuls les deux moteurs d'épaule diffèrent de l'URDF."""
    raw = PoppyHumanoidEnv(floor_noise=False)
    try:
        assert sorted(_frames_differing_from_urdf(raw.model)) == [
            "l_shoulder_motor", "r_shoulder_motor"]
    finally:
        raw.close()


def test_tous_les_reperes_comme_l_urdf(env: PoppyHumanoidEnv) -> None:
    assert _frames_differing_from_urdf(env.model) == []


def test_bras_le_long_du_corps(env: PoppyHumanoidEnv) -> None:
    """Pose debout : coudes sous les épaules, du bon côté, sans toucher le corps."""
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.init_qpos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, data)
    for side, outwards in (("l", 1.0), ("r", -1.0)):
        shoulder = data.xpos[env.model.body(f"{side}_shoulder").id]
        elbow = data.xpos[env.model.body(f"{side}_forearm").id]
        arm = elbow - shoulder
        assert arm[2] < -0.12, f"{side} : coude pas sous l'épaule ({arm})"
        assert outwards * arm[0] > 0.0, f"{side} : coude vers l'intérieur ({arm})"
    for contact in data.contact[:data.ncon]:
        names = {env.model.geom(contact.geom1).name, env.model.geom(contact.geom2).name}
        assert "floor" in names, "contact entre deux parties du robot à la pose debout"


def test_pose_debout_symetrique(env: PoppyHumanoidEnv) -> None:
    """La pose debout est son propre reflet : le miroir ne change rien à action nulle."""
    spec = build_mirror_spec(env)
    joints = env.init_qpos[7:]
    mirrored = spec.joint_sign * joints[spec.joint_perm]
    np.testing.assert_allclose(mirrored, joints, atol=0.05)
