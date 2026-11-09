"""Tests de la marche de référence (piste 3).

Les modes de défaillance qui comptent sont silencieux : des angles rangés dans
un autre ordre que les actionneurs, une phase décalée d'un demi-cycle (la
jambe gauche imitée par la droite), ou une copie de l'URDF pour Placo qui ne
correspond plus au MJCF simulé. Aucun ne lèverait d'erreur ; tous
entraîneraient la politique à imiter autre chose que la référence vérifiée.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.generate_reference_walk import _URDF_PLACO, build_placo_urdf
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv
from src.environments.reference_motion import load_reference


@pytest.fixture(scope="module")
def env() -> PoppyHumanoidEnv:
    environment = PoppyHumanoidEnv(floor_noise=False, fix_left_knee_axis=True)
    yield environment
    environment.close()


def test_articulations_dans_l_ordre_des_actionneurs(env: PoppyHumanoidEnv) -> None:
    reference = load_reference()
    actuated = tuple(env.model.joint(env.model.actuator_trnid[i, 0]).name
                     for i in range(env.model.nu))
    assert reference.joint_names == actuated


def test_phase_dans_la_convention_de_l_horloge() -> None:
    """φ = 0 : le pied droit se pose. Appui droit seul au premier demi-cycle,
    gauche seul au second, double appui juste après chaque pose."""
    reference = load_reference()
    assert list(reference.contacts_at(0.01)) == [1.0, 1.0]
    assert list(reference.contacts_at(0.25)) == [1.0, 0.0]
    assert list(reference.contacts_at(0.51)) == [1.0, 1.0]
    assert list(reference.contacts_at(0.75)) == [0.0, 1.0]
    # Pied au sol 55 % de son cycle, comme reward.phase_stance_ratio.
    assert reference.contacts.mean(axis=0) == pytest.approx([0.55, 0.55], abs=0.01)


def test_reference_dans_les_butees(env: PoppyHumanoidEnv) -> None:
    reference = load_reference()
    low, high = env.model.jnt_range[1:, 0], env.model.jnt_range[1:, 1]
    assert np.all(reference.qpos >= low) and np.all(reference.qpos <= high)


def test_bassin_roule_autour_de_la_verticale() -> None:
    """Le bassin roule à chaque pas, mais autour de zéro, et le tronc reste droit.

    Garde-fou contre une dérive du solveur cinématique : on a eu une
    référence au bassin penché de 12° en permanence, tronc tordu de 32° pour
    compenser, que ni les couples ni le rejeu ne signalaient.
    """
    reference = load_reference()
    roll = np.array([np.arcsin(reference.base_at(k / reference.n_samples)[2][0])
                     for k in range(reference.n_samples)])
    assert abs(np.degrees(roll.mean())) < 1.0
    assert np.degrees(np.abs(roll).max()) < 10.0
    torso = [reference.joint_names.index(n) for n in ("abs_x", "bust_x")]
    assert np.degrees(np.abs(reference.qpos[:, torso]).max()) < 1.0


def test_interpolation_periodique() -> None:
    reference = load_reference()
    q_end, _ = reference.joints_at(1.0 - 1e-9)
    q_start, _ = reference.joints_at(0.0)
    assert q_end == pytest.approx(q_start, abs=1e-6)
    q_mid, _ = reference.joints_at(0.5 / reference.n_samples)
    assert q_mid == pytest.approx(0.5 * (reference.qpos[0] + reference.qpos[1]))


def test_copie_urdf_a_jour(env: PoppyHumanoidEnv) -> None:
    """La copie pour Placo se régénère à l'identique depuis le MJCF.

    Si ce test rougit, le MJCF a changé : relancer
    ``scripts/generate_reference_walk.py``, puis ``check_reference_walk.py``.
    """
    assert _URDF_PLACO.read_text(encoding="utf-8") == build_placo_urdf(env.model)
