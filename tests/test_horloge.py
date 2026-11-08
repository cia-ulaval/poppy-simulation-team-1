"""Tests de l'horloge de marche (environment.gait_period).

Le point qui compte : sans horloge, l'observation garde ses 63 dimensions et
les modèles existants se chargent comme avant. Avec horloge, elle en a 65, le
miroir sait les refléter, et la prime d'alternance reste bornée.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import RewardConfig
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv


def test_sans_horloge_l_observation_garde_63_dimensions() -> None:
    env = PoppyHumanoidEnv(floor_noise=False)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (63,)
        assert obs.shape == (63,)
        _, _, _, _, info = env.step(np.zeros(25, dtype=np.float32))
        assert info["phase_contact_reward"] == 0.0
    finally:
        env.close()


def test_horloge_ajoute_sin_cos_de_la_phase() -> None:
    env = PoppyHumanoidEnv(floor_noise=False, gait_period=0.8)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.shape == (65,)
        # Phase nulle au reset : sin 0, cos 1.
        assert obs[-2:] == pytest.approx([0.0, 1.0])
        for _ in range(20):  # 20 pas de 10 ms = 0,2 s = un quart de cycle
            obs, *_ = env.step(np.zeros(25, dtype=np.float32))
        assert obs[-2:] == pytest.approx([1.0, 0.0], abs=1e-6)
        env.reset(seed=1)
        assert env._gait_phase() == 0.0
    finally:
        env.close()


def test_miroir_de_l_horloge() -> None:
    """Le miroir d'un instant du cycle est le même instant pour l'autre jambe :
    sin et cos changent de signe, et l'observation reste de 65 dimensions."""
    env = PoppyHumanoidEnv(floor_noise=False, gait_period=0.8,
                           fix_left_knee_axis=True)
    try:
        spec = env.mirror_spec()
        assert spec.obs_perm.shape == (65,)
        assert list(spec.obs_perm[-2:]) == [63, 64]
        assert list(spec.obs_sign[-2:]) == [-1.0, -1.0]
        obs, _ = env.reset(seed=0)
        assert spec.mirror_obs(spec.mirror_obs(obs)) == pytest.approx(obs)
    finally:
        env.close()


def test_prime_d_alternance_bornee() -> None:
    weights = RewardConfig(phase_contact_weight=0.5)
    env = PoppyHumanoidEnv(floor_noise=False, gait_period=0.8, reward=weights)
    try:
        env.reset(seed=0)
        for _ in range(50):
            _, _, _, _, info = env.step(np.zeros(25, dtype=np.float32))
            assert 0.0 <= info["phase_contact_reward"] <= 0.5
    finally:
        env.close()
