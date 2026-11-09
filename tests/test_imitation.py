"""Tests de la récompense d'imitation de la marche de référence (piste 3).

Ce qui compte : nulle par défaut (les modèles et les configs existants ne
voient aucune différence), maximale quand le robot est exactement sur la
référence, et nettement plus basse quand la phase est décalée d'un demi-cycle
— sans quoi une erreur de phase ferait imiter la jambe gauche par la droite,
sans rien signaler.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.config import (
    RewardConfig,
    as_evaluation_config,
    load_yaml,
    make_poppy_env_config,
)
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv
from src.environments.reference_motion import load_reference

_CONFIGS = Path(__file__).parent.parent / "configs"
_PERIOD = 0.7
_WEIGHTS = RewardConfig(
    imitation_pos_weight=1.0, imitation_upper_weight=1.0, imitation_vel_weight=1.0,
    imitation_contact_weight=1.0, imitation_base_weight=1.0)


def _imitation_env(**kwargs) -> PoppyHumanoidEnv:
    options = dict(floor_noise=False, fix_left_knee_axis=True, gait_period=_PERIOD,
                   reward=_WEIGHTS)
    options.update(kwargs)
    return PoppyHumanoidEnv(**options)


def _place_on_reference(env: PoppyHumanoidEnv, phase: float) -> np.ndarray:
    """Met l'état exactement sur la référence à ``phase`` ; rend les contacts attendus."""
    reference = load_reference()
    q, qd = reference.joints_at(phase)
    height, quat, _ = reference.base_at(phase)
    qpos = env.init_qpos.copy()
    qpos[2], qpos[3:7], qpos[7:] = height, quat, q
    qvel = np.zeros(env.model.nv)
    qvel[6:] = qd
    env.set_state(qpos, qvel)
    mujoco.mj_forward(env.model, env.data)
    return reference.contacts_at(phase) > 0.5


def test_nulle_par_defaut() -> None:
    assert RewardConfig().uses_imitation is False
    env = PoppyHumanoidEnv(floor_noise=False)
    try:
        assert env._reference is None
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(25, dtype=np.float32))
        assert info["imitation_reward"] == 0.0
    finally:
        env.close()


def test_refuse_un_environnement_qui_ne_correspond_pas() -> None:
    with pytest.raises(ValueError, match="fix_left_knee_axis"):
        _imitation_env(fix_left_knee_axis=False)
    with pytest.raises(ValueError, match="gait_period"):
        _imitation_env(gait_period=0.8)


def test_maximale_sur_la_reference_et_basse_en_opposition_de_phase() -> None:
    env = _imitation_env()
    try:
        env.reset(seed=0)
        for phase in (0.1, 0.3, 0.6, 0.85):
            contacts = _place_on_reference(env, phase)
            env._phase_time = phase * _PERIOD
            assert env._imitation_reward(contacts) == pytest.approx(5.0, abs=1e-3)
            # Même pose, horloge décalée d'un demi-cycle : jambes inversées.
            env._phase_time = (phase + 0.5) * _PERIOD
            shifted = env._imitation_reward(contacts)
            assert shifted < 4.0
        # Les angles des jambes seuls : 1,0 en phase, ~0,3 en opposition.
        env._reward = replace(RewardConfig(), imitation_pos_weight=1.0)
        contacts = _place_on_reference(env, 0.3)
        env._phase_time = 0.8 * _PERIOD
        assert env._imitation_reward(contacts) < 0.5
    finally:
        env.close()


def test_depart_sur_la_reference() -> None:
    """reference_init : phase tirée au hasard, robot posé dans la pose de cette phase."""
    env = _imitation_env(reference_init=1.0)
    reference = load_reference()
    try:
        phases = []
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            phase = env._gait_phase()
            phases.append(phase)
            angle = 2.0 * np.pi * phase
            assert obs[-2:] == pytest.approx([np.sin(angle), np.cos(angle)])
            q_ref, _ = reference.joints_at(phase)
            assert np.abs(env.data.qpos[7:] - q_ref).max() < 0.011
            # Pieds au sol, pas dedans : la semelle d'appui à moins d'un cm.
            feet = [env.model.body(n).id for n in ("r_foot", "l_foot")]
            assert env.data.xpos[feet, 2].min() == pytest.approx(0.035, abs=0.01)
        assert len(set(np.round(phases, 3))) == 5
    finally:
        env.close()


def test_config_imitation_se_charge() -> None:
    config = make_poppy_env_config(load_yaml(_CONFIGS / "palier1_imitation.yaml"))
    assert config.reward.uses_imitation
    assert 0.0 < config.reference_init < 1.0
    assert as_evaluation_config(config).reference_init == 0.0
    assert config.fix_left_knee_axis is True
    assert config.gait_period == pytest.approx(load_reference().period)
    env = PoppyHumanoidEnv(floor_noise=False, reward=config.reward,
                           fix_left_knee_axis=True, gait_period=config.gait_period,
                           reference_init=1.0)
    try:
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.zeros(25, dtype=np.float32))
        assert info["imitation_reward"] > 0.0
        assert np.isfinite(reward)
    finally:
        env.close()
