"""Tests de la récompense configurable du palier 1.

Le point qui compte : un YAML sans section ``reward:`` doit donner exactement
l'ancienne récompense. Sans cette garantie, les modèles existants seraient
évalués sur une récompense qui n'est plus la leur, sans aucune erreur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import (
    RewardConfig,
    as_evaluation_config,
    load_yaml,
    make_poppy_env_config,
    make_reward_config,
)
from src.environments.poppy_humanoid_env import (
    PoppyHumanoidEnv,
    _restitution_to_dampratio,
)

_CONFIGS = Path(__file__).parent.parent / "configs"


def test_config_historique_garde_la_recompense_par_defaut() -> None:
    """poppy_robust.yaml n'a pas de section reward: elle prend les anciens poids."""
    config = make_poppy_env_config(load_yaml(_CONFIGS / "poppy_robust.yaml"))
    assert config.reward == RewardConfig()
    assert config.normalize_action_space is False


def test_cle_de_recompense_inconnue_leve_une_erreur() -> None:
    """Une faute de frappe ne doit pas entraîner 10 M de pas avec le défaut."""
    with pytest.raises(ValueError, match="healty"):
        make_reward_config({"healty": 0.5})


def test_palier1_se_charge_et_survit_a_l_evaluation() -> None:
    """La configuration d'évaluation garde la récompense et l'espace d'action."""
    config = make_poppy_env_config(load_yaml(_CONFIGS / "palier1.yaml"))
    assert config.reward.gait_single == 0.0
    assert config.reward.air_time_weight > 0.0
    evaluation = as_evaluation_config(config)
    assert evaluation.reward == config.reward
    assert evaluation.normalize_action_space is True
    assert evaluation.domain_randomization.enabled is False


def test_espace_d_action_normalise_sur_demande() -> None:
    """Le drapeau donne l'espace [-1, 1] du contrat, à la même forme."""
    env = PoppyHumanoidEnv(floor_noise=False, normalize_action_space=True)
    try:
        assert env.action_space.shape == (25,)
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
    finally:
        env.close()


def test_nouveaux_termes_presents_et_finis() -> None:
    """Les termes du palier 1 sont rapportés, et nuls quand leur poids l'est."""
    env = PoppyHumanoidEnv(floor_noise=False)
    try:
        env.reset(seed=0)
        _, reward, _, _, info = env.step(np.zeros(25, dtype=np.float32))
        assert np.isfinite(reward)
        for term in ("air_time_reward", "yaw_rate_cost", "joint_limit_cost",
                     "torso_upright_reward", "vel_tracking_reward",
                     "torso_tilt_cost", "base_height_cost", "ang_vel_xy_cost",
                     "feet_slide_cost"):
            assert info[term] == 0.0, term
        # Debout à la pose de départ, la poitrine est droite par construction.
        assert info["torso_uprightness"] == pytest.approx(1.0, abs=0.02)
    finally:
        env.close()


def test_suivi_de_vitesse_en_cloche() -> None:
    """Le suivi de vitesse paie au plus à la cible, et presque rien à l'arrêt
    comme à la vitesse d'une chute : c'est ce qui retire l'intérêt du plongeon."""
    weights = RewardConfig(vel_tracking_weight=1.5, target_vel=0.25,
                           vel_tracking_sigma=0.2)
    env = PoppyHumanoidEnv(floor_noise=False, reward=weights)
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(25, dtype=np.float32))
        expected = 1.5 * np.exp(-((info["forward_vel"] - 0.25) / 0.2) ** 2)
        assert info["vel_tracking_reward"] == pytest.approx(expected)
    finally:
        env.close()

    def cloche(v: float) -> float:
        return 1.5 * float(np.exp(-((v - 0.25) / 0.2) ** 2))

    assert cloche(0.25) == pytest.approx(1.5)
    assert cloche(0.0) < 0.5 * cloche(0.25)
    assert cloche(0.8) < 0.01


def test_suivi_de_vitesse_filtre() -> None:
    """Avec un filtre, le suivi porte sur la vitesse lissée, partie de zéro au
    reset : un balancement d'avant en arrière ne touche plus la cible."""
    weights = RewardConfig(vel_tracking_weight=1.5, target_vel=0.25,
                           vel_tracking_sigma=0.2, vel_tracking_tau=1.0)
    env = PoppyHumanoidEnv(floor_noise=False, reward=weights)
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(25, dtype=np.float32))
        alpha = env.dt / (1.0 + env.dt)
        smoothed = alpha * info["forward_vel"]
        expected = 1.5 * np.exp(-((smoothed - 0.25) / 0.2) ** 2)
        assert info["vel_tracking_reward"] == pytest.approx(expected)
    finally:
        env.close()


def test_palier1_marche_se_charge() -> None:
    """La config révisée active le suivi de vitesse et coupe la vitesse brute."""
    cfg = load_yaml(_CONFIGS / "palier1_marche.yaml")
    weights = make_reward_config(cfg.get("reward"))
    assert weights.forward_weight == 0.0
    assert weights.vel_tracking_weight > 0.0
    assert weights.air_time_target == 0.0
    assert weights.vel_tracking_tau > 0.0


def test_restitution_vers_amortissement() -> None:
    """Pas de rebond : amortissement critique ; plus de rebond : moins amorti."""
    assert _restitution_to_dampratio(0.0) == 1.0
    ratios = [_restitution_to_dampratio(e) for e in (0.1, 0.3, 0.5)]
    assert all(0.0 < r < 1.0 for r in ratios)
    assert ratios == sorted(ratios, reverse=True)
