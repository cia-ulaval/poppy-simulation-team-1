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
                     "torso_upright_reward"):
            assert info[term] == 0.0, term
        # Debout à la pose de départ, la poitrine est droite par construction.
        assert info["torso_uprightness"] == pytest.approx(1.0, abs=0.02)
    finally:
        env.close()


def test_restitution_vers_amortissement() -> None:
    """Pas de rebond : amortissement critique ; plus de rebond : moins amorti."""
    assert _restitution_to_dampratio(0.0) == 1.0
    ratios = [_restitution_to_dampratio(e) for e in (0.1, 0.3, 0.5)]
    assert all(0.0 < r < 1.0 for r in ratios)
    assert ratios == sorted(ratios, reverse=True)
