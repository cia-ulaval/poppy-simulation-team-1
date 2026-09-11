"""Les trois tests de base de l'environnement Poppy.

Ces tests sont le filet de sécurité de tout le reste : le sprint 2 prévoit de
modifier l'espace d'observation en profondeur (politique asymétrique acteur /
critique). Sans eux, une modification qui décale les indices passerait inaperçue
— les formes resteraient plausibles et l'entraînement continuerait de tourner
en produisant des résultats faux.

Trois tests seulement, volontairement. Mieux vaut trois tests qui tournent
qu'une suite ambitieuse jamais terminée.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

# Contrat de l'environnement, documenté en tête de poppy_humanoid_env.py :
#   observation = qpos[2:] (30) + qvel (31) + contacts pieds (2)
#   action      = position cible des 25 articulations, normalisée dans [-1, 1]
EXPECTED_OBS_DIM = 63
EXPECTED_ACTION_DIM = 25


@pytest.fixture
def env() -> PoppyHumanoidEnv:
    """Environnement déterministe : sol fixe, aucun rendu."""
    environment = PoppyHumanoidEnv(floor_noise=False, render_mode=None)
    yield environment
    environment.close()


def test_env_creation(env: PoppyHumanoidEnv) -> None:
    """L'environnement se crée et annonce les dimensions attendues."""
    assert env.observation_space.shape == (EXPECTED_OBS_DIM,), (
        f"L'espace d'observation annonce {env.observation_space.shape}, "
        f"attendu ({EXPECTED_OBS_DIM},). Un modèle entraîné sur l'ancienne "
        f"dimension ne pourra plus être rechargé."
    )
    assert env.action_space.shape == (EXPECTED_ACTION_DIM,)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Incohérence connue entre le contrat et le code. La docstring du module "
        "et _action_to_torque (poppy_humanoid_env.py:230) annoncent des actions "
        "normalisées dans [-1, 1] : target = init + action * range. Mais "
        "action_space est hérité de MujocoEnv, donc calé sur actuator_ctrlrange "
        "(±3.1, ±7.3, ±1.8 selon l'articulation). Stable-Baselines3 borne les "
        "actions à action_space, pas à [-1, 1] : une politique peut donc émettre "
        "2.5 et viser 2.5 fois au-delà de la limite mécanique, saturant les "
        "actionneurs. Changer action_space rendra incompatibles tous les modèles "
        "déjà entraînés — c'est une décision d'équipe, pas une correction à "
        "glisser au passage. Retirer ce marqueur une fois tranché."
    ),
)
def test_action_space_is_normalised(env: PoppyHumanoidEnv) -> None:
    """L'espace d'action devrait être normalisé, comme le contrat l'annonce.

    C'est la normalisation qui rend une politique transférable vers le robot
    réel, dont les limites mécaniques diffèrent de celles du modèle simulé.
    """
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_step_shapes(env: PoppyHumanoidEnv) -> None:
    """Après une action, l'observation a la bonne taille et la récompense est finie.

    Une récompense NaN ne fait pas échouer l'entraînement : elle se propage
    silencieusement dans les gradients et la politique cesse d'apprendre sans
    qu'aucune erreur ne soit levée.
    """
    env.reset(seed=0)
    # Action tirée dans [-1, 1], le contrat réel de _action_to_torque — et non
    # via action_space.sample(), qui produit des valeurs hors contrat tant que
    # test_action_space_is_normalised reste en échec attendu.
    rng = np.random.default_rng(0)
    action = rng.uniform(-1.0, 1.0, size=EXPECTED_ACTION_DIM).astype(np.float32)
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == (EXPECTED_OBS_DIM,)
    assert np.all(np.isfinite(observation)), "L'observation contient NaN ou inf"

    assert isinstance(reward, (int, float, np.floating))
    assert np.isfinite(reward), f"Récompense non finie : {reward}"

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Les huit termes de récompense sont ce que lit scripts/evaluate.py pour
    # distinguer un robot qui marche d'un robot qui tient la pose.
    for term in ("healthy_reward", "gait_reward", "ctrl_cost", "x_position"):
        assert term in info, f"Terme absent de l'info du pas : {term}"
        assert np.isfinite(info[term]), f"Terme non fini : {term}"


def test_reset_deterministic(env: PoppyHumanoidEnv) -> None:
    """Deux reset() avec la même graine donnent exactement le même état.

    Sans cette garantie, deux entraînements ne sont pas comparables : on ne
    peut jamais dire si un écart vient d'une modification ou du hasard.
    """
    first, _ = env.reset(seed=1234)
    second, _ = env.reset(seed=1234)

    np.testing.assert_array_equal(
        first,
        second,
        err_msg="Deux reset() avec la même graine donnent des états différents",
    )

    # Et une graine différente doit donner autre chose, sinon la graine est
    # ignorée et le test ci-dessus ne prouve rien.
    third, _ = env.reset(seed=4321)
    assert not np.array_equal(first, third), "La graine n'a aucun effet sur reset()"
