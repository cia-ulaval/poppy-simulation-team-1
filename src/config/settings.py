"""Objets de configuration d'environnement.

Volontairement réduit à trois dataclasses. Les hyperparamètres d'algorithme
(taux d'apprentissage, taille de lot, architecture du réseau) ne sont pas ici :
ils vivent dans les YAML de ``configs/`` et sont passés tels quels à
Stable-Baselines3 par ``scripts/train_poppy.py``. Les dupliquer en dataclasses
créait deux sources de vérité, dont une que personne ne mettait à jour.

Ce module n'importe volontairement ni ``torch`` ni ``numpy`` : il est chargé
par l'évaluation et par les tests, qui n'ont pas à payer l'import de PyTorch
pour lire un seuil de hauteur.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Randomisation du sol, ré-échantillonnée à chaque ``reset()``.

    Attributes:
        enabled: Active la randomisation. Désactivée, deux exécutions du même
            modèle donnent le même résultat — c'est ce qu'on veut en
            évaluation.
        friction_range: Bornes du coefficient de friction de glissement.
        restitution_range: Bornes du coefficient de rebond.
    """

    enabled: bool = True
    friction_range: tuple[float, float] = (0.5, 1.5)
    restitution_range: tuple[float, float] = (0.0, 0.3)


@dataclass(frozen=True)
class PoppyEnvironmentConfig:
    """Configuration de l'environnement Poppy Humanoid.

    Attributes:
        terminate_when_unhealthy: Termine l'épisode quand le robot tombe.
        healthy_z_range: Hauteur du bassin (m) considérée comme debout.
        n_envs: Environnements parallèles. Au-delà de ce que la machine peut
            lancer, ``SubprocVecEnv`` meurt sur un ``BrokenPipeError``.
        normalize_obs: Normalise les observations.
        normalize_reward: Normalise les récompenses. Toujours faux en
            évaluation : une récompense normalisée n'est pas comparable d'un
            entraînement à l'autre.
        clip_obs: Bornage des observations normalisées.
        gamma: Facteur d'actualisation, utilisé par la normalisation.
        frame_skip: Pas de simulation par pas de politique (dt = 2 ms x
            frame_skip).
        domain_randomization: Réglages de randomisation du sol.
    """

    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = (0.25, 0.70)
    n_envs: int = 8
    normalize_obs: bool = True
    normalize_reward: bool = False
    clip_obs: float = 10.0
    gamma: float = 0.99
    frame_skip: int = 5
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration d'un environnement Gymnasium générique.

    Sert au seul drapeau ``train_poppy.py --baseline``, qui entraîne sur
    ``Humanoid-v5`` pour vérifier que le pipeline apprend quelque chose. Les
    valeurs par défaut sont celles d'``Humanoid-v5`` — noter que son bassin est
    à ~1,4 m, contre ~0,4 m pour Poppy : les deux ``healthy_z_range`` n'ont
    rien à voir et ne sont pas interchangeables.
    """

    env_id: str = "Humanoid-v5"
    terminate_when_unhealthy: bool = True
    healthy_z_range: tuple[float, float] = (1.0, 2.0)
    n_envs: int = 8
    normalize_obs: bool = True
    normalize_reward: bool = False
    clip_obs: float = 10.0
    gamma: float = 0.99
