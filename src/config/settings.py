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
    # Corrections des défauts de docs/TECHNIQUE.md §8, désactivées par défaut
    # pour que les YAML existants gardent exactement leur physique.
    restitution_in_solref: bool = False
    push_as_force: bool = False


@dataclass(frozen=True)
class TerrainConfig:
    """Sol à relief aléatoire (palier 2), régénéré à chaque ``reset()``.

    Désactivé, le modèle MuJoCo est exactement celui du MJCF. Activé, un champ
    de hauteur (``hfield``) est ajouté par le code et le plan ``floor`` cesse
    de toucher le robot. À ``difficulty = 0`` le relief est plat : c'est le
    banc d'essai commun.

    Attributes:
        enabled: Ajoute le champ de hauteur au modèle.
        difficulty: Intensité du relief, de 0 (plat) à 1 (toutes les
            amplitudes ci-dessous au maximum). Le curriculum la fait monter.
        size: Côté du terrain (m), centré sur le point de départ.
        resolution: Taille d'une case (m).
        slope_amplitude: Amplitude des ondulations longues (m) — pentes douces.
        bump_height: Hauteur maximale des bosses et des creux (m).
        step_height: Hauteur maximale des petites marches (m).
        roughness: Rugosité case par case (m).
        flat_radius: Rayon de la zone plate au départ (m).
    """

    enabled: bool = False
    difficulty: float = 0.0
    size: float = 20.0
    resolution: float = 0.05
    slope_amplitude: float = 0.05
    bump_height: float = 0.04
    step_height: float = 0.03
    roughness: float = 0.005
    flat_radius: float = 0.6


@dataclass(frozen=True)
class PushConfig:
    """Poussées sur le bassin (palier 3), utilisées quand ``push_as_force``.

    Sans ``push_as_force``, l'ancien comportement s'applique : un « coup » d'un
    pas de 10 ms, de 1 à 5 unités, écrit comme un couple.

    Attributes:
        probability: Probabilité, à chaque pas sans poussée en cours, d'en
            commencer une.
        force_range: Bornes de l'intensité (N), avant ``scale``.
        duration: Durée d'une poussée (s).
        scale: Multiplicateur de l'intensité, de 0 à 1. Le curriculum le
            fait monter.
    """

    probability: float = 0.02
    force_range: tuple[float, float] = (1.0, 5.0)
    duration: float = 0.01
    scale: float = 1.0


@dataclass(frozen=True)
class RewardConfig:
    """Poids de la récompense, par pas de 10 ms.

    Les valeurs par défaut sont **exactement** la récompense codée en dur
    jusqu'en octobre 2026 : un YAML sans section ``reward:`` entraîne et évalue
    comme avant. Les termes ajoutés pour le palier 1 valent 0 par défaut.
    Détail et justification : ``spec/recompense-et-paliers.md``.

    Attributes:
        forward_weight: Poids de la vitesse d'avance, projetée sur le cap.
        max_forward_vel: Plafond de la vitesse récompensée (m/s).
        healthy: Bonus par pas debout.
        upright_weight: Poids de la verticalité du bassin.
        gait_single: Bonus d'appui simple, payé à chaque pas.
        gait_double: Bonus de double appui, payé à chaque pas.
        lateral_weight: Pénalité de vitesse latérale (valeur absolue).
        ctrl_weight: Pénalité Σ action².
        action_rate_weight: Pénalité Σ (action − action précédente)².
        joint_vel_weight: Pénalité Σ vitesse articulaire².
        yaw_rate_weight: Pénalité sur le carré de la vitesse de lacet.
        yaw_rate_tau: Constante de temps (s) du filtre appliqué à la vitesse
            de lacet avant pénalité. 0 : vitesse instantanée. Un filtre d'une
            seconde ne garde que le virage de fond, pas l'oscillation du bassin
            à chaque pas.
        air_time_weight: Poids du temps de vol, payé quand un pied se repose.
        air_time_target: Temps de vol neutre (s) ; en dessous, le terme est
            négatif.
        air_time_max: Plafond du temps de vol récompensé (s), contre le saut.
        joint_limit_weight: Pénalité de dépassement des butées douces (rad).
        joint_limit_soft: Fraction de la course au-delà de laquelle une
            articulation est pénalisée.
        torso_upright_weight: Poids de la verticalité de la poitrine, mesurée
            par rapport à la pose debout.
    """

    forward_weight: float = 5.0
    max_forward_vel: float = 0.5
    healthy: float = 1.0
    upright_weight: float = 1.0
    gait_single: float = 0.3
    gait_double: float = 0.1
    lateral_weight: float = 0.5
    ctrl_weight: float = 0.1
    action_rate_weight: float = 0.05
    joint_vel_weight: float = 0.005
    yaw_rate_weight: float = 0.0
    yaw_rate_tau: float = 0.0
    air_time_weight: float = 0.0
    air_time_target: float = 0.25
    air_time_max: float = 0.5
    joint_limit_weight: float = 0.0
    joint_limit_soft: float = 0.95
    torso_upright_weight: float = 0.0


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
        reward: Poids de la récompense.
        normalize_action_space: Annonce un espace d'action ``[-1, 1]``,
            conforme au contrat. Faux par défaut : le corriger change
            ``action_space`` et les modèles existants ne se chargent plus sans
            le forcer (voir ``docs/TECHNIQUE.md`` §8).
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
    reward: RewardConfig = field(default_factory=RewardConfig)
    normalize_action_space: bool = False
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    pushes: PushConfig = field(default_factory=PushConfig)
    # Le genou gauche du MJCF plie à l'envers : voir PoppyHumanoidEnv.
    # Faux par défaut, c'est un changement de physique.
    fix_left_knee_axis: bool = False


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
