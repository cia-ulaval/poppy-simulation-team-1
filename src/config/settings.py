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
        difficulty_mix: À chaque ``reset()``, tire le relief en trois tiers :
            plat, au hasard entre 0 et ``difficulty``, ou ``difficulty``
            (palier 3 : les poussées sur tous les sols, sans oublier aucun
            d'eux). Désactivé, tous les terrains sont à ``difficulty``.
    """

    enabled: bool = False
    difficulty: float = 0.0
    difficulty_mix: bool = False
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
        vel_tracking_weight: Poids du suivi d'une vitesse cible : le terme
            vaut ``poids * exp(-((v - cible) / largeur)²)``, maximal à la
            cible et qui retombe aussi bien plus lent que plus vite. Une chute
            vers l'avant, trop rapide, n'y rapporte presque rien.
        target_vel: Vitesse d'avance visée (m/s), projetée sur le cap.
        vel_tracking_sigma: Largeur de la cloche (m/s). Trop étroite, un robot
            immobile ne reçoit aucun signal pour se mettre en marche.
        vel_tracking_tau: Constante de temps (s) du filtre appliqué à la
            vitesse avant le suivi. 0 : vitesse instantanée, qu'un robot qui
            se balance d'avant en arrière touche à chaque oscillation sans
            avancer (pilote du 7 nov. 2026). Filtrée, seule l'avance paie.
        torso_tilt_weight: Pénalité d'inclinaison de la poitrine,
            ``poids * (1 - verticalité)``, soit ~θ²/2. Contrairement à
            ``torso_upright_weight``, un bonus qui gonfle aussi la prime pour
            rester debout, elle ne coûte rien droit et pèse dès que le robot
            se penche pour se jeter en avant. Sur le robot, l'IMU du bassin et
            les encodeurs du buste donnent cette inclinaison.
        base_height_weight: Pénalité ``poids * (z - cible)²`` sur la hauteur
            du bassin (termes courants de legged_gym / Unitree) : un robot qui
            plonge ou s'accroupit descend.
        base_height_target: Hauteur visée du bassin (m) ; ~0,41 debout.
        ang_vel_xy_weight: Pénalité ``poids * (ωx² + ωy²)`` sur le roulis et
            le tangage du bassin : contre le balancement et le jeté en avant.
            Mesurable par le gyroscope de l'IMU.
        feet_slide_weight: Pénalité ``poids * Σ |v_pied|`` sur la vitesse
            horizontale d'un pied au sol : un pas doit poser le pied, pas le
            traîner.
        phase_contact_weight: Prime d'alternance, active seulement avec
            l'horloge de marche (``environment.gait_period``). Chaque pied a sa
            phase, décalée d'un demi-cycle ; il doit être au sol pendant
            ``phase_stance_ratio`` du cycle et en l'air le reste. Le terme
            vaut ``poids * fraction des pieds conformes`` (recette des
            humanoïdes G1 / H1 d'unitree_rl_gym).
        phase_stance_ratio: Part du cycle où un pied doit être au sol ; un peu
            plus de 0,5 laisse un court double appui.
        imitation_pos_weight: Imitation de la marche de référence (piste 3),
            active seulement avec l'horloge de marche. Angles des dix
            articulations des jambes : ``poids * exp(-échelle * Σ (q - q̂)²)``,
            q̂ lu dans ``assets/poppy_humanoid/reference_walk.npz`` à la phase
            courante. Noyau exponentiel de DeepMimic (Peng et al. 2018) ;
            Disney (BDX) et Open Duck imitent les mêmes grandeurs.
        imitation_pos_scale: Échelle du noyau des angles (1/rad²). 5 : une
            erreur de 0,1 rad sur chacune des dix articulations rapporte 0,61.
        imitation_upper_weight: Même noyau, même échelle, sur les quinze
            articulations du haut du corps (tronc, tête, bras), que la
            référence tient près de la pose debout.
        imitation_vel_weight: Vitesses articulaires des jambes :
            ``poids * exp(-échelle * Σ (q̇ - q̇̂)²)``.
        imitation_vel_scale: Échelle du noyau des vitesses (s²/rad²).
        imitation_contact_weight: Contacts des pieds : ``poids * fraction des
            pieds dont le contact est celui de la référence``. Mesurable par
            les capteurs de pression sous les pieds.
        imitation_base_weight: Hauteur et inclinaison du bassin :
            ``poids * exp(-échelle_h * (z - ẑ)² - échelle_i * |g - ĝ|²)``, g étant
            la verticale exprimée dans le repère du bassin (ce que mesure
            l'IMU, indépendant du cap). La référence fait rouler le bassin de
            ±10° à chaque pas : c'est ainsi que Poppy, sans roulis de
            cheville, porte son poids sur le pied d'appui.
        imitation_height_scale: Échelle de la hauteur du bassin (1/m²).
        imitation_tilt_scale: Échelle de l'inclinaison du bassin (sans unité ;
            |g - ĝ|² vaut à peu près l'angle d'écart au carré, en rad²).
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
    vel_tracking_weight: float = 0.0
    target_vel: float = 0.25
    vel_tracking_sigma: float = 0.2
    vel_tracking_tau: float = 0.0
    torso_tilt_weight: float = 0.0
    base_height_weight: float = 0.0
    base_height_target: float = 0.41
    ang_vel_xy_weight: float = 0.0
    feet_slide_weight: float = 0.0
    phase_contact_weight: float = 0.0
    phase_stance_ratio: float = 0.55
    imitation_pos_weight: float = 0.0
    imitation_pos_scale: float = 5.0
    imitation_upper_weight: float = 0.0
    imitation_vel_weight: float = 0.0
    imitation_vel_scale: float = 0.05
    imitation_contact_weight: float = 0.0
    imitation_base_weight: float = 0.0
    imitation_height_scale: float = 200.0
    imitation_tilt_scale: float = 20.0

    @property
    def uses_imitation(self) -> bool:
        """Vrai si un terme d'imitation de la marche de référence est actif."""
        return any(weight > 0.0 for weight in (
            self.imitation_pos_weight, self.imitation_upper_weight,
            self.imitation_vel_weight, self.imitation_contact_weight,
            self.imitation_base_weight))


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
    # Bras du MJCF orientés comme dans l'URDF, pose debout bras le long du
    # corps : voir PoppyHumanoidEnv. Faux par défaut, changement de physique.
    fix_arm_frames: bool = False
    # Horloge de marche : période (s) d'un cycle de deux pas. 0 = pas
    # d'horloge, observation de 63 dimensions. Au-delà, l'observation gagne
    # sin et cos de la phase (65 dimensions) : c'est un changement du contrat
    # d'observation, les modèles entraînés sans horloge ne s'y chargent pas.
    # Sur le robot, la phase vient de l'horloge de l'ordinateur de bord.
    gait_period: float = 0.0
    # Départ sur la marche de référence (piste 3) : probabilité qu'un épisode
    # commence à une phase tirée au hasard, dans la pose de la référence à
    # cette phase (« reference state initialization » de DeepMimic). Les
    # autres partent de la pose debout, phase nulle, comme le vrai robot.
    # 0 par défaut ; toujours 0 en évaluation (as_evaluation_config).
    reference_init: float = 0.0


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
