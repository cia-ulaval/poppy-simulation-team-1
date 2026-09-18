"""Construction des environnements vectorisés, pour l'entraînement comme pour
l'évaluation.

Trois fonctions de module, pas une classe : il n'y a aucun état à porter d'un
appel au suivant. La fabrique précédente était une classe dont seules les
méthodes statiques étaient appelées ; sa machinerie d'instance (dossier
temporaire, ``__del__``) ne servait que la génération d'obstacles procéduraux,
supprimée avec le banc d'essai multi-algorithmes.

Un environnement d'évaluation se construit avec les mêmes fonctions qu'un
environnement d'entraînement, puis ``training = False`` et
``norm_reward = False``. C'est volontaire : deux constructeurs distincts
finissent par diverger, et un modèle évalué dans un environnement différent de
celui de son entraînement donne des métriques fausses sans qu'aucune erreur ne
soit levée.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from src.config.settings import EnvironmentConfig, PoppyEnvironmentConfig
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv, register_poppy_env

# Durée maximale d'un épisode, en pas de politique. À 10 ms par pas
# (frame_skip=5 x 2 ms), cela fait 10 s de temps simulé.
_MAX_EPISODE_STEPS = 1000


def _vectorise(
    make_env: Callable[[int], Callable[[], gym.Env]],
    n_envs: int,
    seed: int,
    use_subprocess: bool,
    *,
    norm_obs: bool,
    norm_reward: bool,
    clip_obs: float,
    gamma: float,
) -> VecNormalize:
    """Vectorise ``n_envs`` environnements et les enveloppe de ``VecNormalize``.

    Args:
        make_env: Fabrique de thunk, appelée avec le rang de chaque
            sous-environnement.
        n_envs: Nombre d'environnements parallèles.
        seed: Graine aléatoire globale.
        use_subprocess: Utilise ``SubprocVecEnv`` (vrai parallélisme) dès que
            ``n_envs > 1``. Sur une machine ordinaire, lancer trop de processus
            MuJoCo fait mourir l'entraînement sur un ``BrokenPipeError`` :
            c'est au YAML de rester raisonnable.
        norm_obs: Normalise les observations.
        norm_reward: Normalise les récompenses.
        clip_obs: Bornage des observations normalisées.
        gamma: Facteur d'actualisation, utilisé par la normalisation.

    Returns:
        L'environnement vectorisé et normalisé.
    """
    set_random_seed(seed)
    env_fns = [make_env(rank) for rank in range(n_envs)]

    if n_envs > 1 and use_subprocess:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    return VecNormalize(
        vec_env,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        gamma=gamma,
    )


def make_poppy_env(
    config: PoppyEnvironmentConfig,
    n_envs: int | None = None,
    seed: int = 0,
    use_subprocess: bool = True,
) -> VecNormalize:
    """Construit l'environnement vectorisé du Poppy Humanoid.

    Chaque sous-environnement est un ``PoppyHumanoidEnv`` dont la randomisation
    du sol suit ``config.domain_randomization``.

    Args:
        config: Configuration d'environnement Poppy, issue du YAML.
        n_envs: Nombre d'environnements parallèles. ``None`` prend
            ``config.n_envs``.
        seed: Graine aléatoire ; le sous-environnement de rang ``r`` reçoit
            ``seed + r``.
        use_subprocess: Vrai parallélisme par processus séparés.

    Returns:
        L'environnement vectorisé, enveloppé de ``VecNormalize``.
    """
    register_poppy_env()

    dr = config.domain_randomization
    n = config.n_envs if n_envs is None else n_envs

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = PoppyHumanoidEnv(
                floor_noise=dr.enabled,
                friction_range=dr.friction_range,
                restitution_range=dr.restitution_range,
                healthy_z_range=config.healthy_z_range,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                frame_skip=config.frame_skip,
            )
            env = TimeLimit(env, max_episode_steps=_MAX_EPISODE_STEPS)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        return _init

    return _vectorise(
        make_env,
        n,
        seed,
        use_subprocess,
        norm_obs=config.normalize_obs,
        norm_reward=config.normalize_reward,
        clip_obs=config.clip_obs,
        gamma=config.gamma,
    )


def make_baseline_env(
    config: EnvironmentConfig,
    n_envs: int | None = None,
    seed: int = 0,
    use_subprocess: bool = True,
) -> VecNormalize:
    """Construit l'environnement Humanoid-v5 de Gymnasium.

    Sert uniquement à valider le pipeline (``train_poppy.py --baseline``) : si
    l'entraînement ne progresse pas sur un environnement de référence connu, le
    problème n'est pas dans le modèle Poppy.

    ``xml_file`` n'est volontairement pas transmis : Humanoid-v5 connaît le
    sien. L'ancienne fabrique passait ``xml_file=None`` quand la génération
    d'obstacles était désactivée — c'est-à-dire toujours — et
    ``expand_model_path`` échouait sur un ``AttributeError``. Le drapeau
    ``--baseline`` ne fonctionnait donc plus du tout.

    Args:
        config: Configuration d'environnement générique.
        n_envs: Nombre d'environnements parallèles. ``None`` prend
            ``config.n_envs``.
        seed: Graine aléatoire ; le sous-environnement de rang ``r`` reçoit
            ``seed + r``.
        use_subprocess: Vrai parallélisme par processus séparés.

    Returns:
        L'environnement vectorisé, enveloppé de ``VecNormalize``.
    """
    n = config.n_envs if n_envs is None else n_envs

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env = gym.make(
                config.env_id,
                terminate_when_unhealthy=config.terminate_when_unhealthy,
                healthy_z_range=config.healthy_z_range,
            )
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        return _init

    return _vectorise(
        make_env,
        n,
        seed,
        use_subprocess,
        norm_obs=config.normalize_obs,
        norm_reward=config.normalize_reward,
        clip_obs=config.clip_obs,
        gamma=config.gamma,
    )


def load_normalized_env(
    vec_normalize_path: str,
    base_env: DummyVecEnv,
    training: bool = False,
) -> VecNormalize:
    """Recharge des statistiques de normalisation enregistrées.

    ``norm_reward`` est forcé à ``False`` : une récompense normalisée n'est
    comparable qu'à l'intérieur d'un même entraînement, jamais d'un modèle à
    l'autre. Une évaluation doit montrer les récompenses telles quelles.

    Args:
        vec_normalize_path: Fichier ``.pkl`` enregistré à l'entraînement.
        base_env: Environnement vectorisé à envelopper.
        training: Continue de mettre à jour les statistiques. Faux en
            évaluation, sinon les chiffres dépendent de l'ordre des épisodes.

    Returns:
        L'environnement enveloppé des statistiques rechargées.
    """
    env = VecNormalize.load(vec_normalize_path, base_env)
    env.training = training
    env.norm_reward = False
    return env
