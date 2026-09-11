"""Construction des objets de configuration à partir des YAML de ``configs/``.

Ce module existe pour que l'entraînement et l'évaluation lisent un fichier de
configuration **de la même façon**. Un modèle évalué dans un environnement qui
ne correspond pas à celui de son entraînement donne des métriques fausses sans
qu'aucune erreur ne soit levée : mêmes formes, mêmes types, résultats
silencieusement dénués de sens.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config.settings import DomainRandomizationConfig, PoppyEnvironmentConfig


def load_yaml(path: Path) -> dict:
    """Charge un fichier de configuration YAML.

    Args:
        path: Chemin du fichier.

    Returns:
        Le contenu du fichier, ou un dictionnaire vide si le fichier est vide.
    """
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_poppy_env_config(cfg: dict) -> PoppyEnvironmentConfig:
    """Construit la configuration d'environnement Poppy depuis un YAML chargé.

    Args:
        cfg: Contenu d'un fichier de ``configs/``, tel que rendu par
            :func:`load_yaml`. Les sections absentes prennent leurs valeurs
            par défaut.

    Returns:
        La configuration d'environnement correspondante.
    """
    env_cfg = cfg.get("environment", {})
    dr_cfg = cfg.get("domain_randomization", {})
    return PoppyEnvironmentConfig(
        terminate_when_unhealthy=env_cfg.get("terminate_when_unhealthy", True),
        healthy_z_range=tuple(env_cfg.get("healthy_z_range", [0.25, 0.70])),
        n_envs=env_cfg.get("n_envs", 16),
        normalize_obs=env_cfg.get("normalize_obs", True),
        normalize_reward=env_cfg.get("normalize_reward", False),
        clip_obs=env_cfg.get("clip_obs", 10.0),
        gamma=env_cfg.get("gamma", 0.99),
        frame_skip=env_cfg.get("frame_skip", 5),
        domain_randomization=DomainRandomizationConfig(
            enabled=dr_cfg.get("enabled", True),
            friction_range=tuple(dr_cfg.get("friction_range", [0.5, 1.5])),
            restitution_range=tuple(dr_cfg.get("restitution_range", [0.0, 0.3])),
        ),
    )


def as_evaluation_config(
    config: PoppyEnvironmentConfig,
    floor_noise: bool = False,
) -> PoppyEnvironmentConfig:
    """Dérive une configuration d'évaluation depuis une configuration d'entraînement.

    Trois différences, et seulement trois : un environnement au lieu de N, les
    récompenses jamais normalisées — sinon les chiffres affichés ne sont pas
    comparables d'un entraînement à l'autre — et la randomisation du sol
    désactivée par défaut, pour que deux exécutions du même modèle donnent le
    même résultat.

    Args:
        config: Configuration issue du YAML d'entraînement.
        floor_noise: Réactive la randomisation du sol, pour mesurer la
            robustesse plutôt que la performance nominale.

    Returns:
        La configuration d'évaluation correspondante.
    """
    return PoppyEnvironmentConfig(
        terminate_when_unhealthy=config.terminate_when_unhealthy,
        healthy_z_range=config.healthy_z_range,
        n_envs=1,
        normalize_obs=config.normalize_obs,
        normalize_reward=False,
        clip_obs=config.clip_obs,
        gamma=config.gamma,
        frame_skip=config.frame_skip,
        domain_randomization=DomainRandomizationConfig(
            enabled=floor_noise,
            friction_range=config.domain_randomization.friction_range,
            restitution_range=config.domain_randomization.restitution_range,
        ),
    )
