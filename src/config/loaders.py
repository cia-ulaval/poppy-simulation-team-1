"""Construction des objets de configuration à partir des YAML de ``configs/``.

Ce module existe pour que l'entraînement et l'évaluation lisent un fichier de
configuration **de la même façon**. Un modèle évalué dans un environnement qui
ne correspond pas à celui de son entraînement donne des métriques fausses sans
qu'aucune erreur ne soit levée : mêmes formes, mêmes types, résultats
silencieusement dénués de sens.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import yaml

from src.config.settings import (
    DomainRandomizationConfig,
    PoppyEnvironmentConfig,
    PushConfig,
    RewardConfig,
    TerrainConfig,
)


def load_yaml(path: Path) -> dict:
    """Charge un fichier de configuration YAML.

    Args:
        path: Chemin du fichier.

    Returns:
        Le contenu du fichier, ou un dictionnaire vide si le fichier est vide.
    """
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _make_section(cls, section: dict | None, name: str):
    """Construit une dataclass de réglages depuis une section du YAML.

    Une clé inconnue lève une erreur au lieu d'être ignorée : une faute de
    frappe dans un poids entraînerait sinon 10 M de pas avec la valeur par
    défaut, sans que rien ne le signale.

    Args:
        cls: La dataclass à construire.
        section: Contenu de la section, ou ``None`` si absente.
        name: Nom de la section, pour le message d'erreur.

    Returns:
        Les réglages, ceux non précisés gardant leur valeur par défaut.

    Raises:
        ValueError: Si la section contient une clé inconnue.
    """
    section = section or {}
    types = {f.name: f.type for f in fields(cls)}
    unknown = sorted(set(section) - set(types))
    if unknown:
        raise ValueError(
            f"Clés inconnues dans {name}: {', '.join(unknown)}. "
            f"Connues : {', '.join(sorted(types))}"
        )
    values = {}
    for key, value in section.items():
        kind = str(types[key])
        if kind.startswith("tuple"):
            values[key] = tuple(float(v) for v in value)
        elif kind == "bool":
            values[key] = bool(value)
        else:
            values[key] = float(value)
    return cls(**values)


def make_reward_config(section: dict | None) -> RewardConfig:
    """Construit les poids de la récompense depuis la section ``reward:``."""
    return _make_section(RewardConfig, section, "reward")


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
            restitution_in_solref=bool(dr_cfg.get("restitution_in_solref", False)),
            push_as_force=bool(dr_cfg.get("push_as_force", False)),
        ),
        reward=make_reward_config(cfg.get("reward")),
        normalize_action_space=bool(env_cfg.get("normalize_action_space", False)),
        terrain=_make_section(TerrainConfig, cfg.get("terrain"), "terrain"),
        pushes=_make_section(PushConfig, cfg.get("pushes"), "pushes"),
        fix_left_knee_axis=bool(env_cfg.get("fix_left_knee_axis", False)),
        fix_arm_frames=bool(env_cfg.get("fix_arm_frames", False)),
        gait_period=float(env_cfg.get("gait_period", 0.0)),
        reference_init=float(env_cfg.get("reference_init", 0.0)),
    )


def as_evaluation_config(
    config: PoppyEnvironmentConfig,
    floor_noise: bool = False,
    terrain_difficulty: float = 0.0,
) -> PoppyEnvironmentConfig:
    """Dérive une configuration d'évaluation depuis une configuration d'entraînement.

    Quatre différences, et seulement quatre : un environnement au lieu de N,
    les récompenses jamais normalisées — sinon les chiffres affichés ne sont
    pas comparables d'un entraînement à l'autre —, la randomisation du sol
    désactivée par défaut, pour que deux exécutions du même modèle donnent le
    même résultat, et le départ toujours debout (``reference_init`` à 0) : le
    banc d'essai commun, et le vrai robot, partent de la pose debout.

    Args:
        config: Configuration issue du YAML d'entraînement.
        floor_noise: Réactive la randomisation du sol, pour mesurer la
            robustesse plutôt que la performance nominale.
        terrain_difficulty: Relief du sol accidenté, s'il est activé. 0 par
            défaut : le banc d'essai commun est plat.

    Returns:
        La configuration d'évaluation correspondante.
    """
    return replace(
        config,
        n_envs=1,
        reference_init=0.0,
        normalize_reward=False,
        domain_randomization=replace(config.domain_randomization, enabled=floor_noise),
        # Relief fixé : le tirage en trois tiers ne sert qu'à l'entraînement.
        terrain=replace(config.terrain, difficulty=terrain_difficulty, difficulty_mix=False),
    )
