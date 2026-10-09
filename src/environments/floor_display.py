"""Couleur du sol selon sa friction, pour voir la randomisation à l'écran.

Partagé par ``scripts/viewer.py`` et ``scripts/visualize.py``. Ne touche qu'à
``geom_rgba`` : l'affichage change, la physique non.

Volontairement hors de ``poppy_humanoid_env.py`` : l'environnement ne sait rien
de la façon dont on le regarde.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

SLIPPERY_RGBA = np.array([0.75, 0.88, 1.00, 1.0])  # bleu clair : glissant
GRIPPY_RGBA = np.array([0.05, 0.15, 0.45, 1.0])    # bleu marine : adhérent

_FLOOR_GEOM_NAME = "floor"


def floor_color(friction: float, friction_range: tuple[float, float]) -> NDArray:
    """Couleur d'un sol : bleu clair si glissant, bleu marine si adhérent.

    Args:
        friction: Friction de glissement tirée pour le sol.
        friction_range: Plage du tirage, ``(min, max)``. Le minimum donne le bleu
            clair, le maximum le bleu marine.

    Returns:
        La couleur RGBA, interpolée linéairement entre les deux extrêmes.
    """
    low, high = friction_range
    t = float(np.clip((friction - low) / (high - low), 0.0, 1.0))
    return SLIPPERY_RGBA + t * (GRIPPY_RGBA - SLIPPERY_RGBA)


def color_floor(env: PoppyHumanoidEnv, friction_range: tuple[float, float]) -> float:
    """Recolore le sol d'après la friction tirée au dernier ``reset()``.

    À appeler après chaque ``reset()`` : c'est là que la friction change.

    Args:
        env: L'environnement nu, sans enveloppe (``.unwrapped``).
        friction_range: Plage du tirage, celle passée à l'environnement.

    Returns:
        La friction du sol, pour l'afficher.
    """
    friction = env.get_floor_randomization_info()["floor_slide_friction"]
    floor_id = env.model.geom(_FLOOR_GEOM_NAME).id
    env.model.geom_rgba[floor_id] = floor_color(friction, friction_range)
    return friction
