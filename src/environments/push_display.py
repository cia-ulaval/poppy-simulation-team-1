"""Poussées visibles dans les vidéos : une flèche qui arrive sur le bassin, et un impact.

Pendant une poussée, une flèche cyan vient du côté d'où l'on pousse et touche le
bassin ; un éclat cyan marque le point d'impact. Les deux restent affichés un peu
après la poussée (0,15 s, soit 4 images à 25 images/s, serait trop bref à l'œil),
en s'estompant.

Ne fait qu'ajouter des marqueurs au rendu : la physique ne change pas. La force est
lue dans ``data.xfrc_applied`` du bassin, celle que la simulation applique
vraiment, quelle que soit la façon dont la poussée a été déclenchée.

Volontairement hors de ``poppy_humanoid_env.py``, comme ``floor_display.py`` :
l'environnement ne sait rien de la façon dont on le regarde.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

CYAN = np.array([0.0, 0.9, 1.0, 1.0])

_PELVIS_BODY_ID = 1
_PELVIS_RADIUS = 0.06       # m : la pointe s'arrête sur le bassin, pas en son centre
_METRES_PER_NEWTON = 0.04   # 10 N donnent une flèche de 40 cm
_MIN_LENGTH = 0.15          # m : une petite poussée reste visible
_SHAFT_RADIUS = 0.022       # m
_IMPACT_RADIUS = 0.05       # m


def _frame_pointing(direction: NDArray) -> NDArray:
    """Rotation dont l'axe z local suit ``direction`` (horizontale), aplatie ligne par ligne.

    Une flèche MuJoCo (``mjGEOM_ARROW``) part de ``pos`` le long de son axe z local.
    """
    z = direction / np.linalg.norm(direction)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]).flatten()


class PushMarker:
    """Suit la poussée en cours et la dessine à chaque image.

    Args:
        dt: Durée d'un pas de simulation entre deux images (s).
        hold: Durée d'affichage après la fin de la poussée (s).
    """

    def __init__(self, dt: float, hold: float = 0.5) -> None:
        self._hold_steps = max(1, int(round(hold / dt)))
        self._force = np.zeros(2)
        self._pushing = False
        self._left = 0

    def update(self, data: mujoco.MjData) -> None:
        """Lit la force appliquée au bassin ; à appeler une fois par pas."""
        force = np.asarray(data.xfrc_applied[_PELVIS_BODY_ID, 0:2], dtype=np.float64)
        self._pushing = bool(np.linalg.norm(force) > 1e-9)
        if self._pushing:
            self._force = force.copy()
            self._left = self._hold_steps
        elif self._left > 0:
            self._left -= 1

    @property
    def visible(self) -> bool:
        """Vrai pendant la poussée et le temps d'affichage qui suit."""
        return self._pushing or self._left > 0

    def markers(self, data: mujoco.MjData) -> list[dict[str, Any]]:
        """Les marqueurs à ajouter à l'image : flèche puis impact, ou rien.

        Returns:
            Des dictionnaires pour ``add_marker`` des viewers Gymnasium.
        """
        if not self.visible:
            return []
        magnitude = float(np.linalg.norm(self._force))
        direction = np.array([self._force[0], self._force[1], 0.0]) / magnitude
        # Pleine intensité pendant la poussée, puis s'estompe.
        fade = 1.0 if self._pushing else self._left / self._hold_steps
        pelvis = np.asarray(data.xpos[_PELVIS_BODY_ID], dtype=np.float64)
        impact = pelvis - direction * _PELVIS_RADIUS
        length = max(_MIN_LENGTH, _METRES_PER_NEWTON * magnitude)
        tail = impact - direction * length
        arrow = {
            "type": mujoco.mjtGeom.mjGEOM_ARROW,
            "pos": tail,
            "mat": _frame_pointing(direction),
            "size": np.array([_SHAFT_RADIUS, _SHAFT_RADIUS, length]),
            "rgba": CYAN * np.array([1.0, 1.0, 1.0, 0.25 + 0.75 * fade]),
        }
        # L'éclat grossit pendant la poussée puis se dissipe.
        radius = _IMPACT_RADIUS * (1.0 if self._pushing else 1.0 + 0.6 * (1.0 - fade))
        flash = {
            "type": mujoco.mjtGeom.mjGEOM_SPHERE,
            "pos": impact,
            "size": np.array([radius, radius, radius]),
            "rgba": CYAN * np.array([1.0, 1.0, 1.0, 0.55 * fade]),
        }
        return [arrow, flash]

    def draw(self, viewer: Any, data: mujoco.MjData) -> None:
        """Ajoute les marqueurs à la prochaine image de ``viewer``."""
        for marker in self.markers(data):
            viewer.add_marker(**marker)
