"""Poussées visibles dans les vidéos : un coup de flèche sur le bassin, et son onde de choc.

Pendant une poussée :
- une flèche cyan lumineuse arrive du côté d'où l'on pousse et vient frapper le
  bassin (elle glisse jusqu'au contact pendant les premières images) ;
- la force est écrite au-dessus du point d'impact, par exemple « 10 N » ;
- un éclat marque le point d'impact, et un anneau s'en écarte comme une onde de
  choc.
Le tout reste affiché un peu après la poussée, en s'estompant : 0,15 s de
poussée, soit 4 images à 25 images/s, serait trop bref à l'œil.

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
_STRIKE_DISTANCE = 0.15     # m : d'où la pointe part avant de frapper
_STRIKE_STEPS = 5           # images pour arriver au contact
_FLASH_RADIUS = 0.035       # m
_RING_RADII = (0.05, 0.16)  # m : l'onde de choc s'élargit de l'un à l'autre
_RING_THICKNESS = 0.004     # m : demi-épaisseur de l'anneau (un disque très plat)
_EMISSION = 0.7             # lumineux : la couleur ne dépend presque plus de l'éclairage
_LABEL_HEIGHT = 0.12        # m : la force est écrite au-dessus de l'impact


def _frame_pointing(direction: NDArray) -> NDArray:
    """Rotation dont l'axe z local suit ``direction`` (horizontale), aplatie ligne par ligne.

    Une flèche MuJoCo (``mjGEOM_ARROW``) part de ``pos`` le long de son axe z local ;
    un cylindre a son axe le long de z.
    """
    z = direction / np.linalg.norm(direction)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]).flatten()


def _enable_extras(viewer: Any) -> None:
    """Permet aux marqueurs d'un viewer Gymnasium d'avoir ``emission`` et ``label``.

    Gymnasium ne transmet que la forme, la position, l'orientation et la couleur
    d'un marqueur. On complète la géométrie qu'il vient d'ajouter à la scène.
    """
    if getattr(viewer, "_push_display_extras", False):
        return
    add = viewer._add_marker_to_scene

    def add_with_extras(marker: dict) -> None:
        add(marker)
        geom = viewer.scn.geoms[viewer.scn.ngeom - 1]
        geom.emission = float(marker.get("emission", 0.0))
        geom.label = str(marker.get("label", ""))

    viewer._add_marker_to_scene = add_with_extras
    viewer._push_display_extras = True


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
        self._since_start = 0
        self._left = 0

    def update(self, data: mujoco.MjData) -> None:
        """Lit la force appliquée au bassin ; à appeler une fois par pas."""
        force = np.asarray(data.xfrc_applied[_PELVIS_BODY_ID, 0:2], dtype=np.float64)
        pushing = bool(np.linalg.norm(force) > 1e-9)
        if pushing:
            new_push = not self._pushing and self._left == 0
            self._since_start = 0 if new_push else self._since_start + 1
            self._force = force.copy()
            self._left = self._hold_steps
        elif self._left > 0:
            self._since_start += 1
            self._left -= 1
        self._pushing = pushing

    @property
    def visible(self) -> bool:
        """Vrai pendant la poussée et le temps d'affichage qui suit."""
        return self._pushing or self._left > 0

    def markers(self, data: mujoco.MjData) -> list[dict[str, Any]]:
        """Les marqueurs à ajouter à l'image : flèche, éclat, onde de choc ; ou rien.

        Returns:
            Des dictionnaires pour ``add_marker`` des viewers Gymnasium, avec en
            plus ``emission`` et ``label`` (voir ``draw``).
        """
        if not self.visible:
            return []
        magnitude = float(np.linalg.norm(self._force))
        direction = np.array([self._force[0], self._force[1], 0.0]) / magnitude
        frame = _frame_pointing(direction)
        # 1 pendant la poussée, puis décroît jusqu'à 0.
        fade = 1.0 if self._pushing else self._left / self._hold_steps
        # 0 au début de la poussée, 1 à la fin de l'affichage.
        age = min(1.0, self._since_start / (self._hold_steps + _STRIKE_STEPS))
        pelvis = np.asarray(data.xpos[_PELVIS_BODY_ID], dtype=np.float64)
        impact = pelvis - direction * _PELVIS_RADIUS

        # La flèche part de plus loin et vient frapper le bassin.
        approach = max(0.0, 1.0 - self._since_start / _STRIKE_STEPS)
        tip = impact - direction * _STRIKE_DISTANCE * approach
        length = max(_MIN_LENGTH, _METRES_PER_NEWTON * magnitude)
        markers = [{
            "type": mujoco.mjtGeom.mjGEOM_ARROW,
            "pos": tip - direction * length,
            "mat": frame,
            "size": np.array([_SHAFT_RADIUS, _SHAFT_RADIUS, length]),
            "rgba": CYAN * np.array([1.0, 1.0, 1.0, 0.3 + 0.7 * fade]),
            "emission": _EMISSION,
        }, {
            # La force, écrite au-dessus du point d'impact : à la base de la
            # flèche, elle sortait de l'image quand la poussée vient de loin.
            "type": mujoco.mjtGeom.mjGEOM_SPHERE,
            "pos": impact - direction * 0.08 + np.array([0.0, 0.0, _LABEL_HEIGHT]),
            "size": np.full(3, 0.001),
            "rgba": np.zeros(4),
            "label": f"{magnitude:.0f} N",
        }]
        if approach == 0.0:
            # Éclat au contact, tant que la flèche appuie.
            if self._pushing:
                markers.append({
                    "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                    "pos": impact,
                    "size": np.full(3, _FLASH_RADIUS),
                    "rgba": np.array([0.85, 1.0, 1.0, 0.9]),
                    "emission": 1.0,
                })
            # Onde de choc : un anneau face à la flèche, qui s'élargit et s'efface.
            radius = _RING_RADII[0] + (_RING_RADII[1] - _RING_RADII[0]) * age
            markers.append({
                "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                "pos": impact,
                "mat": frame,
                "size": np.array([radius, _RING_THICKNESS, 0.0]),
                "rgba": CYAN * np.array([1.0, 1.0, 1.0, 0.5 * (1.0 - age)]),
                "emission": _EMISSION,
            })
        return markers

    def draw(self, viewer: Any, data: mujoco.MjData) -> None:
        """Ajoute les marqueurs à la prochaine image de ``viewer``."""
        _enable_extras(viewer)
        for marker in self.markers(data):
            viewer.add_marker(**marker)
