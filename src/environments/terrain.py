"""Sol à relief aléatoire pour le palier 2 (``spec/recompense-et-paliers.md``).

Le terrain est un champ de hauteur MuJoCo (``hfield``) ajouté **par le code** au
modèle chargé depuis le MJCF : ``assets/`` n'est pas modifié, et sans terrain
le modèle reste exactement celui du fichier. Le plan ``floor`` est gardé, pour
que la randomisation de la friction continue de le trouver, mais il ne touche
plus le robot et n'est plus dessiné.

Les hauteurs sont régénérées à chaque ``reset()`` : bosses et creux, petites
marches, ondulations longues (pentes douces) et rugosité. Toutes sont
proportionnelles à ``difficulty`` : à 0 le terrain est parfaitement plat, ce
qui garde le banc d'essai commun. Une zone plate entoure le point de départ,
pour que le robot ne naisse pas dans une bosse.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.config.settings import TerrainConfig

TERRAIN_NAME = "terrain"
_FLOOR_GEOM_NAME = "floor"

# Le champ de hauteur couvre [-_HEIGHT_SPAN, +_HEIGHT_SPAN] autour de z = 0 :
# il faut de la marge sous zéro pour creuser des creux.
_HEIGHT_SPAN = 0.25
_BASE_THICKNESS = 0.05


@dataclass
class TerrainGrid:
    """Hauteurs courantes du terrain, et de quoi les écrire dans le modèle.

    Attributes:
        config: Réglages du terrain.
        hfield_id: Indice du champ de hauteur dans le modèle.
        n: Nombre de cases par côté.
        heights: Hauteurs en mètres, ``(n, n)``, ligne = y, colonne = x.
        difficulty: Difficulté appliquée à la prochaine régénération.
    """

    config: TerrainConfig
    hfield_id: int
    n: int
    heights: NDArray
    difficulty: float

    @property
    def half(self) -> float:
        return 0.5 * self.config.size

    def height_at(self, x: float, y: float) -> float:
        """Hauteur du terrain sous le point ``(x, y)`` (case la plus proche)."""
        col = int(round((x + self.half) / self.config.size * (self.n - 1)))
        row = int(round((y + self.half) / self.config.size * (self.n - 1)))
        if not (0 <= row < self.n and 0 <= col < self.n):
            return 0.0
        return float(self.heights[row, col])

    def bind(self, model: mujoco.MjModel) -> None:
        """Retrouve le champ de hauteur dans le modèle compilé."""
        self.hfield_id = model.hfield(TERRAIN_NAME).id

    def regenerate(self, model: mujoco.MjModel, rng: np.random.Generator) -> None:
        """Tire un nouveau terrain et l'écrit dans ``model.hfield_data``."""
        self.heights = generate_heights(self.config, self.difficulty, self.n, rng)
        normalised = (self.heights + _HEIGHT_SPAN) / (2.0 * _HEIGHT_SPAN)
        adr = model.hfield_adr[self.hfield_id]
        model.hfield_data[adr: adr + self.n * self.n] = normalised.ravel()


def generate_heights(
    config: TerrainConfig,
    difficulty: float,
    n: int,
    rng: np.random.Generator,
) -> NDArray:
    """Tire les hauteurs d'un terrain.

    Args:
        config: Amplitudes et taille du terrain.
        difficulty: Facteur appliqué à toutes les amplitudes, dans ``[0, 1]``.
        n: Nombre de cases par côté.
        rng: Générateur aléatoire de l'environnement.

    Returns:
        Les hauteurs en mètres, ``(n, n)``, bornées à ±25 cm.
    """
    heights = np.zeros((n, n))
    d = float(np.clip(difficulty, 0.0, 1.0))
    if d == 0.0:
        return heights

    half = 0.5 * config.size
    axis = np.linspace(-half, half, n)
    x, y = np.meshgrid(axis, axis)

    # Pentes douces : deux ondulations de 3 à 8 m de longueur d'onde.
    for _ in range(2):
        wavelength = rng.uniform(3.0, 8.0)
        direction = rng.uniform(-np.pi, np.pi)
        phase = rng.uniform(-np.pi, np.pi)
        along = x * np.cos(direction) + y * np.sin(direction)
        heights += 0.5 * d * config.slope_amplitude * np.sin(
            2.0 * np.pi * along / wavelength + phase
        )

    # Bosses et creux : gaussiennes de 15 à 50 cm de rayon.
    for _ in range(80):
        cx, cy = rng.uniform(-half, half, size=2)
        sigma = rng.uniform(0.15, 0.5)
        amplitude = rng.uniform(-1.0, 1.0) * d * config.bump_height
        heights += amplitude * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2)
        )

    # Petites marches : plateaux rectangulaires à bords francs.
    for _ in range(60):
        cx, cy = rng.uniform(-half, half, size=2)
        wx, wy = rng.uniform(0.3, 1.2, size=2)
        step = rng.uniform(-1.0, 1.0) * d * config.step_height
        inside = (np.abs(x - cx) < wx / 2) & (np.abs(y - cy) < wy / 2)
        heights[inside] += step

    # Rugosité case par case.
    heights += rng.uniform(-1.0, 1.0, size=heights.shape) * d * config.roughness

    # Zone plate au départ, raccordée en douceur sur 40 cm.
    radius = np.hypot(x, y)
    blend = np.clip((radius - config.flat_radius) / 0.4, 0.0, 1.0)
    heights *= blend * blend * (3.0 - 2.0 * blend)

    return np.clip(heights, -_HEIGHT_SPAN, _HEIGHT_SPAN)


def add_terrain(spec: mujoco.MjSpec, config: TerrainConfig) -> TerrainGrid:
    """Ajoute le champ de hauteur à un modèle en cours de construction.

    Args:
        spec: Le MJCF Poppy chargé en ``MjSpec``, pas encore compilé.
        config: Réglages du terrain.

    Returns:
        La grille qui pilotera les hauteurs. Son ``hfield_id`` n'est connu
        qu'après compilation : appeler :meth:`TerrainGrid.bind`.
    """
    n = int(round(config.size / config.resolution)) + 1
    spec.add_hfield(
        name=TERRAIN_NAME,
        nrow=n,
        ncol=n,
        size=[0.5 * config.size, 0.5 * config.size, 2.0 * _HEIGHT_SPAN, _BASE_THICKNESS],
        userdata=[0.5] * (n * n),
    )
    floor = spec.geom(_FLOOR_GEOM_NAME)
    spec.worldbody.add_geom(
        name=TERRAIN_NAME,
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=TERRAIN_NAME,
        pos=[0.0, 0.0, -_HEIGHT_SPAN],
        material=floor.material,
        friction=floor.friction,
        condim=floor.condim,
        contype=floor.contype,
        conaffinity=floor.conaffinity,
    )
    # Le plan reste dans le modèle, mais ne touche plus rien et ne se voit plus.
    floor.contype = 0
    floor.conaffinity = 0
    floor.group = 4
    return TerrainGrid(
        config=config,
        hfield_id=-1,
        n=n,
        heights=np.zeros((n, n)),
        difficulty=config.difficulty,
    )
