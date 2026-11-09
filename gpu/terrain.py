"""Sol à relief du palier 2 sur MuJoCo Warp : un terrain par environnement, en tuiles.

Sur CPU (``src/environments/terrain.py``), chaque environnement a son champ
de hauteur de 20 m × 20 m, retiré à chaque ``reset()``. MuJoCo Warp partage
``hfield_data`` entre tous les mondes : un seul relief pour tous. En
revanche ``geom_pos`` est propre à chaque monde. D'où ce montage :

* **un** champ de hauteur, fait de ``T × T`` tuiles de ``n × n`` cases, une
  tuile par environnement (``n = 401`` cases de 5 cm pour 20 m) ;
* dans le monde ``i``, le geom du terrain est **décalé** pour que la tuile
  ``i`` soit centrée sur l'origine. Chaque robot part donc de ``(0, 0)``
  sur son propre relief, comme sur CPU, et les coordonnées ``x, y`` du
  bassin sont celles du CPU ;
* au reset d'un environnement, seule sa tuile est retirée, avec le même
  générateur que le CPU (mêmes tirages, même ordre, mêmes formules), à la
  difficulté courante du curriculum.

Les tuiles sont jointives : une case de 5 cm sépare le bord d'une tuile du
bord de la suivante. Un robot ne quitte pas sa tuile en un épisode (10 s à
0,25 m/s, 2,5 m, pour 10 m de demi-côté). Le relief étant borné à ±25 cm, le
CPU, qui n'a rien au-delà de son terrain, ne fait pas mieux.

Mémoire : ``4 × n²`` octets par environnement, 643 Ko pour 20 m à 5 cm, soit
165 Mo pour 256 environnements et 2,6 Go pour 4096, en un seul tableau sur le
GPU (et autant dans le ``MjModel`` CPU que compile mjlab).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from src.config.settings import TerrainConfig
from src.environments.terrain import _BASE_THICKNESS, _HEIGHT_SPAN, TERRAIN_NAME

_FLOOR_GEOM_NAME = "floor"
_N_WAVES = 2
_N_BUMPS = 80
_N_STEPS = 60
_CHUNK = 16  # tuiles retirées à la fois : borne la mémoire temporaire


def cells_per_side(config: TerrainConfig) -> int:
    """Nombre de cases par côté d'une tuile, comme ``add_terrain``."""
    return int(round(config.size / config.resolution)) + 1


@dataclass(frozen=True)
class TileLayout:
    """Disposition des tuiles dans le champ de hauteur partagé.

    Attributes:
        n: Cases par côté d'une tuile.
        tiles: Tuiles par côté du champ (``T``).
        spacing: Pas de la grille, m.
        half: Demi-côté d'une tuile, m (10 m pour 20 m).
    """

    n: int
    tiles: int
    spacing: float
    half: float

    @classmethod
    def for_envs(cls, config: TerrainConfig, num_envs: int) -> TileLayout:
        n = cells_per_side(config)
        return cls(n=n, tiles=max(1, math.ceil(math.sqrt(num_envs))),
                   spacing=config.size / (n - 1), half=0.5 * config.size)

    @property
    def cells(self) -> int:
        """Cases par côté du champ partagé."""
        return self.tiles * self.n

    @property
    def field_half(self) -> float:
        """Demi-côté du champ partagé, m."""
        return 0.5 * (self.cells - 1) * self.spacing

    def tile_origin(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Ligne et colonne de la première case de la tuile de chaque environnement."""
        return (env_ids // self.tiles) * self.n, (env_ids % self.tiles) * self.n

    def geom_offset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Position ``(x, y)`` du geom du terrain qui centre la tuile sur l'origine.

        La case ``(ligne, colonne)`` du champ est en ``−S + colonne × pas``
        dans le repère du geom ; la première case de la tuile doit tomber en
        ``−half`` dans le monde.
        """
        row0, col0 = self.tile_origin(env_ids)
        shift = self.field_half - self.half
        return torch.stack([shift - col0 * self.spacing, shift - row0 * self.spacing], dim=-1)


def add_tiled_terrain(spec: mujoco.MjSpec, config: TerrainConfig, layout: TileLayout) -> None:
    """``add_terrain`` en tuiles : même geom, mêmes propriétés de contact, champ ``T`` fois plus grand.

    Le champ est créé sans données (hauteurs nulles) ; :class:`TiledTerrain`
    les écrit sur le GPU.
    """
    half = layout.field_half
    spec.add_hfield(
        name=TERRAIN_NAME,
        nrow=layout.cells,
        ncol=layout.cells,
        size=[half, half, 2.0 * _HEIGHT_SPAN, _BASE_THICKNESS],
        # MjSpec exige des données ; un tableau numpy évite une liste Python
        # de dizaines de millions de flottants.
        userdata=np.zeros(layout.cells * layout.cells, dtype=np.float32),
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
    floor.contype = 0
    floor.conaffinity = 0
    floor.group = 4


# ── Générateur : celui de src/environments/terrain.py, en lot ───────────────


@dataclass
class TerrainDraws:
    """Les tirages de ``generate_heights`` pour ``B`` tuiles, dans leur ordre.

    Toutes les grandeurs sont celles du CPU avant multiplication par la
    difficulté ; formes ``(B, …)``.
    """

    wavelength: torch.Tensor   # (B, 2), m
    direction: torch.Tensor    # (B, 2), rad
    phase: torch.Tensor        # (B, 2), rad
    bump_xy: torch.Tensor      # (B, 80, 2), m
    bump_sigma: torch.Tensor   # (B, 80), m
    bump_amp: torch.Tensor     # (B, 80), dans [−1, 1]
    step_xy: torch.Tensor      # (B, 60, 2), m
    step_wh: torch.Tensor      # (B, 60, 2), m
    step_amp: torch.Tensor     # (B, 60), dans [−1, 1]
    rough: torch.Tensor        # (B, n, n), dans [−1, 1]

    @classmethod
    def sample(cls, batch: int, n: int, half: float, device: str) -> TerrainDraws:
        """Tire les paramètres de ``batch`` tuiles, avec le générateur de torch."""
        def u(low: float, high: float, *shape: int) -> torch.Tensor:
            return torch.empty((batch, *shape), device=device).uniform_(low, high)

        return cls(
            wavelength=u(3.0, 8.0, _N_WAVES), direction=u(-math.pi, math.pi, _N_WAVES),
            phase=u(-math.pi, math.pi, _N_WAVES),
            bump_xy=u(-half, half, _N_BUMPS, 2), bump_sigma=u(0.15, 0.5, _N_BUMPS),
            bump_amp=u(-1.0, 1.0, _N_BUMPS),
            step_xy=u(-half, half, _N_STEPS, 2), step_wh=u(0.3, 1.2, _N_STEPS, 2),
            step_amp=u(-1.0, 1.0, _N_STEPS),
            rough=u(-1.0, 1.0, n, n),
        )

    def take(self, index: slice) -> TerrainDraws:
        return TerrainDraws(**{k: v[index] for k, v in self.__dict__.items()})


def build_heights(
    draws: TerrainDraws, config: TerrainConfig, difficulty: float, n: int
) -> torch.Tensor:
    """``generate_heights`` en lot, à partir de tirages donnés.

    Calcule dans le type des tirages : float32 à l'entraînement, float64
    pour comparer au CPU case par case (en float32, une case tombant à 1e-7
    du bord d'une marche peut basculer d'un côté ou de l'autre).

    Returns:
        Hauteurs ``(B, n, n)`` en m, ligne = y, colonne = x, bornées à ±25 cm.
    """
    batch = draws.rough.shape[0]
    device, dtype = draws.rough.device, draws.rough.dtype
    d = float(np.clip(difficulty, 0.0, 1.0))
    heights = torch.zeros((batch, n, n), device=device, dtype=dtype)
    if d == 0.0:
        return heights
    half = 0.5 * config.size
    axis = torch.linspace(-half, half, n, device=device, dtype=torch.float64).to(dtype)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    x, y = x[None], y[None]

    for k in range(_N_WAVES):
        theta = draws.direction[:, k, None, None]
        along = x * torch.cos(theta) + y * torch.sin(theta)
        heights += 0.5 * d * config.slope_amplitude * torch.sin(
            2.0 * math.pi * along / draws.wavelength[:, k, None, None]
            + draws.phase[:, k, None, None])

    for k in range(_N_BUMPS):
        cx = draws.bump_xy[:, k, 0, None, None]
        cy = draws.bump_xy[:, k, 1, None, None]
        sigma = draws.bump_sigma[:, k, None, None]
        amplitude = draws.bump_amp[:, k, None, None] * d * config.bump_height
        heights += amplitude * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))

    for k in range(_N_STEPS):
        cx = draws.step_xy[:, k, 0, None, None]
        cy = draws.step_xy[:, k, 1, None, None]
        wx = draws.step_wh[:, k, 0, None, None]
        wy = draws.step_wh[:, k, 1, None, None]
        inside = ((x - cx).abs() < wx / 2) & ((y - cy).abs() < wy / 2)
        heights += inside * (draws.step_amp[:, k, None, None] * d * config.step_height)

    heights += draws.rough * d * config.roughness

    radius = torch.hypot(x, y)
    blend = torch.clamp((radius - config.flat_radius) / 0.4, 0.0, 1.0)
    heights *= blend * blend * (3.0 - 2.0 * blend)
    return torch.clamp(heights, -_HEIGHT_SPAN, _HEIGHT_SPAN)


# ── Le terrain vivant, sur le GPU ───────────────────────────────────────────


class TiledTerrain:
    """Les tuiles du champ partagé : retirage au reset, hauteur sous un point.

    Args:
        config: Réglages du terrain (YAML).
        layout: Disposition des tuiles.
        hfield_data: Vue torch de ``model.hfield_data`` (partagé par les mondes).
        hfield_adr: Adresse du champ dans ``hfield_data``.
    """

    def __init__(
        self, config: TerrainConfig, layout: TileLayout, hfield_data: torch.Tensor, hfield_adr: int
    ) -> None:
        self.config = config
        self.layout = layout
        size = layout.cells * layout.cells
        self._grid = hfield_data[hfield_adr:hfield_adr + size].view(layout.cells, layout.cells)
        self.difficulty = float(config.difficulty)
        self._grid.fill_(0.5)  # hauteur nulle partout

    def tiles(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Vue indexée ``(B, n, n)`` des données normalisées des tuiles (copie)."""
        n = self.layout.n
        row0, col0 = self.layout.tile_origin(env_ids)
        r = row0[:, None, None] + torch.arange(n, device=env_ids.device)[None, :, None]
        c = col0[:, None, None] + torch.arange(n, device=env_ids.device)[None, None, :]
        return self._grid[r, c]

    def write(self, env_ids: torch.Tensor, heights: torch.Tensor) -> None:
        """Écrit des hauteurs ``(B, n, n)`` en m dans les tuiles, normalisées comme sur CPU."""
        n = self.layout.n
        row0, col0 = self.layout.tile_origin(env_ids)
        r = row0[:, None, None] + torch.arange(n, device=env_ids.device)[None, :, None]
        c = col0[:, None, None] + torch.arange(n, device=env_ids.device)[None, None, :]
        self._grid[r, c] = (heights + _HEIGHT_SPAN) / (2.0 * _HEIGHT_SPAN)

    def regenerate(self, env_ids: torch.Tensor) -> None:
        """Retire le relief des tuiles ``env_ids`` à la difficulté courante."""
        n = self.layout.n
        for start in range(0, len(env_ids), _CHUNK):
            ids = env_ids[start:start + _CHUNK]
            if self.difficulty <= 0.0:
                heights = torch.zeros((len(ids), n, n), device=ids.device)
            else:
                draws = TerrainDraws.sample(len(ids), n, self.layout.half, str(ids.device))
                heights = build_heights(draws, self.config, self.difficulty, n)
            self.write(ids, heights)

    def height_at(self, env_ids: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """``TerrainGrid.height_at`` en lot : case la plus proche, 0 hors de la tuile.

        Args:
            env_ids: ``(B,)`` environnements.
            xy: ``(B, 2)`` points, m, dans le repère du monde.

        Returns:
            ``(B,)`` hauteurs du sol, m.
        """
        lay = self.layout
        scale = (lay.n - 1) / self.config.size
        col = torch.round((xy[:, 0].double() + lay.half) * scale).long()
        row = torch.round((xy[:, 1].double() + lay.half) * scale).long()
        inside = (row >= 0) & (row < lay.n) & (col >= 0) & (col < lay.n)
        row0, col0 = lay.tile_origin(env_ids)
        r = row0 + row.clamp(0, lay.n - 1)
        c = col0 + col.clamp(0, lay.n - 1)
        height = self._grid[r, c] * (2.0 * _HEIGHT_SPAN) - _HEIGHT_SPAN
        return torch.where(inside, height, 0.0)
