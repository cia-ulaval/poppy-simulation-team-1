"""Marche de référence, indexée par la phase de l'horloge de marche (piste 3).

Le fichier ``assets/poppy_humanoid/reference_walk.npz`` est produit par
``scripts/generate_reference_walk.py`` (avec Placo) et vérifié par
``scripts/check_reference_walk.py`` (avec MuJoCo). Ce module ne fait que le
lire : numpy seul, pas de Placo à l'entraînement.

Convention de phase : celle de ``PoppyHumanoidEnv._gait_phase``. φ = 0 au
moment où le pied droit se pose ; le pied gauche suit un demi-cycle plus tard.

Rien n'est lu à l'import : :func:`load_reference` lit le fichier quand on
l'appelle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_REFERENCE_PATH = (
    Path(__file__).parent.parent.parent / "assets" / "poppy_humanoid" / "reference_walk.npz"
)
# Même marche, générée sur le modèle aux bras orientés comme l'URDF
# (``environment.fix_arm_frames``), pose debout bras le long du corps.
ARMS_URDF_REFERENCE_PATH = DEFAULT_REFERENCE_PATH.with_name("reference_walk_bras_urdf.npz")


@dataclass(frozen=True)
class ReferenceMotion:
    """Un cycle de marche de deux pas, échantillonné régulièrement en phase.

    Attributes:
        joint_names: Noms des 25 articulations, dans l'ordre des actionneurs du
            MJCF (celui de ``qpos[7:]`` et de l'action).
        qpos: Angles articulaires, rad, forme ``(N, 25)``.
        qvel: Vitesses articulaires, rad/s, forme ``(N, 25)``.
        contacts: Pieds au sol, 1 ou 0, forme ``(N, 2)``, ordre (droit, gauche)
            comme les contacts de l'observation.
        base_pos: Position du bassin, m, forme ``(N, 3)``, dans un monde où le
            robot marche vers −y (son avant à cap nul) ; x et y relatifs au
            début du cycle, z au-dessus du sol.
        base_quat: Orientation du bassin (w, x, y, z), forme ``(N, 4)``, même
            monde.
        feet_height: Hauteur de la semelle au-dessus du sol, m, forme
            ``(N, 2)``, ordre (droit, gauche).
        period: Durée du cycle, s.
        speed: Vitesse d'avance moyenne, m/s.
        fix_arm_frames: Vrai si elle a été générée avec ``fix_arm_frames``.
    """

    joint_names: tuple[str, ...]
    qpos: NDArray
    qvel: NDArray
    contacts: NDArray
    base_pos: NDArray
    base_quat: NDArray
    feet_height: NDArray
    period: float
    speed: float
    fix_arm_frames: bool = False

    @property
    def n_samples(self) -> int:
        """Nombre d'échantillons N sur le cycle."""
        return int(self.qpos.shape[0])

    def _bracket(self, phase: float) -> tuple[int, int, float]:
        """Indices encadrant ``phase`` sur la grille périodique, et le poids."""
        x = (float(phase) % 1.0) * self.n_samples
        i0 = int(np.floor(x)) % self.n_samples
        return i0, (i0 + 1) % self.n_samples, x - np.floor(x)

    def joints_at(self, phase: float) -> tuple[NDArray, NDArray]:
        """Angles (rad) et vitesses (rad/s) de référence à ``phase``.

        Interpolation linéaire, périodique.

        Returns:
            ``(qpos, qvel)``, chacun de forme ``(25,)``.
        """
        i0, i1, w = self._bracket(phase)
        q = (1.0 - w) * self.qpos[i0] + w * self.qpos[i1]
        qd = (1.0 - w) * self.qvel[i0] + w * self.qvel[i1]
        return q, qd

    def base_at(self, phase: float) -> tuple[float, NDArray, NDArray]:
        """Pose du bassin de référence à ``phase``, interpolée.

        Returns:
            ``(hauteur, quaternion, verticale)`` : hauteur au-dessus du sol (m),
            orientation (w, x, y, z) normée, forme ``(4,)``, dans le monde où le
            robot marche vers −y, et verticale du monde exprimée dans le repère
            du bassin, forme ``(3,)`` — ce que mesure l'accéléromètre d'une IMU
            au repos, indépendant du cap.
        """
        i0, i1, w = self._bracket(phase)
        height = float((1.0 - w) * self.base_pos[i0, 2] + w * self.base_pos[i1, 2])
        quat = (1.0 - w) * self.base_quat[i0] + w * self.base_quat[i1]
        quat = quat / np.linalg.norm(quat)
        qw, qx, qy, qz = quat
        # Troisième ligne de la matrice de rotation : R^T · (0, 0, 1).
        up = np.array([2.0 * (qx * qz - qw * qy),
                       2.0 * (qy * qz + qw * qx),
                       1.0 - 2.0 * (qx * qx + qy * qy)])
        return height, quat, up

    def contacts_at(self, phase: float) -> NDArray:
        """Pieds au sol (1/0, ordre droit, gauche) à ``phase``, échantillon le plus proche."""
        i0, i1, w = self._bracket(phase)
        return self.contacts[i1 if w >= 0.5 else i0]


def load_reference(path: Path = DEFAULT_REFERENCE_PATH) -> ReferenceMotion:
    """Lit une marche de référence ``.npz``.

    Args:
        path: Fichier produit par ``scripts/generate_reference_walk.py``.

    Returns:
        La référence, en ``float64``.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si les tableaux n'ont pas les formes attendues.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        reference = ReferenceMotion(
            joint_names=tuple(str(n) for n in data["joint_names"]),
            qpos=data["qpos"].astype(np.float64),
            qvel=data["qvel"].astype(np.float64),
            contacts=data["contacts"].astype(np.float64),
            base_pos=data["base_pos"].astype(np.float64),
            base_quat=data["base_quat"].astype(np.float64),
            feet_height=data["feet_height"].astype(np.float64),
            period=float(data["period"]),
            speed=float(data["speed"]),
            fix_arm_frames=bool(data["fix_arm_frames"]) if "fix_arm_frames" in data else False,
        )
    n = reference.n_samples
    expected = {
        "qpos": (n, 25), "qvel": (n, 25), "contacts": (n, 2),
        "base_pos": (n, 3), "base_quat": (n, 4), "feet_height": (n, 2),
    }
    for name, shape in expected.items():
        if getattr(reference, name).shape != shape:
            raise ValueError(
                f"{path} : {name} a la forme {getattr(reference, name).shape}, "
                f"attendu {shape}")
    if len(reference.joint_names) != 25:
        raise ValueError(f"{path} : {len(reference.joint_names)} articulations, attendu 25")
    return reference
