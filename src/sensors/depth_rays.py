"""Lancer de rayons horizontaux dans MuJoCo, depuis la tête du robot.

Mesure les distances aux obstacles directement dans le simulateur, sans passer
par une image. C'est la seule approche de perception du dossier qui ne dépende
d'aucune caméra : une caméra stéréo remplacerait ``depth_server`` et
``depth_client``, pas ce module.

En pause avec le reste de la vision, et **écrit pour Humanoid-v5, pas pour
Poppy** — voir le corps ``torso`` ci-dessous. Voir README.md de ce dossier.
"""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

# Décalage de la « tête » par rapport au repère du corps de référence, en
# mètres, dans ce repère.
_DEFAULT_HEAD_OFFSET = np.array([0.0, 0.0, 0.19])

# Les distances sont repliées en 8 secteurs de 8 rayons, dont on garde le
# minimum : c'est l'obstacle le plus proche de chaque secteur.
_N_SECTORS = 8
_RAYS_PER_SECTOR = 8


def cast_horizontal_rays(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_rays: int = _N_SECTORS * _RAYS_PER_SECTOR,
    fov: float = np.pi / 2,
    max_distance: float = 10.0,
    head_offset: NDArray | None = None,
    body_name: str = "torso",
) -> NDArray:
    """Lance ``n_rays`` rayons horizontaux et renvoie la distance par secteur.

    Args:
        model: Modèle MuJoCo.
        data: État courant de la simulation.
        n_rays: Nombre de rayons, réparti uniformément dans le champ. Doit
            valoir 64 : le repliement en secteurs est figé.
        fov: Champ de vision total, en radians, centré sur l'avant du corps.
        max_distance: Portée maximale, en mètres. Un rayon qui ne touche rien
            renvoie cette valeur.
        head_offset: Position de la tête dans le repère du corps de référence,
            en mètres. ``None`` prend (0, 0, 0.19).
        body_name: Corps servant de repère. Le défaut ``"torso"`` est celui
            d'Humanoid-v5 ; **le modèle Poppy n'a pas de corps de ce nom**, ses
            corps du haut sont ``chest``, ``bust_motors``, ``neck`` et
            ``head``. Passer explicitement le bon nom pour Poppy.

    Returns:
        Une distance par secteur, forme ``(8,)``, en mètres : la plus proche
        de chaque groupe de 8 rayons.

    Raises:
        ValueError: Si ``body_name`` n'existe pas dans le modèle, ou si
            ``n_rays`` ne vaut pas 64.
    """
    if n_rays != _N_SECTORS * _RAYS_PER_SECTOR:
        raise ValueError(
            f"n_rays doit valoir {_N_SECTORS * _RAYS_PER_SECTOR} : le "
            f"repliement en {_N_SECTORS} secteurs est figé. Reçu {n_rays}."
        )

    if head_offset is None:
        head_offset = _DEFAULT_HEAD_OFFSET

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        disponibles = [model.body(i).name for i in range(model.nbody)]
        raise ValueError(
            f"Aucun corps nommé {body_name!r} dans ce modèle. "
            f"Corps disponibles : {', '.join(disponibles)}"
        )

    body_pos = data.xpos[body_id].copy()
    body_mat = data.xmat[body_id].reshape(3, 3).copy()

    head_pos = body_pos + body_mat @ head_offset

    vec = np.zeros((n_rays, 3), dtype=np.float64)
    angles = np.linspace(-fov / 2, fov / 2, n_rays)
    for i, angle in enumerate(angles):
        local_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
        world_dir = body_mat @ local_dir
        vec[i] = world_dir / np.linalg.norm(world_dir)

    geomgroup = None
    flg_static = 1
    bodyexclude = body_id

    geomid = np.zeros(n_rays, dtype=np.int32)
    distances = np.zeros(n_rays, dtype=np.float64)
    normal = np.zeros((n_rays, 3), dtype=np.float64)

    mujoco.mj_multiRay(
        model,
        data,
        head_pos,
        vec.flatten(),
        geomgroup,
        flg_static,
        bodyexclude,
        geomid,
        distances,
        normal.flatten(),
        n_rays,
        max_distance,
    )

    # mj_multiRay renvoie -1 quand le rayon ne touche rien.
    distances[distances < 0] = max_distance

    return distances.reshape(_N_SECTORS, _RAYS_PER_SECTOR).min(axis=1)
