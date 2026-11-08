"""Symétrie gauche ↔ droite de Poppy, pour l'augmentation miroir.

Proposition 2 de ``spec/recompense-et-paliers.md``, contre la boiterie : la
politique doit répondre à la situation vue dans un miroir par l'action vue
dans un miroir. Aucune dimension n'est ajoutée à l'observation.

Le miroir est le plan sagittal du robot. La gauche de Poppy est son **+x
local** : le miroir est donc la réflexion x → −x, en repère monde comme en
repère du bassin (au départ, les deux sont alignés à un lacet près, et la
réflexion d'un monde plat reste un monde plat).

Les tables ne sont pas écrites à la main : elles sont **calculées** sur le
modèle. Chaque articulation ``r_…`` est appariée à ``l_…`` ; le signe vient de
la comparaison de leurs axes de rotation, en position debout. Une
articulation centrale (tronc, tête) est appariée à elle-même. Le test
``tests/test_mirror.py`` vérifie le résultat sur la dynamique : un état et
une action miroirs donnent l'état miroir.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

# Réflexion x → −x. Un vecteur (vitesse) devient M v ; un pseudo-vecteur (axe
# de rotation, vitesse angulaire) devient det(M) M ω = −M ω.
_REFLECT = np.array([-1.0, 1.0, 1.0])
_REFLECT_PSEUDO = np.array([1.0, -1.0, -1.0])

_N_ROOT_QPOS = 7
_N_ROOT_QVEL = 6


@dataclass(frozen=True)
class MirrorSpec:
    """Tables de la symétrie, prêtes à appliquer à des lots.

    Attributes:
        joint_perm: Pour chaque articulation, l'indice de sa partenaire.
        joint_sign: Le signe à appliquer à l'angle de la partenaire.
        obs_perm: Permutation des 63 dimensions de l'observation.
        obs_sign: Signe de chaque dimension de l'observation, après
            permutation.
        joint_init: Pose debout (cible d'une action nulle).
        range_up: Course au-dessus de la pose debout (action = +1).
        range_down: Course en dessous (action = −1).
    """

    joint_perm: NDArray
    joint_sign: NDArray
    obs_perm: NDArray
    obs_sign: NDArray
    joint_init: NDArray
    range_up: NDArray
    range_down: NDArray

    def mirror_obs(self, obs: NDArray) -> NDArray:
        """Observation vue dans le miroir. Accepte un lot ``(..., 63)``."""
        return obs[..., self.obs_perm] * self.obs_sign

    def mirror_action(self, action: NDArray) -> NDArray:
        """Action vue dans le miroir, exactement, en passant par les cibles.

        L'action est une position cible relative à la pose debout, avec une
        course différente au-dessus et en dessous. Le miroir porte sur les
        angles cibles, puis l'action est recalculée — sans quoi une
        articulation dont les deux courses diffèrent serait mal reflétée.
        """
        target = self.joint_init + np.where(
            action >= 0, action * self.range_up, action * self.range_down
        )
        mirrored = target[..., self.joint_perm] * self.joint_sign
        delta = mirrored - self.joint_init
        return np.where(delta >= 0, delta / self.range_up, delta / self.range_down)

    def mirror_qpos(self, qpos: NDArray) -> NDArray:
        """État complet ``qpos`` (avec x, y) vu dans le miroir."""
        out = qpos.copy()
        out[0] = -qpos[0]
        w, qx, qy, qz = qpos[3:7]
        out[3:7] = (w, qx, -qy, -qz)
        out[7:] = qpos[7:][self.joint_perm] * self.joint_sign
        return out

    def mirror_qvel(self, qvel: NDArray) -> NDArray:
        """Vitesses ``qvel`` vues dans le miroir."""
        out = qvel.copy()
        out[0:3] = qvel[0:3] * _REFLECT
        out[3:6] = qvel[3:6] * _REFLECT_PSEUDO
        out[6:] = qvel[6:][self.joint_perm] * self.joint_sign
        return out


def _partner(name: str) -> str:
    if name.startswith("r_"):
        return "l_" + name[2:]
    if name.startswith("l_"):
        return "r_" + name[2:]
    return name


def build_mirror_spec(env) -> MirrorSpec:
    """Calcule les tables miroir d'un ``PoppyHumanoidEnv``.

    Args:
        env: L'environnement, dont on lit le modèle, la pose debout et les
            courses de l'action.

    Returns:
        Les tables.

    Raises:
        ValueError: Si une articulation n'a pas de partenaire, ou si l'axe
            d'une paire n'est pas le reflet de l'autre (MJCF asymétrique).
    """
    model = env.model
    names = [model.joint(i).name for i in range(1, model.njnt)]
    n = len(names)

    standing = mujoco.MjData(model)
    standing.qpos[:] = env.init_qpos
    mujoco.mj_kinematics(model, standing)
    axes = standing.xaxis[1:]  # axes des articulations en repère monde

    perm = np.empty(n, dtype=int)
    sign = np.empty(n)
    for i, name in enumerate(names):
        partner = _partner(name)
        if partner not in names:
            raise ValueError(f"Articulation sans partenaire miroir : {name}")
        j = names.index(partner)
        reflected = axes[i] * _REFLECT_PSEUDO
        cosine = float(reflected @ axes[j])
        if abs(cosine) < 0.95:
            raise ValueError(
                f"Axes de {name} et {partner} non symétriques (cos = {cosine:.2f})"
            )
        perm[i] = j
        sign[i] = np.sign(cosine)

    # Les butées doivent aussi se refléter, sinon le miroir demande à une
    # articulation des angles qu'elle ne peut pas atteindre. C'est le cas du
    # genou gauche du MJCF tant que fix_left_knee_axis est faux.
    low, high = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    for i, name in enumerate(names):
        j = perm[i]
        mirrored = np.sort(sign[i] * np.array([low[j], high[j]]))
        if np.max(np.abs(mirrored - [low[i], high[i]])) > 0.3:
            raise ValueError(
                f"Butées de {name} non symétriques de celles de {names[j]} : "
                f"[{low[i]:.2f}, {high[i]:.2f}] contre [{mirrored[0]:.2f}, "
                f"{mirrored[1]:.2f}]. Pour les genoux, activer "
                f"environment.fix_left_knee_axis."
            )

    # Observation : z (1), quaternion (4), angles (25), vitesse linéaire (3),
    # vitesse angulaire du bassin (3), vitesses articulaires (25), contacts (2).
    obs_perm = np.concatenate([
        [0], [1, 2, 3, 4], 5 + perm,
        [30, 31, 32], [33, 34, 35], 36 + perm,
        [62, 61],
    ])
    obs_sign = np.concatenate([
        [1.0], [1.0, 1.0, -1.0, -1.0], sign,
        _REFLECT, _REFLECT_PSEUDO, sign,
        [1.0, 1.0],
    ])
    # Horloge de marche : le miroir d'un instant du cycle est le même instant
    # pour l'autre jambe, soit un demi-cycle plus loin. sin et cos changent
    # de signe.
    if getattr(env, "_gait_period", 0.0) > 0.0:
        obs_perm = np.concatenate([obs_perm, [63, 64]])
        obs_sign = np.concatenate([obs_sign, [-1.0, -1.0]])
    return MirrorSpec(
        joint_perm=perm,
        joint_sign=sign,
        obs_perm=obs_perm,
        obs_sign=obs_sign,
        joint_init=env._joint_init.copy(),
        range_up=env._range_up.copy(),
        range_down=env._range_down.copy(),
    )
