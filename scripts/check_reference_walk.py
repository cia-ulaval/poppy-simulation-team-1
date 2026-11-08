"""Vérifie dans MuJoCo que la marche de référence est physiquement plausible.

Placo produit une marche *cinématiquement* propre à partir d'un modèle très
simplifié (pendule inversé linéaire). Rien ne dit qu'elle est réalisable par le
vrai corps multi-articulé de Poppy, ni par ses moteurs. Ce script le mesure,
sans Placo, sur le modèle même de l'entraînement (MJCF, genou gauche corrigé) :

1. **Rejeu cinématique** : les pieds traversent-ils le sol ? le pied d'appui
   reste-t-il à plat et immobile ? une articulation touche-t-elle sa butée ?
2. **Dynamique inverse** (``mujoco.mj_inverse``, contacts désactivés) : la
   force généralisée qu'exige le mouvement. Ses six composantes sur la base
   flottante doivent venir du sol : on les attribue au(x) pied(s) d'appui, ce
   qui donne le torseur de chaque pied, donc son centre de pression (doit
   rester sous la semelle), son frottement requis, et par différence **les
   couples moteurs**, comparés aux limites ``ctrlrange`` du MJCF.
3. **Rejeu dynamique en boucle ouverte** (facultatif) : la référence envoyée
   telle quelle au régulateur PD de l'environnement. Sans politique pour
   corriger, le robot finit par tomber ; combien de temps il tient dit l'écart
   entre la référence et ce que le PD sait suivre.

Usage::

    python scripts/check_reference_walk.py
    python scripts/check_reference_walk.py --video logs/reference/rejeu.mp4 --camera cote
    python scripts/check_reference_walk.py --open-loop-video logs/reference/boucle_ouverte.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np
from numpy.typing import NDArray

from scripts.generate_reference_walk import (
    FOOT_LENGTH,
    FOOT_RPY,
    FOOT_WIDTH,
    SOLE_LEFT,
    SOLE_RIGHT,
    _rpy_matrix,
)
from src.environments.poppy_humanoid_env import CAMERAS, PoppyHumanoidEnv, set_camera
from src.environments.reference_motion import (
    DEFAULT_REFERENCE_PATH,
    ReferenceMotion,
    load_reference,
)

# Pieds dans l'ordre des contacts de l'observation : (droit, gauche).
_FEET = ("r_foot", "l_foot")
_SOLES = (SOLE_RIGHT, SOLE_LEFT)
_SOLE_ROTATION = _rpy_matrix(FOOT_RPY)   # axes de la semelle dans le repère du pied
_LEG_PREFIXES = ("r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle")
# Poids des moments face aux forces quand le double appui partage le torseur :
# un moment de 1 N·m « coûte » comme une force de 20 N, soit un bras de levier
# de 5 cm, l'ordre de la demi-semelle.
_MOMENT_SCALE = 20.0


def trajectory_qpos(reference: ReferenceMotion, n_cycles: int) -> NDArray:
    """État complet ``qpos`` de ``n_cycles`` cycles enchaînés, forme ``(n_cycles·N, 32)``.

    La base avance d'un cycle à l'autre ; les articulations se répètent.
    """
    advance = np.array([0.0, -reference.speed * reference.period, 0.0])
    rows = []
    for cycle in range(n_cycles):
        for k in range(reference.n_samples):
            rows.append(np.concatenate([
                reference.base_pos[k] + cycle * advance,
                reference.base_quat[k] / np.linalg.norm(reference.base_quat[k]),
                reference.qpos[k],
            ]))
    return np.asarray(rows)


def finite_differences(model: mujoco.MjModel, qpos: NDArray,
                       dt: float) -> tuple[NDArray, NDArray]:
    """Vitesses et accélérations généralisées par différences centrées.

    Returns:
        ``(qvel, qacc)``, forme ``(T, nv)`` ; la première et la dernière ligne
        sont nulles (pas de voisin).
    """
    n = len(qpos)
    forward = np.zeros((n - 1, model.nv))
    for k in range(n - 1):
        mujoco.mj_differentiatePos(model, forward[k], dt, qpos[k], qpos[k + 1])
    qvel = np.zeros((n, model.nv))
    qacc = np.zeros((n, model.nv))
    qvel[1:-1] = 0.5 * (forward[:-1] + forward[1:])
    qacc[1:-1] = (forward[1:] - forward[:-1]) / dt
    return qvel, qacc


def _foot_vertices(model: mujoco.MjModel) -> list[NDArray]:
    """Sommets des maillages de collision de chaque pied, dans le repère du pied."""
    vertices = []
    for name in _FEET:
        body = model.body(name).id
        parts = []
        for g in range(model.ngeom):
            if (model.geom_bodyid[g] != body or model.geom_contype[g] == 0
                    or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mesh = model.geom_dataid[g]
            start = model.mesh_vertadr[mesh]
            v = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
            parts.append(v @ rotation.reshape(3, 3).T + model.geom_pos[g])
        vertices.append(np.concatenate(parts))
    return vertices


def kinematic_check(env: PoppyHumanoidEnv, reference: ReferenceMotion) -> dict[str, float]:
    """Rejeu cinématique d'un cycle : sol, appui à plat, glissement, butées.

    Returns:
        Mesures en m (et rad pour les butées) : ``penetration_max``,
        ``stance_height_max`` (écart du pied d'appui au sol), ``stance_slip_max``
        (déplacement horizontal du pied d'appui sur le cycle),
        ``swing_height_max``, ``other_contacts`` (géométries hors pieds sous
        le sol), et par articulation la marge aux butées.
    """
    model, data = env.model, env.data
    vertices = _foot_vertices(model)
    # Deux cycles : l'appui gauche chevauche φ = 0, il n'est entier que sur deux.
    qpos = trajectory_qpos(reference, 2)
    contacts = np.tile(reference.contacts, (2, 1)) > 0.5
    foot_ids = [model.body(n).id for n in _FEET]
    lowest = np.zeros((len(qpos), 2))
    sole_xy = np.zeros((len(qpos), 2, 2))
    other_contacts = 0
    for k, q in enumerate(qpos):
        data.qpos[:] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for i, body in enumerate(foot_ids):
            rotation = data.xmat[body].reshape(3, 3)
            world = vertices[i] @ rotation.T + data.xpos[body]
            lowest[k, i] = world[:, 2].min()
            sole_xy[k, i] = (data.xpos[body] + rotation @ np.asarray(_SOLES[i]))[:2]
        for c in range(data.ncon):
            contact = data.contact[c]
            bodies = {model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]}
            if contact.dist < -1e-3 and not bodies & set(foot_ids):
                other_contacts += 1
    slip = 0.0
    for i in range(2):
        # Chaque appui ininterrompu : le pied ne doit pas bouger de sa pose.
        edges = np.flatnonzero(np.diff(contacts[:, i].astype(int))) + 1
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            if contacts[start, i]:
                xy = sole_xy[start:stop, i]
                slip = max(slip, float(np.linalg.norm(xy - xy[0], axis=1).max()))
    low, high = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    margin = np.minimum(reference.qpos.min(0) - low, high - reference.qpos.max(0))
    return {
        "penetration_max": float(max(0.0, -lowest.min())),
        "stance_height_max": float(np.abs(lowest[contacts]).max()),
        "stance_slip_max": slip,
        "swing_height_max": float(lowest[~contacts].max()) if (~contacts).any() else 0.0,
        "other_contacts": float(other_contacts),
        "joint_margin": margin,
        "joint_range": high - low,
    }


def _stance_jacobians(model: mujoco.MjModel, data: mujoco.MjData,
                      in_contact: NDArray) -> list[tuple[int, NDArray, NDArray, NDArray]]:
    """Pour chaque pied d'appui : (pied, jacobienne 6 × nv au centre de la
    semelle, position de ce centre, axes de la semelle dans le monde)."""
    stance = []
    for i, name in enumerate(_FEET):
        if not in_contact[i]:
            continue
        body = model.body(name).id
        rotation = data.xmat[body].reshape(3, 3)
        point = data.xpos[body] + rotation @ np.asarray(_SOLES[i])
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, point, body)
        stance.append((i, np.vstack([jacp, jacr]), point, rotation @ _SOLE_ROTATION))
    return stance


def inverse_dynamics(env: PoppyHumanoidEnv, reference: ReferenceMotion) -> dict[str, NDArray]:
    """Couples moteurs et torseurs au sol qu'exige la référence.

    Trois cycles sont enchaînés pour dériver sans effet de bord ; seul celui
    du milieu est mesuré.

    Returns:
        ``torque`` (N·m, ``(N, 25)``), ``armature_torque`` (N·m, sa part due
        à l'armature du MJCF, inertie de rotor ramenée à l'articulation),
        ``cop`` (m, ``(N, 2, 2)``, position avant / gauche du centre de
        pression dans chaque semelle, NaN hors appui), ``normal_force``
        (N, ``(N, 2)``), ``friction`` (rapport tangentiel / normal requis,
        ``(N, 2)``).
    """
    model, data = env.model, env.data
    n = reference.n_samples
    dt = reference.period / n
    qpos = trajectory_qpos(reference, 3)
    qvel, qacc = finite_differences(model, qpos, dt)
    saved_flags = model.opt.disableflags
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    torque = np.zeros((n, model.nu))
    armature = model.dof_armature[6:] * qacc[n:2 * n, 6:]
    cop = np.full((n, 2, 2), np.nan)
    normal = np.zeros((n, 2))
    friction = np.zeros((n, 2))
    scale = np.diag([1.0, 1.0, 1.0, _MOMENT_SCALE, _MOMENT_SCALE, _MOMENT_SCALE])
    try:
        for k in range(n):
            j = n + k
            data.qpos[:] = qpos[j]
            data.qvel[:] = qvel[j]
            data.qacc[:] = qacc[j]
            mujoco.mj_inverse(model, data)
            required = data.qfrc_inverse.copy()
            stance = _stance_jacobians(model, data, reference.contacts[k] > 0.5)
            # Torseurs (force, moment) des pieds d'appui, au centre des semelles :
            # leurs six premières composantes généralisées doivent égaler celles
            # que réclame la base flottante. Plus petite norme pondérée.
            base = np.hstack([jac[:, :6].T @ np.linalg.inv(scale) for _, jac, _, _ in stance])
            wrenches = np.linalg.lstsq(base, required[:6], rcond=None)[0]
            external = np.zeros(model.nv)
            for s, (i, jac, _, axes) in enumerate(stance):
                wrench = np.linalg.inv(scale) @ wrenches[6 * s:6 * s + 6]
                external += jac.T @ wrench
                force, moment = axes.T @ wrench[:3], axes.T @ wrench[3:]
                normal[k, i] = force[2]
                if force[2] > 1e-6:
                    cop[k, i] = (-moment[1] / force[2], moment[0] / force[2])
                    friction[k, i] = float(np.hypot(force[0], force[1]) / force[2])
            torque[k] = (required - external)[6:]
    finally:
        model.opt.disableflags = saved_flags
    return {"torque": torque, "armature_torque": armature, "cop": cop,
            "normal_force": normal, "friction": friction}


def open_loop(env: PoppyHumanoidEnv, reference: ReferenceMotion, seconds: float,
              video: Path | None, camera: str) -> float:
    """Envoie la référence telle quelle au PD de l'environnement.

    L'état initial est celui de la référence à φ = 0 ; l'action vise l'angle de
    référence à la phase courante.

    Returns:
        Durée tenue avant la chute, s (``seconds`` si le robot ne tombe pas).
    """
    env.reset(seed=0)
    data = env.data
    data.qpos[:] = trajectory_qpos(reference, 1)[0]
    data.qvel[:] = 0.0
    data.qvel[6:] = reference.qvel[0]
    mujoco.mj_forward(env.model, data)
    frames = []
    steps = int(round(seconds / env.dt))
    for step in range(steps):
        phase = ((step + 1) * env.dt / reference.period) % 1.0
        target, _ = reference.joints_at(phase)
        offset = target - env._joint_init
        action = np.where(offset >= 0, offset / env._range_up, offset / env._range_down)
        _, _, terminated, _, _ = env.step(action)
        if video is not None:
            frames.append(env.render())
        if terminated:
            break
    if video is not None:
        _write_video(frames, video, env)
    return (step + 1) * env.dt


def replay_video(env: PoppyHumanoidEnv, reference: ReferenceMotion, n_cycles: int,
                 path: Path) -> None:
    """Rejeu cinématique, au temps réel de l'environnement (une image par 10 ms)."""
    qpos = trajectory_qpos(reference, n_cycles)
    stride = max(1, int(round(env.dt / (reference.period / reference.n_samples))))
    frames = []
    for q in qpos[::stride]:
        env.data.qpos[:] = q
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        frames.append(env.render())
    _write_video(frames, path, env)


def _write_video(frames: list[NDArray], path: Path, env: PoppyHumanoidEnv) -> None:
    import imageio.v2 as imageio  # rendu vidéo seulement

    path.parent.mkdir(parents=True, exist_ok=True)
    fps = int(round(1.0 / env.dt))
    imageio.mimsave(str(path), frames, fps=fps)
    print(f"Vidéo écrite : {path} ({len(frames)} images, {fps} fps)")


def report(env: PoppyHumanoidEnv, reference: ReferenceMotion,
           kinematics: dict, dynamics: dict) -> None:
    """Affiche le bilan."""
    model = env.model
    names = [model.joint(i).name for i in range(1, model.njnt)]
    print(f"Référence : {reference.n_samples} échantillons, cycle {reference.period} s, "
          f"{reference.speed:.3f} m/s")
    print("-" * 66)
    print("Rejeu cinématique")
    print(f"  pied sous le sol (max)          {kinematics['penetration_max'] * 1000:6.2f} mm")
    print(f"  pied d'appui hors du sol (max)  {kinematics['stance_height_max'] * 1000:6.2f} mm")
    print(f"  glissement du pied d'appui      {kinematics['stance_slip_max'] * 1000:6.2f} mm")
    print(f"  levée du pied de vol (max)      {kinematics['swing_height_max'] * 100:6.2f} cm")
    print(f"  autres corps sous le sol        {kinematics['other_contacts']:6.0f}")
    print("-" * 66)
    torque = dynamics["torque"]
    limit = model.actuator_ctrlrange[:, 1]
    print(f"{'articulation':<14}{'|τ| max':>9}{'RMS':>8}{'limite':>8}{'max/lim':>9}"
          f"{'marge butée':>13}")
    for j, name in enumerate(names):
        if not name.startswith(_LEG_PREFIXES) and np.abs(torque[:, j]).max() < 0.5 * limit[j]:
            continue
        peak = np.abs(torque[:, j]).max()
        rms = np.sqrt(np.mean(torque[:, j] ** 2))
        margin = np.degrees(kinematics["joint_margin"][j])
        flag = "  ← DÉPASSE" if peak > limit[j] else ""
        print(f"{name:<14}{peak:9.2f}{rms:8.2f}{limit[j]:8.1f}{peak / limit[j]:8.0%}"
              f"{margin:11.1f}°{flag}")
    others = [n for j, n in enumerate(names) if not n.startswith(_LEG_PREFIXES)
              and np.abs(torque[:, j]).max() < 0.5 * limit[j]]
    print(f"  (haut du corps, tous sous 50 % de leur limite : {len(others)} articulations)")
    print("-" * 66)
    cop = dynamics["cop"]
    single = (reference.contacts > 0.5).sum(axis=1) == 1
    for label, mask in (("appui simple", single), ("double appui", ~single)):
        c = cop[mask]
        fwd = np.nanmax(np.abs(c[..., 0])) / (FOOT_LENGTH / 2)
        side = np.nanmax(np.abs(c[..., 1])) / (FOOT_WIDTH / 2)
        print(f"Centre de pression, {label:<13} avant-arrière {fwd:5.0%} de la demi-semelle, "
              f"latéral {side:5.0%}")
    stance = reference.contacts > 0.5
    print(f"Force normale minimale en appui    {dynamics['normal_force'][stance].min():6.2f} N "
          f"(poids {model.body_mass.sum() * 9.81:.1f} N)")
    print(f"Frottement requis (max)            {dynamics['friction'][stance].max():6.2f}")
    kp_error = np.abs(torque).max(0) / 8.0
    worst = int(np.argmax(kp_error))
    print(f"Erreur de suivi qu'imposerait le PD (kp = 8 N·m/rad) : "
          f"{np.degrees(kp_error[worst]):.0f}° sur {names[worst]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--video", type=Path, default=None,
                        help="Rejeu cinématique en .mp4")
    parser.add_argument("--cycles", type=int, default=4,
                        help="Cycles rejoués dans la vidéo (défaut: 4)")
    parser.add_argument("--camera", choices=sorted(CAMERAS), default="cote")
    parser.add_argument("--open-loop", type=float, default=5.0,
                        help="Durée du rejeu dynamique en boucle ouverte, s (0 : aucun)")
    parser.add_argument("--open-loop-video", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reference = load_reference(args.reference)
    render = args.video is not None or args.open_loop_video is not None
    env = PoppyHumanoidEnv(floor_noise=False, fix_left_knee_axis=True,
                           render_mode="rgb_array" if render else None)
    set_camera(env, args.camera)
    kinematics = kinematic_check(env, reference)
    dynamics = inverse_dynamics(env, reference)
    report(env, reference, kinematics, dynamics)
    if args.video is not None:
        replay_video(env, reference, args.cycles, args.video)
    if args.open_loop > 0:
        held = open_loop(env, reference, args.open_loop, args.open_loop_video, args.camera)
        print(f"Boucle ouverte sur le PD : tient {held:.2f} s sur {args.open_loop:.1f} s")
    env.close()


if __name__ == "__main__":
    main()
