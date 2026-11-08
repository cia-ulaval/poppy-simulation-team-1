"""Génère la marche de référence de Poppy avec Placo (piste 3).

Disney (robot BDX, RSS 2024) et Open Duck Mini font apprendre à leur politique
une démarche *imitée* : un générateur de marche paramétrique (Placo, de Rhoban)
calcule une marche cinématiquement propre — modèle du pendule inversé linéaire
(LIPM), pied d'appui posé à plat, pied de vol sur une courbe douce — et la
récompense d'entraînement paie la politique pour la suivre. Ce script produit
cette référence pour Poppy, une fois pour toutes.

Placo n'est qu'une dépendance de **génération** : l'entraînement lit le fichier
produit (``assets/poppy_humanoid/reference_walk.npz``) avec numpy seul.

Ce que fait le script
---------------------
1. Construit une copie de l'URDF adaptée à Placo
   (``assets/poppy_humanoid/poppy_humanoid_placo.urdf``) : voir
   :func:`build_placo_urdf`. L'URDF et le MJCF d'origine ne sont pas touchés.
2. Planifie une marche droite à pas constants, la déroule avec le solveur
   cinématique de Placo, et en extrait **un cycle de deux pas** en régime établi.
3. Rééchantillonne ce cycle sur une grille de phase régulière, dans la
   convention de l'horloge de marche de l'environnement (voir plus bas), et
   l'écrit en ``.npz``.

La vérification physique (rejeu dans MuJoCo, couples par dynamique inverse,
vidéo) est dans ``scripts/check_reference_walk.py``, qui n'a pas besoin de Placo.

Convention de phase
-------------------
Celle de ``PoppyHumanoidEnv._gait_phase`` et du terme ``phase_contact`` :
pied droit à la phase, pied gauche décalé d'un demi-cycle, chaque pied au sol
pendant ``phase_stance_ratio`` de son propre cycle. Donc, pour une phase φ :

* φ = 0      le pied droit se pose (début du double appui) ;
* φ = 0,5 − r  le pied gauche décolle... avec r le double appui, etc.

Concrètement, avec un double appui de 5 % du cycle : double appui sur
[0 ; 0,05), appui droit seul sur [0,05 ; 0,5), double appui sur [0,5 ; 0,55),
appui gauche seul sur [0,55 ; 1).

Usage (dans le conteneur ``dev``, Placo installé à la volée)::

    docker compose --profile dev run --rm dev bash -c \\
        "pip install -r requirements/reference.txt && \\
         python scripts/generate_reference_walk.py"

Axes : l'avant de Poppy est son −y local et sa gauche son +x (voir
``CLAUDE.md``). Placo suppose x vers l'avant, y à gauche, z en haut : les repères
ajoutés à l'URDF portent cette convention-là, pas celle des liens de Poppy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import numpy as np
from numpy.typing import NDArray

_ROOT = Path(__file__).parent.parent
_ASSETS = _ROOT / "assets" / "poppy_humanoid"
_URDF_SOURCE = _ASSETS / "poppy_humanoid.urdf"
_URDF_PLACO = _ASSETS / "poppy_humanoid_placo.urdf"
_OUTPUT = _ASSETS / "reference_walk.npz"

# ── Ce que la copie de l'URDF change ─────────────────────────────────
# Centre de l'empreinte de la semelle, dans le repère du lien du pied
# (mesuré sur les maillages de collision r_foot_c / l_foot_c du MJCF) :
# la semelle est à 3,5 cm sous la cheville, le long du −y local ; le centre de
# l'empreinte est 2,9 cm devant la cheville (+z local) et 2,0 cm vers
# l'intérieur (±x local). Empreinte : 14,8 cm de long, 5,3 cm de large.
SOLE_RIGHT = (0.0197, -0.035, 0.0292)
SOLE_LEFT = (-0.0197, -0.035, 0.0292)
FOOT_LENGTH = 0.148   # m
FOOT_WIDTH = 0.053    # m

# Orientation des repères ajoutés, en roulis-tangage-lacet URDF.
# Pied : dans le repère du lien du pied (cheville à 0), x local = gauche du
# robot, y local = haut, z local = avant. Le repère Placo veut x = avant,
# y = gauche, z = haut : x_placo = z_lien, y_placo = x_lien, z_placo = y_lien.
FOOT_RPY = (-math.pi / 2, -math.pi / 2, 0.0)
# Tronc : le bassin a x = gauche, y = arrière, z = haut. Un lacet de −90° donne
# x = avant (−y du bassin), y = gauche (+x du bassin).
_TRUNK_RPY = (0.0, 0.0, -math.pi / 2)


def _rpy_matrix(rpy: tuple[float, float, float]) -> NDArray:
    """Matrice de rotation d'un triplet URDF ``(roulis, tangage, lacet)`` en rad.

    Convention URDF : ``R = Rz(lacet) · Ry(tangage) · Rx(roulis)``.

    Returns:
        Matrice 3 × 3.
    """
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rpy_from_matrix(rotation: NDArray) -> tuple[float, float, float]:
    """Triplet URDF ``(roulis, tangage, lacet)`` en rad d'une matrice 3 × 3.

    Inverse de :func:`_rpy_matrix`, y compris au blocage de cardan
    (tangage à ±90°), où le lacet est pris nul.
    """
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        yaw = 0.0
        sign = 1.0 if pitch > 0 else -1.0
        roll = math.atan2(sign * rotation[0, 1], rotation[1, 1])
    rpy = (roll, pitch, yaw)
    if not np.allclose(_rpy_matrix(rpy), rotation, atol=1e-9):
        raise ValueError(f"Conversion en roulis-tangage-lacet ratée : {rotation}")
    return rpy


def _add_fixed_frame(robot: ET.Element, name: str, parent: str,
                     xyz: tuple[float, float, float],
                     rpy: tuple[float, float, float]) -> None:
    """Ajoute un lien sans masse ``name``, fixé à ``parent``."""
    ET.SubElement(robot, "link", {"name": name})
    joint = ET.SubElement(robot, "joint", {"name": f"{name}_frame", "type": "fixed"})
    ET.SubElement(joint, "origin", {
        "xyz": " ".join(f"{v:.6g}" for v in xyz),
        "rpy": " ".join(f"{v:.17g}" for v in rpy),
    })
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": name})


def _fmt(values: Any, digits: int = 17) -> str:
    """Nombres séparés par des espaces, comme l'URDF les écrit."""
    return " ".join(f"{float(v):.{digits}g}" for v in values)


def build_placo_urdf(model: mujoco.MjModel, source: Path = _URDF_SOURCE) -> str:
    """Construit la copie de l'URDF de Poppy que ``placo.HumanoidRobot`` accepte.

    La cinématique est recopiée **du MJCF compilé** (``model``), pas de
    l'URDF : c'est le MJCF qui est simulé, et les deux ne coïncident pas
    partout. Pour chaque articulation, l'origine (position et orientation du
    corps enfant), l'axe et les butées viennent de ``model`` ; masses,
    inerties et maillages restent ceux de l'URDF, identiques au MJCF.

    Ce que cela change par rapport à l'URDF d'origine :

    1. **Bras** : le MJCF écrit ``euler="3.14159 0 ±1.5708"`` pour
       ``*_shoulder_motor``, que MuJoCo lit en angles intrinsèques, alors que
       l'URDF donne les mêmes nombres en roulis-tangage-lacet extrinsèques.
       Les bras du MJCF ne sont donc pas ceux de l'URDF.
    2. **Genou gauche retourné** si ``model`` a été compilé avec
       ``fix_left_knee_axis`` — c'est le cas ici : sans lui, le genou gauche ne
       peut pas fléchir. La référence ne vaut que pour un environnement qui
       l'active.
    3. **Butées de hip_y** à ±1,2 rad, celles du MJCF.
    4. **Repères ajoutés**, liens fixes sans masse : ``right_foot`` et
       ``left_foot`` au centre de l'empreinte de chaque semelle, ``trunk`` à
       l'origine du bassin. Tous orientés x = avant, y = gauche, z = haut,
       comme Placo l'attend — alors que les liens de Poppy ont l'avant en −y.

    Args:
        model: Modèle MuJoCo de l'environnement, genou gauche corrigé.
        source: URDF d'origine.

    Returns:
        Le contenu XML de la copie adaptée.
    """
    tree = ET.parse(source)
    robot = tree.getroot()
    for joint in robot.findall("joint"):
        name = joint.get("name")
        child = model.body(joint.find("child").get("link"))
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, child.quat)
        origin = joint.find("origin")
        origin.set("xyz", _fmt(child.pos))
        origin.set("rpy", _fmt(_rpy_from_matrix(rotation.reshape(3, 3))))
        mj_joint = model.joint(name)
        joint.find("axis").set("xyz", _fmt(mj_joint.axis, 6))
        limit = joint.find("limit")
        limit.set("lower", _fmt(mj_joint.range[:1], 12))
        limit.set("upper", _fmt(mj_joint.range[1:], 12))
    _add_fixed_frame(robot, "right_foot", "r_foot", SOLE_RIGHT, FOOT_RPY)
    _add_fixed_frame(robot, "left_foot", "l_foot", SOLE_LEFT, FOOT_RPY)
    _add_fixed_frame(robot, "trunk", "pelvis", (0.0, 0.0, 0.0), _TRUNK_RPY)
    ET.indent(tree, space="\t")
    header = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<!-- GÉNÉRÉ par scripts/generate_reference_walk.py : ne pas modifier à\n"
        "     la main. Copie de poppy_humanoid.urdf pour Placo, cinématique\n"
        "     recopiée du MJCF compilé avec fix_left_knee_axis ; repères\n"
        "     right_foot, left_foot, trunk (x avant, y gauche, z haut). -->\n"
    )
    return header + ET.tostring(robot, encoding="unicode") + "\n"


# ── La marche ────────────────────────────────────────────────────────

@dataclass
class WalkParams:
    """Réglages de la marche générée. Unités : m, s, rad.

    Les valeurs par défaut sont celles retenues après la vérification des
    couples (voir ``scripts/check_reference_walk.py``).

    Attributes:
        speed: Vitesse d'avance, m/s.
        period: Durée d'un cycle de deux pas, s. Doit valoir
            ``environment.gait_period`` de la configuration d'entraînement.
        double_support: Durée de chaque double appui, en fraction du cycle.
            0,05 donne un pied au sol 55 % de son cycle, comme
            ``reward.phase_stance_ratio: 0.55``.
        com_height: Hauteur du centre de masse au-dessus du sol, m. Debout
            jambes tendues, elle vaut ~0,476 m : la différence fléchit les
            genoux. Plus haut, le genou passe près de l'extension et doit
            accélérer davantage pour lever le pied (0,465 : genou à 4,4 N·m) ;
            plus bas, la cheville approche de sa butée.
        foot_height: Levée du pied de vol, m. 3 cm au lieu de 2 : +0,4 à
            +1,3 N·m au genou de vol.
        foot_rise_ratio: Part de la phase de vol passée à monter et descendre
            verticalement (Placo). Plus elle est courte, plus le pied quitte
            le sol en douceur : 0,3 au lieu de 0,1 ajoute ~0,5 N·m au genou.
        feet_spacing: Écart latéral entre les centres des deux semelles, m.
            0,094 est l'écart naturel, hanches droites : sans roulis de
            cheville, Poppy ne peut pas le changer en gardant le pied à plat.
        trunk_pitch: Tangage du bassin, rad (0 : comme debout).
        zmp_margin: Marge du ZMP à l'intérieur de la semelle, m.
        n_steps: Nombre de pas planifiés ; le cycle est pris au milieu.
        ik_dt: Pas de temps du solveur cinématique, s. Fixe aussi la grille de
            phase : ``period / ik_dt`` échantillons.
        ik_iterations: Résolutions du solveur par pas de temps.
        planning_refine: Pas de la grille de planification de Placo par
            double appui.
    """

    speed: float = 0.25
    period: float = 0.7
    double_support: float = 0.05
    com_height: float = 0.455
    foot_height: float = 0.02
    foot_rise_ratio: float = 0.1
    feet_spacing: float = 0.094
    trunk_pitch: float = 0.0
    zmp_margin: float = 0.015
    n_steps: int = 16
    ik_dt: float = 0.005
    ik_iterations: int = 3
    planning_refine: int = 5

    @property
    def step_length(self) -> float:
        """Avance d'un pas, m."""
        return self.speed * self.period / 2.0


_CLI_PARAMS = ("speed", "period", "double_support", "com_height", "foot_height",
               "foot_rise_ratio", "feet_spacing", "trunk_pitch", "zmp_margin")


def _placo_parameters(placo: Any, params: WalkParams) -> Any:
    """Paramètres ``placo.HumanoidParameters`` d'une marche droite à pas constants."""
    p = placo.HumanoidParameters()
    double_support = params.double_support * params.period
    single_support = params.period / 2.0 - double_support
    p.single_support_duration = single_support
    # Grille de planification : le ZMP n'est contraint dans la semelle qu'à ses
    # nœuds. Avec un double appui d'un seul pas de grille (35 ms), il sortait
    # de la semelle entre deux nœuds au début de chaque appui simple. Un double
    # appui dure ici exactement planning_refine pas de grille.
    p.single_support_timesteps = max(1, round(single_support / double_support)) * (
        params.planning_refine)
    p.double_support_ratio = double_support / single_support
    p.startend_double_support_ratio = 3.0
    # L'horizon couvre toute la marche : pas de replanification.
    total = (params.n_steps + 4) * params.period / 2.0
    p.planned_timesteps = int(math.ceil(total / p.dt())) + 10
    p.replan_timesteps = 10
    p.walk_com_height = params.com_height
    p.walk_foot_height = params.foot_height
    p.walk_trunk_pitch = params.trunk_pitch
    p.walk_foot_rise_ratio = params.foot_rise_ratio
    p.foot_length = FOOT_LENGTH
    p.foot_width = FOOT_WIDTH
    p.feet_spacing = params.feet_spacing
    p.zmp_margin = params.zmp_margin
    p.foot_zmp_target_x = 0.0
    p.foot_zmp_target_y = 0.0
    p.walk_max_dtheta = 1.0
    p.walk_max_dy = 0.05
    p.walk_max_dx_forward = max(0.12, params.step_length)
    p.walk_max_dx_backward = 0.05
    return p


def run_placo_walk(params: WalkParams, urdf: Path,
                   standing: dict[str, float]) -> dict[str, NDArray]:
    """Planifie et déroule la marche avec Placo.

    Le haut du corps (tronc, tête, bras) est tenu à la pose debout de
    l'environnement. Le **roulis du bassin est laissé libre** : sans roulis de
    cheville, c'est le seul moyen pour Poppy de déporter son centre de masse
    au-dessus du pied d'appui en gardant le pied à plat — il roule autour de
    la hanche d'appui, de ±6° environ, et le haut du corps roule avec lui.
    Roulis bloqué, le CoM manque sa trajectoire latérale de 1,3 cm et la
    marche est irréalisable.

    Essayé et abandonné : garder la poitrine droite (tâche d'orientation sur
    ``chest``, roulis du tronc ``abs_x`` / ``bust_x`` libéré). Sans rappel du
    tronc, le solveur dérive vers un bassin penché de 12° en permanence et un
    tronc tordu de 32° ; avec rappel, le bassin ne roule plus et le centre de
    pression reste hors de la semelle tout le cycle.

    Args:
        params: Réglages de la marche.
        urdf: Copie de l'URDF pour Placo (:func:`build_placo_urdf`).
        standing: Pose debout, angle (rad) par nom d'articulation, dans l'ordre
            des actionneurs du MJCF.

    Returns:
        Tableaux échantillonnés tous les ``ik_dt`` : ``t`` (s), ``q`` (rad,
        ``(T, 25)``, ordre de ``standing``), ``T_fbase`` (``(T, 4, 4)``, monde
        Placo : x avant), ``support`` (``(T, 2)``, droit, gauche), ``feet_z``
        (m, ``(T, 2)``), ``com_error`` (m, ``(T, 3)``).
    """
    import placo  # dépendance de génération seulement : requirements/reference.txt

    names = list(standing)
    robot = placo.HumanoidRobot(str(urdf), placo.Flags.ignore_collisions)
    p = _placo_parameters(placo, params)
    solver = placo.KinematicsSolver(robot)
    solver.enable_joint_limits(True)
    solver.enable_velocity_limits(False)
    solver.dt = params.ik_dt
    tasks = placo.WalkTasks()
    tasks.initialize_tasks(solver, robot)
    tasks.trunk_orientation_task.mask.set_axises("yz", "local")   # roulis libre
    legs = [n for n in names if n[2:].startswith(("hip", "knee", "ankle"))]
    upper_body = solver.add_joints_task()
    upper_body.set_joints({n: v for n, v in standing.items() if n not in legs})
    upper_body.configure("upper_body", "soft", 10.0)

    for name, value in standing.items():
        robot.set_joint(name, value)
    # Genoux pré-fléchis : partir jambes tendues met le solveur en singularité.
    robot.set_joint("r_knee_y", -0.5)
    robot.set_joint("l_knee_y", 0.5)
    robot.update_kinematics()
    tasks.reach_initial_pose(np.eye(4), p.feet_spacing, p.walk_com_height, p.walk_trunk_pitch)
    solver.enable_velocity_limits(True)

    planner = placo.FootstepsPlannerRepetitive(p)
    planner.configure(params.step_length, 0.0, 0.0, params.n_steps)
    footsteps = planner.plan(
        placo.HumanoidRobot_Side.left,
        placo.flatten_on_floor(robot.get_T_world_left()),
        placo.flatten_on_floor(robot.get_T_world_right()),
    )
    supports = placo.FootstepsPlanner.make_supports(
        footsteps, 0.0, True, p.has_double_support(), True)
    trajectory = placo.WalkPatternGenerator(robot, p).plan(supports, robot.com_world(), 0.0)

    record: dict[str, list] = {k: [] for k in
                               ("t", "q", "T_fbase", "support", "feet_z", "com_error")}
    for k in range(int(np.floor(trajectory.t_end / params.ik_dt))):
        t = k * params.ik_dt
        tasks.update_tasks_from_trajectory(trajectory, t)
        for _ in range(params.ik_iterations):
            robot.update_kinematics()
            solver.solve(True)
        robot.update_kinematics()
        # Les changements d'appui tombent pile sur la grille : un décalage
        # infime évite que l'arrondi flottant les place d'un côté ou de l'autre.
        t_support = t + 1e-7
        if trajectory.support_is_both(t_support):
            support = (1.0, 1.0)
        elif trajectory.support_side(t_support) == placo.HumanoidRobot_Side.left:
            support = (0.0, 1.0)
        else:
            support = (1.0, 0.0)
        record["t"].append(t)
        record["q"].append([robot.get_joint(n) for n in names])
        record["T_fbase"].append(robot.get_T_world_fbase().copy())
        record["support"].append(support)
        record["feet_z"].append([robot.get_T_world_right()[2, 3],
                                 robot.get_T_world_left()[2, 3]])
        record["com_error"].append(robot.com_world() - trajectory.get_p_world_CoM(t))
    return {k: np.asarray(v, dtype=np.float64) for k, v in record.items()}


def _right_touchdowns(support: NDArray) -> NDArray:
    """Indices où le pied droit se pose : son contact passe de 0 à 1."""
    right = support[:, 0] > 0.5
    return np.flatnonzero(right[1:] & ~right[:-1]) + 1


def extract_cycle(walk: dict[str, NDArray], params: WalkParams,
                  names: list[str]) -> dict[str, NDArray]:
    """Extrait un cycle en régime établi, de la pose du pied droit à la suivante.

    Le cycle est pris au milieu de la marche, loin du démarrage et de l'arrêt.
    L'écart entre ce cycle et le suivant mesure ce qu'il reste de transitoire.

    Returns:
        Les tableaux de :class:`src.environments.reference_motion.ReferenceMotion`,
        plus ``periodicity_error`` (rad, écart articulaire maximal entre deux
        cycles consécutifs) et ``com_error_max`` (m).
    """
    n = int(round(params.period / params.ik_dt))
    starts = _right_touchdowns(walk["support"])
    starts = starts[starts + 2 * n < len(walk["t"])]
    if len(starts) < 3:
        raise RuntimeError("Marche trop courte pour extraire un cycle établi")
    i0 = int(starts[len(starts) // 2])
    if starts[len(starts) // 2 + 1] - i0 != n:
        raise RuntimeError(
            f"Cycle de {(starts[len(starts) // 2 + 1] - i0) * params.ik_dt:.3f} s, "
            f"attendu {params.period} s")
    sl = slice(i0, i0 + n)
    q = walk["q"][sl]
    dt = params.period / n
    qvel = (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2.0 * dt)

    # Monde Placo (x avant) → monde de l'environnement (avant = −y) : lacet −90°.
    to_env = _rpy_matrix((0.0, 0.0, -math.pi / 2))
    positions = walk["T_fbase"][:, :3, 3] @ to_env.T
    base_pos = positions[sl].copy()
    base_pos[:, :2] -= base_pos[0, :2]
    base_quat = np.zeros((n, 4))
    for k, frame in enumerate(walk["T_fbase"][sl]):
        mujoco.mju_mat2Quat(base_quat[k], (to_env @ frame[:3, :3]).ravel())
    advance = positions[i0, 1] - positions[i0 + n, 1]   # l'avant est −y
    return {
        "joint_names": np.array(names),
        "qpos": q,
        "qvel": qvel,
        "contacts": walk["support"][sl],
        "base_pos": base_pos,
        "base_quat": base_quat,
        "feet_height": walk["feet_z"][sl],
        "period": np.float64(params.period),
        "speed": np.float64(advance / params.period),
        "periodicity_error": np.float64(np.abs(walk["q"][i0 + n:i0 + 2 * n] - q).max()),
        "com_error_max": np.float64(np.abs(walk["com_error"][sl]).max()),
    }


def save_reference(cycle: dict[str, NDArray], params: WalkParams, path: Path) -> None:
    """Écrit la référence en ``.npz`` compressé, tableaux flottants sur 32 bits.

    Les réglages sont joints en JSON (``params_json``) : le fichier dit
    comment le refaire.
    """
    arrays = {
        k: (v.astype(np.float32) if v.dtype.kind == "f" and v.ndim > 0 else v)
        for k, v in cycle.items()
    }
    arrays["params_json"] = np.array(json.dumps(asdict(params), sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _parse_args() -> argparse.Namespace:
    defaults = WalkParams()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    parser.add_argument("--urdf-only", action="store_true",
                        help="Écrit seulement la copie de l'URDF pour Placo.")
    for name in _CLI_PARAMS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float,
                            default=getattr(defaults, name))
    return parser.parse_args()


def main() -> None:
    from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

    args = _parse_args()
    env = PoppyHumanoidEnv(floor_noise=False, fix_left_knee_axis=True)
    _URDF_PLACO.write_text(build_placo_urdf(env.model), encoding="utf-8")
    print(f"URDF pour Placo : {_URDF_PLACO.relative_to(_ROOT)}")
    if args.urdf_only:
        return
    params = WalkParams(**{k: getattr(args, k) for k in _CLI_PARAMS})
    names = [env.model.joint(i).name for i in range(1, env.model.njnt)]
    standing = dict(zip(names, env.init_qpos[7:].tolist(), strict=True))
    walk = run_placo_walk(params, _URDF_PLACO, standing)
    cycle = extract_cycle(walk, params, names)
    save_reference(cycle, params, args.output)
    print(f"Référence : {args.output} ({args.output.stat().st_size / 1024:.1f} Ko)")
    print(f"  vitesse {float(cycle['speed']):.3f} m/s, pas {params.step_length * 100:.2f} cm, "
          f"cycle {params.period} s, {len(cycle['qpos'])} échantillons")
    print(f"  écart entre deux cycles {float(cycle['periodicity_error']):.1e} rad, "
          f"erreur de CoM {float(cycle['com_error_max']) * 1000:.2f} mm")
    low, high = env.model.jnt_range[1:, 0], env.model.jnt_range[1:, 1]
    margin = np.minimum(cycle["qpos"].min(0) - low, high - cycle["qpos"].max(0))
    worst = int(np.argmin(margin))
    print(f"  plus petite marge aux butées : {names[worst]} {np.degrees(margin[worst]):.1f}°")


if __name__ == "__main__":
    main()
