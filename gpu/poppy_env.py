"""Poppy Humanoid sur mjlab : la même MDP que ``PoppyHumanoidEnv``, en lots sur le GPU.

Rien n'est réinventé ici. Les constantes (pose debout, courses, gains PD,
butées douces, repère de la poitrine) sont **lues** sur une
``PoppyHumanoidEnv`` construite avec la même configuration, et le pas de
temps suit l'ordre exact de ``PoppyHumanoidEnv.step`` :

1. poussée aléatoire éventuelle sur le bassin (couple, défaut historique) ;
2. action bornée à l'espace d'action annoncé, comme le fait SB3 ;
3. cible ``pose debout + action × course``, couple PD calculé **une fois**
   sur l'état de début de pas, puis tenu pendant les ``frame_skip``
   sous-pas ;
4. ``rne_postconstraint`` après le dernier sous-pas, comme Gymnasium, pour
   les forces de contact des pieds ;
5. récompense, terminaison, puis observation.

Les grandeurs dérivées (``xpos``, ``xmat``, ``cfrc_ext``) lues par la
récompense ont un sous-pas de retard sur ``qpos``, exactement comme dans
Gymnasium, qui ne refait pas ``mj_forward`` après ``mj_step``. Les
contacts observés sont ceux de la récompense, et valent 0 juste après un
reset (Gymnasium ne les calcule pas dans ``reset``).

Axes : l'avant de Poppy est son −y local, sa gauche son +x
(``tests/test_heading.py``). La vitesse est projetée sur le cap du bassin.

Formes : ``N`` environnements, 25 articulations ; angles en rad, vitesses
en m/s et rad/s, forces en N, couples en N·m, temps en s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers import (
    ActionTerm,
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.event_manager import requires_model_fields
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from src.config.settings import PoppyEnvironmentConfig
from src.environments.poppy_humanoid_env import (
    _CONTACT_THRESHOLD,
    _KD,
    _KP,
    _LEFT_KNEE,
    _LOCAL_FORWARD,
    _LOCAL_LEFT,
    _MODEL_PATH,
    _N_JOINTS,
    _PELVIS_BODY_ID,
    PoppyHumanoidEnv,
)

ROBOT = "robot"
"""Nom de l'entité dans la scène mjlab ; préfixe ``robot/`` de ses éléments."""

_OBS_GROUP = "policy"

# Bornes de la randomisation d'origine (poppy_humanoid_env.py), écrites en dur
# là-bas aussi : bruit de l'état initial, lacet initial, masses, poussées.
_RESET_NOISE = 0.01
_YAW_RANGE = 0.26
_MASS_SCALE = (0.85, 1.15)
_PUSH_PROBABILITY = 0.02
_PUSH_RANGE = (1.0, 5.0)


def reference_env(config: PoppyEnvironmentConfig) -> PoppyHumanoidEnv:
    """Construit la ``PoppyHumanoidEnv`` CPU décrite par une configuration.

    Mêmes arguments que ``env_factory.make_poppy_env`` ; c'est elle qui fait
    foi pour toutes les constantes du portage.

    Args:
        config: Configuration d'environnement, issue d'un YAML de ``configs/``.

    Returns:
        L'environnement CPU, non vectorisé, sans ``TimeLimit``.
    """
    dr = config.domain_randomization
    return PoppyHumanoidEnv(
        floor_noise=dr.enabled,
        friction_range=dr.friction_range,
        restitution_range=dr.restitution_range,
        healthy_z_range=config.healthy_z_range,
        terminate_when_unhealthy=config.terminate_when_unhealthy,
        frame_skip=config.frame_skip,
        reward=config.reward,
        normalize_action_space=config.normalize_action_space,
        restitution_in_solref=dr.restitution_in_solref,
        push_as_force=dr.push_as_force,
        terrain=config.terrain,
        pushes=config.pushes,
        fix_left_knee_axis=config.fix_left_knee_axis,
        gait_period=config.gait_period,
    )


def check_supported(config: PoppyEnvironmentConfig) -> None:
    """Refuse les options que ce portage ne reproduit pas encore.

    Mieux vaut échouer au lancement qu'entraîner silencieusement une autre
    physique : le sol accidenté (palier 2) et les deux corrections de
    ``docs/TECHNIQUE.md`` §8 (restitution dans ``solref``, poussée en force)
    ne sont pas portés.

    Raises:
        NotImplementedError: Si la configuration en demande une.
    """
    dr = config.domain_randomization
    missing = [
        name for name, used in (
            ("terrain.enabled", config.terrain.enabled),
            ("domain_randomization.restitution_in_solref", dr.restitution_in_solref),
            ("domain_randomization.push_as_force", dr.push_as_force),
        ) if used
    ]
    if missing:
        raise NotImplementedError(
            "Non porté sur GPU : " + ", ".join(missing)
            + ". Entraîner cette configuration avec scripts/train_poppy.py."
        )


def _poppy_spec(fix_left_knee_axis: bool) -> mujoco.MjSpec:
    """Charge le MJCF Poppy, genou gauche corrigé comme dans ``PoppyHumanoidEnv``."""
    spec = mujoco.MjSpec.from_file(str(_MODEL_PATH))
    if fix_left_knee_axis:
        knee = spec.joint(_LEFT_KNEE)
        knee.axis = -np.asarray(knee.axis)
    return spec


@dataclass
class MjcfOptions(MujocoCfg):
    """Options du solveur recopiées du modèle CPU, champ par champ.

    ``MjSpec.attach`` ne transmet pas la balise ``<option>`` du MJCF à la
    scène mjlab, et ``MujocoCfg`` n'accepte que les intégrateurs Euler et
    implicitfast. Or Poppy est en **RK4** : sans cette recopie, la scène
    compilée tournerait en Euler, avec les valeurs par défaut de mjlab
    (``ccd_iterations`` à 50 au lieu de 35, entre autres).

    Attributes:
        source: Les options du modèle CPU, telles que les rend
            :func:`option_fields` (types simples, sérialisables).
    """

    source: dict | None = None

    def apply(self, model: mujoco.MjModel) -> None:
        """Écrit dans ``model.opt`` toutes les options de ``source``."""
        if self.source is None:
            raise ValueError("MjcfOptions.source : options du modèle CPU attendues.")
        for name, value in self.source.items():
            current = getattr(model.opt, name)
            if isinstance(current, np.ndarray):
                current[...] = value
            else:
                setattr(model.opt, name, type(current)(value))


# ── État partagé entre les termes ───────────────────────────────────────────


@dataclass
class PoppyState:
    """Constantes et tampons par environnement, partagés par les termes.

    Les termes mjlab sont des objets séparés ; ce que ``PoppyHumanoidEnv``
    garde dans ``self`` (action précédente, filtres, temps de vol…) vit ici,
    une ligne par environnement.
    """

    # Constantes, lues sur la PoppyHumanoidEnv de référence.
    dt: float
    gait_period: float
    floor_noise: bool
    reward: object
    healthy_z: tuple[float, float]
    terminate_when_unhealthy: bool
    friction_range: tuple[float, float]
    restitution_range: tuple[float, float]
    init_qpos: torch.Tensor          # (nq,)
    joint_init: torch.Tensor         # (25,) rad
    range_up: torch.Tensor           # (25,) rad
    range_down: torch.Tensor         # (25,) rad
    torque_low: torch.Tensor         # (25,) N·m
    torque_high: torch.Tensor        # (25,) N·m
    action_low: torch.Tensor         # (25,) bornes de l'espace d'action
    action_high: torch.Tensor        # (25,)
    soft_low: torch.Tensor           # (25,) rad
    soft_high: torch.Tensor          # (25,) rad
    chest_local_up: torch.Tensor     # (3,)
    init_body_mass: torch.Tensor     # (nbody,) kg
    pelvis_id: int
    chest_id: int
    feet_ids: torch.Tensor           # (2,) (droit, gauche)
    floor_geom_id: int
    # Tampons, (N, …).
    action: torch.Tensor = field(init=False)
    prev_action: torch.Tensor = field(init=False)
    has_prev: torch.Tensor = field(init=False)
    xy_before: torch.Tensor = field(init=False)
    feet_xy_prev: torch.Tensor = field(init=False)
    feet_need_refresh: torch.Tensor = field(init=False)
    foot_force: torch.Tensor = field(init=False)
    yaw_rate_filtered: torch.Tensor = field(init=False)
    forward_vel_filtered: torch.Tensor = field(init=False)
    air_time: torch.Tensor = field(init=False)
    was_in_contact: torch.Tensor = field(init=False)
    info: dict = field(init=False, default_factory=dict)

    def allocate(self, num_envs: int, device: str) -> None:
        """Alloue les tampons, dans l'état d'un environnement juste remis à zéro."""
        def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.zeros((num_envs, *shape), device=device, dtype=dtype)

        self.action = zeros(_N_JOINTS)
        self.prev_action = zeros(_N_JOINTS)
        self.has_prev = zeros(dtype=torch.bool)
        self.xy_before = zeros(2)
        self.feet_xy_prev = zeros(2, 2)
        self.feet_need_refresh = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.foot_force = zeros(2)
        self.yaw_rate_filtered = zeros()
        self.forward_vel_filtered = zeros()
        self.air_time = zeros(2)
        self.was_in_contact = torch.ones((num_envs, 2), device=device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        """Remet les tampons d'épisode à zéro, comme ``reset_model``."""
        self.has_prev[env_ids] = False
        self.feet_need_refresh[env_ids] = True
        self.foot_force[env_ids] = 0.0
        self.yaw_rate_filtered[env_ids] = 0.0
        self.forward_vel_filtered[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.was_in_contact[env_ids] = True


def _tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)


def build_state(
    config: PoppyEnvironmentConfig, mj_model: mujoco.MjModel, device: str
) -> PoppyState:
    """Lit les constantes sur la ``PoppyHumanoidEnv`` CPU de même configuration.

    Args:
        config: Configuration d'environnement.
        mj_model: Modèle compilé de la scène mjlab, pour retrouver les
            identifiants des corps sous leur préfixe ``robot/``.
        device: Périphérique torch.

    Returns:
        L'état, sans ses tampons (voir :meth:`PoppyState.allocate`).
    """
    ref = reference_env(config)
    try:
        def body_id(cpu_id: int) -> int:
            return mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{ROBOT}/{ref.model.body(cpu_id).name}"
            )

        floor = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{ROBOT}/floor")
        return PoppyState(
            dt=float(ref.dt),
            gait_period=float(ref._gait_period),
            floor_noise=bool(ref._floor_noise),
            reward=ref._reward,
            healthy_z=tuple(float(v) for v in ref._healthy_z_range),
            terminate_when_unhealthy=bool(ref._terminate_when_unhealthy),
            friction_range=tuple(float(v) for v in ref._friction_range),
            restitution_range=tuple(float(v) for v in ref._restitution_range),
            init_qpos=_tensor(ref.init_qpos, device),
            joint_init=_tensor(ref._joint_init, device),
            range_up=_tensor(ref._range_up, device),
            range_down=_tensor(ref._range_down, device),
            torque_low=_tensor(ref._torque_low, device),
            torque_high=_tensor(ref._torque_high, device),
            action_low=_tensor(ref.action_space.low, device),
            action_high=_tensor(ref.action_space.high, device),
            soft_low=_tensor(ref._soft_low, device),
            soft_high=_tensor(ref._soft_high, device),
            chest_local_up=_tensor(ref._chest_local_up, device),
            init_body_mass=_tensor(ref._init_body_mass, device),
            pelvis_id=body_id(_PELVIS_BODY_ID),
            chest_id=body_id(ref._chest_id),
            feet_ids=torch.tensor(
                [body_id(ref._r_foot_id), body_id(ref._l_foot_id)], device=device
            ),
            floor_geom_id=floor,
        )
    finally:
        ref.close()


def poppy_state(env: ManagerBasedRlEnv) -> PoppyState:
    """L'état partagé de l'environnement, créé par le terme d'action."""
    return env.poppy_state  # type: ignore[attr-defined]


# ── Action : PD en Python, comme _action_to_torque ─────────────────────────


class PoppyPdAction(ActionTerm):
    """Action → cible articulaire → couple PD, tenu pendant les sous-pas.

    ``cible = pose debout + action × course`` (course au-dessus ou en dessous
    selon le signe), ``couple = kp (cible − q) − kd q̇``, borné aux limites
    des moteurs. L'action est d'abord bornée à l'espace d'action annoncé
    (±3,1 à ±7,3 sans normalisation : le défaut de ``docs/TECHNIQUE.md`` §8
    est reproduit, pas corrigé), ce que SB3 fait avant ``env.step``.
    """

    def __init__(self, cfg: PoppyPdActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        state = build_state(cfg.config, env.sim.mj_model, env.device)
        state.allocate(env.num_envs, env.device)
        env.poppy_state = state  # type: ignore[attr-defined]
        self._state = state
        self._raw = torch.zeros((env.num_envs, _N_JOINTS), device=env.device)
        self._torque = torch.zeros_like(self._raw)

    @property
    def action_dim(self) -> int:
        return _N_JOINTS

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        """Calcule le couple d'un pas, sur l'état de début de pas."""
        s = self._state
        data = self._env.sim.data
        self._raw[:] = actions
        action = torch.clamp(actions, s.action_low, s.action_high)
        s.action[:] = action
        s.xy_before[:] = data.qpos[:, 0:2]
        # Gymnasium lit la position des pieds « avant » sur les données du pas
        # précédent, en retard d'un sous-pas : on la garde de la récompense
        # précédente, sauf juste après un reset où elle est à jour des deux côtés.
        fresh = s.feet_need_refresh
        if bool(fresh.any()):
            s.feet_xy_prev[fresh] = data.xpos[fresh][:, s.feet_ids, 0:2]
            s.feet_need_refresh[:] = False
        if s.floor_noise:
            _historical_push(self._env, s)
        target = torch.where(
            action >= 0,
            s.joint_init + action * s.range_up,
            s.joint_init + action * s.range_down,
        )
        torque = _KP * (target - data.qpos[:, 7:]) - _KD * data.qvel[:, 6:]
        self._torque[:] = torch.clamp(torque, s.torque_low, s.torque_high)

    def apply_actions(self) -> None:
        """Le même couple à chaque sous-pas (``do_simulation(torque, frame_skip)``)."""
        self._env.sim.data.ctrl[:] = self._torque

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._state.reset(slice(None) if env_ids is None else env_ids)
        self._raw[env_ids if env_ids is not None else slice(None)] = 0.0


@dataclass(kw_only=True)
class PoppyPdActionCfg(ActionTermCfg):
    """Configuration de :class:`PoppyPdAction`."""

    config: PoppyEnvironmentConfig
    entity_name: str = ROBOT

    def build(self, env: ManagerBasedRlEnv) -> PoppyPdAction:
        return PoppyPdAction(self, env)


def _historical_push(env: ManagerBasedRlEnv, s: PoppyState) -> None:
    """``_maybe_apply_push`` sans ``push_as_force`` : un couple de 1 à 5 N·m.

    Avec une probabilité de 2 % par pas, écrit un couple horizontal aléatoire
    dans ``xfrc_applied[bassin, 3:5]`` (l'intention était une force : défaut
    connu, reproduit) ; sinon remet ces deux composantes à zéro.
    """
    n = env.num_envs
    device = env.device
    push = torch.rand(n, device=device) < _PUSH_PROBABILITY
    magnitude = torch.empty(n, device=device).uniform_(*_PUSH_RANGE)
    angle = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    torque_xy = torch.stack([magnitude * torch.cos(angle), magnitude * torch.sin(angle)], -1)
    torque_xy = torch.where(push[:, None], torque_xy, torch.zeros_like(torque_xy))
    env.sim.data.xfrc_applied[:, s.pelvis_id, 3:5] = torque_xy


# ── Observation : qpos[2:], qvel, contacts des pieds, horloge ──────────────


def poppy_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """L'observation de ``_get_obs`` : 63 dimensions, 65 avec l'horloge.

    Returns:
        ``(N, 63 ou 65)`` : ``qpos[2:]`` (30), ``qvel`` (31), force de contact
        des pieds droit et gauche en N (2), puis sin et cos de la phase.
    """
    s = poppy_state(env)
    data = env.sim.data
    parts = [data.qpos[:, 2:], data.qvel[:, :], s.foot_force]
    if s.gait_period > 0.0:
        angle = 2.0 * math.pi * _gait_phase(env, s)
        parts += [torch.sin(angle)[:, None], torch.cos(angle)[:, None]]
    return torch.cat(parts, dim=-1)


def _gait_phase(env: ManagerBasedRlEnv, s: PoppyState) -> torch.Tensor:
    """Phase de l'horloge dans [0, 1), 0 au reset : ``(pas × dt / période) mod 1``."""
    elapsed = env.episode_length_buf.to(torch.float64) * s.dt
    return torch.remainder(elapsed / s.gait_period, 1.0).to(torch.float32)


# ── Terminaison : bassin hors de la plage de hauteur ───────────────────────


def poppy_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Vrai quand la hauteur du bassin sort de ``healthy_z_range`` (sol plat, z = 0)."""
    s = poppy_state(env)
    if not s.terminate_when_unhealthy:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    z = env.sim.data.qpos[:, 2]
    return ~((s.healthy_z[0] < z) & (z < s.healthy_z[1]))


# ── Récompense : tous les termes de PoppyHumanoidEnv.step ──────────────────


def _ground_direction(vector: torch.Tensor) -> torch.Tensor:
    """Partie horizontale normalisée d'une direction ``(N, 3)`` → ``(N, 2)``."""
    flat = vector[:, 0:2]
    norm = torch.linalg.norm(flat, dim=-1, keepdim=True)
    fallback = torch.as_tensor(_LOCAL_FORWARD[:2], dtype=flat.dtype, device=flat.device)
    return torch.where(norm < 1e-8, fallback.expand_as(flat), flat / norm.clamp_min(1e-12))


class PoppyReward:
    """La récompense de ``PoppyHumanoidEnv.step``, terme pour terme.

    Un seul terme mjlab, de poids 1, qui rend la somme : plusieurs termes
    partagent un état (filtres, temps de vol, action précédente) et doivent
    être calculés dans l'ordre d'origine. Le détail est rangé dans
    ``poppy_state(env).info`` sous les noms de l'``info`` CPU.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg, env

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids  # l'état est remis à zéro par le terme d'action

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        s = poppy_state(env)
        data = env.sim.data
        # Forces de contact, comme mj_rnePostConstraint après mj_step.
        with wp.ScopedDevice(env.sim.wp_device):
            mjwarp.rne_postconstraint(env.sim.wp_model, env.sim.wp_data)
        phase = _gait_phase(env, s) if s.gait_period > 0.0 else None
        return compute_reward(
            s,
            qpos=data.qpos[:, :],
            qvel=data.qvel[:, :],
            pelvis_rot=data.xmat[:, s.pelvis_id].reshape(-1, 3, 3),
            chest_rot=data.xmat[:, s.chest_id].reshape(-1, 3, 3),
            feet_xy=data.xpos[:, s.feet_ids, 0:2],
            foot_force=torch.linalg.norm(data.cfrc_ext[:, s.feet_ids, 3:6], dim=-1),
            phase=phase,
        )


_LOCAL_AXES: dict[str, torch.Tensor] = {}


def _local_axis(name: str, device: torch.device) -> torch.Tensor:
    key = f"{name}@{device}"
    if key not in _LOCAL_AXES:
        values = _LOCAL_FORWARD if name == "forward" else _LOCAL_LEFT
        _LOCAL_AXES[key] = torch.as_tensor(values, dtype=torch.float32, device=device)
    return _LOCAL_AXES[key]


def compute_reward(  # noqa: PLR0915 - un terme par bloc, dans l'ordre de step()
    s: PoppyState,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    pelvis_rot: torch.Tensor,
    chest_rot: torch.Tensor,
    feet_xy: torch.Tensor,
    foot_force: torch.Tensor,
    phase: torch.Tensor | None,
) -> torch.Tensor:
    """Récompense d'un pas, à partir des grandeurs que lit ``PoppyHumanoidEnv.step``.

    Fonction pure de l'état de simulation (hors tampons de ``s``, mis à jour
    comme dans ``step``) : ``gpu/tests`` la nourrit de données MuJoCo C pour
    vérifier qu'elle rend la récompense CPU, indépendamment du moteur.

    Args:
        s: État partagé ; lit ``action``, ``xy_before``, ``feet_xy_prev`` et
            les filtres, et les met à jour.
        qpos: ``(N, nq)`` après le pas.
        qvel: ``(N, nv)`` après le pas.
        pelvis_rot: ``(N, 3, 3)`` orientation du bassin (un sous-pas de retard).
        chest_rot: ``(N, 3, 3)`` orientation de la poitrine.
        feet_xy: ``(N, 2, 2)`` position horizontale des pieds (droit, gauche), m.
        foot_force: ``(N, 2)`` norme de la force de contact des pieds, N.
        phase: ``(N,)`` phase de l'horloge après le pas, ou ``None`` sans horloge.

    Returns:
        ``(N,)`` la récompense ; le détail va dans ``s.info``.
    """
    w = s.reward
    dt = s.dt
    s.foot_force[:] = foot_force
    rotation = pelvis_rot
    local_forward = _local_axis("forward", qpos.device)
    local_left = _local_axis("left", qpos.device)

    # Avance, sur le cap du bassin.
    velocity = (qpos[:, 0:2] - s.xy_before) / dt
    forward_dir = _ground_direction(rotation @ local_forward)
    left_dir = _ground_direction(rotation @ local_left)
    forward_vel = (velocity * forward_dir).sum(-1)
    capped_vel = forward_vel.clamp(-w.max_forward_vel, w.max_forward_vel)
    forward_reward = w.forward_weight * capped_vel

    # Suivi de vitesse cible, sur la vitesse filtrée.
    if w.vel_tracking_weight > 0.0:
        if w.vel_tracking_tau > 0.0:
            alpha = dt / (w.vel_tracking_tau + dt)
            s.forward_vel_filtered += alpha * (forward_vel - s.forward_vel_filtered)
        else:
            s.forward_vel_filtered[:] = forward_vel
        vel_error = (s.forward_vel_filtered - w.target_vel) / w.vel_tracking_sigma
        vel_tracking_reward = w.vel_tracking_weight * torch.exp(-vel_error ** 2)
    else:
        vel_tracking_reward = torch.zeros_like(forward_vel)

    # Debout, verticalité du bassin.
    z = qpos[:, 2]
    healthy = (s.healthy_z[0] < z) & (z < s.healthy_z[1])
    healthy_reward = torch.where(healthy, w.healthy, 0.0)
    quat = qpos[:, 3:7]
    uprightness = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright_reward = w.upright_weight * uprightness

    # Appui simple / double.
    in_contact = foot_force > _CONTACT_THRESHOLD     # (droit, gauche)
    single = in_contact[:, 0] != in_contact[:, 1]
    double = in_contact[:, 0] & in_contact[:, 1]
    gait_reward = torch.where(
        single, w.gait_single, torch.where(double, w.gait_double, 0.0)
    )

    # Temps de vol, payé à la pose du pied.
    touchdown = in_contact & ~s.was_in_contact
    landed = torch.clamp(s.air_time, max=w.air_time_max)
    air_time_reward = w.air_time_weight * ((landed - w.air_time_target) * touchdown).sum(-1)
    s.air_time[:] = torch.where(in_contact, 0.0, s.air_time + dt)
    s.was_in_contact[:] = in_contact

    lateral_vel = (velocity * left_dir).sum(-1)
    lateral_cost = w.lateral_weight * lateral_vel.abs()

    action = s.action
    ctrl_cost = w.ctrl_weight * (action ** 2).sum(-1)
    action_rate_cost = torch.where(
        s.has_prev,
        w.action_rate_weight * ((action - s.prev_action) ** 2).sum(-1),
        0.0,
    )
    s.prev_action[:] = action
    s.has_prev[:] = True

    joint_vel_cost = w.joint_vel_weight * (qvel[:, 6:] ** 2).sum(-1)

    # Lacet, filtré.
    ang_vel_world = (rotation @ qvel[:, 3:6, None])[:, :, 0]
    yaw_rate = ang_vel_world[:, 2]
    if w.yaw_rate_tau > 0.0:
        alpha = dt / (w.yaw_rate_tau + dt)
        s.yaw_rate_filtered += alpha * (yaw_rate - s.yaw_rate_filtered)
    else:
        s.yaw_rate_filtered[:] = yaw_rate
    yaw_rate_cost = w.yaw_rate_weight * s.yaw_rate_filtered ** 2

    q = qpos[:, 7:]
    joint_limit_cost = w.joint_limit_weight * (
        torch.clamp(s.soft_low - q, min=0.0) + torch.clamp(q - s.soft_high, min=0.0)
    ).sum(-1)

    torso_uprightness = (chest_rot @ s.chest_local_up)[:, 2]
    torso_upright_reward = w.torso_upright_weight * torso_uprightness
    torso_tilt_cost = w.torso_tilt_weight * (1.0 - torso_uprightness)

    base_height_cost = w.base_height_weight * (z - w.base_height_target) ** 2
    ang_vel_xy_cost = w.ang_vel_xy_weight * (ang_vel_world[:, 0] ** 2 + ang_vel_world[:, 1] ** 2)
    feet_speed = torch.linalg.norm(feet_xy - s.feet_xy_prev, dim=-1) / dt
    s.feet_xy_prev[:] = feet_xy
    feet_slide_cost = w.feet_slide_weight * (feet_speed * in_contact).sum(-1)

    phase_contact_reward = torch.zeros_like(z)
    if phase is not None and w.phase_contact_weight > 0.0:
        leg_phase = torch.remainder(
            phase[:, None] + torch.tensor([0.0, 0.5], device=qpos.device), 1.0
        )
        is_stance = leg_phase < w.phase_stance_ratio
        phase_contact_reward = w.phase_contact_weight * (
            (in_contact == is_stance).float().mean(-1)
        )

    reward = (forward_reward + healthy_reward + upright_reward
              + gait_reward
              - lateral_cost - ctrl_cost
              - action_rate_cost - joint_vel_cost
              + air_time_reward + torso_upright_reward
              - yaw_rate_cost - joint_limit_cost
              + vel_tracking_reward - torso_tilt_cost
              - base_height_cost - ang_vel_xy_cost - feet_slide_cost
              + phase_contact_reward)

    s.info = {
        "forward_vel": forward_vel, "capped_vel": capped_vel,
        "lateral_vel": lateral_vel, "uprightness": uprightness,
        "healthy_reward": healthy_reward, "gait_reward": gait_reward,
        "ctrl_cost": ctrl_cost, "action_rate_cost": action_rate_cost,
        "joint_vel_cost": joint_vel_cost, "air_time_reward": air_time_reward,
        "yaw_rate_cost": yaw_rate_cost, "joint_limit_cost": joint_limit_cost,
        "torso_upright_reward": torso_upright_reward,
        "vel_tracking_reward": vel_tracking_reward, "torso_tilt_cost": torso_tilt_cost,
        "base_height_cost": base_height_cost, "ang_vel_xy_cost": ang_vel_xy_cost,
        "feet_slide_cost": feet_slide_cost, "phase_contact_reward": phase_contact_reward,
        "torso_uprightness": torso_uprightness, "yaw_rate": yaw_rate,
        "heading": torch.atan2(forward_dir[:, 1], forward_dir[:, 0]),
        "r_contact": in_contact[:, 0].float(), "l_contact": in_contact[:, 1].float(),
        "touchdowns": touchdown.sum(-1).float(),
        "x_position": qpos[:, 0], "y_position": qpos[:, 1],
        "reward": reward,
    }
    return reward


# ── Reset et randomisation de domaine ──────────────────────────────────────


@requires_model_fields("geom_friction", "geom_solimp", "body_mass")
def poppy_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """``reset_model`` : état initial bruité, lacet, sol et masses randomisés.

    Sans randomisation (évaluation), seul le bruit de ±0,01 sur ``qpos`` et
    ``qvel`` s'applique, quaternion compris, comme sur CPU. Avec : lacet
    initial de ±0,26 rad, friction du sol (glissement, et torsion et roulis
    proportionnels), « restitution » écrite dans ``geom_solimp[sol, 4]``
    (défaut connu : c'est l'exposant de l'impédance, reproduit tel quel), et
    chaque masse à ±15 % de sa valeur d'origine, sans recalcul des
    constantes dérivées — comme ``_randomize_masses``.
    """
    s = poppy_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    device = env.device
    data = env.sim.data
    model = env.sim.model
    nq, nv = s.init_qpos.shape[0], data.qvel.shape[1]

    def uniform(low: float, high: float, *shape: int) -> torch.Tensor:
        return torch.empty((n, *shape), device=device).uniform_(low, high)

    qpos = s.init_qpos + uniform(-_RESET_NOISE, _RESET_NOISE, nq)
    qvel = uniform(-_RESET_NOISE, _RESET_NOISE, nv)
    if s.floor_noise:
        half = 0.5 * uniform(-_YAW_RANGE, _YAW_RANGE)
        qpos[:, 3] = torch.cos(half)
        qpos[:, 4:6] = 0.0
        qpos[:, 6] = torch.sin(half)
    data.qpos[env_ids] = qpos
    data.qvel[env_ids] = qvel
    data.xfrc_applied[env_ids] = 0.0
    data.ctrl[env_ids] = 0.0

    if s.floor_noise:
        floor = s.floor_geom_id
        friction = uniform(*s.friction_range)
        model.geom_friction[env_ids, floor] = torch.stack(
            [friction, 0.005 * friction, 0.0001 * friction], dim=-1
        )
        model.geom_solimp[env_ids, floor, 4] = uniform(*s.restitution_range)
        nbody = s.init_body_mass.shape[0]
        scale = uniform(*_MASS_SCALE, nbody - 1)
        model.body_mass[env_ids, 1:] = s.init_body_mass[1:] * scale


# ── Assemblage ─────────────────────────────────────────────────────────────


def make_env_cfg(
    config: PoppyEnvironmentConfig, num_envs: int, seed: int | None = None
) -> ManagerBasedRlEnvCfg:
    """Configuration mjlab de l'environnement Poppy décrit par un YAML de ``configs/``.

    Args:
        config: Configuration d'environnement CPU (``make_poppy_env_config``).
        num_envs: Nombre de mondes simulés en parallèle sur le GPU.
        seed: Graine de torch et de Warp.

    Returns:
        La configuration, à passer à ``ManagerBasedRlEnv``.

    Raises:
        NotImplementedError: Si la configuration demande une option non portée.
    """
    check_supported(config)
    ref = reference_env(config)
    try:
        source = option_fields(ref.model.opt)
        step_dt = float(ref.dt)
    finally:
        ref.close()

    fix_knee = config.fix_left_knee_axis
    # La pose initiale vient de poppy_reset ; init_state ne sert qu'au keyframe
    # que mjlab ajoute à la scène, et que rien ici ne relit.
    robot = EntityCfg(spec_fn=lambda: _poppy_spec(fix_knee))

    return ManagerBasedRlEnvCfg(
        decimation=config.frame_skip,
        scene=SceneCfg(num_envs=num_envs, entities={ROBOT: robot}),
        observations={
            _OBS_GROUP: ObservationGroupCfg(
                terms={"poppy": ObservationTermCfg(func=poppy_observation)},
                concatenate_terms=True,
            )
        },
        actions={"joints": PoppyPdActionCfg(config=config)},
        events={"reset": EventTermCfg(func=poppy_reset, mode="reset")},
        rewards={"poppy": RewardTermCfg(func=PoppyReward, weight=1.0)},
        terminations={
            "fallen": TerminationTermCfg(func=poppy_fallen),
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        sim=SimulationCfg(mujoco=MjcfOptions(source=source, timestep=source["timestep"])),
        seed=seed,
        episode_length_s=1000 * step_dt,
        scale_rewards_by_dt=False,
    )


def option_fields(opt: mujoco.MjOption) -> dict[str, object]:
    """Les options d'un modèle en types simples : pour les recopier et les comparer."""
    out: dict[str, object] = {}
    for name in dir(opt):
        if name.startswith("_"):
            continue
        value = getattr(opt, name)
        if callable(value):
            continue
        out[name] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


__all__ = [
    "ROBOT",
    "MjcfOptions",
    "PoppyPdAction",
    "PoppyPdActionCfg",
    "PoppyReward",
    "PoppyState",
    "check_supported",
    "make_env_cfg",
    "option_fields",
    "poppy_observation",
    "poppy_reset",
    "poppy_state",
    "reference_env",
]
