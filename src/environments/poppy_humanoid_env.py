"""
Custom Gymnasium environment for the Poppy Humanoid robot (MuJoCo).

The environment follows the same API as Humanoid-v5 (gymnasium) but is
adapted to the Poppy Humanoid's kinematics (25 DOFs, ~83 cm, ~3.5 kg).

Domain randomization
--------------------
When ``floor_noise=True`` (default), the floor's friction and restitution
are re-sampled from a configurable range at every ``reset()``.  This makes
the policy more robust to surface variability encountered on the real robot.

Control
-------
Actions in [-1, 1] are mapped to **target joint positions** within each
joint's mechanical limits.  A PD controller converts these targets into
torques, emulating the real Dynamixel servo behaviour.

Observation space (63-dim):
    - qpos[2:]        : 30 values  (z + quat(4) + 25 joints; skip global x,y)
    - qvel            : 31 values  (6 root + 25 joints)
    - foot_contacts   : 2 values   (left foot, right foot contact force)
    - total           : 63 values

Action space (25-dim):
    Target joint positions for each of the 25 revolute joints,
    normalised to [-1, 1] (rescaled to joint limits internally).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle


_ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "poppy_humanoid"
_MODEL_PATH = _ASSETS_DIR / "poppy_humanoid.xml"

# Number of actuated joints
_N_JOINTS = 25

# 30 (qpos[2:]) + 31 (qvel) + 2 (foot contacts) = 63
_OBS_DIM = 30 + 31 + 2

_DEFAULT_HEALTHY_Z_RANGE = (0.25, 0.70)

_FLOOR_GEOM_NAME = "floor"

# ── Natural standing pose corrections ───────────────────────
# At qpos=0 the URDF shoulder frames cause the arms to cross in
# front of the body, and the torso/ankles are not balanced.
# These offsets define the true "home" standing pose:
#   • Arms hang at the robot's sides
#   • Torso leaned slightly backward to center the CoM over the feet
#   • Ankles tilted to compensate for CoM offset
# Values were found by grid-search for maximum passive stability
# (robot stands ~700 steps with action=0).
_STANDING_POSE_OVERRIDES: Dict[str, float] = {
    # Arms at sides (fixes crossed-arm URDF bug)
    "l_shoulder_y":   2.214,
    "l_shoulder_x":  -1.536,
    "r_shoulder_y":  -2.251,
    "r_shoulder_x":  -1.564,
    # Torso balance (prevents forward tilt)
    "abs_y":         -0.190,
    "bust_y":        -0.008,
    # Ankle balance (prevents lateral drift)
    "r_ankle_y":      0.026,
    "l_ankle_y":      0.026,
}

# ── Initial pelvis height ───────────────────────────────────
# Chosen so the feet just touch the ground at the standing pose.
_INIT_PELVIS_Z = 0.41

# ── PD controller gains ─────────────────────────────────────
# Tuned so that kp * typical_error ≈ max_torque (3.1 Nm).
# kd provides damping that replaces what the real Dynamixel
# servo PID would provide internally.
_KP = 8.0    # Nm/rad  — proportional gain
_KD = 0.5    # Nm·s/rad — derivative gain


class PoppyHumanoidEnv(MujocoEnv, EzPickle):
    """
    MuJoCo environment for the Poppy Humanoid robot with optional
    floor domain randomization and PD position control.

    Parameters
    ----------
    floor_noise : bool
        Enable floor friction / restitution randomization on each reset.
    friction_range : tuple[float, float]
        (min, max) for the slide-friction coefficient of the floor.
    restitution_range : tuple[float, float]
        (min, max) for the floor restitution (bounciness) coefficient.
    healthy_z_range : tuple[float, float]
        Pelvis z-height (in world frame) considered healthy.
    terminate_when_unhealthy : bool
        Terminate episode when the robot falls.
    forward_reward_weight : float
        Weight for the forward (x) velocity reward.
    healthy_reward : float
        Per-step bonus for staying upright.
    ctrl_cost_weight : float
        Weight for the L2 control cost.
    render_mode : str or None
        "human" or "rgb_array" or None.
    frame_skip : int
        Number of MuJoCo simulation steps per ``step()`` call.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        floor_noise: bool = True,
        friction_range: Tuple[float, float] = (0.5, 1.5),
        restitution_range: Tuple[float, float] = (0.0, 0.3),
        healthy_z_range: Tuple[float, float] = _DEFAULT_HEALTHY_Z_RANGE,
        terminate_when_unhealthy: bool = True,
        forward_reward_weight: float = 5.0,
        healthy_reward: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        max_forward_vel: float = 0.5,
        action_rate_cost_weight: float = 0.05,
        joint_vel_cost_weight: float = 0.005,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
    ):
        EzPickle.__init__(
            self,
            floor_noise=floor_noise,
            friction_range=friction_range,
            restitution_range=restitution_range,
            healthy_z_range=healthy_z_range,
            terminate_when_unhealthy=terminate_when_unhealthy,
            forward_reward_weight=forward_reward_weight,
            healthy_reward=healthy_reward,
            ctrl_cost_weight=ctrl_cost_weight,
            max_forward_vel=max_forward_vel,
            action_rate_cost_weight=action_rate_cost_weight,
            joint_vel_cost_weight=joint_vel_cost_weight,
            render_mode=render_mode,
            frame_skip=frame_skip,
        )

        self._floor_noise = floor_noise
        self._friction_range = friction_range
        self._restitution_range = restitution_range
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._forward_reward_weight = forward_reward_weight
        self._healthy_reward = healthy_reward
        self._ctrl_cost_weight = ctrl_cost_weight
        self._max_forward_vel = max_forward_vel
        self._action_rate_cost_weight = action_rate_cost_weight
        self._joint_vel_cost_weight = joint_vel_cost_weight

        # Track previous action for smoothness penalty
        self._prev_action: Optional[NDArray] = None

        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            model_path=str(_MODEL_PATH),
            frame_skip=frame_skip,
            observation_space=obs_space,
            render_mode=render_mode,
        )

        # ── Fix init_qpos so action=0 → natural standing pose ───────
        # The raw URDF has arms crossed at qpos=0 and unbalanced CoM;
        # we patch init_qpos so the PD controller targets the correct
        # home position (arms at sides, torso centered).
        self.init_qpos[2] = _INIT_PELVIS_Z
        joint_names = [self.model.joint(i).name
                       for i in range(1, self.model.njnt)]
        for jname, value in _STANDING_POSE_OVERRIDES.items():
            idx = joint_names.index(jname)
            self.init_qpos[7 + idx] = value

        # ── Cache joint limits for action → target position mapping ──
        # model.jnt_range shape: (n_joints_total, 2).
        # Index 0 is the free joint (no limits), so we skip it.
        self._joint_low = self.model.jnt_range[1:, 0].copy()
        self._joint_high = self.model.jnt_range[1:, 1].copy()

        # action=0 → default standing pose (init_qpos for joints)
        # action=+1 → upper joint limit, action=-1 → lower joint limit
        self._joint_init = self.init_qpos[7:].copy()  # skip free-joint (7 values)
        self._range_up = self._joint_high - self._joint_init    # available range upward
        self._range_down = self._joint_init - self._joint_low   # available range downward

        # ── Cache actuator torque limits for PD clipping ─────────────
        self._torque_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._torque_high = self.model.actuator_ctrlrange[:, 1].copy()

        # ── Cache foot body IDs for contact sensing ──────────────────
        self._r_foot_id = self.model.body("r_foot").id
        self._l_foot_id = self.model.body("l_foot").id

        # ── Cache initial body masses for DR mass randomization ──────
        self._init_body_mass = self.model.body_mass.copy()

    @property
    def is_healthy(self) -> bool:
        z = self.data.qpos[2]
        return float(self._healthy_z_range[0]) < z < float(self._healthy_z_range[1])

    @property
    def terminated(self) -> bool:
        return self._terminate_when_unhealthy and not self.is_healthy

    def _action_to_torque(self, action: NDArray) -> NDArray:
        """
        Convert normalised action [-1, 1] to PD torque.

        1.  action → target joint position within mechanical limits
            action=0 → default standing pose (init_qpos)
            action=+1 → upper joint limit
            action=-1 → lower joint limit
        2.  PD:  τ = kp · (target − q) − kd · q̇
        3.  Clip to actuator force limits
        """
        # Map [-1, 1] → joint position (asymmetric around init pose)
        target = np.where(
            action >= 0,
            self._joint_init + action * self._range_up,
            self._joint_init + action * self._range_down,
        )

        # Current joint state (skip free-joint: 7 qpos, 6 qvel)
        q = self.data.qpos[7:]
        qd = self.data.qvel[6:]

        # PD control
        torque = _KP * (target - q) - _KD * qd

        # Clip to actuator force limits
        return np.clip(torque, self._torque_low, self._torque_high)

    def _get_foot_contacts(self) -> NDArray:
        """Return contact force magnitude for each foot via cfrc_ext.

        cfrc_ext shape is (nbody, 6): [torque(3), force(3)].
        We take the force magnitude (last 3 components).
        VecNormalize handles scaling during training.
        """
        r_force = np.linalg.norm(self.data.cfrc_ext[self._r_foot_id, 3:])
        l_force = np.linalg.norm(self.data.cfrc_ext[self._l_foot_id, 3:])
        return np.array([r_force, l_force], dtype=np.float64)

    def _get_uprightness(self) -> float:
        """Return how upright the torso is (1.0 = perfectly vertical).

        The root quaternion is qpos[3:7] = (w, x, y, z).
        The world up-vector [0, 0, 1] rotated by the inverse quaternion
        gives the up-vector in the body frame.  The z-component of that
        vector is the cosine of the tilt angle: 1.0 when vertical.
        """
        quat = self.data.qpos[3:7]  # (w, x, y, z)
        w, x, y, z = quat
        # z-component of world-up rotated into body frame
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return float(up_z)

    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, bool, Dict]:
        xy_before = self.data.qpos[:2].copy()

        # Random external push (domain randomization)
        if self._floor_noise:
            self._maybe_apply_push()

        # PD position control: action [-1,1] → target pos → torque
        torque = self._action_to_torque(action)
        self.do_simulation(torque, self.frame_skip)

        xy_after = self.data.qpos[:2].copy()
        dt = self.dt

        # ── Forward velocity reward (CAPPED) ────────────────────
        # Cap at max_forward_vel to prevent the policy from
        # learning to take huge strides for extra speed.
        forward_vel = (xy_after[0] - xy_before[0]) / dt
        capped_vel = np.clip(forward_vel, -self._max_forward_vel,
                             self._max_forward_vel)
        forward_reward = self._forward_reward_weight * capped_vel

        # ── Alive bonus ─────────────────────────────────────────
        healthy_reward = self._healthy_reward if self.is_healthy else 0.0

        # ── Uprightness bonus ───────────────────────────────────
        uprightness = self._get_uprightness()
        upright_reward = 1.0 * uprightness  # max 1.0/step when perfectly vertical

        # ── Foot contact alternation reward ─────────────────────
        # Reward proper gait: exactly one foot on the ground at a time
        foot_contacts = self._get_foot_contacts()
        r_contact = foot_contacts[0] > 1.0   # right foot touching
        l_contact = foot_contacts[1] > 1.0   # left foot touching
        if r_contact != l_contact:
            # One foot on ground, one in air → walking gait
            gait_reward = 0.3
        elif r_contact and l_contact:
            # Both feet on ground → standing/double support (ok but less)
            gait_reward = 0.1
        else:
            # Both feet in air → airborne/falling
            gait_reward = 0.0

        # ── Lateral velocity penalty ────────────────────────────
        lateral_vel = (xy_after[1] - xy_before[1]) / dt
        lateral_cost = 0.5 * abs(lateral_vel)

        # ── Control cost ────────────────────────────────────────
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))

        # ── Action rate penalty (smoothness) ────────────────────
        # Penalise large changes between consecutive actions to
        # prevent jerky movements and huge step-to-step changes.
        if self._prev_action is not None:
            action_rate_cost = self._action_rate_cost_weight * np.sum(
                np.square(action - self._prev_action)
            )
        else:
            action_rate_cost = 0.0
        self._prev_action = action.copy()

        # ── Joint velocity penalty ──────────────────────────────
        # Penalise fast joint movements (prevents wild swinging)
        joint_vel = self.data.qvel[6:]  # skip root 6-dof
        joint_vel_cost = self._joint_vel_cost_weight * np.sum(
            np.square(joint_vel)
        )

        reward = (forward_reward + healthy_reward + upright_reward
                  + gait_reward
                  - lateral_cost - ctrl_cost
                  - action_rate_cost - joint_vel_cost)

        observation = self._get_obs()
        terminated = self.terminated
        info = {
            "forward_vel": forward_vel,
            "capped_vel": capped_vel,
            "lateral_vel": lateral_vel,
            "uprightness": uprightness,
            "healthy_reward": healthy_reward,
            "gait_reward": gait_reward,
            "ctrl_cost": ctrl_cost,
            "action_rate_cost": action_rate_cost,
            "joint_vel_cost": joint_vel_cost,
            "x_position": xy_after[0],
            "y_position": xy_after[1],
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_obs(self) -> NDArray:
        qpos = self.data.qpos[2:].copy()       # shape (30,)
        qvel = self.data.qvel.copy()            # shape (31,)
        foot_contacts = self._get_foot_contacts()  # shape (2,)
        return np.concatenate([qpos, qvel, foot_contacts])

    def _maybe_apply_push(self) -> None:
        """Apply a random external force to the pelvis with low probability.

        This forces the policy to learn active balance recovery instead of
        relying on a single leaned posture.  Pushes happen ~2% of steps.
        """
        if self.np_random.random() < 0.02:
            # Random horizontal force on pelvis (body index 1)
            force_mag = self.np_random.uniform(1.0, 5.0)  # Newtons
            angle = self.np_random.uniform(-np.pi, np.pi)
            fx = force_mag * np.cos(angle)
            fy = force_mag * np.sin(angle)
            self.data.xfrc_applied[1, 3] = fx
            self.data.xfrc_applied[1, 4] = fy
        else:
            # Clear any previous push
            self.data.xfrc_applied[1, 3:5] = 0.0

    def _randomize_masses(self) -> None:
        """Scale each body mass by ±15% to prevent overfitting to exact dynamics."""
        for i in range(1, self.model.nbody):  # skip world body (0)
            scale = self.np_random.uniform(0.85, 1.15)
            self.model.body_mass[i] = self._init_body_mass[i] * scale

    def reset_model(self) -> NDArray:
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nv
        )

        if self._floor_noise:
            # Randomise initial yaw (rotation around z) up to ±15°
            yaw = self.np_random.uniform(-0.26, 0.26)
            # qpos[3:7] = quaternion (w, x, y, z) — apply yaw rotation
            half = yaw / 2.0
            # Compose with existing quaternion: rotate around world z
            qpos[3] = np.cos(half)   # w
            qpos[4] = 0.0            # x
            qpos[5] = 0.0            # y
            qpos[6] = np.sin(half)   # z

        self.set_state(qpos, qvel)

        if self._floor_noise:
            self._randomize_floor()
            self._randomize_masses()

        # Clear any leftover push forces
        self.data.xfrc_applied[:] = 0.0

        # Reset action smoothness tracking
        self._prev_action = None

        return self._get_obs()

    def _randomize_floor(self) -> None:
        """
        Re-sample floor friction and restitution at every episode reset.
        """
        floor_id = self.model.geom(name=_FLOOR_GEOM_NAME).id

        slide_friction = self.np_random.uniform(
            self._friction_range[0], self._friction_range[1]
        )
        self.model.geom_friction[floor_id, 0] = slide_friction
        self.model.geom_friction[floor_id, 1] = 0.005 * slide_friction
        self.model.geom_friction[floor_id, 2] = 0.0001 * slide_friction

        restitution = self.np_random.uniform(
            self._restitution_range[0], self._restitution_range[1]
        )
        self.model.geom_solimp[floor_id, 4] = restitution

    def get_floor_randomization_info(self) -> Dict[str, float]:
        """Return current floor friction and restitution (for logging)."""
        floor_id = self.model.geom(name=_FLOOR_GEOM_NAME).id
        return {
            "floor_slide_friction": float(self.model.geom_friction[floor_id, 0]),
            "floor_restitution": float(self.model.geom_solimp[floor_id, 4]),
        }


def register_poppy_env() -> None:
    """Register PoppyHumanoid-v0 with Gymnasium if not already registered."""
    env_id = "PoppyHumanoid-v0"
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point="src.environments.poppy_humanoid_env:PoppyHumanoidEnv",
            max_episode_steps=1000,
        )
