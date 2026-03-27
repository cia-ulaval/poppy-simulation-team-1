"""
Custom Gymnasium environment for the Poppy Humanoid robot (MuJoCo).

The environment follows the same API as Humanoid-v5 (gymnasium) but is
adapted to the Poppy Humanoid's kinematics (25 DOFs, ~83 cm, ~3.5 kg).

Domain randomization
--------------------
When ``floor_noise=True`` (default), the floor's friction and restitution
are re-sampled from a configurable range at every ``reset()``.  This makes
the policy more robust to surface variability encountered on the real robot.

Observation space (70-dim):
    - qpos[2:]   : 30 values  (z + quat(4) + 25 joints; skip global x,y)
    - qvel        : 31 values  (6 root + 25 joints)
    - cinert      : not included (keep obs compact)
    - total       : 61 values

Action space (25-dim):
    Torque commands for each of the 25 revolute joints,
    normalised to [-1, 1] (rescaled to effort limits internally by MuJoCo).
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

_OBS_DIM = 30 + 31  # = 61

_DEFAULT_HEALTHY_Z_RANGE = (0.25, 0.70)

_FLOOR_GEOM_NAME = "floor"


class PoppyHumanoidEnv(MujocoEnv, EzPickle):
    """
    MuJoCo environment for the Poppy Humanoid robot with optional
    floor domain randomization.

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
        # render_fps must equal 1 / (timestep * frame_skip)
        # = 1 / (0.002 * 5) = 100 Hz
        "render_fps": 100,
    }

    def __init__(
        self,
        floor_noise: bool = True,
        friction_range: Tuple[float, float] = (0.5, 1.5),
        restitution_range: Tuple[float, float] = (0.0, 0.3),
        healthy_z_range: Tuple[float, float] = _DEFAULT_HEALTHY_Z_RANGE,
        terminate_when_unhealthy: bool = True,
        forward_reward_weight: float = 1.25,
        healthy_reward: float = 5.0,
        ctrl_cost_weight: float = 0.1,
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

    @property
    def is_healthy(self) -> bool:
        z = self.data.qpos[2]
        return float(self._healthy_z_range[0]) < z < float(self._healthy_z_range[1])

    @property
    def terminated(self) -> bool:
        return self._terminate_when_unhealthy and not self.is_healthy


    def step(self, action: NDArray) -> Tuple[NDArray, float, bool, bool, Dict]:
        xy_before = self.data.qpos[:2].copy()

        self.do_simulation(action, self.frame_skip)

        xy_after = self.data.qpos[:2].copy()
        dt = self.dt

        # Forward velocity reward (x-axis)
        forward_vel = (xy_after[0] - xy_before[0]) / dt
        forward_reward = self._forward_reward_weight * forward_vel

        # Alive bonus
        healthy_reward = self._healthy_reward if self.is_healthy else 0.0

        # Control cost
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))

        reward = forward_reward + healthy_reward - ctrl_cost

        observation = self._get_obs()
        terminated = self.terminated
        info = {
            "forward_vel": forward_vel,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "x_position": xy_after[0],
            "y_position": xy_after[1],
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_obs(self) -> NDArray:
        qpos = self.data.qpos[2:].copy()   # shape (30,)
        qvel = self.data.qvel.copy()        # shape (31,)
        return np.concatenate([qpos, qvel])

    def reset_model(self) -> NDArray:
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        if self._floor_noise:
            self._randomize_floor()

        return self._get_obs()

    def _randomize_floor(self) -> None:
        """
        Re-sample floor friction and restitution at every episode reset.

        MuJoCo's geom_friction has shape (ngeom, 3):
            column 0 → slide friction (tangential)
            column 1 → spin friction  (torsional)
            column 2 → roll friction

        We randomize the slide friction uniformly within ``friction_range``
        and keep spin/roll friction at their default ratios relative to slide.
        We also randomize the restitution (``geom_solref`` / ``geom_solimp``
        energy parameter) for the floor geom.
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
