"""Miroir gauche ↔ droite pour l'extension de symétrie de rsl_rl.

Mêmes tables que ``MirrorPPO`` : elles sont **calculées** par
``src.environments.mirror.build_mirror_spec`` sur la ``PoppyHumanoidEnv`` de
même configuration (paires ``r_…`` / ``l_…``, signes tirés des axes, horloge
décalée d'un demi-cycle). Rien n'est réécrit à la main.

Comme ``MirrorPPO._mirror_obs``, le reflet porte sur l'observation **brute** :
on dénormalise avec les statistiques de ``VecNormalizeWrapper``, on reflète,
on renormalise et on borne. L'action est reflétée en passant par les angles
cibles, à cause des courses asymétriques autour de la pose debout.

rsl_rl appelle :func:`mirror_batch` avec ``obs`` et/ou ``actions`` et attend
en retour l'original suivi du reflet, concaténés sur la première dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

from gpu.poppy_env import reference_env
from gpu.vecnormalize import OBS_GROUP, VecNormalizeWrapper
from src.environments.mirror import build_mirror_spec


@dataclass(frozen=True)
class MirrorTables:
    """Tables du miroir, en tenseurs sur le périphérique de l'environnement."""

    obs_perm: torch.Tensor     # (D,) indices
    obs_sign: torch.Tensor     # (D,)
    joint_perm: torch.Tensor   # (25,) indices
    joint_sign: torch.Tensor   # (25,)
    joint_init: torch.Tensor   # (25,) rad
    range_up: torch.Tensor     # (25,) rad
    range_down: torch.Tensor   # (25,) rad


def mirror_tables(env: VecNormalizeWrapper) -> MirrorTables:
    """Tables du miroir de l'environnement, calculées une fois puis gardées."""
    cached = getattr(env, "_mirror_tables", None)
    if cached is not None:
        return cached
    config = env.unwrapped.action_manager.get_term("joints").cfg.config
    ref = reference_env(config)
    try:
        spec = build_mirror_spec(ref)
    finally:
        ref.close()

    def t(values, dtype=torch.float32) -> torch.Tensor:  # noqa: ANN001
        return torch.as_tensor(values, dtype=dtype, device=env.device)

    tables = MirrorTables(
        obs_perm=t(spec.obs_perm, torch.long), obs_sign=t(spec.obs_sign),
        joint_perm=t(spec.joint_perm, torch.long), joint_sign=t(spec.joint_sign),
        joint_init=t(spec.joint_init), range_up=t(spec.range_up),
        range_down=t(spec.range_down),
    )
    env._mirror_tables = tables  # type: ignore[attr-defined]
    return tables


def mirror_obs(env: VecNormalizeWrapper, obs: torch.Tensor) -> torch.Tensor:
    """Observation normalisée ``(n, D)`` vue dans le miroir, normalisée à son tour."""
    m = mirror_tables(env)
    if not env.norm_obs:
        return obs[:, m.obs_perm] * m.obs_sign
    assert env.obs_rms is not None
    mean = env.obs_rms.mean.to(torch.float32)
    std = torch.sqrt(env.obs_rms.var + env.epsilon).to(torch.float32)
    raw = obs * std + mean
    mirrored = raw[:, m.obs_perm] * m.obs_sign
    return torch.clamp((mirrored - mean) / std, -env.clip_obs, env.clip_obs)


def mirror_action(env: VecNormalizeWrapper, action: torch.Tensor) -> torch.Tensor:
    """Action ``(n, 25)`` vue dans le miroir, en passant par les angles cibles."""
    m = mirror_tables(env)
    target = m.joint_init + torch.where(action >= 0, action * m.range_up, action * m.range_down)
    delta = target[:, m.joint_perm] * m.joint_sign - m.joint_init
    return torch.where(delta >= 0, delta / m.range_up, delta / m.range_down)


def mirror_batch(
    env: VecNormalizeWrapper,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Fonction d'augmentation pour ``rsl_rl.extensions.Symmetry``.

    Returns:
        ``(obs, actions)`` de taille double : originaux puis reflets.
    """
    obs_out = None
    if obs is not None:
        policy = obs[OBS_GROUP]
        obs_out = TensorDict(
            {OBS_GROUP: torch.cat([policy, mirror_obs(env, policy)], dim=0)},
            batch_size=[policy.shape[0] * 2],
        )
    act_out = None
    if actions is not None:
        act_out = torch.cat([actions, mirror_action(env, actions)], dim=0)
    return obs_out, act_out
