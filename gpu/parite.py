"""Parité pas à pas entre ``PoppyHumanoidEnv`` (MuJoCo C) et le portage mjlab (MuJoCo Warp).

Deux contrôles, sans entraînement :

1. **Modèle** : la scène compilée par mjlab porte exactement la physique du
   modèle CPU (masses, inerties, axes, butées, amortissements, contacts,
   moteurs, options du solveur, RK4 compris).
2. **Dynamique et récompense** : depuis le même état initial, la même suite
   d'actions donne, pas à pas, le même état, la même observation et les
   mêmes termes de récompense. La simulation est chaotique (contacts) et
   MuJoCo Warp calcule en float32 : on mesure l'écart sur les premiers pas,
   là où il doit rester à l'échelle des arrondis, et on montre comment il
   croît ensuite.

Usage (image GPU) ::

    python -m gpu.parite --config configs/exp_a_ancien_cadre_nouvelle_recompense.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from gpu.poppy_env import ROBOT, make_env_cfg, option_fields, poppy_state, reference_env
from src.config import load_yaml, make_poppy_env_config
from src.config.loaders import as_evaluation_config
from src.config.settings import PoppyEnvironmentConfig

# Champs du modèle qui font la physique. Les noms changent (préfixe robot/),
# pas l'ordre des corps, articulations, géométries et moteurs.
MODEL_FIELDS = (
    "body_parentid", "body_mass", "body_inertia", "body_pos", "body_quat",
    "body_ipos", "body_iquat", "jnt_type", "jnt_axis", "jnt_pos", "jnt_range",
    "jnt_limited", "dof_damping", "dof_armature", "dof_frictionloss",
    "geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat",
    "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_contype",
    "geom_conaffinity", "geom_condim", "geom_margin", "geom_gap", "geom_priority",
    "mesh_vertnum", "actuator_trnid", "actuator_gear", "actuator_ctrlrange",
    "actuator_ctrllimited", "actuator_gaintype", "actuator_biastype", "qpos0",
)

# Termes comparés à chaque pas, sous les noms de l'info CPU.
STEP_TERMS = (
    "forward_vel", "lateral_vel", "uprightness", "torso_uprightness", "yaw_rate",
    "healthy_reward", "gait_reward", "ctrl_cost", "action_rate_cost",
    "joint_vel_cost", "air_time_reward", "yaw_rate_cost", "joint_limit_cost",
    "vel_tracking_reward", "torso_tilt_cost", "base_height_cost",
    "ang_vel_xy_cost", "feet_slide_cost", "phase_contact_reward", "imitation_reward",
)


def compare_models(cpu: mujoco.MjModel, gpu: mujoco.MjModel) -> list[str]:
    """Liste les champs physiques qui diffèrent entre deux modèles compilés.

    Returns:
        Une ligne par écart ; vide si les deux modèles sont identiques.
    """
    problems = []
    for name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh"):
        if getattr(cpu, name) != getattr(gpu, name):
            problems.append(f"{name} : {getattr(cpu, name)} contre {getattr(gpu, name)}")
    if problems:
        return problems
    for name in MODEL_FIELDS:
        a, b = np.asarray(getattr(cpu, name)), np.asarray(getattr(gpu, name))
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-9):
            diff = np.max(np.abs(a.astype(float) - b.astype(float))) if a.shape == b.shape else "forme"
            problems.append(f"{name} : écart max {diff}")
    cpu_opt, gpu_opt = option_fields(cpu.opt), option_fields(gpu.opt)
    for key, value in cpu_opt.items():
        if not np.allclose(np.asarray(value, float), np.asarray(gpu_opt[key], float)):
            problems.append(f"opt.{key} : {value} contre {gpu_opt[key]}")
    return problems


def _sync_initial_state(env: ManagerBasedRlEnv, qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Place tous les mondes dans l'état CPU donné, comme juste après un reset."""
    data = env.sim.data
    data.qpos[:] = torch.as_tensor(qpos, dtype=torch.float32, device=env.device)
    data.qvel[:] = torch.as_tensor(qvel, dtype=torch.float32, device=env.device)
    data.xfrc_applied[:] = 0.0
    env.sim.forward()
    poppy_state(env).reset(slice(None))


def run_dynamics(
    config: PoppyEnvironmentConfig, steps: int, seed: int, num_envs: int = 2
) -> dict[str, np.ndarray]:
    """Déroule la même suite d'actions sur CPU et sur GPU, sans randomisation.

    Les actions sont tirées dans ``[-0.3, 0.3]`` autour de la pose debout :
    assez pour faire bouger toutes les articulations et poser les pieds,
    pas assez pour que le robot tombe en quelques pas.

    Args:
        config: Configuration d'environnement (la randomisation est coupée).
        steps: Nombre de pas de 10 ms.
        seed: Graine du reset CPU et des actions.
        num_envs: Mondes GPU ; tous reçoivent le même état et la même action.

    Returns:
        Par grandeur, l'écart absolu pas par pas ``(steps,)``, plus
        ``qpos_cpu`` et ``qpos_gpu`` ``(steps, nq)`` pour inspection.
    """
    config = as_evaluation_config(config)
    cpu = reference_env(config)
    obs_cpu, _ = cpu.reset(seed=seed)

    env = ManagerBasedRlEnv(make_env_cfg(config, num_envs=num_envs, seed=seed), device="cuda:0")
    env.reset()
    _sync_initial_state(env, cpu.data.qpos.copy(), cpu.data.qvel.copy())
    obs_gpu = env.observation_manager.compute(update_history=True)["policy"][0].cpu().numpy()

    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {k: [] for k in ("qpos", "qvel", "obs", "reward", *STEP_TERMS)}
    out["obs0"] = [float(np.max(np.abs(obs_cpu - obs_gpu)))]
    qpos_cpu, qpos_gpu = [], []
    for _ in range(steps):
        action = rng.uniform(-0.3, 0.3, size=25)
        obs_c, rew_c, term_c, _, info_c = cpu.step(action)
        batch = torch.as_tensor(action, dtype=torch.float32, device=env.device).repeat(num_envs, 1)
        obs_g, rew_g, term_g, _, _ = env.step(batch)
        info_g = {k: v[0].item() for k, v in poppy_state(env).info.items()}
        qp_g = env.sim.data.qpos[0].cpu().numpy()
        qv_g = env.sim.data.qvel[0].cpu().numpy()
        out["qpos"].append(float(np.max(np.abs(cpu.data.qpos - qp_g))))
        out["qvel"].append(float(np.max(np.abs(cpu.data.qvel - qv_g))))
        out["obs"].append(float(np.max(np.abs(obs_c - obs_g["policy"][0].cpu().numpy()))))
        out["reward"].append(abs(rew_c - rew_g[0].item()))
        for key in STEP_TERMS:
            out[key].append(abs(float(info_c[key]) - info_g[key]))
        qpos_cpu.append(cpu.data.qpos.copy())
        qpos_gpu.append(qp_g)
        if term_c or bool(term_g[0]):
            break
    env.close()
    cpu.close()
    result = {k: np.asarray(v) for k, v in out.items()}
    result["qpos_cpu"] = np.asarray(qpos_cpu)
    result["qpos_gpu"] = np.asarray(qpos_gpu)
    return result


def check_reset_randomization(config: PoppyEnvironmentConfig, num_envs: int = 512) -> dict:
    """Statistiques de la randomisation de domaine au reset, côté GPU.

    Returns:
        Bornes observées de la friction du sol, de ``solimp[4]``, des
        rapports de masse et du lacet initial, à comparer aux plages du YAML.
    """
    env = ManagerBasedRlEnv(make_env_cfg(config, num_envs=num_envs, seed=0), device="cuda:0")
    env.reset()
    s = poppy_state(env)
    model, data = env.sim.model, env.sim.data
    friction = model.geom_friction[:, s.floor_geom_id].cpu().numpy()
    solimp = model.geom_solimp[:, s.floor_geom_id, 4].cpu().numpy()
    mass = (model.body_mass[:, 1:] / s.init_body_mass[1:]).cpu().numpy()
    quat = data.qpos[:, 3:7].cpu().numpy()
    yaw = 2.0 * np.arctan2(quat[:, 3], quat[:, 0])
    env.close()
    return {
        "friction": (friction[:, 0].min(), friction[:, 0].max()),
        "friction_ratio": (np.max(np.abs(friction[:, 1] / friction[:, 0] - 0.005)),
                           np.max(np.abs(friction[:, 2] / friction[:, 0] - 0.0001))),
        "solimp4": (solimp.min(), solimp.max()),
        "mass_ratio": (mass.min(), mass.max()),
        "yaw": (yaw.min(), yaw.max()),
    }


def main() -> None:
    """Affiche les trois contrôles de parité."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path,
                        default=Path("configs/exp_a_ancien_cadre_nouvelle_recompense.yaml"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    config = make_poppy_env_config(load_yaml(args.config))

    cpu = reference_env(config)
    env = ManagerBasedRlEnv(make_env_cfg(config, num_envs=2, seed=0), device="cuda:0")
    problems = compare_models(cpu.model, env.sim.mj_model)
    print(f"Entité mjlab : {ROBOT}, {env.sim.mj_model.nbody} corps, "
          f"intégrateur {env.sim.mj_model.opt.integrator} (RK4 = {int(mujoco.mjtIntegrator.mjINT_RK4)})")
    print("Modèle :", "identique" if not problems else "\n  " + "\n  ".join(problems))
    env.close()
    cpu.close()

    result = run_dynamics(config, args.steps, args.seed)
    n = len(result["qpos"])
    print(f"\nDynamique, {n} pas (écart absolu max, CPU float64 contre GPU float32)")
    print(f"  observation au reset : {result['obs0'][0]:.2e}")
    for window in (1, 5, 20, 50, 100, 200):
        if window > n:
            break
        print(f"  pas 1–{window:<3} qpos {result['qpos'][:window].max():.2e}  "
              f"qvel {result['qvel'][:window].max():.2e}  "
              f"récompense {result['reward'][:window].max():.2e}")
    print("\nTermes de récompense, écart max sur les 20 premiers pas")
    for key in STEP_TERMS:
        print(f"  {key:<22} {result[key][:20].max():.2e}")

    print("\nRandomisation au reset (512 mondes)")
    for key, (lo, hi) in check_reset_randomization(config).items():
        print(f"  {key:<16} [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
