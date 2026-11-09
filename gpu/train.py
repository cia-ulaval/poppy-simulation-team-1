"""Entraîne Poppy sur GPU : mjlab (MuJoCo Warp) et rsl_rl, réglés comme le run SB3.

Usage (image GPU) ::

    python -m gpu.train --config gpu/configs/exp_a.yaml
    python -m gpu.train --config gpu/configs/exp_a.yaml --num-envs 64 --timesteps 200000

L'environnement, la récompense et la randomisation viennent du YAML CPU
désigné par ``base_config``. Ses sections ``ppo`` et ``network`` sont
traduites en réglages rsl_rl, terme à terme (voir :func:`runner_cfg`).
Les checkpoints ``model_<itération>.pt`` contiennent les statistiques de
normalisation ; ``gpu/export_sb3.py`` les convertit en ``.zip`` SB3.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.runners import OnPolicyRunner

from gpu.poppy_env import make_env_cfg
from gpu.vecnormalize import OBS_GROUP, VecNormalizeWrapper
from src.config import load_yaml, make_poppy_env_config

_ACTIVATIONS = {"relu": "relu", "tanh": "tanh", "elu": "elu"}


def load_gpu_config(path: Path) -> tuple[dict, dict]:
    """Charge un YAML de ``gpu/configs/`` et le YAML CPU qu'il désigne.

    Returns:
        ``(gpu_cfg, base_cfg)``.
    """
    gpu_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_cfg = load_yaml(Path(gpu_cfg["base_config"]))
    return gpu_cfg, base_cfg


def runner_cfg(
    base_cfg: dict, gpu_cfg: dict, num_envs: int, num_steps: int, iterations: int, seed: int
) -> dict:
    """Traduit les sections ``ppo`` et ``network`` du YAML CPU pour rsl_rl.

    Correspondances avec ``stable_baselines3.PPO`` :

    * ``batch_size`` (taille de mini-lot) → ``num_mini_batches`` =
      transitions par mise à jour / ``batch_size`` ;
    * ``n_epochs`` → ``num_learning_epochs`` ; ``clip_range`` →
      ``clip_param`` ; ``vf_coef`` → ``value_loss_coef`` (même perte
      quadratique moyenne, non bornée) ; ``ent_coef`` → ``entropy_coef`` ;
    * taux constant → ``schedule: fixed`` ; avantages normalisés par
      mini-lot, comme SB3 ;
    * ``log_std_init`` → écart type initial ``exp(log_std_init)``,
      paramétré en log comme SB3 ; réseaux séparés pour la politique et la
      valeur, comme ``net_arch: {pi, vf}`` ;
    * norme de gradient bornée globalement et Adam à ``eps = 1e-5`` :
      :class:`gpu.ppo.SB3PPO` ;
    * ``target_kl`` (arrêt anticipé de SB3) et ``symmetry_loss_weight``
      (passe miroir de ``MirrorPPO``) : :class:`gpu.ppo.SB3PPO` aussi.
    """
    ppo = base_cfg["ppo"]
    net = base_cfg.get("network", {})
    if isinstance(ppo.get("learning_rate"), str) or ppo.get("lr_schedule", "constant") != "constant":
        raise NotImplementedError("Seul un taux d'apprentissage constant est porté.")
    batch = num_envs * num_steps
    if batch % int(ppo["batch_size"]):
        raise ValueError(f"{batch} transitions ne se découpent pas en mini-lots de {ppo['batch_size']}.")
    hidden = list(net.get("hidden_layers", [64, 64]))
    activation = _ACTIVATIONS[net.get("activation", "tanh")]
    return {
        "seed": seed,
        "num_steps_per_env": num_steps,
        "max_iterations": iterations,
        "save_interval": max(1, round(gpu_cfg["gpu"].get("checkpoint_freq", 500_000) / batch)),
        "obs_groups": {"actor": [OBS_GROUP], "critic": [OBS_GROUP]},
        "logger": "tensorboard",
        "check_for_nan": True,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": hidden,
            "activation": activation,
            "obs_normalization": False,  # fait par VecNormalizeWrapper, comme SB3
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": math.exp(float(net.get("log_std_init", 0.0))),
                "std_type": "log",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": hidden,
            "activation": activation,
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "gpu.ppo:SB3PPO",
            "num_learning_epochs": int(ppo["n_epochs"]),
            "num_mini_batches": batch // int(ppo["batch_size"]),
            "clip_param": float(ppo["clip_range"]),
            "gamma": float(ppo["gamma"]),
            "lam": float(ppo["gae_lambda"]),
            "value_loss_coef": float(ppo["vf_coef"]),
            "entropy_coef": float(ppo["ent_coef"]),
            "learning_rate": float(ppo["learning_rate"]),
            "max_grad_norm": float(ppo["max_grad_norm"]),
            "use_clipped_value_loss": False,
            "schedule": "fixed",
            "desired_kl": None,
            "normalize_advantage_per_mini_batch": True,
            "optimizer": "adam",
            "rnd_cfg": None,
            "symmetry_cfg": None,  # pas la symétrie de rsl_rl : celle de MirrorPPO
            "target_kl": None if ppo.get("target_kl") is None else float(ppo["target_kl"]),
            "sb3_symmetry_weight": float(ppo.get("symmetry_loss_weight", 0.0)),
        },
    }


class PoppyRunner(OnPolicyRunner):
    """``OnPolicyRunner`` qui range la normalisation dans chaque checkpoint."""

    def save(self, path: str, infos: dict | None = None) -> None:
        infos = dict(infos or {})
        infos["vec_normalize"] = self.env.state_dict()
        infos["timesteps"] = ((self.current_learning_iteration + 1)
                              * self.env.num_envs * self.cfg["num_steps_per_env"])
        super().save(path, infos)


def build(
    gpu_cfg: dict, base_cfg: dict, num_envs: int, seed: int, device: str
) -> VecNormalizeWrapper:
    """Construit l'environnement mjlab enveloppé, prêt pour rsl_rl."""
    env_config = make_poppy_env_config(base_cfg)
    env = ManagerBasedRlEnv(make_env_cfg(env_config, num_envs=num_envs, seed=seed), device=device)
    env_section = base_cfg.get("environment", {})
    return VecNormalizeWrapper(
        env,
        norm_obs=bool(env_section.get("normalize_obs", True)),
        norm_reward=bool(env_section.get("normalize_reward", False)),
        clip_obs=float(env_section.get("clip_obs", 10.0)),
        gamma=float(env_section.get("gamma", 0.99)),
    )


def parse_args() -> argparse.Namespace:
    """Analyse la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraîne Poppy sur GPU (mjlab + rsl_rl).")
    parser.add_argument("--config", type=Path, default=Path("gpu/configs/exp_a.yaml"))
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps-per-env", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Pas au total (défaut : training.total_timesteps du YAML CPU)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    """Entraîne et écrit les checkpoints, la configuration et le débit mesuré."""
    args = parse_args()
    gpu_cfg, base_cfg = load_gpu_config(args.config)
    num_envs = args.num_envs or int(gpu_cfg["gpu"]["num_envs"])
    num_steps = args.steps_per_env or int(gpu_cfg["gpu"]["num_steps_per_env"])
    timesteps = args.timesteps or int(base_cfg["training"]["total_timesteps"])
    seed = args.seed if args.seed is not None else int(base_cfg["training"].get("seed", 42))
    iterations = math.ceil(timesteps / (num_envs * num_steps))
    log_dir = args.log_dir or Path("logs/gpu") / f"{args.config.stem}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    env = build(gpu_cfg, base_cfg, num_envs, seed, args.device)
    cfg = runner_cfg(base_cfg, gpu_cfg, num_envs, num_steps, iterations, seed)
    (log_dir / "config.json").write_text(json.dumps(
        {"gpu_config": str(args.config), "base_config": gpu_cfg["base_config"],
         "num_envs": num_envs, "num_steps_per_env": num_steps, "iterations": iterations,
         "seed": seed, "runner": cfg}, indent=2, default=str), encoding="utf-8")

    runner = PoppyRunner(env, cfg, log_dir=str(log_dir), device=args.device)
    start = time.time()
    runner.learn(num_learning_iterations=iterations)
    elapsed = time.time() - start
    done = iterations * num_envs * num_steps
    summary = {"timesteps": done, "seconds": elapsed, "steps_per_second": done / elapsed,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (log_dir / "throughput.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{done} pas en {elapsed / 60:.1f} min, {done / elapsed:.0f} pas/s — {log_dir}")


if __name__ == "__main__":
    main()
