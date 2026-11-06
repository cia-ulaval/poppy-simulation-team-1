#!/usr/bin/env python3
"""
Training script for the Poppy Humanoid robot with floor domain randomization.

Usage:
    python scripts/train_poppy.py
    python scripts/train_poppy.py --config configs/poppy_robust.yaml
    python scripts/train_poppy.py --config configs/poppy_robust.yaml --timesteps 5000000
    python scripts/train_poppy.py --config configs/poppy_robust.yaml --no-floor-noise
    python scripts/train_poppy.py --config configs/palier1.yaml         --init-from models/2026-09-19_recompense-corrigee/best_model.zip
"""
import argparse
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn as nn
import yaml
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from src.config import (
    EnvironmentConfig,
    as_evaluation_config,
    load_yaml,
    make_poppy_env_config,
)
from src.environments.env_factory import make_baseline_env, make_poppy_env

_ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
}

_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu":  nn.ELU,
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RL on Poppy Humanoid with floor domain randomization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default robust training (PPO, 20M steps)
  python scripts/train_poppy.py

  # Custom config file
  python scripts/train_poppy.py --config configs/poppy_robust.yaml

  # Override timesteps from CLI
  python scripts/train_poppy.py --timesteps 5000000

  # Disable floor noise to measure its impact
  python scripts/train_poppy.py --no-floor-noise
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: configs/poppy_robust.yaml, "
             "or configs/humanoid_baseline.yaml with --baseline)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=list(_ALGO_MAP.keys()),
        default=None,
        help="Override algorithm from YAML",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from YAML",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override n_envs from YAML",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from YAML",
    )
    parser.add_argument(
        "--no-floor-noise",
        action="store_true",
        help="Disable floor domain randomization",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use standard Humanoid-v5 instead of Poppy (for pipeline validation)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Override log directory from YAML",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Repart des poids d'un modèle existant (.zip) au lieu de zéro. "
             "Ses statistiques de normalisation des observations sont reprises "
             "aussi ; celles des récompenses repartent de zéro, la récompense "
             "ayant pu changer.",
    )
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> Path:
    """Choisit le YAML par défaut selon la cible.

    Sans ce choix, ``--baseline`` chargeait ``configs/poppy_robust.yaml``, dont
    le ``healthy_z_range`` vaut [0.25, 0.70] m — la hauteur de bassin de Poppy.
    Le bassin d'Humanoid-v5 est à ~1.4 m : l'épisode se terminait au premier
    pas, et le « test du pipeline » ne testait rien.
    """
    if args.config is not None:
        return args.config
    return Path(
        "configs/humanoid_baseline.yaml" if args.baseline
        else "configs/poppy_robust.yaml"
    )


def build_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI overrides (CLI takes priority)."""
    if args.algorithm is not None:
        cfg["algorithm"] = args.algorithm
    if args.timesteps is not None:
        cfg["training"]["total_timesteps"] = args.timesteps
    if args.n_envs is not None:
        cfg["environment"]["n_envs"] = args.n_envs
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.no_floor_noise:
        cfg["domain_randomization"]["enabled"] = False
    if args.log_dir is not None:
        cfg["training"]["log_dir"] = str(args.log_dir)
    return cfg


# La construction de la configuration d'environnement vit dans
# src/config/loaders.py : l'évaluation doit lire le YAML exactement pareil.


def make_policy_kwargs(cfg: dict, algo_name: str) -> dict:
    net = cfg.get("network", {})
    layers = net.get("hidden_layers", [256, 256])
    activation = _ACTIVATION_MAP.get(net.get("activation", "relu"), nn.ReLU)
    is_on_policy = algo_name in ("PPO", "A2C")
    if is_on_policy:
        kwargs = {
            "net_arch": {"pi": layers, "vf": layers},
            "activation_fn": activation,
        }
    else:
        kwargs = {
            "net_arch": {"pi": layers, "qf": layers},
            "activation_fn": activation,
        }
    # Optional policy init params (critical for humanoid tasks)
    if "log_std_init" in net:
        kwargs["log_std_init"] = float(net["log_std_init"])
    if "ortho_init" in net:
        kwargs["ortho_init"] = bool(net["ortho_init"])
    return kwargs


def linear_schedule(initial: float):
    """Taux qui décroît linéairement de ``initial`` à 0 sur l'entraînement.

    Stable-Baselines3 appelle le schéma avec la part restante de
    l'entraînement : 1 au premier pas, 0 au dernier.
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial
    return schedule


def make_algo_kwargs(cfg: dict, algo_name: str) -> dict:
    key = algo_name.lower()
    algo_cfg = dict(cfg.get(key, {}))
    # « linear » : learning_rate est la valeur de départ, qui décroît jusqu'à 0.
    schedule = algo_cfg.pop("learning_rate_schedule", "constant")
    # Convert learning_rate from string like "1.0e-4" → float
    if "learning_rate" in algo_cfg:
        algo_cfg["learning_rate"] = float(algo_cfg["learning_rate"])
        if schedule == "linear":
            algo_cfg["learning_rate"] = linear_schedule(algo_cfg["learning_rate"])
        elif schedule != "constant":
            raise ValueError(
                f"learning_rate_schedule inconnu : {schedule!r} (constant | linear)"
            )
    return algo_cfg


def find_vec_normalize(model_path: Path) -> Path:
    """Retrouve le ``vec_normalize.pkl`` rangé à côté d'un modèle."""
    for name in ("vec_normalize.pkl", "vec_normalize_final.pkl"):
        candidate = model_path.parent / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Pas de statistiques de normalisation à côté de {model_path}. Sans "
        f"elles, la politique reprise verrait des observations mal centrées."
    )


def load_pretrained(
    algo_class,
    model_path: Path,
    train_env: VecNormalize,
    algo_kwargs: dict,
    **kwargs,
):
    """Charge un modèle existant pour continuer à l'entraîner.

    Les hyperparamètres du YAML remplacent ceux enregistrés dans le modèle.
    Les espaces enregistrés sont remplacés par ceux de l'environnement : la
    forme est vérifiée à la main, seules les bornes de l'action peuvent
    différer (espace d'action normalisé ou non, docs/TECHNIQUE.md §8).

    Les statistiques d'observation du modèle sont recopiées dans
    ``train_env`` : la politique a appris sur des observations centrées sur
    cette moyenne-là.
    """
    with open(find_vec_normalize(model_path), "rb") as handle:
        previous = pickle.load(handle)
    if previous.obs_rms.mean.shape != train_env.observation_space.shape:
        raise ValueError(
            f"Observation de {previous.obs_rms.mean.shape} dimensions dans "
            f"{model_path}, {train_env.observation_space.shape} ici."
        )
    train_env.obs_rms = previous.obs_rms

    custom_objects = {
        "observation_space": train_env.observation_space,
        "action_space": train_env.action_space,
        # Les schémas enregistrés sont des fonctions sérialisées, fragiles
        # d'une version de Python à l'autre ; ils sont reconstruits depuis
        # les kwargs de toute façon.
        "lr_schedule": lambda _: 0.0,
        "learning_rate": algo_kwargs.get("learning_rate", 3e-4),
        "clip_range": algo_kwargs.get("clip_range", 0.2),
    }
    return algo_class.load(
        str(model_path),
        env=train_env,
        custom_objects=custom_objects,
        **algo_kwargs,
        **kwargs,
    )


def main() -> int:
    args = parse_args()

    # Load YAML
    config_path = resolve_config_path(args)
    cfg = load_yaml(config_path)
    cfg = build_config(cfg, args)

    algo_name  = cfg.get("algorithm", "PPO").upper()
    train_cfg  = cfg.get("training", {})
    total_ts   = train_cfg.get("total_timesteps", 20_000_000)
    seed       = train_cfg.get("seed", 42)
    ckpt_freq  = train_cfg.get("checkpoint_freq", 200_000)
    eval_freq  = train_cfg.get("eval_freq", 10_000)
    n_eval_ep  = train_cfg.get("n_eval_episodes", 5)

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir    = Path(train_cfg.get("log_dir", "./logs/poppy")) / timestamp
    figs_dir   = Path(cfg.get("figs_dir", "./figs/poppy"))
    log_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # La configuration effective (YAML + options de la ligne de commande) est
    # gardée à côté du modèle : sans elle, impossible de savoir plus tard avec
    # quelle récompense il a été entraîné.
    with open(log_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {**cfg, "_source": str(config_path),
             "_init_from": str(args.init_from) if args.init_from else None},
            handle, allow_unicode=True, sort_keys=False,
        )

    dr_enabled = cfg.get("domain_randomization", {}).get("enabled", True)

    print("\n" + "=" * 70)
    print("  Poppy Humanoid — Robust Training")
    print("=" * 70)
    print(f"  Algorithm       : {algo_name}")
    print(f"  Total timesteps : {total_ts:,}")
    print(f"  Seed            : {seed}")
    print(f"  Config          : {config_path}")
    print(f"  Log dir         : {log_dir}")
    print(f"  Init from       : {args.init_from or 'zéro'}")
    print(f"  Floor noise     : {'ON' if dr_enabled else 'OFF'}")
    if dr_enabled:
        dr = cfg.get("domain_randomization", {})
        print(f"    friction range    = {dr.get('friction_range')}")
        print(f"    restitution range = {dr.get('restitution_range')}")
    print("=" * 70 + "\n")

    # ── Build environments ──────────────────────────────────────────────
    use_baseline = args.baseline
    env_cfg = cfg.get("environment", {})
    n_envs = env_cfg.get("n_envs", 16)

    if use_baseline:
        # Standard Humanoid-v5 (no Poppy XML)
        base_config = EnvironmentConfig(
            env_id=env_cfg.get("env_id", "Humanoid-v5"),
            terminate_when_unhealthy=env_cfg.get("terminate_when_unhealthy", True),
            healthy_z_range=tuple(env_cfg.get("healthy_z_range", [1.0, 2.0])),
            n_envs=n_envs,
            normalize_obs=env_cfg.get("normalize_obs", True),
            normalize_reward=env_cfg.get("normalize_reward", True),
            clip_obs=env_cfg.get("clip_obs", 10.0),
            gamma=env_cfg.get("gamma", 0.99),
        )
        train_env = make_baseline_env(base_config, n_envs=n_envs, seed=seed)
        eval_env = make_baseline_env(
            base_config, n_envs=1, seed=seed + 1000, use_subprocess=False,
        )
        eval_env.training = False     # Don't update normalization stats during eval
        eval_env.norm_reward = False  # Show true rewards
        print(f"  Env               : {base_config.env_id} (baseline)")
    else:
        # Custom Poppy Humanoid
        env_config = make_poppy_env_config(cfg)
        train_env = make_poppy_env(config=env_config, seed=seed)
        # Même dérivation que scripts/evaluate.py : un environnement, pas de
        # normalisation des récompenses, pas de randomisation du sol.
        eval_env_config = as_evaluation_config(env_config)
        eval_env = make_poppy_env(
            config=eval_env_config,
            n_envs=1,
            seed=seed + 1000,
            use_subprocess=False,
        )
        eval_env.training = False     # Don't update normalization stats during eval
        eval_env.norm_reward = False  # Show true rewards
        print("  Env               : PoppyHumanoid-v0")

    # ── Build model ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device          : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    AlgoClass    = _ALGO_MAP[algo_name]
    policy_kwargs = make_policy_kwargs(cfg, algo_name)
    algo_kwargs  = make_algo_kwargs(cfg, algo_name)

    if args.init_from is not None:
        if use_baseline:
            raise SystemExit("--init-from ne s'applique pas à --baseline.")
        # policy_kwargs n'est pas transmis : l'architecture est celle du
        # modèle repris, pas celle du YAML.
        model = load_pretrained(
            AlgoClass,
            args.init_from,
            train_env,
            algo_kwargs,
            seed=seed,
            verbose=1,
            tensorboard_log=str(log_dir),
            device=device,
        )
    else:
        model = AlgoClass(
            "MlpPolicy",
            train_env,
            seed=seed,
            verbose=1,
            tensorboard_log=str(log_dir),
            policy_kwargs=policy_kwargs,
            device=device,
            **algo_kwargs,
        )

    # ── Callbacks ───────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(ckpt_freq // n_envs, 1),
        save_path=str(log_dir),
        name_prefix=f"poppy_{algo_name.lower()}",
        save_vecnormalize=True,
    )
    # Custom EvalCallback that:
    # 1. Syncs VecNormalize stats from train → eval before each evaluation
    # 2. Saves VecNormalize with best model
    class SyncedEvalCallback(EvalCallback):
        """EvalCallback with VecNormalize sync + save."""
        def _on_step(self) -> bool:
            # Sync normalization stats before eval runs
            if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
                sync_envs_normalization(self.training_env, self.eval_env)

            prev_best = self.best_mean_reward
            result = super()._on_step()

            # Save VecNormalize alongside best model
            if self.best_mean_reward > prev_best and self.best_model_save_path is not None:
                vec_norm_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")
                if isinstance(self.training_env, VecNormalize):
                    self.training_env.save(vec_norm_path)
                    if self.verbose >= 1:
                        print(f"  Saved VecNormalize → {vec_norm_path}")
            return result

    eval_cb = SyncedEvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_ep,
        deterministic=True,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # ── Train ───────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps=total_ts,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # ── Save final model ────────────────────────────────────────────────
    final_path = log_dir / f"poppy_{algo_name.lower()}_final"
    model.save(str(final_path))
    train_env.save(str(log_dir / "vec_normalize_final.pkl"))
    print(f"\nModel saved → {final_path}.zip")
    print(f"Training time: {elapsed / 60:.1f} min")
    print(f"\nTensorBoard:\n  tensorboard --logdir={log_dir}\n")

    train_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
