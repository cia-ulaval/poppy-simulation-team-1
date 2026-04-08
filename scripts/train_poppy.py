#!/usr/bin/env python3
"""
Training script for the Poppy Humanoid robot with floor domain randomization.

Usage:
    python scripts/train_poppy.py
    python scripts/train_poppy.py --config configs/poppy_robust.yaml
    python scripts/train_poppy.py --config configs/poppy_robust.yaml --timesteps 5000000
    python scripts/train_poppy.py --config configs/poppy_robust.yaml --no-floor-noise
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

import torch.nn as nn

from src.config import (
    DomainRandomizationConfig,
    PoppyEnvironmentConfig,
    EnvironmentConfig,
)
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from src.environments.env_factory import HumanoidEnvFactory
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv


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


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
        default=Path("configs/poppy_robust.yaml"),
        help="YAML config file (default: configs/poppy_robust.yaml)",
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
    return parser.parse_args()


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


def make_env_config(cfg: dict) -> PoppyEnvironmentConfig:
    env_cfg = cfg.get("environment", {})
    dr_cfg  = cfg.get("domain_randomization", {})
    return PoppyEnvironmentConfig(
        terminate_when_unhealthy=env_cfg.get("terminate_when_unhealthy", True),
        healthy_z_range=tuple(env_cfg.get("healthy_z_range", [0.25, 0.70])),
        n_envs=env_cfg.get("n_envs", 16),
        normalize_obs=env_cfg.get("normalize_obs", True),
        normalize_reward=env_cfg.get("normalize_reward", False),
        clip_obs=env_cfg.get("clip_obs", 10.0),
        gamma=env_cfg.get("gamma", 0.99),
        frame_skip=env_cfg.get("frame_skip", 5),
        domain_randomization=DomainRandomizationConfig(
            enabled=dr_cfg.get("enabled", True),
            friction_range=tuple(dr_cfg.get("friction_range", [0.5, 1.5])),
            restitution_range=tuple(dr_cfg.get("restitution_range", [0.0, 0.3])),
        ),
    )


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


def make_algo_kwargs(cfg: dict, algo_name: str) -> dict:
    key = algo_name.lower()
    algo_cfg = cfg.get(key, {})
    # Convert learning_rate from string like "1.0e-4" → float
    if "learning_rate" in algo_cfg:
        algo_cfg["learning_rate"] = float(algo_cfg["learning_rate"])
    return algo_cfg


def main() -> int:
    args = parse_args()

    # Load YAML
    cfg = load_yaml(args.config)
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

    dr_enabled = cfg.get("domain_randomization", {}).get("enabled", True)

    print("\n" + "=" * 70)
    print("  Poppy Humanoid — Robust Training")
    print("=" * 70)
    print(f"  Algorithm       : {algo_name}")
    print(f"  Total timesteps : {total_ts:,}")
    print(f"  Seed            : {seed}")
    print(f"  Log dir         : {log_dir}")
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
        factory = HumanoidEnvFactory(base_config)
        train_env = factory.create_training_env(
            n_envs=n_envs, seed=seed, use_subprocess=True,
        )
        eval_env = factory.create_eval_env(seed=seed + 1000)
        print(f"  Env               : {base_config.env_id} (baseline)")
    else:
        # Custom Poppy Humanoid
        env_config = make_env_config(cfg)
        train_env = HumanoidEnvFactory.create_poppy_training_env(
            config=env_config,
            seed=seed,
        )
        eval_env_config = PoppyEnvironmentConfig(
            terminate_when_unhealthy=env_config.terminate_when_unhealthy,
            healthy_z_range=env_config.healthy_z_range,
            n_envs=1,
            normalize_obs=env_config.normalize_obs,
            normalize_reward=False,
            clip_obs=env_config.clip_obs,
            gamma=env_config.gamma,
            frame_skip=env_config.frame_skip,
            domain_randomization=DomainRandomizationConfig(enabled=False),
        )
        eval_env = HumanoidEnvFactory.create_poppy_training_env(
            config=eval_env_config,
            n_envs=1,
            seed=seed + 1000,
            use_subprocess=False,
        )
        eval_env.training = False     # Don't update normalization stats during eval
        eval_env.norm_reward = False  # Show true rewards
        print(f"  Env               : PoppyHumanoid-v0")

    # ── Build model ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device          : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    AlgoClass    = _ALGO_MAP[algo_name]
    policy_kwargs = make_policy_kwargs(cfg, algo_name)
    algo_kwargs  = make_algo_kwargs(cfg, algo_name)

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
