"""Convertit un checkpoint rsl_rl en modèle Stable-Baselines3 évaluable sur CPU.

Le modèle entraîné sur GPU doit passer **le même banc** que les modèles CPU :
``scripts/evaluate.py`` et ``gpu/criteres.py``, sur ``PoppyHumanoidEnv`` et
MuJoCo C. On écrit donc, à côté du checkpoint :

* ``<nom>.zip`` : un ``PPO`` SB3 de même architecture (``net_arch`` séparés
  ``pi`` / ``vf``), dont on recopie les poids couche par couche, et
  ``log_std`` ;
* ``<nom>_vecnormalize.pkl`` : un ``VecNormalize`` qui porte les statistiques
  d'observation et de retour du checkpoint.

Usage (image GPU, qui a SB3 et Gymnasium) ::

    python -m gpu.export_sb3 logs/gpu/<run>/model_1220.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn

from gpu.poppy_env import reference_env
from src.config import load_yaml, make_poppy_env_config
from src.config.loaders import as_evaluation_config

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


def _copy_mlp(state: dict, hidden: nn.Sequential, head: nn.Linear, n_hidden: int) -> None:
    """Recopie ``mlp.0, mlp.2, …`` (rsl_rl) dans les couches cachées et la tête SB3."""
    linears = [m for m in hidden if isinstance(m, nn.Linear)]
    if len(linears) != n_hidden:
        raise ValueError(f"{len(linears)} couches cachées SB3 pour {n_hidden} rsl_rl")
    for i, layer in enumerate(linears):
        layer.weight.data.copy_(state[f"mlp.{2 * i}.weight"])
        layer.bias.data.copy_(state[f"mlp.{2 * i}.bias"])
    head.weight.data.copy_(state[f"mlp.{2 * n_hidden}.weight"])
    head.bias.data.copy_(state[f"mlp.{2 * n_hidden}.bias"])


def export(checkpoint: Path, base_config: Path, out_stem: Path | None = None) -> tuple[Path, Path]:
    """Écrit le ``.zip`` SB3 et le ``.pkl`` de normalisation d'un checkpoint.

    Args:
        checkpoint: ``model_<it>.pt`` écrit par ``gpu/train.py``.
        base_config: YAML CPU de l'entraînement (``base_config`` du YAML GPU).
        out_stem: Chemin sans extension ; par défaut, celui du checkpoint.

    Returns:
        ``(zip, pkl)``.
    """
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = load_yaml(base_config)
    net = cfg.get("network", {})
    hidden = list(net.get("hidden_layers", [64, 64]))
    config = as_evaluation_config(make_poppy_env_config(cfg))

    venv = DummyVecEnv([lambda: reference_env(config)])
    model = PPO(
        "MlpPolicy", venv, device="cpu",
        policy_kwargs={
            "net_arch": {"pi": hidden, "vf": hidden},
            "activation_fn": _ACTIVATIONS[net.get("activation", "tanh")],
            "log_std_init": float(net.get("log_std_init", 0.0)),
            "ortho_init": False,
        },
    )
    policy = model.policy
    actor, critic = saved["actor_state_dict"], saved["critic_state_dict"]
    _copy_mlp(actor, policy.mlp_extractor.policy_net, policy.action_net, len(hidden))
    _copy_mlp(critic, policy.mlp_extractor.value_net, policy.value_net, len(hidden))
    policy.log_std.data.copy_(actor["distribution.log_std_param"])

    stem = out_stem or checkpoint.with_suffix("")
    zip_path = stem.with_suffix(".zip")
    model.save(str(zip_path))

    norm = saved["infos"]["vec_normalize"]
    vec = VecNormalize(venv, norm_obs=norm["norm_obs"], norm_reward=norm["norm_reward"],
                       clip_obs=norm["clip_obs"], clip_reward=norm["clip_reward"],
                       gamma=norm["gamma"], epsilon=norm["epsilon"])
    for name in ("obs_rms", "ret_rms"):
        rms = getattr(vec, name)
        rms.mean = norm[name]["mean"].numpy().astype(np.float64)
        rms.var = norm[name]["var"].numpy().astype(np.float64)
        rms.count = float(norm[name]["count"])
    pkl_path = stem.parent / f"{stem.name}_vecnormalize.pkl"
    vec.save(str(pkl_path))
    venv.close()

    meta = {"checkpoint": str(checkpoint), "timesteps": saved["infos"].get("timesteps"),
            "iteration": saved.get("iter"),
            "action_std_mean": float(math.exp(float(actor["distribution.log_std_param"].mean())))}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return zip_path, pkl_path


def main() -> None:
    """Exporte un ou plusieurs checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--base-config", type=Path,
                        default=Path("configs/exp_a_ancien_cadre_nouvelle_recompense.yaml"))
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        zip_path, pkl_path = export(checkpoint, args.base_config)
        print(f"{checkpoint} → {zip_path}, {pkl_path}")


if __name__ == "__main__":
    main()
