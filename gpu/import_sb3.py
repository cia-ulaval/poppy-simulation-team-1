"""Modèle SB3 → acteur et critique rsl_rl : le ``--init-from`` de ``train_poppy.py`` sur GPU.

``scripts/train_poppy.py --init-from <modele.zip>`` (``load_pretrained``)
repart d'un modèle SB3 avec :

* ses poids (politique, valeur, ``log_std``) ;
* l'état de son Adam, que ``PPO.load`` restaure avec les poids ;
* ses statistiques de normalisation des **observations** ; celles des
  récompenses repartent de zéro, la récompense ayant pu changer ;
* un compteur de pas remis à zéro (``reset_num_timesteps``) : le
  curriculum repart du début.

:func:`load_into` fait la même chose dans un ``PoppyRunner``. C'est
l'inverse de ``gpu/export_sb3.py``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from stable_baselines3 import PPO

# Paramètre SB3 → (réseau rsl_rl, nom dans ce réseau).
_PARAM_MAP = {
    "mlp_extractor.policy_net.0": ("actor", "mlp.0"),
    "mlp_extractor.policy_net.2": ("actor", "mlp.2"),
    "action_net": ("actor", "mlp.4"),
    "mlp_extractor.value_net.0": ("critic", "mlp.0"),
    "mlp_extractor.value_net.2": ("critic", "mlp.2"),
    "value_net": ("critic", "mlp.4"),
}


def find_vec_normalize(model_zip: Path) -> Path:
    """Le ``.pkl`` de normalisation rangé à côté d'un modèle SB3, comme ``train_poppy.py``."""
    for name in ("vec_normalize.pkl", "vec_normalize_final.pkl"):
        path = model_zip.parent / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Pas de vec_normalize.pkl à côté de {model_zip}")


def _rsl_name(sb3_name: str) -> tuple[str, str]:
    if sb3_name == "log_std":
        return "actor", "distribution.log_std_param"
    stem, _, leaf = sb3_name.rpartition(".")
    net, prefix = _PARAM_MAP[stem]
    return net, f"{prefix}.{leaf}"


def load_into(runner, model_zip: Path) -> dict[str, int]:  # noqa: ANN001 - PoppyRunner
    """Recopie poids, état d'Adam et statistiques d'observation d'un modèle SB3.

    Args:
        runner: ``gpu.train.PoppyRunner`` avec un algorithme ``SB3PPO``, avant
            ``learn``.
        model_zip: Modèle SB3 (``best_model.zip`` d'un run CPU, par exemple).

    Returns:
        Le nombre de paramètres et d'états d'Adam recopiés.
    """
    model = PPO.load(str(model_zip), device="cpu", custom_objects={
        "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0,
        "learning_rate": 0.0,
    })
    policy = model.policy
    alg = runner.alg
    nets = {"actor": dict(alg.actor.named_parameters()),
            "critic": dict(alg.critic.named_parameters())}
    sb3_params = list(policy.named_parameters())
    targets = []
    with torch.no_grad():
        for name, param in sb3_params:
            net, rsl = _rsl_name(name)
            target = nets[net][rsl]
            if target.shape != param.shape:
                raise ValueError(f"{name} {tuple(param.shape)} → {net}.{rsl} {tuple(target.shape)}")
            target.copy_(param.to(target.device))
            targets.append(target)

    # État d'Adam : SB3 l'indexe dans l'ordre de policy.parameters().
    state = policy.optimizer.state_dict()["state"]
    copied = 0
    for index, target in enumerate(targets):
        if index not in state:
            continue
        opt = alg.optimizer.owner(target)
        src = state[index]
        device = target.device
        opt.state[target] = {
            "step": src["step"].detach().to(device=device, dtype=torch.float32).reshape(()),
            "exp_avg": src["exp_avg"].detach().to(device),
            "exp_avg_sq": src["exp_avg_sq"].detach().to(device),
        }
        copied += 1

    with open(find_vec_normalize(model_zip), "rb") as handle:
        previous = pickle.load(handle)
    env = runner.env
    rms = env.obs_rms
    rms.mean.copy_(torch.as_tensor(previous.obs_rms.mean, dtype=rms.mean.dtype))
    rms.var.copy_(torch.as_tensor(previous.obs_rms.var, dtype=rms.var.dtype))
    rms.count = float(previous.obs_rms.count)
    return {"params": len(targets), "adam_states": copied}
