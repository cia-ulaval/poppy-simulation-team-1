"""Évaluation croisée : un modèle SB3 (CPU ou exporté) marche dans le simulateur GPU.

Le banc de référence reste le CPU (``gpu/criteres.py``). Ce script fait
l'inverse : il prend un modèle entraîné sur MuJoCo C et le fait marcher sur
MuJoCo Warp, sans entraînement. Si le contact pied-sol, plan ou champ de
hauteur, se comporte autrement sur GPU, la survie et la démarche le
montrent ici avant tout entraînement.

Mêmes conditions que le banc CPU : configuration d'évaluation
(``as_evaluation_config`` : sans randomisation, départ debout), relief à la
difficulté demandée, actions déterministes (moyenne de la politique, bornée
à l'espace d'action), observations normalisées par le ``.pkl`` du modèle.
Les ``N`` épisodes tournent en parallèle, un par environnement ; chaque
environnement ne compte que son premier épisode. Les résumés d'épisode sont
ceux de ``scripts/evaluate.py`` (``_summarise``), et les critères ceux de
``gpu/criteres.py``, M7 compris.

Usage (image GPU) ::

    python -m gpu.eval_gpu --config configs/palier2_hplaco.yaml \\
        --model <run>/poppy_ppo_9000000_steps.zip --terrain 0.3 --terrain 0.6 --terrain 1.0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from stable_baselines3 import PPO

from gpu.poppy_env import make_env_cfg, poppy_state
from src.config import load_yaml, make_poppy_env_config
from src.config.loaders import as_evaluation_config

sys.path.insert(0, str(Path(__file__).parent))
from criteres import _row, criteria, vec_normalize_for  # noqa: E402

_KEYS = ("forward_vel", "lateral_vel", "heading", "r_contact", "l_contact",
         "touchdowns", "torso_uprightness", "reward")


def _summary(trace: dict[str, list[float]], steps: int, survived: bool, dt: float) -> dict:
    """``_summarise`` et ``_gait_metrics`` de ``scripts/evaluate.py``, pour un épisode."""
    heading = np.unwrap(np.asarray(trace["heading"]))
    right = np.asarray(trace["r_contact"]) > 0.5
    left = np.asarray(trace["l_contact"]) > 0.5
    return {
        "reward": float(np.sum(trace["reward"])),
        "steps": float(steps),
        "survived": float(survived),
        "avance": float(np.sum(trace["forward_vel"])) * dt,
        "derive": float(np.sum(trace["lateral_vel"])) * dt,
        "cap_deg": float(np.degrees(heading[-1] - heading[0])),
        "appui_droit": float(np.mean(right & ~left)),
        "appui_gauche": float(np.mean(left & ~right)),
        "cadence": float(np.sum(trace["touchdowns"])) / (steps * dt),
        "tronc_mean": float(np.mean(trace["torso_uprightness"])),
    }


def evaluate(
    model_zip: Path, config_path: Path, terrain: float, episodes: int, seed: int,
    env: ManagerBasedRlEnv | None = None,
) -> dict:
    """``episodes`` épisodes déterministes d'un modèle SB3 dans le simulateur GPU.

    Returns:
        Les critères M1–M7 (``gpu/criteres.criteria`` et M7), plus la
        récompense moyenne.
    """
    config = make_poppy_env_config(load_yaml(config_path))
    config = as_evaluation_config(config, terrain_difficulty=terrain)
    if config.terrain.enabled:
        config = replace(config, terrain=replace(config.terrain, difficulty=terrain))
    own = env is None
    if own:
        env = ManagerBasedRlEnv(make_env_cfg(config, num_envs=episodes, seed=seed),
                                device="cuda:0")
    device = env.device
    s = poppy_state(env)
    if s.terrain is not None:
        s.terrain.difficulty = terrain

    model = PPO.load(str(model_zip), device=device, custom_objects={
        "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0, "learning_rate": 0.0})
    with open(vec_normalize_for(model_zip), "rb") as handle:
        norm = pickle.load(handle)
    mean = torch.as_tensor(norm.obs_rms.mean, dtype=torch.float32, device=device)
    std = torch.sqrt(torch.as_tensor(norm.obs_rms.var, dtype=torch.float32, device=device)
                     + norm.epsilon)
    low = torch.as_tensor(model.action_space.low, device=device)
    high = torch.as_tensor(model.action_space.high, device=device)

    mj = env.sim.mj_model
    arm_ids = {side: (mj.body(f"robot/{side}_shoulder").id, mj.body(f"robot/{side}_forearm").id)
               for side in ("l", "r")}
    obs, _ = env.reset()
    n = env.num_envs
    active = torch.ones(n, dtype=torch.bool, device=device)
    traces = [{k: [] for k in _KEYS} for _ in range(n)]
    drops = {"l": [], "r": []}
    results: list[dict | None] = [None] * n
    steps = np.zeros(n, dtype=int)
    for _ in range(int(env.max_episode_length) + 1):
        xpos = env.sim.data.xpos
        for side, (shoulder, forearm) in arm_ids.items():
            drop = (xpos[:, shoulder, 2] - xpos[:, forearm, 2])[active]
            drops[side].extend(drop.cpu().tolist())
        with torch.no_grad():
            x = torch.clamp((obs["policy"] - mean) / std, -norm.clip_obs, norm.clip_obs)
            action = model.policy.get_distribution(x).distribution.mean
            action = torch.clamp(action, low, high)
        obs, reward, terminated, truncated, _ = env.step(action)
        info = {k: v.cpu().numpy() for k, v in s.info.items() if k in _KEYS}
        info["reward"] = reward.cpu().numpy()
        done = (terminated | truncated).cpu().numpy()
        for i in np.flatnonzero(active.cpu().numpy()):
            for k in _KEYS:
                traces[i][k].append(float(info[k][i]))
            steps[i] += 1
            if done[i]:
                results[i] = _summary(traces[i], int(steps[i]), bool(truncated[i]), s.dt)
                active[i] = False
        if not bool(active.any()):
            break
    if own:
        env.close()
    out = criteria(results)
    out["M7_l"] = float(np.mean(drops["l"]))
    out["M7_r"] = float(np.mean(drops["r"]))
    out["M7"] = min(out["M7_l"], out["M7_r"])
    out["M7_ok"] = out["M7"] >= 0.10
    out["reward"] = float(np.mean([r["reward"] for r in results]))
    return out


def main() -> None:
    """Imprime le tableau M1–M7 de chaque modèle, à chaque difficulté."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/palier2_hplaco.yaml"))
    parser.add_argument("--terrain", type=float, action="append", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    print("| Modèle | M1 | M2 vitesse | M3 tronc | M4 \\|cap\\| | M5 D−G | "
          "M6 \\|latéral\\| (max) | Score | M7 coude sous épaule |")
    print("|---|---|---|---|---|---|---|---|---|")
    results = {}
    for terrain in args.terrain or [0.0]:
        config = make_poppy_env_config(load_yaml(args.config))
        config = as_evaluation_config(config, terrain_difficulty=terrain)
        if config.terrain.enabled:
            config = replace(config, terrain=replace(config.terrain, difficulty=terrain))
        env = ManagerBasedRlEnv(make_env_cfg(config, num_envs=args.episodes, seed=args.seed),
                                device="cuda:0")
        for model in args.model:
            torch.manual_seed(args.seed)  # même relief pour chaque modèle
            c = evaluate(model, args.config, terrain, args.episodes, args.seed, env=env)
            name = f"{model.parent.parent.name}/{model.stem} @ {terrain:g}"
            results[name] = c
            print(_row(name, c), flush=True)
        env.close()
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
