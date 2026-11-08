"""Critères M1–M6 de ``docs/journal-experiences.md`` §2, sur le banc CPU.

Tourne dans l'image CPU (``poppy-train``), sans mjlab : les épisodes sont
ceux de ``scripts/evaluate.py`` (``make_eval_env`` puis ``run_episodes``),
sur ``PoppyHumanoidEnv`` et MuJoCo C, sol plat, sans randomisation, actions
déterministes. Un modèle GPU exporté par ``gpu/export_sb3.py`` y est mesuré
exactement comme les modèles CPU.

| Critère | Seuil |
|---|---|
| M1 sans chute | ≥ 90 % |
| M2 vitesse vers l'avant | 0,20 à 0,35 m/s |
| M3 tronc | ≥ 0,985 |
| M4 dérive de cap | \\|cap\\| < 15°, moyenne épisode par épisode |
| M5 symétrie d'appui | \\|D − G\\| < 10 points |
| M6 dérive latérale | \\|latéral\\| < 0,3 m en 10 s, moyenne épisode par épisode |

Usage (image CPU) ::

    python gpu/criteres.py --config configs/exp_a_ancien_cadre_nouvelle_recompense.yaml \\
        --model logs/gpu/<run>/model_1220.zip [--model …] --episodes 10 --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts.evaluate import load_model, run_episodes
from src.config import load_yaml, make_poppy_env_config
from src.environments.env_factory import make_eval_env

_DT = 0.01  # s, frame_skip 5 × 2 ms


def criteria(episodes: list[dict[str, float]]) -> dict[str, float | bool]:
    """Calcule M1–M6 et le score sur des épisodes résumés par ``run_episodes``.

    Returns:
        Les six mesures, leur verdict (``M1_ok``…) et ``score`` sur 6.
    """
    def mean(key: str) -> float:
        return float(np.mean([e[key] for e in episodes]))

    speed = float(np.mean([e["avance"] / (e["steps"] * _DT) for e in episodes]))
    lateral = [abs(e["derive"]) for e in episodes]
    heading = [abs(e["cap_deg"]) for e in episodes]
    out: dict[str, float | bool] = {
        "M1": mean("survived"),
        "M2": speed,
        "M3": mean("tronc_mean"),
        "M4": float(np.mean(heading)),
        "M5": 100.0 * abs(mean("appui_droit") - mean("appui_gauche")),
        "M6": float(np.mean(lateral)),
        "M6_max": float(np.max(lateral)),
        "steps": mean("steps"),
    }
    out["M1_ok"] = out["M1"] >= 0.9
    out["M2_ok"] = 0.20 <= out["M2"] <= 0.35
    out["M3_ok"] = out["M3"] >= 0.985
    out["M4_ok"] = out["M4"] < 15.0
    out["M5_ok"] = out["M5"] < 10.0
    out["M6_ok"] = out["M6"] < 0.3
    out["score"] = int(sum(bool(out[f"M{i}_ok"]) for i in range(1, 7)))
    return out


def measure(model_zip: Path, config: Path, episodes: int, seed: int) -> dict:
    """Déroule ``episodes`` épisodes d'un modèle exporté et rend ses critères."""
    pkl = model_zip.parent / f"{model_zip.stem}_vecnormalize.pkl"
    env = make_eval_env(
        config=make_poppy_env_config(load_yaml(config)), seed=seed,
        vec_normalize_path=pkl if pkl.exists() else None,
    )
    try:
        runs = run_episodes(load_model(model_zip, "PPO"), env, episodes,
                            deterministic=True, video_path=None, seed=seed)
    finally:
        env.close()
    return criteria(runs)


def _row(name: str, c: dict) -> str:
    mark = {True: " ", False: "✗"}
    return (f"| {name} | {100 * c['M1']:.0f} %{mark[c['M1_ok']]} | {c['M2']:.3f}{mark[c['M2_ok']]} "
            f"| {c['M3']:.3f}{mark[c['M3_ok']]} | {c['M4']:.1f}°{mark[c['M4_ok']]} "
            f"| {c['M5']:.1f}{mark[c['M5_ok']]} | {c['M6']:.2f}{mark[c['M6_ok']]} ({c['M6_max']:.2f}) "
            f"| **{c['score']}/6** |")


def main() -> None:
    """Mesure chaque modèle et imprime le tableau du journal."""
    parser = argparse.ArgumentParser(description="Critères M1–M6 sur le banc CPU.")
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/exp_a_ancien_cadre_nouvelle_recompense.yaml"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results = {}
    print("| Modèle | M1 | M2 vitesse | M3 tronc | M4 \\|cap\\| | M5 D−G | "
          "M6 \\|latéral\\| (max) | Score |")
    print("|---|---|---|---|---|---|---|---|")
    for model in args.model:
        c = measure(model, args.config, args.episodes, args.seed)
        results[str(model)] = c
        print(_row(model.stem, c), flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
