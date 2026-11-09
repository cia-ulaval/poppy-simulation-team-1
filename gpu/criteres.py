"""Critères M1–M7 de ``docs/journal-experiences.md`` §2, sur le banc CPU.

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
| M7 bras le long du corps | chaque coude en moyenne ≥ 10 cm sous son épaule |

M7 (palier 1, H-Placo) : hauteur de l'épaule moins celle du coude, axe z du
monde, corps ``{l,r}_shoulder`` contre ``{l,r}_forearm``, lue avant chaque
action de chaque épisode, moyennée sur tous les pas ; le pire des deux bras
doit atteindre 0,10 m. Mêmes corps et même lecture que la mesure CPU de la
session principale.

« Meilleur » modèle d'un run GPU (``--select-best``) : rsl_rl n'a pas
d'``EvalCallback``. On rejoue son critère sur les checkpoints : récompense
brute moyenne de 5 épisodes déterministes, configuration d'évaluation,
graine d'entraînement + 1000 ; le checkpoint de plus forte récompense est
le meilleur. Sur CPU, l'``EvalCallback`` le cherche tous les 50 000 pas ;
ici, parmi les checkpoints, tous les ~250 000 pas.

Usage (image CPU) ::

    python gpu/criteres.py --config configs/exp_a_ancien_cadre_nouvelle_recompense.yaml \\
        --model logs/gpu/<run>/model_1220.zip [--model …] --episodes 10 --seed 1
    python gpu/criteres.py --config gpu/configs/h_placo_cpu.yaml --select-best \\
        --train-seed 42 --model logs/gpu/<run>/model_*.zip
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
from src.environments.env_factory import make_eval_env, single_env

_DT = 0.01  # s, frame_skip 5 × 2 ms
_ELBOW_DROP = 0.10  # m, M7


class _ArmRecorder:
    """Enveloppe un modèle : lit la chute épaule → coude avant chaque action (M7)."""

    def __init__(self, model, env) -> None:  # noqa: ANN001
        self.model = model
        bare = single_env(env).unwrapped
        self._data = bare.data
        self._ids = {side: (bare.model.body(f"{side}_shoulder").id,
                            bare.model.body(f"{side}_forearm").id) for side in ("l", "r")}
        self.drops: dict[str, list[float]] = {"l": [], "r": []}

    def predict(self, obs, deterministic: bool = True):  # noqa: ANN001, ANN201
        for side, (shoulder, forearm) in self._ids.items():
            self.drops[side].append(
                float(self._data.xpos[shoulder][2] - self._data.xpos[forearm][2]))
        return self.model.predict(obs, deterministic=deterministic)


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


def vec_normalize_for(model_zip: Path) -> Path | None:
    """Le ``.pkl`` de normalisation d'un modèle, export GPU ou checkpoint SB3.

    ``<stem>_vecnormalize.pkl`` (``gpu/export_sb3.py``) ; sinon les noms de
    ``train_poppy.py`` : ``poppy_ppo_vecnormalize_<n>_steps.pkl``,
    ``vec_normalize_final.pkl`` et ``best_model/vec_normalize.pkl``.
    """
    folder, stem = model_zip.parent, model_zip.stem
    candidates = [folder / f"{stem}_vecnormalize.pkl"]
    if stem.startswith("poppy_ppo_") and stem.endswith("_steps"):
        candidates.append(folder / stem.replace("poppy_ppo_", "poppy_ppo_vecnormalize_", 1))
    if stem == "poppy_ppo_final":
        candidates.append(folder / "vec_normalize_final.pkl")
    if stem == "best_model":
        candidates.append(folder / "vec_normalize.pkl")
    for path in candidates:
        path = path.with_suffix(".pkl")
        if path.exists():
            return path
    return None


def measure(model_zip: Path, config: Path, episodes: int, seed: int) -> dict:
    """Déroule ``episodes`` épisodes d'un modèle exporté et rend ses critères."""
    env = make_eval_env(
        config=make_poppy_env_config(load_yaml(config)), seed=seed,
        vec_normalize_path=vec_normalize_for(model_zip),
    )
    try:
        recorder = _ArmRecorder(load_model(model_zip, "PPO"), env)
        runs = run_episodes(recorder, env, episodes,
                            deterministic=True, video_path=None, seed=seed)
    finally:
        env.close()
    out = criteria(runs)
    out["M7_l"] = float(np.mean(recorder.drops["l"]))
    out["M7_r"] = float(np.mean(recorder.drops["r"]))
    out["M7"] = min(out["M7_l"], out["M7_r"])
    out["M7_ok"] = out["M7"] >= _ELBOW_DROP
    out["reward"] = float(np.mean([e["reward"] for e in runs]))
    return out


def select_best(models: list[Path], config: Path, train_seed: int, episodes: int = 5) -> Path:
    """Le critère de l'``EvalCallback`` de ``train_poppy.py``, rejoué sur des checkpoints.

    Returns:
        Le modèle de plus forte récompense brute moyenne.
    """
    best, best_reward = None, -np.inf
    for model in models:
        env = make_eval_env(
            config=make_poppy_env_config(load_yaml(config)), seed=train_seed + 1000,
            vec_normalize_path=vec_normalize_for(model),
        )
        try:
            runs = run_episodes(load_model(model, "PPO"), env, episodes,
                                deterministic=True, video_path=None, seed=train_seed + 1000)
        finally:
            env.close()
        reward = float(np.mean([e["reward"] for e in runs]))
        print(f"{model.stem}\t{reward:.1f}", flush=True)
        if reward > best_reward:
            best, best_reward = model, reward
    print(f"meilleur : {best} ({best_reward:.1f})")
    return best


def _row(name: str, c: dict) -> str:
    mark = {True: " ", False: "✗"}
    return (f"| {name} | {100 * c['M1']:.0f} %{mark[c['M1_ok']]} | {c['M2']:.3f}{mark[c['M2_ok']]} "
            f"| {c['M3']:.3f}{mark[c['M3_ok']]} | {c['M4']:.1f}°{mark[c['M4_ok']]} "
            f"| {c['M5']:.1f}{mark[c['M5_ok']]} | {c['M6']:.2f}{mark[c['M6_ok']]} ({c['M6_max']:.2f}) "
            f"| **{c['score']}/6** | G {100 * c['M7_l']:.1f} D {100 * c['M7_r']:.1f} cm"
            f"{mark[c['M7_ok']]} |")


def main() -> None:
    """Mesure chaque modèle et imprime le tableau du journal."""
    parser = argparse.ArgumentParser(description="Critères M1–M7 sur le banc CPU.")
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/exp_a_ancien_cadre_nouvelle_recompense.yaml"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--select-best", action="store_true",
                        help="Choisir le meilleur checkpoint comme l'EvalCallback (5 épisodes)")
    parser.add_argument("--train-seed", type=int, default=42)
    args = parser.parse_args()
    if args.select_best:
        select_best(args.model, args.config, args.train_seed)
        return

    results = {}
    print("| Modèle | M1 | M2 vitesse | M3 tronc | M4 \\|cap\\| | M5 D−G | "
          "M6 \\|latéral\\| (max) | Score | M7 coude sous épaule |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in args.model:
        c = measure(model, args.config, args.episodes, args.seed)
        results[str(model)] = c
        print(_row(model.stem, c), flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
