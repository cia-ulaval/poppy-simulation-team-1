"""Évalue un modèle entraîné sans écran ni robot.

Répond à la question qu'on ne pouvait pas poser au dépôt jusqu'ici : *est-ce
que ce modèle marche ?* — et surtout *pourquoi obtient-il cette récompense ?*

Le détail des huit termes de récompense est la partie utile. Une politique qui
reste debout sans avancer et une politique qui marche obtiennent des totaux
comparables ; seul le détail les distingue. Si ``healthy`` domine et que
``gait`` est proche de zéro, le robot ne marche pas, il tient la pose.

Exemples
--------
    python scripts/evaluate.py --model models/<date>/best_model.zip
    python scripts/evaluate.py --model <modele> --episodes 20 --floor-noise
    python scripts/evaluate.py --model <modele> --video sortie.mp4
    python scripts/evaluate.py --random --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.config import as_evaluation_config, load_yaml, make_poppy_env_config
from src.environments.env_factory import load_normalized_env, make_poppy_env

# Ordre d'essai pour retrouver l'algorithme d'un fichier .zip. Stable-Baselines3
# n'y enregistre pas son nom de façon exploitable : PPO et A2C partagent la même
# classe de politique. On tente donc les classes jusqu'à ce que l'une charge.
_ALGO_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}

# Les huit termes que la récompense additionne, dans l'ordre où ils apparaissent
# dans PoppyHumanoidEnv.step. Les trois derniers sont des coûts (déjà négatifs
# dans la somme, comptés positifs ici).
_REWARD_TERMS = (
    "healthy_reward",
    "gait_reward",
    "capped_vel",
    "lateral_vel",
    "uprightness",
    "ctrl_cost",
    "action_rate_cost",
    "joint_vel_cost",
)


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Évalue un modèle Poppy entraîné, sans écran.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Fichier .zip du modèle entraîné (exclusif avec --random)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Politique aléatoire au lieu d'un modèle : donne le plancher de "
             "comparaison. Sans ce repère, on ne sait pas si la récompense "
             "d'un modèle entraîné est bonne.",
    )
    parser.add_argument(
        "--vec-normalize",
        type=Path,
        default=None,
        help="Statistiques de normalisation (détectées près du modèle si absent)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/poppy_robust.yaml"),
        help="YAML décrivant l'environnement d'entraînement (défaut: configs/poppy_robust.yaml)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=sorted(_ALGO_CLASSES),
        default=None,
        help="Force l'algorithme au lieu de le deviner",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Nombre d'épisodes à dérouler (défaut: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire (défaut: 42)",
    )
    parser.add_argument(
        "--floor-noise",
        action="store_true",
        help="Réactive la randomisation du sol : mesure la robustesse, pas la "
             "performance nominale. Les résultats cessent d'être reproductibles.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Échantillonne les actions au lieu de prendre la moyenne",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Enregistre le premier épisode dans ce fichier .mp4",
    )
    return parser.parse_args()


def find_vec_normalize(model_path: Path, explicit: Optional[Path]) -> Optional[Path]:
    """Retrouve les statistiques de normalisation associées à un modèle.

    Args:
        model_path: Chemin du modèle.
        explicit: Chemin fourni par l'utilisateur, prioritaire s'il existe.

    Returns:
        Le chemin trouvé, ou ``None`` si le modèle a été entraîné sans
        normalisation.

    Raises:
        FileNotFoundError: Si un chemin explicite est fourni mais n'existe pas.
    """
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Statistiques introuvables : {explicit}")
        return explicit

    directory = model_path.parent
    candidates = []

    # Checkpoint intermédiaire d'abord, parce qu'il est plus spécifique.
    # CheckpointCallback écrit le modèle <prefixe>_<N>_steps.zip et sa
    # normalisation <prefixe>_vecnormalize_<N>_steps.pkl, dans le même dossier
    # que le vec_normalize_final.pkl de fin d'entraînement. Chercher le
    # générique en premier ferait donc charger, sans le dire, les statistiques
    # du dernier pas sur un modèle du millionième : mêmes formes, mêmes types,
    # chiffres faux et aucune erreur levée.
    stem = model_path.stem
    if stem.endswith("_steps") and stem.count("_") >= 2:
        prefix, steps, _ = stem.rsplit("_", 2)
        candidates.append(directory / f"{prefix}_vecnormalize_{steps}_steps.pkl")

    candidates += [
        directory / "vec_normalize_final.pkl",
        directory / "vec_normalize.pkl",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_model(model_path: Path, algorithm: Optional[str]) -> Any:
    """Charge un modèle Stable-Baselines3 en déterminant sa classe.

    Args:
        model_path: Fichier .zip du modèle.
        algorithm: Nom d'algorithme imposé, ou ``None`` pour deviner.

    Returns:
        Le modèle chargé.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si aucune classe connue ne parvient à le charger.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    if algorithm is not None:
        return _ALGO_CLASSES[algorithm].load(str(model_path))

    failures: list[str] = []
    for name, klass in _ALGO_CLASSES.items():
        try:
            return klass.load(str(model_path))
        except Exception as exc:  # noqa: BLE001 - on rapporte toutes les tentatives
            failures.append(f"{name}: {type(exc).__name__}")

    raise ValueError(
        f"Aucun algorithme connu n'a pu charger {model_path}. "
        f"Tentatives — {', '.join(failures)}. "
        f"Forcer avec --algorithm si l'algorithme est connu."
    )


class RandomPolicy:
    """Politique aléatoire, présentant l'interface ``predict`` de SB3.

    Elle répond à « que vaut un robot qui agite ses membres au hasard ? ».
    C'est le plancher : une politique entraînée qui ne le dépasse pas n'a rien
    appris, et sans ce repère une récompense de 2 000 ne veut rien dire.

    Les actions sont tirées uniformément dans ``[-1, 1]``, le contrat annoncé
    par ``PoppyHumanoidEnv._action_to_torque`` — et non via
    ``action_space.sample()``, qui produit des valeurs hors contrat tant que
    ``test_action_space_is_normalised`` reste en échec attendu.
    """

    def __init__(self, action_dim: int, seed: int) -> None:
        self._action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    def predict(
        self, observation: Any, deterministic: bool = True
    ) -> tuple[NDArray, None]:
        """Tire une action au hasard. ``deterministic`` est ignoré."""
        action = self._rng.uniform(-1.0, 1.0, size=(1, self._action_dim))
        return action.astype(np.float32), None


def build_env(
    config_path: Path,
    seed: int,
    floor_noise: bool,
    vec_normalize_path: Optional[Path],
    render: bool,
) -> VecNormalize | DummyVecEnv:
    """Construit l'environnement d'évaluation.

    L'environnement est dérivé du YAML d'entraînement : mêmes limites
    articulaires, même ``frame_skip``, même seuil de chute. Sans cela les
    métriques ne seraient pas comparables à celles de l'entraînement.

    Args:
        config_path: YAML décrivant l'environnement.
        seed: Graine aléatoire.
        floor_noise: Active la randomisation du sol.
        vec_normalize_path: Statistiques de normalisation, ou ``None``.
        render: Prépare l'environnement pour le rendu hors écran.

    Returns:
        L'environnement vectorisé, enveloppé de ``VecNormalize`` si des
        statistiques ont été fournies.
    """
    train_config = make_poppy_env_config(load_yaml(config_path))
    eval_config = as_evaluation_config(train_config, floor_noise=floor_noise)

    # make_poppy_env renvoie déjà un VecNormalize ; on ne garde que
    # l'environnement de base pour y appliquer les statistiques enregistrées.
    wrapped = make_poppy_env(
        config=eval_config,
        n_envs=1,
        seed=seed,
        use_subprocess=False,
    )
    base_env = wrapped.venv

    if render:
        _single_env(base_env).unwrapped.render_mode = "rgb_array"

    if vec_normalize_path is None:
        return base_env

    return load_normalized_env(str(vec_normalize_path), base_env, training=False)


def _single_env(vec_env: Any) -> Any:
    """Retourne l'environnement Gymnasium sous les enveloppes vectorielles."""
    env = vec_env
    while hasattr(env, "venv"):
        env = env.venv
    return env.envs[0]


def run_episodes(
    model: Any,
    env: Any,
    n_episodes: int,
    deterministic: bool,
    video_path: Optional[Path],
) -> list[dict[str, float]]:
    """Déroule des épisodes complets et collecte leurs métriques.

    Args:
        model: Modèle chargé.
        env: Environnement vectorisé (un seul sous-environnement).
        n_episodes: Nombre d'épisodes à dérouler.
        deterministic: Prend l'action moyenne plutôt qu'un échantillon.
        video_path: Si fourni, enregistre le premier épisode.

    Returns:
        Une entrée par épisode : récompense, durée, distances, vitesses et
        somme de chaque terme de récompense.
    """
    episodes: list[dict[str, float]] = []
    frames: list[NDArray] = []
    current: dict[str, list[float]] = defaultdict(list)

    obs = env.reset()
    steps = 0

    while len(episodes) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)
        steps += 1

        info = infos[0]
        for key, value in info.items():
            if isinstance(value, (int, float, np.floating)):
                current[key].append(float(value))
        current["reward"].append(float(rewards[0]))

        if video_path is not None and not episodes:
            frames.append(_single_env(env).render())

        if dones[0]:
            episodes.append(_summarise(current, steps, info))
            current = defaultdict(list)
            steps = 0

            if video_path is not None and frames:
                _write_video(frames, video_path, _single_env(env))
                frames = []

    return episodes


def _summarise(
    current: dict[str, list[float]],
    steps: int,
    final_info: dict[str, Any],
) -> dict[str, float]:
    """Agrège les relevés d'un épisode terminé.

    ``truncated`` distingue une chute d'une fin de temps imparti : c'est la
    métrique la plus parlante du lot. Un robot qui tombe systématiquement au
    bout de 40 pas ne marche pas, quelle que soit sa récompense.
    """
    summary: dict[str, float] = {
        "reward": float(np.sum(current["reward"])),
        "steps": float(steps),
        "x_final": current["x_position"][-1] if current["x_position"] else 0.0,
        "y_final": current["y_position"][-1] if current["y_position"] else 0.0,
        "forward_vel_mean": float(np.mean(current["forward_vel"]))
        if current["forward_vel"] else 0.0,
        "uprightness_mean": float(np.mean(current["uprightness"]))
        if current["uprightness"] else 0.0,
        # Monitor place TimeLimit.truncated dans l'info du dernier pas.
        "survived": float(bool(final_info.get("TimeLimit.truncated", False))),
    }
    for term in _REWARD_TERMS:
        summary[f"sum_{term}"] = float(np.sum(current[term])) if current[term] else 0.0
    return summary


def _write_video(frames: list[NDArray], path: Path, env: Any) -> None:
    """Écrit les images collectées dans un fichier vidéo."""
    import imageio.v2 as imageio

    fps = int(env.metadata.get("render_fps", 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps)
    print(f"Vidéo écrite : {path} ({len(frames)} images, {fps} fps)")


def report(episodes: list[dict[str, float]], model: Path | str) -> None:
    """Affiche le bilan de l'évaluation."""

    def stat(key: str) -> tuple[float, float]:
        values = [episode[key] for episode in episodes]
        return float(np.mean(values)), float(np.std(values))

    reward_mean, reward_std = stat("reward")
    steps_mean, steps_std = stat("steps")
    x_mean, x_std = stat("x_final")
    survival = float(np.mean([episode["survived"] for episode in episodes]))

    print()
    print(f"Modèle    : {model}")
    print(f"Épisodes  : {len(episodes)}")
    print("-" * 62)
    print(f"{'Récompense totale':<28} {reward_mean:>10.1f} ± {reward_std:.1f}")
    print(f"{'Durée (pas)':<28} {steps_mean:>10.0f} ± {steps_std:.0f}")
    print(f"{'Distance en x (m)':<28} {x_mean:>10.2f} ± {x_std:.2f}")
    print(f"{'Dérive latérale (m)':<28} {stat('y_final')[0]:>10.2f}")
    print(f"{'Vitesse avant (m/s)':<28} {stat('forward_vel_mean')[0]:>10.3f}")
    print(f"{'Verticalité (0-1)':<28} {stat('uprightness_mean')[0]:>10.3f}")
    print(f"{'Épisodes sans chute':<28} {survival * 100:>9.0f} %")
    print("-" * 62)
    print("Décomposition de la récompense (somme moyenne par épisode)")
    for term in _REWARD_TERMS:
        print(f"  {term:<26} {stat(f'sum_{term}')[0]:>10.1f}")
    print("-" * 62)

    if survival == 0.0:
        print("Aucun épisode mené à terme : le robot tombe systématiquement.")
    elif stat("forward_vel_mean")[0] < 0.05:
        print("Vitesse quasi nulle : le modèle tient la pose, il ne marche pas.")


def main() -> int:
    """Point d'entrée."""
    args = parse_args()

    if (args.model is None) == (not args.random):
        raise SystemExit("Fournir --model <fichier.zip> ou --random, pas les deux.")

    if args.random:
        # Pas de modèle, donc pas de statistiques de normalisation : les
        # récompenses brutes sont précisément ce qu'on veut comme plancher.
        vec_normalize_path = None
        label = "politique aléatoire"
    else:
        label = args.model
        vec_normalize_path = find_vec_normalize(args.model, args.vec_normalize)
        if vec_normalize_path is None:
            print("Aucune statistique de normalisation trouvée : évaluation brute.")
        else:
            print(f"Normalisation : {vec_normalize_path}")

    env = build_env(
        config_path=args.config,
        seed=args.seed,
        floor_noise=args.floor_noise,
        vec_normalize_path=vec_normalize_path,
        render=args.video is not None,
    )

    model = (
        RandomPolicy(action_dim=env.action_space.shape[0], seed=args.seed)
        if args.random
        else load_model(args.model, args.algorithm)
    )

    try:
        episodes = run_episodes(
            model=model,
            env=env,
            n_episodes=args.episodes,
            deterministic=not args.stochastic,
            video_path=args.video,
        )
        report(episodes, label)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
