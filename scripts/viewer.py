"""Ouvre le modèle Poppy dans le viewer MuJoCo, sans politique.

Pour *regarder* le robot : articulations, limites, pose de départ. C'est le
premier réflexe utile quand on découvre le projet, et ça ne demande aucun
modèle entraîné.

Le sol est tiré au hasard à chaque episode
comme à l'entraînement, sa friction varie entre 0,4 et 2,0
Sa couleurchange en focntion de la friciton:
 - bleu clair s'il est glissant
 - bleu marine s'il est adhérent
La touche Entrée lance un nouvel épisode avec un nouveau sol

Pour regarder une **politique** se dérouler, c'est `scripts/visualize.py`

Nécessite un écran : à lancer en natif, pas dans un conteneur (voir
docs/DOCKER.md § Sans Docker).

    python scripts/viewer.py                (Windows, Linux)
    .venv/bin/mjpython scripts/viewer.py    (macOS)

La fenêtre passive (fonction launch_passive) de MuJoCo exige mjpython sur macOS.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

_MODEL_PATH = Path(__file__).parent.parent / "assets" / "poppy_humanoid" / "poppy_humanoid.xml"

_SLIPPERY_RGBA = np.array([0.75, 0.88, 1.00, 1.0])  # bleu clair : glissant
_GRIPPY_RGBA = np.array([0.05, 0.15, 0.45, 1.0])    # bleu marine : adhérent
# Mêmes plages que l'entraînement : configs/poppy_robust.yaml, domain_randomization.
_FRICTION_RANGE = (0.4, 2.0)
_RESTITUTION_RANGE = (0.0, 0.5)
_KEY_ENTER = 257


def _floor_color(friction: float, low: float, high: float) -> np.ndarray:
    """Couleur du sol: bleu clair si glissant, bleu marine si adhérent."""
    t = (friction - low) / (high - low)
    return _SLIPPERY_RGBA + t * (_GRIPPY_RGBA - _SLIPPERY_RGBA)


def _new_episode(env: PoppyHumanoidEnv, floor_id: int, episode: int) -> None:
    """Relance un essai: nouveau sol tiré au hasard et recoloré selon sa friction."""
    env.reset()
    friction = env.get_floor_randomization_info()["floor_slide_friction"]
    env.model.geom_rgba[floor_id] = _floor_color(friction, _FRICTION_RANGE[0], _FRICTION_RANGE[1])
    print(f"Episode {episode} : friction {friction:.2f}")


def main() -> int:
    """Charge le modèle Poppy et ouvre le viewer interactif.

    Returns:
        0 si la fenêtre s'est ouverte, 1 si le modèle est introuvable.
    """
    argparse.ArgumentParser(
        description="Ouvre le modèle Poppy dans le viewer MuJoCo, sans politique.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ne prend aucun argument et ne demande aucun modèle entraîné : c'est le
premier réflexe utile quand on découvre le projet. Clic gauche glissé pour
tourner autour, molette pour zoomer. Entrée : nouvel épisode, avec un
nouveau sol.

Couleur du sol : bleu clair s'il est glissant, bleu marine s'il est
adhérent. Le numéro de l'épisode et la friction s'affichent dans le terminal.

Nécessite un écran, donc un Python installé en natif — voir
docs/DEMARRAGE.md étape 2. Impossible depuis un conteneur sous Windows.
Sous macOS, lancer avec .venv/bin/mjpython — voir docs/DEMARRAGE-mac.md.

Pour regarder une POLITIQUE se dérouler, c'est scripts/visualize.py.
        """,
    ).parse_args()

    if not _MODEL_PATH.exists():
        print(f"Modèle introuvable : {_MODEL_PATH}", file=sys.stderr)
        return 1

    env = PoppyHumanoidEnv(
        floor_noise=True,
        friction_range=_FRICTION_RANGE,
        restitution_range=_RESTITUTION_RANGE,
    )
    model = env.model
    data = env.data

    # njnt compte l'articulation libre de la racine en plus des 25 moteurs.
    print(f"Modèle    : {_MODEL_PATH.name}")
    print(f"Corps     : {model.nbody}")
    print(f"Articulations : {model.njnt} (dont 1 libre pour la racine)")
    print(f"Actionneurs   : {model.nu}")
    print("\nFermer la fenêtre pour quitter.")

    restart = False

    def on_key(keycode: int) -> None:
        nonlocal restart
        if keycode == _KEY_ENTER:
            restart = True

    episode = 1
    floor_id = model.geom("floor").id
    _new_episode(env, floor_id, episode)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            with viewer.lock():
                if restart:
                    restart = False
                    episode += 1
                    _new_episode(env, floor_id, episode)
                env.step(np.zeros(model.nu))
            viewer.sync()
            time.sleep(env.dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
