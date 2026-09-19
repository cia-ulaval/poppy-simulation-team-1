"""Ouvre le modèle Poppy dans le viewer MuJoCo, sans politique.

Pour *regarder* le robot : articulations, limites, pose de départ. C'est le
premier réflexe utile quand on découvre le projet, et ça ne demande aucun
modèle entraîné.

Pour regarder une **politique** se dérouler, c'est `visu.py` à la racine.

Nécessite un écran : à lancer en natif, pas dans un conteneur (voir
docs/DOCKER.md § Sans Docker).

    python scripts/viewer.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer

_MODEL_PATH = Path(__file__).parent.parent / "assets" / "poppy_humanoid" / "poppy_humanoid.xml"


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
tourner autour, molette pour zoomer.

Nécessite un écran, donc un Python installé en natif — voir
docs/DEMARRAGE.md étape 2. Impossible depuis un conteneur sous Windows.

Pour regarder une POLITIQUE se dérouler, c'est scripts/visualize.py.
        """,
    ).parse_args()

    if not _MODEL_PATH.exists():
        print(f"Modèle introuvable : {_MODEL_PATH}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(_MODEL_PATH))
    data = mujoco.MjData(model)

    # njnt compte l'articulation libre de la racine en plus des 25 moteurs.
    print(f"Modèle    : {_MODEL_PATH.name}")
    print(f"Corps     : {model.nbody}")
    print(f"Articulations : {model.njnt} (dont 1 libre pour la racine)")
    print(f"Actionneurs   : {model.nu}")
    print("\nFermer la fenêtre pour quitter.")

    mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
