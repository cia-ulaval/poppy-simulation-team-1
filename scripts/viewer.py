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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer

_MODEL_PATH = Path(__file__).parent.parent / "assets" / "poppy_humanoid" / "poppy_humanoid.xml"


def main() -> int:
    """Charge le modèle Poppy et ouvre le viewer interactif."""
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
