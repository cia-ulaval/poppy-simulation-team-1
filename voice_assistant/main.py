"""Poppy complet : son visage, sa voix, et on lui parle au micro ou au clavier.

Depuis voice_assistant/ :

    python main.py              visage plein écran (le robot)
    python main.py --fenetre    visage dans une fenêtre (un ordinateur)
    python main.py --sans-visage
    python main.py --sans-micro

Au micro : Entrée pour commencer à parler, Entrée pour envoyer. On peut aussi taper
une question. « quit » pour sortir.
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from conversation.assistant import discuter

ECRAN = Path(__file__).resolve().parent.parent / "src" / "ui" / "screen.py"


def main():
    parser = argparse.ArgumentParser(description="Parler à Poppy, avec son visage.")
    parser.add_argument("--fenetre", action="store_true",
                        help="visage dans une fenêtre au lieu du plein écran")
    parser.add_argument("--sans-visage", action="store_true",
                        help="ne pas lancer le visage (s'il tourne déjà, il suit quand même)")
    parser.add_argument("--sans-micro", action="store_true", help="clavier seulement")
    args = parser.parse_args()

    visage = None
    if not args.sans_visage:
        commande = [sys.executable, str(ECRAN)] + (["--fenetre"] if args.fenetre else [])
        visage = subprocess.Popen(commande)

    try:
        asyncio.run(discuter(micro=not args.sans_micro))
    except KeyboardInterrupt:
        pass
    finally:
        if visage is not None:
            visage.terminate()


if __name__ == "__main__":
    main()
