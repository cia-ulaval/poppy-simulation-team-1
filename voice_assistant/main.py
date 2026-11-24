"""Poppy complet : son visage, sa voix, et on lui parle au micro ou au clavier.

Depuis voice_assistant/ :

    python main.py              visage plein écran (le robot)
    python main.py --fenetre    visage dans une fenêtre (un ordinateur)
    python main.py --sans-visage
    python main.py --sans-micro
    python main.py --micro-ouvert   micro toujours ouvert : on parle quand on veut
    python main.py --interrompre    pareil, et parler lui coupe la parole (casque)

Au micro : Entrée pour commencer à parler, Entrée pour envoyer. Micro ouvert : on parle,
et Poppy répond quand on se tait (il n'écoute pas pendant qu'il parle). On peut aussi
taper une question. « quit » pour sortir.
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

# Le python3 fourni par macOS est un 3.9, trop ancien pour le visage (et son tkinter
# aussi) : mieux vaut le dire tout de suite qu'au milieu d'une conversation.
if sys.version_info < (3, 10):  # noqa: UP036 — la voix tourne hors des images Docker
    sys.exit(f"Python {sys.version.split()[0]} est trop ancien : il faut Python 3.10 ou plus "
             "(3.12 de python.org conseillé). Voir voice_assistant/README.md, étape 1.")

from conversation.assistant import discuter  # noqa: E402

ECRAN = Path(__file__).resolve().parent.parent / "src" / "ui" / "screen.py"


def main():
    parser = argparse.ArgumentParser(description="Parler à Poppy, avec son visage.")
    parser.add_argument("--fenetre", action="store_true",
                        help="visage dans une fenêtre au lieu du plein écran")
    parser.add_argument("--sans-visage", action="store_true",
                        help="ne pas lancer le visage (s'il tourne déjà, il suit quand même)")
    parser.add_argument("--sans-micro", action="store_true", help="clavier seulement")
    parser.add_argument("--micro-ouvert", action="store_true",
                        help="micro toujours ouvert : Poppy répond quand on se tait")
    parser.add_argument("--interrompre", action="store_true",
                        help="micro ouvert, et parler coupe la parole à Poppy ; seulement "
                             "avec un casque ou un micro qui n'entend pas le haut-parleur")
    args = parser.parse_args()
    if args.interrompre:
        args.micro_ouvert = True

    visage = None
    if not args.sans_visage:
        commande = [sys.executable, str(ECRAN)] + (["--fenetre"] if args.fenetre else [])
        visage = subprocess.Popen(commande)

    try:
        asyncio.run(discuter(micro=not args.sans_micro, micro_ouvert=args.micro_ouvert,
                             interrompre=args.interrompre))
    except KeyboardInterrupt:
        pass
    finally:
        if visage is not None:
            visage.terminate()


if __name__ == "__main__":
    main()
