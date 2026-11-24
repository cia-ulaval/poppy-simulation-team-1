"""Côté voix du contrat entre la voix et le visage (spec/voix-visage.md).

La voix écrit un petit fichier JSON que src/ui/screen.py relit toutes les 40 ms :

    {"emotion": "happy", "parle": true, "bouche": 0.6}

Emplacement : /dev/shm/poppy_visage si /dev/shm existe (le robot : en mémoire vive,
pour ne pas user la carte SD), sinon poppy_visage dans le dossier temporaire du
système. screen.py applique la même règle. L'écriture passe par un fichier temporaire
puis os.replace, pour que le visage ne lise jamais un fichier à moitié écrit.

Ce module ne dépend que de la bibliothèque standard : il ne demande ni clé ni son.
"""

import json
import os
import tempfile
import threading
from pathlib import Path

# Les noms de EMOTIONS dans src/ui/screen.py, dans le même ordre. GPT ne peut choisir
# qu'un visage de cette liste ; tests/test_voix_visage.py vérifie qu'elle n'a pas
# divergé de screen.py.
EMOTIONS_VISAGE = ("happy", "sad", "thinking", "neutral", "angry", "67")

PERIODE_S = 0.04  # le visage relit le fichier toutes les 40 ms : on écrit au même rythme


def chemin_visage():
    """Fichier partagé avec le visage ; même règle que dans src/ui/screen.py."""
    shm = Path("/dev/shm")
    if shm.is_dir():
        return shm / "poppy_visage"
    return Path(tempfile.gettempdir()) / "poppy_visage"


class Visage:
    """Tient l'état du visage et le réécrit en continu tant que la voix tourne.

    L'écriture continue garde le fichier frais : s'il a plus de 3 s, le visage
    considère que la voix est partie et reprend sa vie propre.
    """

    def __init__(self, emotion="happy", chemin=None):
        self.chemin = Path(chemin) if chemin else chemin_visage()
        self.emotion = emotion
        self.parle = False
        self.bouche = 0.0
        self._verrou = threading.Lock()
        self._arret = threading.Event()
        self._fil = None

    def changer_emotion(self, emotion):
        """Change l'émotion ; un nom inconnu est ignoré et renvoie False."""
        if emotion not in EMOTIONS_VISAGE:
            return False
        with self._verrou:
            self.emotion = emotion
        self.ecrire()
        return True

    def bouche_ouverte(self, parle, bouche=0.0):
        """Appelé par la lecture audio : parle ou non, et ouverture entre 0 et 1."""
        with self._verrou:
            self.parle = bool(parle)
            self.bouche = min(1.0, max(0.0, float(bouche)))

    def etat(self):
        with self._verrou:
            return {"emotion": self.emotion, "parle": self.parle,
                    "bouche": round(self.bouche, 3)}

    def ecrire(self):
        temporaire = self.chemin.with_name(self.chemin.name + ".tmp")
        try:
            temporaire.write_text(json.dumps(self.etat()), encoding="utf-8")
            os.replace(temporaire, self.chemin)
        except OSError:
            pass  # le visage n'est pas indispensable : la voix continue sans lui

    def demarrer(self):
        """Réécrit le fichier toutes les 40 ms, dans un fil à part."""
        if self._fil is not None:
            return self
        self._arret.clear()
        self._fil = threading.Thread(target=self._boucle, name="visage", daemon=True)
        self._fil.start()
        return self

    def arreter(self):
        """Ferme la bouche, arrête l'écriture et retire le fichier."""
        self._arret.set()
        if self._fil is not None:
            self._fil.join(timeout=1)
            self._fil = None
        try:
            self.chemin.unlink()
        except OSError:
            pass

    def _boucle(self):
        while not self._arret.is_set():
            self.ecrire()
            self._arret.wait(PERIODE_S)

    def __enter__(self):
        return self.demarrer()

    def __exit__(self, *exc):
        self.arreter()


if __name__ == "__main__":
    # Essai à la main : python visage.py sad   (visage lancé à côté)
    import sys
    import time

    nom = sys.argv[1] if len(sys.argv) > 1 else "happy"
    with Visage(nom) as visage:
        print(f"Fichier : {visage.chemin} ; émotion {nom}, parle 5 s.")
        debut = time.monotonic()
        while time.monotonic() - debut < 5:
            t = time.monotonic() - debut
            visage.bouche_ouverte(True, abs((t * 3) % 2 - 1))
            time.sleep(PERIODE_S)
