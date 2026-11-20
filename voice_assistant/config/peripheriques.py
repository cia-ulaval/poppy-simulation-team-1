"""Micro et haut-parleur utilisés par la voix.

Par défaut, ceux qui sont choisis dans les réglages son du système : sur un
ordinateur, le micro et le haut-parleur actifs en ce moment. Pour en imposer
d'autres, par exemple ceux du robot, mettre dans voice_assistant/.env :

    POPPY_MICRO=1
    POPPY_HAUT_PARLEUR=2

Un numéro ou un morceau du nom, tels que les affiche `python -m sounddevice`.
Ce module ne demande pas la clé OpenAI : on peut tester le son sans elle.
"""

import math
import os
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from scipy.signal import resample_poly

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GPT_SAMPLE_RATE = 24000  # l'API Realtime parle et écoute en PCM16 mono à 24 kHz


def _peripherique(variable):
    """Numéro ou nom lu dans le .env ; None veut dire « celui par défaut du système »."""
    valeur = os.getenv(variable, "").strip()
    if not valeur:
        return None
    return int(valeur) if valeur.isdigit() else valeur


MICROPHONE_DEVICE = _peripherique("POPPY_MICRO")
SPEAKER_DEVICE = _peripherique("POPPY_HAUT_PARLEUR")


def frequence(device, kind):
    """Fréquence native du périphérique ; kind vaut « input » ou « output »."""
    return int(sd.query_devices(device, kind)["default_samplerate"])


def reechantillonner(audio, depuis, vers):
    """Change la fréquence d'un signal, par exemple de 24 kHz à 48 kHz."""
    if depuis == vers:
        return np.asarray(audio)
    commun = math.gcd(depuis, vers)
    return resample_poly(audio, vers // commun, depuis // commun)


class Reechantillonneur:
    """Change la fréquence d'un flux arrivant par morceaux, sans clic entre eux.

    reechantillonner() traite chaque morceau seul et fait un petit saut à chaque
    jointure : inaudible sur un enregistrement entier, mais un clic toutes les
    quelques dizaines de millisecondes en continu. Ici, on garde le dernier
    échantillon et la position entre deux morceaux, et on interpole linéairement :
    largement assez pour de la voix.
    """

    def __init__(self, depuis, vers):
        self.pas = depuis / vers  # avance dans le signal d'entrée par échantillon de sortie
        self.precedent = 0.0
        self.position = 1.0  # position du prochain échantillon, 0 étant « precedent »

    def __call__(self, morceau):
        morceau = np.asarray(morceau, dtype=np.float32)
        if morceau.size == 0:
            return morceau
        signal = np.concatenate(([self.precedent], morceau))
        fin = len(signal) - 1
        positions = np.arange(self.position, fin, self.pas) if self.position < fin else []
        sortie = np.interp(positions, np.arange(len(signal)), signal).astype(np.float32)
        if len(positions):
            self.position = positions[-1] + self.pas - fin
        else:
            self.position -= fin
        self.precedent = float(signal[-1])
        return sortie
