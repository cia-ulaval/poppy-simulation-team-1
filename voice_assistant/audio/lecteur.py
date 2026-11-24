"""Lecture de la voix au fur et à mesure qu'elle arrive, et bouche qui suit le son.

Speaker.play_realtime_audio() attend d'avoir toute la réponse avant de jouer. Ici,
chaque morceau reçu de GPT va dans une file que le haut-parleur vide à son rythme.
Le volume de chaque bloc est calculé dans le callback du haut-parleur, au moment où le
son sort, pas quand il arrive : GPT envoie plus vite qu'on ne joue.

Essai sans clé, depuis voice_assistant/ : python -m audio.lecteur
"""

import asyncio
import collections
import threading
import time

import numpy as np
import sounddevice as sd
from config.peripheriques import GPT_SAMPLE_RATE, SPEAKER_DEVICE, Reechantillonneur, frequence

# Volume RMS (signal entre -1 et 1) multiplié par ce gain pour l'ouverture de la bouche,
# puis plafonné à 1. Une voix normale tourne autour de 0,05 à 0,15 de RMS.
GAIN_BOUCHE = 5.0
BLOC_S = 0.02  # le haut-parleur demande un bloc toutes les 20 ms
BLOQUE_S = 3.0  # sans demande du haut-parleur depuis ce temps, on le considère bloqué


class LecteurContinu:
    def __init__(self, visage=None, device=SPEAKER_DEVICE):
        self.visage = visage
        self.device = device
        self.sample_rate = frequence(device, "output")
        self._file = collections.deque()  # morceaux float32 à la fréquence du haut-parleur
        self._verrou = threading.Lock()
        self._reechantillonneur = Reechantillonneur(GPT_SAMPLE_RATE, self.sample_rate)
        self._fin_annoncee = threading.Event()
        self._fini = threading.Event()
        self._fini.set()
        self.debut = None  # instant où la question est partie
        self.premier_son = None  # instant où le premier son est sorti
        self._joues = 0  # échantillons de la réponse en cours déjà sortis
        self._dernier_appel = time.monotonic()  # dernier passage dans le callback
        self._fin_instant = 0.0  # instant où la dernière réponse a fini de sortir
        self._flux = self._ouvrir()

    def _ouvrir(self):
        return sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", device=self.device,
            blocksize=int(self.sample_rate * BLOC_S), callback=self._callback,
        )

    @property
    def nom(self):
        return sd.query_devices(self.device, "output")["name"]

    def demarrer(self):
        self._flux.start()
        self._dernier_appel = time.monotonic()
        return self

    def arreter(self):
        self._flux.stop()
        self._flux.close()
        if self.visage is not None:
            self.visage.bouche_ouverte(False)

    def nouvelle_reponse(self):
        """À appeler quand la question part : prépare la file et le chronomètre."""
        with self._verrou:
            self._file.clear()
        self._reechantillonneur = Reechantillonneur(GPT_SAMPLE_RATE, self.sample_rate)
        self._fin_annoncee.clear()
        self._fini.clear()
        self.debut = time.monotonic()
        self.premier_son = None
        self._joues = 0

    def interrompre(self):
        """On a coupé la parole à Poppy : il se tait tout de suite et ferme la bouche.

        Renvoie la durée de voix réellement sortie du haut-parleur, en millisecondes :
        GPT en a besoin pour savoir où sa phrase a été coupée.
        """
        with self._verrou:
            self._file.clear()
            joues = self._joues
        self._fin_annoncee.set()
        if self.visage is not None:
            self.visage.bouche_ouverte(False)
        if not self._fini.is_set():
            self._fin_instant = time.monotonic()
            self._fini.set()
        return int(joues * 1000 / self.sample_rate)

    def ajouter(self, pcm16):
        """Ajoute un morceau PCM16 mono à 24 kHz, tel que l'envoie GPT."""
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        audio = self._reechantillonneur(audio)
        if audio.size:
            with self._verrou:
                self._file.append(audio)

    def fin_reponse(self):
        """Plus rien n'arrivera pour cette réponse : la bouche se fermera à la fin."""
        self._fin_annoncee.set()

    def occupe(self, marge_s=0.0):
        """Vrai pendant une réponse, et marge_s secondes après qu'elle a fini de sortir."""
        return not self._fini.is_set() or time.monotonic() - self._fin_instant < marge_s

    async def attendre_fin(self):
        """Attend que tout ce qui est dans la file soit sorti du haut-parleur.

        Si le haut-parleur cesse de demander du son (un casque Bluetooth qui décroche,
        par exemple), on n'attend pas indéfiniment : on abandonne la fin de la réponse,
        on ferme la bouche et on rouvre le haut-parleur.
        """
        while not await asyncio.to_thread(self._fini.wait, 0.5):
            if time.monotonic() - self._dernier_appel > BLOQUE_S:
                print("\n  (haut-parleur bloqué : fin de la réponse abandonnée, on le rouvre)")
                self._relancer()
                return

    def _relancer(self):
        with self._verrou:
            self._file.clear()
        if self.visage is not None:
            self.visage.bouche_ouverte(False)
        self._fin_instant = time.monotonic()
        self._fini.set()
        try:
            self._flux.abort()
            self._flux.close()
        except sd.PortAudioError:
            pass
        self._flux = self._ouvrir()
        self.demarrer()

    def _callback(self, outdata, frames, time_info, status):
        self._dernier_appel = time.monotonic()
        sortie = outdata[:, 0]
        rempli = 0
        with self._verrou:
            while rempli < frames and self._file:
                morceau = self._file[0]
                n = min(frames - rempli, len(morceau))
                sortie[rempli:rempli + n] = morceau[:n]
                rempli += n
                if n == len(morceau):
                    self._file.popleft()
                else:
                    self._file[0] = morceau[n:]
            vide = not self._file
            self._joues += rempli
        sortie[rempli:] = 0.0

        if rempli:
            if self.premier_son is None:
                self.premier_son = time.monotonic()
            rms = float(np.sqrt(np.mean(sortie[:rempli] ** 2)))
            if self.visage is not None:
                self.visage.bouche_ouverte(True, min(1.0, rms * GAIN_BOUCHE))
        elif vide and self._fin_annoncee.is_set():
            if self.visage is not None:
                self.visage.bouche_ouverte(False)
            if not self._fini.is_set():
                self._fin_instant = time.monotonic()
                self._fini.set()
        elif self.visage is not None and self.visage.parle:
            self.visage.bouche_ouverte(True, 0.0)  # trou dans le flux : bouche fermée

    def __enter__(self):
        return self.demarrer()

    def __exit__(self, *exc):
        self.arreter()


if __name__ == "__main__":
    # Une voix synthétique (une note modulée en volume) jouée par petits morceaux,
    # comme les arrive de GPT. Le visage, lancé à côté, doit ouvrir et fermer la bouche.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from visage import Visage

    async def essai():
        with Visage("happy") as visage, LecteurContinu(visage) as lecteur:
            print(f"Haut-parleur : {lecteur.nom} ({lecteur.sample_rate} Hz)")
            t = np.arange(int(GPT_SAMPLE_RATE * 3)) / GPT_SAMPLE_RATE
            enveloppe = np.clip(np.sin(2 * np.pi * 1.5 * t), 0, None)
            signal = (0.3 * enveloppe * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
            lecteur.nouvelle_reponse()
            for debut in range(0, len(signal), 2400):  # morceaux de 100 ms
                lecteur.ajouter(signal[debut:debut + 2400].tobytes())
                await asyncio.sleep(0.02)
            lecteur.fin_reponse()
            await lecteur.attendre_fin()
            print(f"Premier son après {lecteur.premier_son - lecteur.debut:.2f} s. Terminé.")

    asyncio.run(essai())
