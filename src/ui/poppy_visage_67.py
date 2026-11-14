"""Visage « 67 » de Poppy, plein écran, pour la danse du même nom.

Écrit pour l'écran de la tête (800×480, Raspberry Pi 4) mais tout est mis à l'échelle
de l'écran. Les yeux sont un « 6 » (à gauche de l'écran) et un « 7 » (à droite) ;
sourcils levés, joues roses, grand sourire. Les deux yeux balancent de gauche à droite
au rythme de la danse, et celui du battement en cours gonfle et s'allume.

La cadence vient de ``/tmp/poppy_67``, que ``scripts/robot_danse.py`` (ordre ``67`` du
mode ``spectacle``) remplit de « 6 » quand monte la main du robot qui est à gauche pour
le public (sa main droite), de « 7 » pour l'autre. Sans ce fichier, ou s'il n'a pas
bougé depuis 3 s, le visage garde sa propre cadence. Il ne commande aucun moteur.

    DISPLAY=:0 python3 poppy_visage_67.py [fichier_de_cadence]

Échap pour quitter.
"""

from __future__ import annotations

import math
import random
import sys
import time
import tkinter as tk
from pathlib import Path

CADENCE = Path("/tmp/poppy_67")
FRAIS_S = 3.0  # au-delà, le fichier de cadence est ignoré
TEMPS_LIBRE_MS = 650  # un battement sans cadence : à peu près celui du robot à SPEED 120
IMAGE_MS = 30
JAUNE, JAUNE_PALE, ROSE, BLANC, NOIR = "#ffd34d", "#9c8434", "#ff7aa2", "white", "black"
CHIFFRES = ("6", "7")  # œil gauche de l'écran, œil droit


class Visage:
    """Le dessin et son animation, sur un canevas de ``w`` × ``h`` pixels."""

    def __init__(self, racine: tk.Misc, canevas: tk.Canvas, w: int, h: int,
                 cadence: Path = CADENCE) -> None:
        self.racine, self.c, self.cadence = racine, canevas, cadence
        self.r = int(min(w * 0.12, h * 0.24))  # demi-hauteur d'un œil
        self.jeu = self.r * 0.35  # course du balancement
        self.cy = int(h * 0.34)
        self.cx = (int(w * 0.25), int(w * 0.75))
        self.taille = int(self.r * 2.3)  # hauteur des chiffres, en pixels
        epais = max(6, w // 100)
        c = canevas

        self.joues = [c.create_oval(x - self.r * 0.6, self.cy + self.r * 1.15,
                                    x + self.r * 0.6, self.cy + self.r * 1.45,
                                    fill=ROSE, outline="") for x in self.cx]
        self.yeux = [c.create_text(x, self.cy, text=CHIFFRES[k], fill=JAUNE,
                                   font=("Helvetica", -self.taille, "bold"))
                     for k, x in enumerate(self.cx)]
        self.paupieres = [c.create_line(x - self.r * 0.6, self.cy, x + self.r * 0.6, self.cy,
                                        fill=JAUNE, width=epais * 2, capstyle="round",
                                        state="hidden") for x in self.cx]
        self.sourcils = [c.create_line(0, 0, 0, 0, fill=BLANC, width=epais, capstyle="round")
                         for _ in self.cx]
        bx0, bx1, by0, by1 = int(w * 0.36), int(w * 0.64), int(h * 0.56), int(h * 0.90)
        c.create_arc(bx0, by0, bx1, by1, start=0, extent=-180, style="chord",
                     fill=BLANC, outline=BLANC, width=epais)
        lx, ly = (bx0 + bx1) / 2, by0 + (by1 - by0) * 0.80
        c.create_oval(lx - (bx1 - bx0) * 0.16, ly - (by1 - by0) * 0.13,
                      lx + (bx1 - bx0) * 0.16, ly + (by1 - by0) * 0.10, fill=ROSE, outline="")

        self.phase = "6"
        self.x = 0.0  # balancement actuel, de -1 (gauche de l'écran) à 1 (droite)
        self.debut_battement = time.monotonic()
        self.dernier_signal = -FRAIS_S  # dernière phase lue dans le fichier de cadence
        self.cligne_jusqu_a = 0.0

    def lire_cadence(self) -> str | None:
        """Phase écrite par la danse, ou None si le fichier est absent, vide ou ancien."""
        try:
            if time.time() - self.cadence.stat().st_mtime > FRAIS_S:
                return None
            texte = self.cadence.read_text().strip()
        except OSError:
            return None
        return texte if texte in CHIFFRES else None

    def battre(self, phase: str) -> None:
        self.phase = phase
        self.debut_battement = time.monotonic()

    def suivre_cadence(self) -> None:
        maintenant = time.monotonic()
        phase = self.lire_cadence()
        if phase is not None:
            self.dernier_signal = maintenant
            if phase != self.phase:
                self.battre(phase)
        elif (maintenant - self.dernier_signal > FRAIS_S
              and maintenant - self.debut_battement > TEMPS_LIBRE_MS / 1000):
            self.battre("7" if self.phase == "6" else "6")  # pas de danse : cadence propre

    def dessiner(self) -> None:
        maintenant = time.monotonic()
        rebond = math.exp(-6 * (maintenant - self.debut_battement))  # 1 au battement
        actif = CHIFFRES.index(self.phase)
        cible = -1.0 if actif == 0 else 1.0  # les yeux partent du côté du chiffre battu
        self.x += (cible - self.x) * 0.25  # balancement amorti
        cligne = maintenant < self.cligne_jusqu_a

        for k, x in enumerate(self.cx):
            px = x + self.x * self.jeu
            py = self.cy - (0.2 * self.r * rebond if k == actif else 0.0)
            echelle = (1.0 + 0.3 * rebond) if k == actif else 0.85
            self.c.coords(self.yeux[k], px, py)
            self.c.itemconfig(self.yeux[k], state="hidden" if cligne else "normal",
                              fill=JAUNE if k == actif else JAUNE_PALE,
                              font=("Helvetica", -int(self.taille * echelle), "bold"))
            self.c.coords(self.paupieres[k], px - self.r * 0.6, self.cy,
                          px + self.r * 0.6, self.cy)
            self.c.itemconfig(self.paupieres[k], state="normal" if cligne else "hidden")

            # Sourcils levés ; celui de l'œil battu monte un peu plus.
            cote = 1 if k == 1 else -1
            haut = self.cy - self.r * (1.3 + (0.15 * rebond if k == actif else 0.0))
            pente = self.r * 0.14 * cote
            self.c.coords(self.sourcils[k], px - self.r * 0.6, haut + pente,
                          px + self.r * 0.6, haut - pente)

    def image(self) -> None:
        self.suivre_cadence()
        self.dessiner()
        self.racine.after(IMAGE_MS, self.image)

    def cligner(self) -> None:
        self.cligne_jusqu_a = time.monotonic() + 0.15
        self.racine.after(random.randint(2500, 6000), self.cligner)

    def demarrer(self) -> None:
        self.image()
        self.racine.after(random.randint(2500, 6000), self.cligner)


def main() -> None:
    racine = tk.Tk()
    racine.attributes("-fullscreen", True)
    racine.configure(bg=NOIR, cursor="none")
    w, h = racine.winfo_screenwidth(), racine.winfo_screenheight()
    canevas = tk.Canvas(racine, bg=NOIR, highlightthickness=0)
    canevas.pack(fill="both", expand=True)
    cadence = Path(sys.argv[1]) if len(sys.argv) > 1 else CADENCE
    Visage(racine, canevas, w, h, cadence).demarrer()
    racine.bind("<Escape>", lambda _e: racine.destroy())
    racine.mainloop()


if __name__ == "__main__":
    main()
