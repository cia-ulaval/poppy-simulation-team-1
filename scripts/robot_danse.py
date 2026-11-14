"""Démonstration sur le Poppy réel : debout, coucou et floss de l'ancienne équipe.

Ce script tourne **sur le robot**, dans le même dossier que ``robot_dxl.py`` dont il
reprend les garde-fous : ports lus dans ``POPPY_DXL_PORT_JAMBES`` et
``POPPY_DXL_PORT_HAUT`` sans défaut, vitesse plafonnée, couple limité à 500,
température surveillée, couple coupé puis relu à la sortie, même sur erreur ou signal.
Testé sur le robot le 13 novembre 2026. Procédure et pièges : docs/robot-reel.md,
section « Démonstration ».

Les poses sont celles de l'ancienne équipe (``positions_11.csv`` à ``positions_14.csv``),
versionnées dans configs/robot_poses/. On les copie avec les scripts :

    scp scripts/robot_dxl.py scripts/robot_danse.py configs/robot_poses/*.csv poppy:/tmp/
    ssh -t poppy
    export POPPY_DXL_PORT_JAMBES=/dev/serial/by-id/usb-Xevelabs_USB2AX_<…>-if00
    export POPPY_DXL_PORT_HAUT=/dev/serial/by-id/usb-Xevelabs_USB2AX_<…>-if00
    cd /tmp && python3 robot_danse.py verifier

Sous-commandes :
    verifier      lecture seule : chaque pose est-elle dans les limites EEPROM, et à
                  quelle distance la position actuelle est-elle de debout et de coucou ?
    debout        jambes et tronc en pose debout, bras libres, tenue ``--maintien`` s
                  (20 par défaut, 120 au plus), puis relâchement
    coucou        bras seuls : pose de coucou, le coude droit plie et déplie ``--cycles``
                  fois, retour à la pose de départ, relâchement
    floss         tous les moteurs sauf la tête : pose 11, puis ``--cycles`` fois la
                  suite 11, 12, 13, 12, 11, 14
    spectacle     met debout, tient jambes et tronc, et attend des ordres écrits dans
                  ``/tmp/poppy_ordre`` (lu chaque seconde, puis vidé). Les bras ne sont
                  pris que le temps d'un ordre, puis relâchés en 3 s.

Ordres de ``spectacle``, par exemple ``echo floss > /tmp/poppy_ordre`` :
    coucou        un coucou, puis les bras sont relâchés
    floss         un floss, retour debout, puis les bras sont relâchés
    etat          consigne, position réelle et température des cinq moteurs du torse
    arriere N     penche le torse de N degrés en arrière (0 < N <= 6), moitié sur
                  ``abs_y``, moitié sur ``bust_y`` ; vaut pour debout ET pour le floss
    avant N       idem vers l'avant
    stop          relâche tout et quitte
Un ordre qui échoue est abandonné et le robot revient debout : les jambes restent
tenues. Jambes et tronc ne sont relâchés qu'au ``stop``, sur erreur, sur Ctrl-C ou
signal, ou si un moteur atteint TEMP_ARRET.

Options des commandes qui bougent :
    --confirme      obligatoire : quelqu'un surveille, une main sur l'alimentation
    --tolerance     degrés de mouvement tolérés avant de démarrer (3 par défaut)
    --marge-deg     un moteur hors de ses limites EEPROM d'au plus cette marge est
                    ramené sur sa limite au lieu d'être refusé (2,6° par défaut, 15 au
                    plus). Robot couché, ``bust_y`` en sort : prendre 12.
    --poses DIR     dossier des positions_*.csv (dossier courant par défaut ; vaut
                    aussi pour ``verifier``)
    --cycles, --vitesse   coucou, floss, spectacle : 3 cycles, vitesse 120 (200 au plus)

Pose debout : celle de l'ancienne équipe (positions_11) penche en avant. Le torse est
reculé de 3° : ``abs_y`` 2133 et ``bust_y`` 2229 au lieu de 2150 et 2246. Le même
décalage s'applique à toutes les poses du floss (voir DEBOUT_TORSE).

Positions en ticks : 0-4095 sur 360° (MX), 0-1023 sur 300° (AX).
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import robot_dxl as r

FLOSS = ["11", "12", "13", "12", "11", "14"]
JAMBES_TRONC = r.JAMBES + [31, 32, 33, 34, 35]
BRAS = [41, 42, 43, 44, 51, 52, 53, 54]
# motor_node_test.py de l'équipe précédente : bras posés, puis le coude droit plie et déplie.
COUCOU_BRAS = {41: 2048 + 1024, 42: 2048 + 1024, 43: 2048, 44: 2048,
               51: 2048 - 800, 52: 2048, 53: 2048 + 800, 54: 2048}
COUCOU_COUDE = (3500, 2048)
COUPLE = 500  # comme l'équipe précédente
VITESSE_MAX = 200
TEMP_ALERTE, TEMP_ARRET = 50, 60  # °C ; l'alarme interne des MX coupe vers 70-80
MARGE = 30  # ticks, ~2,6° sur un MX
ORDRES = Path("/tmp/poppy_ordre")
TORSE = (31, 34)  # abs_y, bust_y : sens -1, des ticks en plus penchent le torse en avant
# Meilleure pose debout observée le 13 nov. 2026 : 3° plus en arrière que positions_11
# (2150 et 2246), qui penchait en avant. Le même décalage vaut pour les poses du floss.
DEBOUT_TORSE = {31: 2133, 34: 2229}


def lire_pose(dossier: Path, nom: str) -> dict[int, int]:
    chemin = dossier / f"positions_{nom}.csv"
    if not chemin.is_file():
        sys.exit(f"refusé : {chemin} introuvable (voir --poses)")
    with open(chemin, encoding="utf-8") as f:
        return {int(ligne["motor_id"]): int(ligne["value"]) for ligne in csv.DictReader(f)}


def charger_poses(dossier: Path) -> dict[str, dict[int, int]]:
    """Poses du floss, torse décalé pour que la pose 11 tombe sur DEBOUT_TORSE."""
    poses = {n: lire_pose(dossier, n) for n in sorted(set(FLOSS))}
    decalage = {i: DEBOUT_TORSE[i] - poses["11"][i] for i in TORSE}
    for p in poses.values():
        for i in TORSE:
            p[i] += decalage[i]
    return poses


def pose_debout(poses: dict[str, dict[int, int]]) -> dict[int, int]:
    return {i: poses["11"][i] for i in JAMBES_TRONC}


def hors_limites(robot: r.Robot, cibles: dict[int, int]) -> list[str]:
    return [f"{r.NOMS[i]}={c}" for i, c in cibles.items() if not r.dans_limites(robot, i, c)]


def prendre(robot: r.Robot, ids: list[int], actifs: list[int]) -> None:
    """Couple sur place sur ``ids``. Refuse si l'un d'eux est hors limites."""
    for i in ids:
        actifs.append(i)
        if r.couple_sur_place(robot, i, r.VITESSE_LENTE, COUPLE):
            continue
        # Hors limites : à moins de MARGE ticks, on le ramène sur sa limite. La consigne
        # écrite est DANS les limites, donc pas de rabattement surprise.
        p = robot.lire(i, r.PRESENT, 2)
        borne = min(max(p, robot.lire(i, r.CW_LIMIT, 2)), robot.lire(i, r.CCW_LIMIT, 2))
        if abs(borne - p) > MARGE:
            actifs.remove(i)
            raise SystemExit(f"refusé : {r.NOMS[i]} est hors limites ({p}), replace-le à la main")
        robot.ecrire(i, r.GOAL, borne, 2)
        robot.ecrire(i, r.SPEED, r.VITESSE_LENTE, 2)
        robot.ecrire(i, r.TORQUE_LIMIT, COUPLE, 2)
        robot.ecrire(i, r.TORQUE_ENABLE, 1, 1)
        print(f"  {r.NOMS[i]} ramené de {p} à sa limite {borne}", flush=True)


def aller(robot: r.Robot, cibles: dict[int, int], vitesse: int, delai: float) -> None:
    for i in cibles:
        robot.ecrire(i, r.SPEED, vitesse, 2)
    for i, c in cibles.items():
        robot.ecrire(i, r.GOAL, c, 2)
    r.attendre_arret(robot, list(cibles), delai)
    chauds = [r.NOMS[i] for i in cibles if robot.lire(i, r.TEMPERATURE, 1) >= TEMP_ARRET]
    if chauds:
        raise SystemExit(f"arrêt : trop chaud sur {chauds}")


def ecarts(robot: r.Robot, cibles: dict[int, int]) -> str:
    return ", ".join(
        f"{r.NOMS[i]} {(c - robot.lire(i, r.PRESENT, 2)) * r.deg_par_tick(i):+.0f}°"
        for i, c in cibles.items())


def cmd_verifier(robot: r.Robot, args: argparse.Namespace) -> None:
    floss = charger_poses(args.poses)
    poses = {"debout": pose_debout(floss), "coucou": COUCOU_BRAS}
    poses.update({f"floss {n}": p for n, p in floss.items()})
    for nom, cibles in poses.items():
        print(f"{nom:9s} hors limites : {hors_limites(robot, cibles) or 'aucun'}")
    print("écart actuel → debout :", ecarts(robot, poses["debout"]))
    print("écart actuel → coucou :", ecarts(robot, COUCOU_BRAS))


def cmd_debout(robot: r.Robot, args: argparse.Namespace) -> None:
    cibles = pose_debout(charger_poses(args.poses))
    if hors := hors_limites(robot, cibles):
        sys.exit(f"refusé : pose hors limites {hors}")
    r.verifier_depart(robot, JAMBES_TRONC, args.tolerance)
    actifs: list[int] = []
    try:
        prendre(robot, JAMBES_TRONC, actifs)
        print("== debout (jambes et tronc ; bras libres)", flush=True)
        aller(robot, cibles, r.VITESSE_LENTE, 10)
        print(f"== maintien {args.maintien:.0f} s", flush=True)
        time.sleep(args.maintien)
    finally:
        r.relacher(robot, actifs)


def cmd_coucou(robot: r.Robot, args: argparse.Namespace) -> None:
    if hors := hors_limites(robot, COUCOU_BRAS):
        sys.exit(f"refusé : pose hors limites {hors}")
    r.verifier_depart(robot, BRAS, args.tolerance)
    depart = {i: robot.lire(i, r.PRESENT, 2) for i in BRAS}
    actifs: list[int] = []
    try:
        prendre(robot, BRAS, actifs)
        print("== bras en place", flush=True)
        aller(robot, COUCOU_BRAS, r.VITESSE_LENTE, 10)
        print("== coucou", flush=True)
        for _ in range(args.cycles):
            for c in COUCOU_COUDE:
                aller(robot, {54: c}, args.vitesse, 4)
        print("== retour", flush=True)
        aller(robot, depart, r.VITESSE_LENTE, 10)
    finally:
        r.relacher(robot, actifs)


def cmd_floss(robot: r.Robot, args: argparse.Namespace) -> None:
    poses = charger_poses(args.poses)
    for n, p in poses.items():
        if hors := hors_limites(robot, p):
            sys.exit(f"refusé : pose {n} hors limites {hors}")
    ids = sorted(poses["11"])
    r.verifier_depart(robot, ids, args.tolerance)
    actifs: list[int] = []
    try:
        prendre(robot, ids, actifs)
        print("== mise en place lente", flush=True)
        aller(robot, poses["11"], r.VITESSE_LENTE, 15)
        for k in range(args.cycles):
            print(f"== floss {k + 1}/{args.cycles}", flush=True)
            for n in FLOSS:
                aller(robot, poses[n], args.vitesse, 3)
        aller(robot, poses["11"], r.VITESSE_LENTE, 10)
    finally:
        r.relacher(robot, actifs)


def relacher_bras(robot: r.Robot, actifs: list[int]) -> None:
    """Relâche les bras en 3 s ; jambes et tronc gardent leur couple."""
    bras = [i for i in BRAS if i in actifs]
    for limite in (240, 120, 0):
        for i in bras:
            robot.ecrire(i, r.TORQUE_LIMIT, limite, 2)
        time.sleep(1)
    if restes := r.couper(robot, bras):
        raise SystemExit(f"ALERTE : couple resté sur {restes}")
    for i in bras:
        actifs.remove(i)


def faire_coucou(robot: r.Robot, actifs: list[int], vitesse: int, cycles: int) -> None:
    depart = {i: robot.lire(i, r.PRESENT, 2) for i in BRAS}
    try:
        prendre(robot, BRAS, actifs)
        aller(robot, COUCOU_BRAS, r.VITESSE_LENTE, 10)
        for _ in range(cycles):
            for c in COUCOU_COUDE:
                aller(robot, {54: c}, vitesse, 4)
        aller(robot, depart, r.VITESSE_LENTE, 10)
    finally:
        relacher_bras(robot, actifs)


def faire_floss(robot: r.Robot, actifs: list[int], poses: dict, vitesse: int,
                cycles: int, debout: dict[int, int]) -> None:
    try:
        prendre(robot, BRAS, actifs)
        aller(robot, poses["11"], r.VITESSE_LENTE, 15)
        for _ in range(cycles):
            for n in FLOSS:
                aller(robot, poses[n], vitesse, 3)
        aller(robot, poses["11"], r.VITESSE_LENTE, 10)
        aller(robot, debout, r.VITESSE_LENTE, 10)
    finally:
        relacher_bras(robot, actifs)


def cmd_spectacle(robot: r.Robot, args: argparse.Namespace) -> None:
    """Debout, tient jambes et tronc, et exécute les ordres écrits dans /tmp/poppy_ordre.

    Ordres : coucou, floss, etat, arriere N, avant N, stop (détail en tête du module).
    Jambes et tronc ne sont relâchés qu'au stop, sur erreur, ou au-delà de TEMP_ARRET.
    """
    poses = charger_poses(args.poses)
    debout = pose_debout(poses)
    for nom, p in [("coucou", COUCOU_BRAS), *poses.items()]:
        if hors := hors_limites(robot, p):
            sys.exit(f"refusé : pose {nom} hors limites {hors}")
    r.verifier_depart(robot, JAMBES_TRONC, args.tolerance)
    ORDRES.write_text("")
    actifs: list[int] = []
    alerte = False
    try:
        prendre(robot, JAMBES_TRONC, actifs)
        aller(robot, debout, r.VITESSE_LENTE, 10)
        print("debout, j'attends les ordres", flush=True)
        while True:
            temps = {i: robot.lire(i, r.TEMPERATURE, 1) for i in actifs}
            pire = max(temps, key=temps.get)
            if temps[pire] >= TEMP_ARRET:
                raise SystemExit(f"arrêt : {r.NOMS[pire]} à {temps[pire]}°C")
            if temps[pire] >= TEMP_ALERTE and not alerte:
                alerte = True
                print(f"ALERTE chaleur : {r.NOMS[pire]} à {temps[pire]}°C", flush=True)
            ordre = ORDRES.read_text().strip()
            if ordre:
                ORDRES.write_text("")
                print(f"ordre : {ordre} ({time.strftime('%H:%M:%S')})", flush=True)
            mots = ordre.split()
            try:
                if ordre == "coucou":
                    faire_coucou(robot, actifs, args.vitesse, args.cycles)
                elif ordre == "floss":
                    faire_floss(robot, actifs, poses, args.vitesse, args.cycles, debout)
                elif ordre == "etat":
                    etat_torse(robot, debout)
                elif len(mots) == 2 and mots[0] in ("arriere", "avant"):
                    incliner(robot, debout, poses, mots[0], float(mots[1]))
                elif ordre == "stop":
                    break
                elif ordre:
                    print("ordre inconnu", flush=True)
            except (SystemExit, ValueError) as erreur:  # un ordre raté ne fait pas tomber le robot
                print(f"ordre abandonné : {erreur} ; jambes toujours tenues", flush=True)
                aller(robot, debout, r.VITESSE_LENTE, 10)
            if ordre:
                print(f"fait, j'attends ; max {temps[pire]}°C ({r.NOMS[pire]})", flush=True)
            time.sleep(1)
    finally:
        print("relâchement de tout ; torse final :", flush=True)
        try:
            etat_torse(robot, debout)
        except OSError:
            pass
        r.relacher(robot, actifs)


PAS_MAX_DEG = 6.0


def etat_torse(robot: r.Robot, debout: dict[int, int]) -> None:
    """Consigne, position réelle et température des moteurs du torse."""
    for i in (31, 32, 33, 34, 35):
        p = robot.lire(i, r.PRESENT, 2)
        print(f"  {r.NOMS[i]:7s} consigne {debout[i]} réel {p} "
              f"(affaissement {(p - debout[i]) * r.deg_par_tick(i):+.1f}°) "
              f"{robot.lire(i, r.TEMPERATURE, 1)}°C", flush=True)


def incliner(robot: r.Robot, debout: dict[int, int], poses: dict, sens: str, deg: float) -> None:
    """Penche le torse de ``deg`` degrés en tout, moitié sur abs_y, moitié sur bust_y.

    La nouvelle consigne vaut pour la pose debout et pour toutes les poses du floss.
    """
    if not 0 < deg <= PAS_MAX_DEG:
        raise SystemExit(f"pas refusé : entre 0 et {PAS_MAX_DEG}°")
    d = round(deg / 2 / r.deg_par_tick(31)) * (-1 if sens == "arriere" else 1)
    nouveau = {i: debout[i] + d for i in TORSE}
    if hors := hors_limites(robot, nouveau):
        raise SystemExit(f"limite atteinte {hors}")
    debout.update(nouveau)
    for p in poses.values():
        for i in TORSE:
            p[i] += d
    aller(robot, nouveau, r.VITESSE_LENTE, 6)
    etat_torse(robot, debout)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sous = p.add_subparsers(dest="commande", required=True)
    aide_poses = "dossier des positions_*.csv (dossier courant par défaut)"
    sous.add_parser("verifier").add_argument("--poses", type=Path, default=Path("."),
                                             help=aide_poses)
    for nom in ("debout", "coucou", "floss", "spectacle"):
        s = sous.add_parser(nom)
        s.add_argument("--poses", type=Path, default=Path("."), help=aide_poses)
        s.add_argument("--confirme", action="store_true", required=True)
        s.add_argument("--tolerance", type=float, default=3.0)
        s.add_argument("--marge-deg", type=float, default=2.6,
                       help="moteur hors limites ramené sur sa limite s'il en est à moins (max 15)")
        if nom == "debout":
            s.add_argument("--maintien", type=float, default=20.0)
        else:
            s.add_argument("--cycles", type=int, default=3)
            s.add_argument("--vitesse", type=int, default=120)
    args = p.parse_args()
    if getattr(args, "vitesse", 0) > VITESSE_MAX or not 0 < getattr(args, "cycles", 1) <= 10:
        sys.exit(f"refusé : vitesse max {VITESSE_MAX}, cycles entre 1 et 10")
    if getattr(args, "maintien", 0) > 120:
        sys.exit("refusé : maintien max 120 s")
    global MARGE
    if hasattr(args, "marge_deg"):
        if not 0 <= args.marge_deg <= 15:
            sys.exit("refusé : --marge-deg entre 0 et 15")
        MARGE = round(args.marge_deg / (360 / 4096))
    signal.signal(signal.SIGTERM, r.sortir)
    signal.signal(signal.SIGHUP, r.sortir)
    commandes = {"verifier": cmd_verifier, "debout": cmd_debout,
                 "coucou": cmd_coucou, "floss": cmd_floss, "spectacle": cmd_spectacle}
    try:
        commandes[args.commande](r.ouvrir_robot(), args)
    except KeyboardInterrupt:
        print("\narrêt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
