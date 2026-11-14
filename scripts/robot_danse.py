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
    67 [N]        danse « 67 », N cycles (5 par défaut, 1 à 6), bras seuls, puis les
                  bras sont relâchés ; voir plus bas
    etat          consigne, position réelle et température du torse et des jambes en y
    arriere N     penche le torse de N degrés en arrière (0 < N <= 6), moitié sur
                  ``abs_y``, moitié sur ``bust_y`` ; vaut pour debout ET pour le floss
    avant N       idem vers l'avant
    genoux N      plie les genoux de N degrés (-10 à 10, négatif : déplie), hanches et
                  chevilles de N/2 chacune pour garder pied à plat et torse d'aplomb
    chevilles N   chevilles de N degrés (-6 à 6) : + = tibia vers l'avant
    couple N      couple limite des jambes (300 à 900, sur 1023)
    tete          redresse la tête (head_z et head_y à 512) et la tient jusqu'au stop
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

Pose debout : celle de l'ancienne équipe (positions_11), ``abs_y`` 2150 et ``bust_y``
2246. Le torse y penche légèrement en avant, mais elle tient. Un torse reculé de 3°
(2133/2229) a tenu un temps puis a fait tomber le robot. DEBOUT_TORSE fixe ce torse ;
tout écart à positions_11 est reporté sur les poses du floss. Pour régler en cours de
spectacle : ``arriere N`` et ``avant N``.

Danse « 67 » : coudes au corps, avant-bras vers l'avant, puis les mains montent et
descendent en alternance par les coudes, à ``--vitesse``. Le robot tombe facilement en
avant quand ses bras portent du poids devant lui : les mains restent à 20 cm devant les
épaules au plus, et les épaules bougent avant les coudes, bras encore tendus. Le poignet
ne tourne pas : les paumes ne regardent qu'à moitié vers le haut. À chaque battement, la
phase est écrite dans ``/tmp/poppy_67`` : « 6 » quand monte la main droite du robot, à
gauche pour le public, « 7 » pour l'autre. Le visage ``src/ui/poppy_visage_67.py``, dont
les yeux sont un 6 et un 7, s'y cale.

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
# Meilleure pose debout observée le 13 nov. 2026 : celle de positions_11, décalage nul.
# Elle penche un peu en avant mais tient ; 3° plus en arrière (2133/2229) a fini par
# tomber. Un écart à positions_11 serait reporté sur toutes les poses du floss.
DEBOUT_TORSE = {31: 2150, 34: 2246}

# Danse « 67 ». Pose cherchée dans MuJoCo (fix_left_knee_axis et fix_arm_frames) et
# convertie par configs/robot_calibration.yaml. Angles de simulation : épaules y ±1,30 rad,
# épaules x 1,45, arm_z ±0,25, coudes à 100°, puis 120°/80° en alternance. Coude à 3 cm
# devant l'épaule et 4 cm sur le côté, main à 20 cm devant l'épaule, mains à 12 cm
# l'une de l'autre au plus près ; le centre de masse n'avance que de 0,7 cm par rapport
# aux bras pendants de la simulation. Paumes tournées de ~23° vers le haut, sans plus :
# il n'y a pas de rotation du poignet.
EPAULES_67 = {41: 2895, 42: 2993, 43: 2211, 51: 2895, 52: 1103, 53: 2211}
COUDES_67 = {44: 910, 54: 3186}  # 100° ; l_elbow_y plie vers les ticks bas, r_elbow_y vers les hauts
# « 6 » : main droite du robot en haut (120°), gauche en bas (80°) ; c'est la main de
# gauche pour le public, du côté de l'œil « 6 » du visage. « 7 » : l'inverse.
BATTEMENT_67 = (("6", {44: 1138, 54: 3413}),
                ("7", {44: 683, 54: 2958}))
CYCLES_67 = 5
PHASE_67 = Path("/tmp/poppy_67")  # "6", "7" ou vide : lu par src/ui/poppy_visage_67.py


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
    poses.update({f"67 {k}": p for k, p in enumerate(poses_67())})
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


def bras_au_repos(robot: r.Robot) -> dict[int, int]:
    """Position actuelle des bras, ramenée dans les limites EEPROM.

    Un bras qui pend repose parfois juste hors de ses limites (coude à 2030 pour 2036) :
    y renvoyer le moteur déclenche l'erreur 0x02. On revient donc à cette position-ci.
    """
    depart = {}
    for i in BRAS:
        p = robot.lire(i, r.PRESENT, 2)
        depart[i] = min(max(p, robot.lire(i, r.CW_LIMIT, 2)), robot.lire(i, r.CCW_LIMIT, 2))
    return depart


def poses_67() -> list[dict[int, int]]:
    return [EPAULES_67, COUDES_67, *(cibles for _, cibles in BATTEMENT_67)]


def signaler_67(phase: str) -> None:
    """Donne la cadence au visage ; sans effet sur les moteurs, et jamais bloquant."""
    try:
        PHASE_67.write_text(phase)
    except OSError:
        pass


def faire_67(robot: r.Robot, actifs: list[int], vitesse: int, cycles: int) -> None:
    """Danse « 67 » : bras seuls, jambes, tronc et tête restent tenus.

    Épaules d'abord, bras encore tendus, puis coudes à 100° : la main ne passe jamais
    loin devant. Au retour, coudes d'abord, puis épaules, puis bras relâchés.
    """
    depart = bras_au_repos(robot)
    try:
        prendre(robot, BRAS, actifs)
        aller(robot, EPAULES_67, r.VITESSE_LENTE, 10)
        aller(robot, COUDES_67, r.VITESSE_LENTE, 10)
        for _ in range(cycles):
            for phase, cibles in BATTEMENT_67:
                signaler_67(phase)
                aller(robot, cibles, vitesse, 3)
        signaler_67("")
        aller(robot, COUDES_67, r.VITESSE_LENTE, 6)
        aller(robot, {i: depart[i] for i in COUDES_67}, r.VITESSE_LENTE, 10)
        aller(robot, depart, r.VITESSE_LENTE, 10)
    finally:
        signaler_67("")
        relacher_bras(robot, actifs)


def faire_coucou(robot: r.Robot, actifs: list[int], vitesse: int, cycles: int) -> None:
    depart = bras_au_repos(robot)
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

    Ordres : coucou, floss, 67 [N], etat, arriere N, avant N, genoux N, chevilles N,
    couple N, tete, stop (détail en tête du module). Jambes et tronc ne sont relâchés
    qu'au stop, sur erreur, ou au-delà de TEMP_ARRET.
    """
    poses = charger_poses(args.poses)
    debout = pose_debout(poses)
    for nom, p in [("coucou", COUCOU_BRAS), *poses.items(), *(("67", p) for p in poses_67())]:
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
                elif mots[:1] == ["67"] and len(mots) <= 2:
                    cycles = int(mots[1]) if len(mots) == 2 else CYCLES_67
                    if not 1 <= cycles <= 6:
                        raise SystemExit("67 refusé : entre 1 et 6 cycles")
                    faire_67(robot, actifs, args.vitesse, cycles)
                elif ordre == "floss":
                    faire_floss(robot, actifs, poses, args.vitesse, args.cycles, debout)
                elif ordre == "etat":
                    etat_torse(robot, debout)
                elif len(mots) == 2 and mots[0] in ("arriere", "avant"):
                    incliner(robot, debout, poses, mots[0], float(mots[1]))
                elif len(mots) == 2 and mots[0] == "genoux":
                    plier_genoux(robot, debout, poses, float(mots[1]))
                elif len(mots) == 2 and mots[0] == "chevilles":
                    pencher_chevilles(robot, debout, poses, float(mots[1]))
                elif len(mots) == 2 and mots[0] == "couple":
                    serrer_jambes(robot, int(mots[1]))
                elif ordre == "tete":
                    tenir_tete(robot, actifs)
                elif ordre == "stop":
                    break
                elif ordre:
                    print("ordre inconnu", flush=True)
            # Un ordre raté ne fait pas tomber le robot.
            except (SystemExit, ValueError, OSError) as erreur:
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
    """Consigne, position réelle et température du torse et des jambes en y."""
    for i in (31, 32, 33, 34, 35, 13, 14, 15, 23, 24, 25):
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


def decaler(robot: r.Robot, debout: dict[int, int], poses: dict,
            deltas_deg: dict[int, float]) -> None:
    """Décale des articulations (degrés, en ticks moteur) dans la pose debout et le floss."""
    d = {i: round(v / r.deg_par_tick(i)) for i, v in deltas_deg.items()}
    nouveau = {i: debout[i] + d[i] for i in d}
    if hors := hors_limites(robot, nouveau):
        raise SystemExit(f"limite atteinte {hors}")
    debout.update(nouveau)
    for p in poses.values():
        for i in d:
            p[i] += d[i]
    aller(robot, nouveau, r.VITESSE_LENTE, 6)
    etat_torse(robot, debout)


def plier_genoux(robot: r.Robot, debout: dict[int, int], poses: dict, deg: float) -> None:
    """Plie les genoux de ``deg`` (négatif : déplie), hanches et chevilles de deg/2 chacune.

    Pied à plat et torse d'aplomb : la cuisse avance de deg/2, le tibia aussi.
    Ticks : hanches y + = cuisse en avant ; genou gauche plie vers +, droit vers − ;
    chevilles + = pointe qui monte (tibia vers l'avant).
    """
    if not 0 < abs(deg) <= 10:
        raise SystemExit("pas refusé : genoux entre -10 et 10°")
    h = deg / 2
    decaler(robot, debout, poses, {13: h, 23: h, 14: deg, 24: -deg, 15: h, 25: h})


def pencher_chevilles(robot: r.Robot, debout: dict[int, int], poses: dict, deg: float) -> None:
    """Chevilles de ``deg`` : + = tibia vers l'avant (pointe qui monte), − = vers l'arrière."""
    if not 0 < abs(deg) <= PAS_MAX_DEG:
        raise SystemExit(f"pas refusé : chevilles entre -{PAS_MAX_DEG} et {PAS_MAX_DEG}°")
    decaler(robot, debout, poses, {15: deg, 25: deg})


def tenir_tete(robot: r.Robot, actifs: list[int]) -> None:
    """Redresse la tête (head_z 512, head_y 512) et la garde tenue jusqu'au stop.

    head_y pend souvent sous sa limite basse : on écrit d'abord la vitesse, ce qui
    active le couple et ramène doucement la consigne sur la limite (méthode du
    coucou de robot_dxl, qui a fonctionné), puis on monte à 512.
    """
    for i in (36, 37):
        if i not in actifs:
            actifs.append(i)  # avant toute écriture : chacune peut activer le couple
            robot.ecrire(i, r.SPEED, r.VITESSE_LENTE, 2)
            robot.ecrire(i, r.TORQUE_LIMIT, COUPLE, 2)
            robot.ecrire(i, r.TORQUE_ENABLE, 1, 1)
    r.attendre_arret(robot, [37], 6)
    aller(robot, {36: 512, 37: robot.lire(37, r.CW_LIMIT, 2)}, r.VITESSE_LENTE, 8)
    aller(robot, {37: 512}, r.VITESSE_LENTE, 8)
    for i in (36, 37):
        print(f"  {r.NOMS[i]} à {robot.lire(i, r.PRESENT, 2)} (cible 512)", flush=True)


def serrer_jambes(robot: r.Robot, couple: int) -> None:
    """Couple limite des jambes (registre TORQUE_LIMIT, 1023 = 100 %)."""
    if not 300 <= couple <= 900:
        raise SystemExit("couple refusé : entre 300 et 900")
    for i in r.JAMBES:
        robot.ecrire(i, r.TORQUE_LIMIT, couple, 2)
    print(f"  couple des jambes à {couple} ({couple * 100 // 1023} %)", flush=True)


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
