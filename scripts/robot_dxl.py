"""Outil Dynamixel pour le Poppy réel : lire, tester les sens, poser au zéro, faire coucou.

Ce script tourne **sur le robot** (le Raspberry Pi), sans le reste du dépôt : il ne
dépend que de ``pyserial``, plus ``pyyaml`` pour ``pose-zero``. On le copie et on le
lance là-bas, par exemple :

    scp scripts/robot_dxl.py configs/robot_calibration.yaml poppy:/tmp/
    ssh -t poppy 'export POPPY_DXL_PORT_JAMBES=/dev/serial/by-id/...F04071-if00 \\
        POPPY_DXL_PORT_HAUT=/dev/serial/by-id/...117180-if00; python3 /tmp/robot_dxl.py lire'

Sous-commandes :
    lire          état des 25 moteurs (lecture seule)
    suivre        positions en continu, pour manipuler à la main (lecture seule)
    tester-sens   chaque articulation tourne d'une amplitude puis revient, une à la fois
    pose-zero     met le robot à l'angle 0 de la simulation, par groupes, puis relâche
    coucou        redresse la tête et agite le bras droit

Sécurité (voir CONTRIBUTING.md et docs/robot-reel.md) :
- les ports viennent de ``POPPY_DXL_PORT_JAMBES`` et ``POPPY_DXL_PORT_HAUT``, sans
  valeur par défaut : sans eux, le script refuse de démarrer. Chaque port est vérifié
  en interrogeant un moteur attendu (11 sur les jambes, 31 sur le haut) ;
- toute commande qui bouge exige ``--confirme`` et refuse de démarrer si un moteur a
  déjà du couple, ou si le robot bouge (quelqu'un le manipule) ;
- consigne = position actuelle AVANT tout réglage : sur AX/MX, écrire la vitesse, le
  couple limite ou la consigne **réactive le couple** ;
- un moteur hors de ses limites EEPROM n'est jamais écrit : une consigne hors limites
  est rabattue sur la limite ET active le couple, malgré l'erreur renvoyée ;
- couple bridé, vitesse lente, amplitude plafonnée ;
- à la sortie, même sur erreur, Ctrl-C, SIGTERM ou SIGHUP : couple coupé EN DERNIER,
  puis relu, sur les 25 moteurs.

Positions en ticks : 0-4095 sur 360° (MX), 0-1023 sur 300° (AX). Angles en degrés.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import serial

BAUD = 1_000_000
PING, READ, WRITE = 0x01, 0x02, 0x03
MODEL, CW_LIMIT, CCW_LIMIT = 0, 6, 8
TORQUE_ENABLE, LED, GOAL, SPEED, TORQUE_LIMIT = 24, 25, 30, 32, 34
PRESENT, LOAD, VOLTAGE, TEMPERATURE, MOVING = 36, 40, 42, 43, 46

NOMS = {
    11: "l_hip_x", 12: "l_hip_z", 13: "l_hip_y", 14: "l_knee_y", 15: "l_ankle_y",
    21: "r_hip_x", 22: "r_hip_z", 23: "r_hip_y", 24: "r_knee_y", 25: "r_ankle_y",
    31: "abs_y", 32: "abs_x", 33: "abs_z", 34: "bust_y", 35: "bust_x",
    36: "head_z", 37: "head_y",
    41: "l_shoulder_y", 42: "l_shoulder_x", 43: "l_arm_z", 44: "l_elbow_y",
    51: "r_shoulder_y", 52: "r_shoulder_x", 53: "r_arm_z", 54: "r_elbow_y",
}
JAMBES = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]
HAUT = [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 51, 52, 53, 54]
AX = {36, 37}
MODELES = {12: "AX-12", 18: "AX-18", 29: "MX-28", 310: "MX-64", 320: "MX-106"}
GROUPES = {
    "tronc": [31, 32, 33, 34, 35, 36, 37],
    "bras": [41, 42, 43, 44, 51, 52, 53, 54],
    "jambes": [11, 12, 13, 14, 15, 21, 22, 23, 24, 25],
}
# arm_z : coude plié de 60° vers l'avant pendant le test, sinon la rotation ne se voit
# pas. Coudes directs, zéro 2048 : l_elbow_y plie vers les ticks bas, r_elbow_y vers les hauts.
COUDE_PLIE = {43: (44, 2048 - 683), 53: (54, 2048 + 683)}

VITESSE_LENTE = 25  # registre MOVING_SPEED, ~2,9 tr/min soit ~17 °/s
COUPLE_BRIDE = 450  # registre TORQUE_LIMIT, ~44 % du couple maximal
AMPLITUDE_MAX_DEG = 30.0
TEMPERATURE_MAX = 55  # °C


def deg_par_tick(i: int) -> float:
    """Degrés par tick : 300/1024 pour les AX, 360/4096 pour les MX."""
    return 300 / 1024 if i in AX else 360 / 4096


def zero_par_defaut(i: int) -> int:
    """Ticks au milieu de la course : 512 (AX) ou 2048 (MX)."""
    return 512 if i in AX else 2048


def paquet(dxl_id: int, instr: int, params: tuple[int, ...] = ()) -> bytes:
    """Paquet Dynamixel protocole 1.0 : en-tête, id, longueur, instruction, somme."""
    corps = [dxl_id, len(params) + 2, instr, *params]
    return bytes([0xFF, 0xFF, *corps, (~sum(corps)) & 0xFF])


class Bus:
    """Un bus Dynamixel (protocole 1.0, 1 Mbit/s) sur un port série."""

    def __init__(self, port: str) -> None:
        self.port = port
        self.ser = serial.Serial(port, BAUD, timeout=0.05)

    def _echange(self, i: int, instr: int, params: tuple[int, ...] = (), n: int = 0) -> bytes:
        self.ser.reset_input_buffer()
        self.ser.write(paquet(i, instr, params))
        rep = self.ser.read(6 + n)
        if len(rep) != 6 + n or rep[:2] != b"\xff\xff" or rep[2] != i:
            raise OSError(f"id {i} : pas de réponse valide sur {self.port}")
        if (~sum(rep[2:-1])) & 0xFF != rep[-1]:
            raise OSError(f"id {i} : somme de contrôle invalide")
        if rep[4]:
            raise OSError(f"id {i} : erreur moteur 0x{rep[4]:02x}")
        return rep[5:-1]

    def repond(self, i: int) -> bool:
        """Vrai si le moteur ``i`` répond à un PING."""
        try:
            self._echange(i, PING)
        except OSError:
            return False
        return True

    def lire(self, i: int, adr: int, n: int) -> int:
        """Lit ``n`` octets (1 ou 2, petit-boutiste) à l'adresse ``adr``."""
        d = self._echange(i, READ, (adr, n), n)
        return d[0] if n == 1 else d[0] | d[1] << 8

    def ecrire(self, i: int, adr: int, val: int, n: int) -> None:
        """Écrit ``n`` octets (1 ou 2). Attention : GOAL, SPEED et TORQUE_LIMIT activent le couple."""
        octets = (val & 0xFF,) if n == 1 else (val & 0xFF, (val >> 8) & 0xFF)
        self._echange(i, WRITE, (adr, *octets))


@dataclass
class Robot:
    """Les deux bus, et pour chaque identifiant le bus qui le porte."""

    bus_de: dict[int, Bus]

    @property
    def tous(self) -> list[int]:
        """Les 25 identifiants."""
        return list(self.bus_de)

    def lire(self, i: int, adr: int, n: int) -> int:
        """Lit un registre du moteur ``i``."""
        return self.bus_de[i].lire(i, adr, n)

    def ecrire(self, i: int, adr: int, val: int, n: int) -> None:
        """Écrit un registre du moteur ``i``."""
        self.bus_de[i].ecrire(i, adr, val, n)


def ouvrir_robot() -> Robot:
    """Ouvre les deux bus depuis l'environnement et vérifie qu'ils ne sont pas inversés.

    Raises:
        SystemExit: Si une variable manque ou si un port ne porte pas le bon bus.
    """
    ports = {}
    for bus, var, temoin in (("jambes", "POPPY_DXL_PORT_JAMBES", 11),
                             ("haut", "POPPY_DXL_PORT_HAUT", 31)):
        port = os.environ.get(var)
        if not port:
            sys.exit(f"refusé : {var} n'est pas défini (aucun port par défaut, voir le --help)")
        b = Bus(port)
        if not b.repond(temoin):
            sys.exit(f"refusé : le moteur {temoin} ne répond pas sur {port} ({var})")
        ports[bus] = b
    return Robot({**{i: ports["jambes"] for i in JAMBES}, **{i: ports["haut"] for i in HAUT}})


def couper(robot: Robot, ids: list[int]) -> list[int]:
    """Coupe le couple et le relit. Renvoie les identifiants restés sous couple."""
    restes = []
    for i in ids:
        for _ in range(3):
            try:
                robot.ecrire(i, TORQUE_ENABLE, 0, 1)
                if robot.lire(i, TORQUE_ENABLE, 1) == 0:
                    break
            except OSError:
                time.sleep(0.05)
        else:
            restes.append(i)
    return restes


def couper_et_rapporter(robot: Robot) -> None:
    """Coupe le couple sur les 25 moteurs et dit si c'est fait."""
    restes = couper(robot, robot.tous)
    if restes:
        print(f"ALERTE : couple encore actif sur {restes}", flush=True)
    else:
        print("couple coupé et relu à 0 sur les 25 moteurs", flush=True)


def verifier_depart(robot: Robot, ids: list[int], tolerance_deg: float) -> dict[int, int]:
    """Refuse si un moteur a du couple ou si le robot bouge. Renvoie les positions (ticks).

    Attend jusqu'à 30 s qu'aucune articulation de ``ids`` ne bouge de plus de
    ``tolerance_deg`` en 2 s : un robot suspendu se balance, une main le déplace.
    """
    if any(robot.lire(i, TORQUE_ENABLE, 1) for i in robot.tous):
        sys.exit("refusé : un moteur a déjà du couple, quelqu'un d'autre pilote peut-être")
    debut, p2 = time.time(), {i: robot.lire(i, PRESENT, 2) for i in ids}
    while True:
        p1 = p2
        time.sleep(2)
        p2 = {i: robot.lire(i, PRESENT, 2) for i in ids}
        pire = max(abs(p2[i] - p1[i]) * deg_par_tick(i) for i in ids)
        if pire <= tolerance_deg:
            return p2
        if time.time() - debut > 30:
            sys.exit(f"refusé : le robot bouge encore ({pire:.1f}° en 2 s) après 30 s")


def dans_limites(robot: Robot, i: int, ticks: int) -> bool:
    """Vrai si ``ticks`` est dans les limites EEPROM du moteur ``i``."""
    return robot.lire(i, CW_LIMIT, 2) <= ticks <= robot.lire(i, CCW_LIMIT, 2)


def couple_sur_place(robot: Robot, i: int, vitesse: int, couple: int) -> bool:
    """Met le couple sur ``i`` sans le faire bouger. Faux si le moteur est hors limites.

    L'ordre compte : consigne = position actuelle d'abord, car écrire la vitesse ou
    le couple limite active le couple sur la consigne courante.
    """
    p = robot.lire(i, PRESENT, 2)
    if not dans_limites(robot, i, p):
        return False
    robot.ecrire(i, GOAL, p, 2)
    robot.ecrire(i, SPEED, vitesse, 2)
    robot.ecrire(i, TORQUE_LIMIT, couple, 2)
    robot.ecrire(i, TORQUE_ENABLE, 1, 1)
    return True


def attendre_arret(robot: Robot, ids: list[int], delai_s: float) -> None:
    """Attend que les moteurs ``ids`` aient fini de bouger, au plus ``delai_s`` secondes."""
    debut = time.time()
    while time.time() - debut < delai_s:
        time.sleep(0.2)
        if not any(robot.lire(i, MOVING, 1) for i in ids):
            return


def cmd_lire(robot: Robot, _args: argparse.Namespace) -> None:
    """Affiche l'état des 25 moteurs. Lecture seule."""
    for i in robot.tous:
        modele = MODELES.get(robot.lire(i, MODEL, 2), "?")
        p = robot.lire(i, PRESENT, 2)
        cw, ccw = robot.lire(i, CW_LIMIT, 2), robot.lire(i, CCW_LIMIT, 2)
        hors = "" if cw <= p <= ccw else "  HORS LIMITES"
        print(f"{i:2d} {NOMS[i]:13s} {modele:6s} couple={robot.lire(i, TORQUE_ENABLE, 1)}"
              f" pos={p:4d} ({(p - zero_par_defaut(i)) * deg_par_tick(i):+7.1f}°)"
              f" limites=[{cw},{ccw}] {robot.lire(i, VOLTAGE, 1) / 10:.1f}V"
              f" {robot.lire(i, TEMPERATURE, 1)}°C{hors}")


def cmd_suivre(robot: Robot, args: argparse.Namespace) -> None:
    """Affiche des positions 5 fois par seconde jusqu'à Ctrl-C. Lecture seule."""
    ids = args.ids or [12, 22, 11, 21]
    if any(robot.lire(i, TORQUE_ENABLE, 1) for i in ids):
        sys.exit("refusé : un de ces moteurs a du couple, ne manipule pas le robot")
    print("   ".join(f"{NOMS[i]:>15s}" for i in ids))
    while True:
        print("   ".join(f"{(robot.lire(i, PRESENT, 2) - zero_par_defaut(i)) * deg_par_tick(i):+14.1f}°"
                         for i in ids), flush=True)
        time.sleep(0.2)


def tester_un(robot: Robot, i: int, amplitude_deg: float) -> str:
    """Tourne ``i`` d'``amplitude_deg`` (vers + si possible), revient, coupe. Renvoie le compte rendu."""
    p = robot.lire(i, PRESENT, 2)
    d = round(amplitude_deg / deg_par_tick(i))
    cible = p + d if dans_limites(robot, i, p + d) else p - d
    if not (dans_limites(robot, i, p) and dans_limites(robot, i, cible)):
        return "sauté (hors limites ou pas de place)"
    coude = COUDE_PLIE.get(i)
    if coude and not couple_sur_place(robot, coude[0], 40, COUPLE_BRIDE):
        return "sauté (coude hors limites)"
    if coude:
        robot.ecrire(coude[0], GOAL, coude[1], 2)
        attendre_arret(robot, [coude[0]], 3)
    couple_sur_place(robot, i, 40, COUPLE_BRIDE)
    robot.ecrire(i, LED, 1, 1)
    robot.ecrire(i, GOAL, cible, 2)
    time.sleep(1.5)
    atteint = robot.lire(i, PRESENT, 2)
    robot.ecrire(i, GOAL, p, 2)
    time.sleep(1.2)
    robot.ecrire(i, LED, 0, 1)
    if coude:
        couper(robot, [i])
        robot.ecrire(coude[0], GOAL, robot.lire(coude[0], PRESENT, 2), 2)
    if couper(robot, [i] + ([coude[0]] if coude else [])):
        raise SystemExit(f"ALERTE : couple resté actif autour de {i}")
    signe = "+" if cible > p else "−"
    return f"{signe} : {p}→{atteint} ({(atteint - p) * deg_par_tick(i):+.1f}°) puis retour"


def cmd_tester_sens(robot: Robot, args: argparse.Namespace) -> None:
    """Teste le sens des articulations demandées, une à la fois, 3 s d'écart."""
    if not 0 < args.amplitude <= AMPLITUDE_MAX_DEG:
        sys.exit(f"refusé : amplitude hors de ]0, {AMPLITUDE_MAX_DEG}]°")
    ids = args.ids or GROUPES[args.groupe]
    verifier_depart(robot, ids, args.tolerance)
    try:
        for n, i in enumerate(ids, 1):
            time.sleep(3)
            print(f"  {n}. {NOMS[i]:13s} {tester_un(robot, i, args.amplitude)}", flush=True)
    finally:
        couper_et_rapporter(robot)


def charger_zeros(chemin: Path) -> dict[int, int]:
    """Zéros (ticks) par identifiant, lus dans le YAML de calibration."""
    import yaml  # seulement pour pose-zero : les autres commandes s'en passent

    with chemin.open(encoding="utf-8") as fichier:
        articulations = yaml.safe_load(fichier)["articulations"]
    return {int(v["id"]): int(v["zero"]) for v in articulations.values()}


def relacher(robot: Robot, actifs: list[int]) -> None:
    """Baisse le couple des moteurs actifs en 3 s, puis le coupe sur les 25."""
    for limite in (240, 180, 120, 60, 0):
        for i in actifs:
            try:
                robot.ecrire(i, TORQUE_LIMIT, limite, 2)
            except OSError as erreur:
                print("  ", erreur)
        time.sleep(0.6)
    couper_et_rapporter(robot)


def poser_groupe(robot: Robot, ids: list[int], zeros: dict[int, int], max_deg: float,
                 actifs: list[int]) -> None:
    """Met le couple sur ``ids`` et amène au zéro ceux qui en sont à moins de ``max_deg``."""
    for i in ids:
        p = robot.lire(i, PRESENT, 2)
        ecart = (zeros[i] - p) * deg_par_tick(i)
        if not dans_limites(robot, i, zeros[i]):
            print(f"  {NOMS[i]:13s} zéro {zeros[i]} hors limites : laissé libre", flush=True)
            continue
        actifs.append(i)  # avant toute écriture : chacune peut activer le couple
        if not couple_sur_place(robot, i, VITESSE_LENTE, COUPLE_BRIDE):
            actifs.remove(i)
            print(f"  {NOMS[i]:13s} hors limites ({p}) : laissé libre", flush=True)
        elif abs(ecart) > max_deg:
            print(f"  {NOMS[i]:13s} à {ecart:+.0f}° du zéro : tenu sur place", flush=True)
        else:
            robot.ecrire(i, GOAL, zeros[i], 2)
    attendre_arret(robot, ids, 8)
    for i in ids:
        if i in actifs:
            reste = (zeros[i] - robot.lire(i, PRESENT, 2)) * deg_par_tick(i)
            print(f"  {NOMS[i]:13s} reste {reste:+5.1f}° du zéro"
                  f" {robot.lire(i, TEMPERATURE, 1)}°C", flush=True)


def cmd_pose_zero(robot: Robot, args: argparse.Namespace) -> None:
    """Pose au zéro de la calibration, groupe par groupe, maintient, puis relâche."""
    if not 0 <= args.maintien <= 300:
        sys.exit("refusé : maintien hors de [0, 300] s")
    zeros = charger_zeros(args.calibration)
    groupes = args.groupes or list(GROUPES)
    exclus = set(args.exclure)
    verifier_depart(robot, [i for g in groupes for i in GROUPES[g]], args.tolerance)
    actifs: list[int] = []
    try:
        for g in groupes:
            print(f"== {g}", flush=True)
            poser_groupe(robot, [i for i in GROUPES[g] if i not in exclus], zeros,
                         args.max_deg, actifs)
        print(f"== maintien {args.maintien:.0f} s", flush=True)
        debut = time.time()
        while time.time() - debut < args.maintien:
            chauds = [i for i in actifs if robot.lire(i, TEMPERATURE, 1) >= TEMPERATURE_MAX]
            if chauds:
                raise SystemExit(f"arrêt : température trop haute sur {chauds}")
            time.sleep(1)
    finally:
        relacher(robot, actifs)


def cmd_coucou(robot: Robot, args: argparse.Namespace) -> None:
    """Redresse la tête puis agite le bras droit trois fois (démonstration)."""
    verifier_depart(robot, HAUT, args.tolerance)
    bras = 52  # r_shoulder_x, inversé : monter les ticks lève le bras ; 2048 = horizontal
    depart = robot.lire(bras, PRESENT, 2)
    try:
        for i in (36, 37, bras):
            # head_y pend souvent sous sa limite : vitesse d'abord (couple sur place).
            robot.ecrire(i, SPEED, VITESSE_LENTE, 2)
            robot.ecrire(i, TORQUE_LIMIT, COUPLE_BRIDE, 2)
            robot.ecrire(i, TORQUE_ENABLE, 1, 1)
        for i, cible in ((36, 512), (37, robot.lire(37, CW_LIMIT, 2)), (37, 512), (bras, 2560)):
            robot.ecrire(i, GOAL, cible, 2)
            attendre_arret(robot, [i], 10)
        robot.ecrire(bras, SPEED, 60, 2)
        for cible in (2720, 2400) * 3:
            robot.ecrire(bras, GOAL, cible, 2)
            attendre_arret(robot, [bras], 3)
        robot.ecrire(bras, SPEED, VITESSE_LENTE, 2)
        robot.ecrire(bras, GOAL, depart, 2)
        attendre_arret(robot, [bras], 12)
    finally:
        relacher(robot, [36, 37, bras])


def sortir(signum: int, _frame: object) -> None:
    """Transforme SIGTERM et SIGHUP en sortie propre, pour que les ``finally`` coupent le couple."""
    raise SystemExit(f"signal {signum} reçu")


def construire_parseur() -> argparse.ArgumentParser:
    """Arguments de la ligne de commande."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = p.add_subparsers(dest="commande", required=True)
    sous.add_parser("lire", help="état des 25 moteurs, lecture seule")
    s = sous.add_parser("suivre", help="positions en continu, lecture seule")
    s.add_argument("--ids", type=int, nargs="+", help="défaut : les quatre hanches x et z")
    for nom in ("tester-sens", "pose-zero", "coucou"):
        s = sous.add_parser(nom)
        s.add_argument("--confirme", action="store_true", required=True,
                       help="obligatoire : quelqu'un tient l'alimentation, robot suspendu")
        s.add_argument("--tolerance", type=float, default=2.0,
                       help="mouvement toléré en 2 s avant de démarrer, en degrés (défaut 2)")
        if nom == "tester-sens":
            s.add_argument("--groupe", choices=GROUPES, default="tronc")
            s.add_argument("--ids", type=int, nargs="+", help="sous-ensemble, ex. 33 35 36")
            s.add_argument("--amplitude", type=float, default=30.0, help="degrés, max 30")
        if nom == "pose-zero":
            s.add_argument("--calibration", type=Path, default=Path("robot_calibration.yaml"))
            s.add_argument("--groupes", choices=GROUPES, nargs="+")
            s.add_argument("--exclure", type=int, nargs="*", default=[])
            s.add_argument("--max-deg", type=float, default=45.0,
                           help="au-delà, le moteur est tenu sur place au lieu d'être déplacé")
            s.add_argument("--maintien", type=float, default=30.0, help="secondes, max 300")
    return p


def main() -> int:
    """Point d'entrée."""
    args = construire_parseur().parse_args()
    signal.signal(signal.SIGTERM, sortir)
    signal.signal(signal.SIGHUP, sortir)
    commandes = {"lire": cmd_lire, "suivre": cmd_suivre, "tester-sens": cmd_tester_sens,
                 "pose-zero": cmd_pose_zero, "coucou": cmd_coucou}
    try:
        commandes[args.commande](ouvrir_robot(), args)
    except KeyboardInterrupt:
        print("\narrêt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
