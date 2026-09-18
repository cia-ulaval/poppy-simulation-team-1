"""Le format publié sur le websocket, épinglé.

Ce test existe pour une raison précise : c'est le seul endroit du dépôt où un
bug ne produit ni exception ni test rouge, mais un mouvement sur un robot
physique. Publier des angles désalignés de leurs identifiants enverrait des
consignes parfaitement valides aux mauvaises articulations.

``roslibpy`` est bouchonné plutôt qu'installé : on vérifie *notre* charge utile,
pas la bibliothèque, et l'image de test n'a pas besoin d'une dépendance réseau.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

# Les 25 articulations motorisées du modèle Poppy, dans l'ordre du MJCF.
JOINTS = [
    "r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
    "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y",
    "abs_y", "abs_x", "abs_z", "bust_y", "bust_x",
    "head_z", "head_y",
    "l_shoulder_y", "l_shoulder_x", "l_arm_z", "l_elbow_y",
    "r_shoulder_y", "r_shoulder_x", "r_arm_z", "r_elbow_y",
]


@pytest.fixture
def publisher(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Un MotorArrayPublisher dont le lien rosbridge est bouchonné.

    Yields:
        Le couple (publisher, envois), où ``envois`` collecte les messages
        effectivement transmis au sujet.
    """
    envois: list[dict] = []

    class _Ros:
        def __init__(self, host: str, port: int) -> None:
            self.host, self.port = host, port

        def run(self, timeout: float | None = None) -> None:
            pass

        def terminate(self) -> None:
            pass

    class _Topic:
        def __init__(self, node: Any, name: str, message_type: str) -> None:
            self.name = name

        def publish(self, message: dict) -> None:
            envois.append(message)

        def unadvertise(self) -> None:
            pass

    faux = types.ModuleType("roslibpy")
    faux.Ros = _Ros
    faux.Topic = _Topic
    faux.Message = dict
    monkeypatch.setitem(sys.modules, "roslibpy", faux)
    monkeypatch.delitem(sys.modules, "src.robot.ros_publisher", raising=False)

    from src.robot.ros_publisher import MotorArrayPublisher

    yield MotorArrayPublisher(), envois


def test_charge_publiee(publisher) -> None:
    """La charge contient les 25 moteurs et leurs 25 angles, alignés.

    Les valeurs attendues sont celles que publiait le code d'avant le
    nettoyage, qui recevait 26 identifiants et retirait le premier. Cette
    liste est la preuve que le passage à 25 identifiants sans compensation n'a
    rien changé sur le fil.
    """
    pub, envois = publisher
    angles = [round(0.01 * i, 4) for i in range(25)]

    pub.publish(motor_ids=JOINTS, angles_rad=angles)

    assert len(envois) == 1
    charge = json.loads(envois[0]["data"])
    assert charge["motor_ids"] == JOINTS
    assert charge["angles_rad"] == angles


def test_longueurs_incoherentes_refusees(publisher) -> None:
    """Un décalage entre moteurs et angles doit lever, pas publier."""
    pub, envois = publisher

    with pytest.raises(ValueError, match="ne correspond pas"):
        pub.publish(motor_ids=JOINTS, angles_rad=[0.0] * 24)

    assert envois == [], "rien ne doit partir quand les longueurs divergent"


def test_port_invalide_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un port hors plage est refusé à la construction.

    La cible se lit dans l'environnement : une valeur absurde doit échouer
    tout de suite, pas au premier message.
    """
    faux = types.ModuleType("roslibpy")
    faux.Ros = faux.Topic = faux.Message = dict
    monkeypatch.setitem(sys.modules, "roslibpy", faux)
    monkeypatch.delitem(sys.modules, "src.robot.ros_publisher", raising=False)
    monkeypatch.setenv("POPPY_ROSBRIDGE_PORT", "70000")

    from src.robot.ros_publisher import MotorArrayPublisher

    with pytest.raises(ValueError, match="entre 1 et 65535"):
        MotorArrayPublisher()
