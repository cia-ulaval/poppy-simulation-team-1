"""Publication des commandes moteur vers un pont rosbridge websocket."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import roslibpy

_TOPIC = "/poppy_motor_state"
_MESSAGE_TYPE = "std_msgs/String"


class MotorArrayPublisher:
    """Publie les commandes moteur du Poppy vers un pont rosbridge websocket."""

    def __init__(self, node_name: str = "poppy_motor_pub") -> None:
        """Initialise le client rosbridge.

        La cible est lue depuis ``POPPY_ROSBRIDGE_HOST`` et
        ``POPPY_ROSBRIDGE_PORT``, avec des valeurs par défaut sûres : le faux
        robot de la pile compose, jamais l'adresse d'un robot réel.

        Args:
            node_name: Nom du nœud, conservé pour compatibilité.

        Raises:
            ValueError: Si le port ou le délai d'attente lus dans
                l'environnement ne sont pas exploitables.
        """
        self._enabled = False

        host = os.environ.get("POPPY_ROSBRIDGE_HOST", "rosbridge")
        port_raw = os.environ.get("POPPY_ROSBRIDGE_PORT", "9090")

        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(
                f"POPPY_ROSBRIDGE_PORT doit être un entier, reçu {port_raw!r}"
            ) from exc

        if not 1 <= port <= 65535:
            raise ValueError(
                f"POPPY_ROSBRIDGE_PORT doit être entre 1 et 65535, reçu {port}"
            )

        timeout_raw = os.environ.get("POPPY_ROSBRIDGE_TIMEOUT_S", "10.0")
        try:
            timeout_s = float(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                f"POPPY_ROSBRIDGE_TIMEOUT_S doit être un nombre, reçu {timeout_raw!r}"
            ) from exc

        if timeout_s <= 0:
            raise ValueError(
                f"POPPY_ROSBRIDGE_TIMEOUT_S doit être strictement positif, "
                f"reçu {timeout_s}"
            )

        self._node = roslibpy.Ros(host=host, port=port)
        self._publisher = roslibpy.Topic(self._node, _TOPIC, _MESSAGE_TYPE)
        self._node.run(timeout=timeout_s)
        self._enabled = True

    def publish(self, motor_ids: Sequence[str], angles_rad: Sequence[float]) -> None:
        """Publie les identifiants et angles des moteurs.

        Args:
            motor_ids: Identifiants des 25 moteurs, forme ``(N,)``.
            angles_rad: Angles cibles en radians, forme ``(N,)``, dans le même
                ordre que ``motor_ids``.

        Raises:
            ValueError: Si les deux séquences n'ont pas la même longueur.
                Publier des angles désalignés de leurs moteurs enverrait au
                robot des consignes valides mais destinées aux mauvaises
                articulations.
        """
        if not self._enabled:
            return

        if len(motor_ids) != len(angles_rad):
            raise ValueError(
                f"Le nombre d'angles ({len(angles_rad)}) ne correspond pas au "
                f"nombre de moteurs ({len(motor_ids)})"
            )

        message = roslibpy.Message(
            {
                "data": json.dumps(
                    {
                        "motor_ids": list(motor_ids),
                        "angles_rad": [float(value) for value in angles_rad],
                    }
                )
            }
        )
        self._publisher.publish(message)

    def close(self) -> None:
        """Ferme le sujet et le lien vers le pont."""
        if not self._enabled:
            return

        self._publisher.unadvertise()
        self._node.terminate()
