from __future__ import annotations

import json
import os
from collections.abc import Sequence

# import rclpy
import roslibpy

# from std_msgs.msg import String


class MotorArrayPublisher:
    """Publie les commandes moteur du Poppy vers un pont rosbridge websocket."""

    def __init__(self, node_name: str = "poppy_motor_pub") -> None:
        """Initialise le client rosbridge.

        La cible est lue depuis ``POPPY_ROSBRIDGE_HOST`` et
        ``POPPY_ROSBRIDGE_PORT`` avec des valeurs par défaut sûres.

        Args:
            node_name: Nom du nœud, conservé pour compatibilité.
        """
        self._enabled = False
        self._owns_context = False
        # self._rclpy = None
        self._node = None
        self._publisher = None
        self._msg_type = None
        # self._rclpy = rclpy
        # self._msg_type = String

        # if not rclpy.ok():
        #     rclpy.init()
        #     self._owns_context = True

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
        self._publisher = roslibpy.Topic(
            self._node, "/poppy_motor_state", "std_msgs/String"
        )
        self._node.run(timeout=timeout_s)
        # self._node = rclpy.create_node(node_name)
        # self._publisher = self._node.create_publisher(String, "/poppy_motor_state", 10)
        self._enabled = True

    def publish(self, motor_ids: Sequence[str], angles_rad: Sequence[float]) -> None:
        """Publie un message contenant les identifiants et angles des moteurs.

        Args:
            motor_ids: Identifiants des moteurs, forme ``(N,)``. Le premier
                élément est ignoré par compatibilité avec le format historique.
            angles_rad: Angles cibles en radians, forme ``(N,)``.

        Raises:
            ValueError: Si ``angles_rad`` contient des valeurs non convertibles
                en ``float``, ou si le nombre d'angles ne correspond pas au
                nombre de moteurs (``len(motor_ids) - 1``).
        """
        if not self._enabled:
            return

        if len(motor_ids) - 1 != len(angles_rad):
            raise ValueError(
                f"Le nombre d'angles ({len(angles_rad)}) ne correspond pas au "
                f"nombre de moteurs ({len(motor_ids) - 1})"
            )

        # msg = self._msg_type()
        msg = roslibpy.Message(
            {
                "data": json.dumps(
                    {
                        "motor_ids": motor_ids[1:],
                        "angles_rad": [float(value) for value in angles_rad],
                    }
                )
            }
        )
        self._publisher.publish(msg)

        # self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        if not self._enabled:
            return

        self._publisher.unadvertise()

        self._node.terminate()
        # if self._owns_context:
        #     self._rclpy.shutdown()
