from __future__ import annotations
import rclpy
from std_msgs.msg import String
import json


class MotorArrayPublisher:
    def __init__(self, node_name: str = "poppy_motor_pub") -> None:
        self._enabled = False
        self._owns_context = False
        self._rclpy = None
        self._node = None
        self._publisher = None
        self._msg_type = None
        self._rclpy = rclpy
        self._msg_type = String

        if not rclpy.ok():
            rclpy.init()
            self._owns_context = True

        self._node = rclpy.create_node(node_name)
        self._publisher = self._node.create_publisher(String, "/poppy_motor_state", 10)
        self._enabled = True

    def publish(self, motor_ids, angles_rad) -> None:
        if not self._enabled:
            return

        msg = self._msg_type()
        msg.data = json.dumps(
            {
                "motor_ids": [int(value) for value in motor_ids],
                "angles_rad": [float(value) for value in angles_rad],
            }
        )
        self._publisher.publish(msg)

        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        if not self._enabled:
            return

        self._node.destroy_node()
        if self._owns_context:
            self._rclpy.shutdown()
