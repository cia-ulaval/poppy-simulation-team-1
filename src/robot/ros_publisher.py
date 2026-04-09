from __future__ import annotations

import json

# import rclpy
import roslibpy

# from std_msgs.msg import String


class MotorArrayPublisher:
    def __init__(self, node_name: str = "poppy_motor_pub") -> None:
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

        self._node = roslibpy.Ros(host="10.242.180.129", port=9090)
        self._node.run()
        # self._node = rclpy.create_node(node_name)
        self._publisher = self._node.run()
        self._publisher = roslibpy.Topic(
            self._node, "/poppy_motor_state", "std_msgs/String"
        )
        # self._publisher = self._node.create_publisher(String, "/poppy_motor_state", 10)
        self._enabled = True

    def publish(self, motor_ids, angles_rad) -> None:
        if not self._enabled:
            return

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
