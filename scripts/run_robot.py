import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robot.robot_config import RobotConfig
from src.robot.pypot_adapter import PypotAdapter
from src.robot.action_mapper import ControlMode
from src.robot.robot_controller import RobotController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_robot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy SB3 model on Poppy Humanoid via pypot",
    )

    parser.add_argument("--model", type=Path, required=True, help="Path to trained model (.zip)")
    # stats de normalisation, généré pendant l'entrainement normalement
    parser.add_argument("--vec-normalize", type=Path, default=None, help="Path to VecNormalize stats (.pkl)")
    # déclaration du hardware etc, dans Poppy standard, JSON par défaut dans le package poppy-humanoid
    parser.add_argument("--config-file", type=str, required=True, help="Path to pypot robot JSON config file")
    parser.add_argument("--steps", type=int, default=None, help="Number of control steps")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--freq", type=float, default=50.0, help="Control frequency in Hz (default: 50)")
    parser.add_argument("--gain", type=float, default=30.0, help="Action gain (default: 30)")
    parser.add_argument("--mode", choices=["delta", "direct"], default="delta", help="Control mode")
    # si on disable il va penser toujours être droit 
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import pypot.robot
    except ImportError:
        logger.error("pypot is not installed. Install with: pip install pypot")
        sys.exit(1)

    config = RobotConfig(
        control_freq_hz=args.freq,
        action_gain=args.gain,
        has_imu=not args.no_imu,
    )

    logger.info("Connecting to robot via %s", args.config_file)
    robot = pypot.robot.from_json(args.config_file)
    adapter = PypotAdapter(robot, config=config)
    logger.info("Robot connected.")

    control_mode = ControlMode.DELTA if args.mode == "delta" else ControlMode.DIRECT
    logger.info("Loading model from %s", args.model)
    controller = RobotController.from_checkpoint(
        model_path=args.model,
        adapter=adapter,
        vec_normalize_path=args.vec_normalize,
        config=config,
        control_mode=control_mode,
    )

    # Run the whole controller
    logger.info("Starting control loop (freq=%.0fHz, mode=%s)", args.freq, args.mode)
    try:
        controller.run(n_steps=args.steps, duration_s=args.duration)
    finally:
        controller.close()
        logger.info("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
