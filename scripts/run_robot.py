import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments.poppy_humanoid_env import PoppyHumanoidEnv
from src.robot.simulation_adapter import SimulationAdapter

MODEL_CLASSES = [TD3, SAC, PPO, A2C]


def load_model(model_path: Path):
    for cls in MODEL_CLASSES:
        try:
            return cls.load(str(model_path))
        except Exception:
            continue
    raise ValueError(f"Could not load model from {model_path}")


def load_vec_normalize(model_path: Path, vec_path: Path | None) -> VecNormalize | None:
    path = vec_path or model_path.parent / "vec_normalize.pkl"
    if not path.exists():
        return None

    def make_env():
        env = PoppyHumanoidEnv(floor_noise=False, render_mode=None)
        return TimeLimit(env, max_episode_steps=1000)

    dummy = DummyVecEnv([make_env])
    vn = VecNormalize.load(str(path), dummy)
    vn.training = False
    vn.norm_reward = False
    return vn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Déroule une politique en simulation et publie les positions "
                    "articulaires vers un pont rosbridge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sécurité
--------
La cible par défaut est le FAUX robot de la pile compose, jamais un robot
réel. Pour viser un vrai Poppy il faut définir POPPY_ROSBRIDGE_HOST
explicitement, et quelqu'un doit avoir la main sur l'alimentation.

La boucle est ouverte : rien n'est relu du robot. Si un moteur bloque,
personne ne le sait.

Variables d'environnement
-------------------------
  POPPY_ROSBRIDGE_HOST      hôte du pont          (défaut: rosbridge)
  POPPY_ROSBRIDGE_PORT      port websocket        (défaut: 9090)
  POPPY_ROSBRIDGE_TIMEOUT_S délai de connexion    (défaut: 10.0)
  POPPY_CONTROL_PERIOD_S    attente entre deux commandes, en secondes
                            (défaut: 5.0 — c'est un réglage de sécurité,
                            pas un régime de marche)

Exemples
--------
  # Contre le faux robot de la pile compose
  docker compose --profile mock up -d
  docker compose --profile robot run --rm bridge python scripts/run_robot.py --model models/2026-04-08_23-00-52/best_model.zip

  # Écouter ce qui est publié, dans un autre terminal
  docker compose exec rosbridge bash -lc "source /opt/ros/humble/setup.bash && ros2 topic echo /poppy_motor_state"
        """,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Fichier .zip de la politique à dérouler",
    )
    parser.add_argument(
        "--vec-normalize",
        type=Path,
        default=None,
        help="Statistiques de normalisation. Sans elles, la politique reçoit "
             "des observations à la mauvaise échelle et produit des commandes "
             "absurdes, sans qu'aucune erreur ne soit levée.",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    vec_normalize = load_vec_normalize(args.model, args.vec_normalize)
    adapter = SimulationAdapter()
    obs = adapter.reset()

    try:
        while True:
            if vec_normalize is not None:
                obs_input = vec_normalize.normalize_obs(obs.reshape(1, -1)).flatten()
            else:
                obs_input = obs

            action, _ = model.predict(obs_input.reshape(1, -1), deterministic=True)
            obs, _ = adapter.step(action.flatten())
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        adapter.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
