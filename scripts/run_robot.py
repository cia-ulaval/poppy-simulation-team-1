import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import TD3, SAC, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

from src.robot.SimulationAdaptater import SimulationAdapter
from src.environments.poppy_humanoid_env import PoppyHumanoidEnv

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
    parser = argparse.ArgumentParser(description="MuJoCo simulation + ROS publishing")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vec-normalize", type=Path, default=None)
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
