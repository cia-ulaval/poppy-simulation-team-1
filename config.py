import yaml
import os
from pathlib import Path


class Config:
    # Load hyperparameters from YAML config file

    def __init__(self, config_path="configs/ppo_humanoid.yaml"):
        self.config_path = config_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)

        self._load_algorithm_config()
        self._load_network_config()
        self._load_training_config()
        self._load_environment_config()
        self._create_directories()

        print(f"[OK] Configuration loaded from: {config_path}")

    def _load_algorithm_config(self):
        algo = self.raw_config.get('algorithm', {})

        self.algorithm_name = algo.get('name', 'PPO')
        self.learning_rate = algo.get('learning_rate', 3e-4)
        self.n_steps = algo.get('n_steps', 2048)
        self.batch_size = algo.get('batch_size', 64)
        self.n_epochs = algo.get('n_epochs', 10)
        self.gamma = algo.get('gamma', 0.99)
        self.gae_lambda = algo.get('gae_lambda', 0.95)
        self.clip_range = algo.get('clip_range', 0.2)
        self.ent_coef = algo.get('ent_coef', 0.0)
        self.vf_coef = algo.get('vf_coef', 0.5)
        self.max_grad_norm = algo.get('max_grad_norm', 0.5)

    def _load_network_config(self):
        network = self.raw_config.get('network', {})

        self.policy_layers = network.get('pi', [256, 256, 128])
        self.value_layers = network.get('vf', [256, 256, 128])
        self.activation = network.get('activation', 'relu')
        self.network_arch = dict(pi=self.policy_layers, vf=self.value_layers)

    def _load_training_config(self):
        training = self.raw_config.get('training', {})

        self.total_timesteps = training.get('total_timesteps', 10_000_000)
        self.n_envs = training.get('n_envs', 8)
        self.eval_freq = training.get('eval_freq', 10000)
        self.save_freq = training.get('save_freq', 50000)
        self.log_interval = training.get('log_interval', 1)

        self.log_dir = "configs/logs"
        self.models_dir = "configs/models"
        self.tensorboard_dir = "configs/tensorboard_logs"

    def _load_environment_config(self):
        env = self.raw_config.get('environment', {})

        self.env_name = env.get('name', 'Humanoid-v5')
        self.terminate_when_unhealthy = env.get('terminate_when_unhealthy', True)
        self.healthy_z_range = tuple(env.get('healthy_z_range', [1.0, 2.0]))

    def _create_directories(self):
        for directory in [self.log_dir, self.models_dir, self.tensorboard_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_ppo_kwargs(self):
        return {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'verbose': 1,
            'tensorboard_log': self.tensorboard_dir,
        }

    def get_env_kwargs(self):
        return {
            'terminate_when_unhealthy': self.terminate_when_unhealthy,
            'healthy_z_range': self.healthy_z_range,
        }

    def print_summary(self):
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)

        print(f"\nAlgorithm: {self.algorithm_name}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Network: {self.policy_layers}")
        print(f"   Batch size: {self.batch_size}")

        print(f"\nTraining:")
        print(f"   Total timesteps: {self.total_timesteps:,}")
        print(f"   Parallel envs: {self.n_envs}")
        print(f"   Eval frequency: {self.eval_freq:,} steps")
        print(f"   Save frequency: {self.save_freq:,} steps")

        print(f"\nEnvironment: {self.env_name}")
        print(f"   Healthy Z range: {self.healthy_z_range}")

        print(f"\nDirectories:")
        print(f"   Logs: {self.log_dir}")
        print(f"   Models: {self.models_dir}")
        print(f"   Tensorboard: {self.tensorboard_dir}")

        print("\n" + "=" * 60 + "\n")


# Global singleton instance
config = Config()


if __name__ == "__main__":
    cfg = Config()
    cfg.print_summary()
