"""
Main training script for Poppy Humanoid with Isaac Gym and rl_games PPO
"""

import os
import yaml
import argparse
from datetime import datetime

# CRITICAL: Import isaacgym BEFORE torch
from tasks.poppy_humanoid import PoppyHumanoid

# Now safe to import torch and rl_games
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder


class RLGPUEnv(vecenv.IVecEnv):
    """Wrapper for Isaac Gym environment to work with rl_games"""

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = None
        self.config_name = config_name

    def initialize(self, config_name, num_actors):
        """Initialize environment"""
        # Load task config
        task_cfg_path = f"cfg/task/{config_name}.yaml"
        with open(task_cfg_path, 'r') as f:
            task_cfg = yaml.safe_load(f)

        # Override numEnvs if specified
        task_cfg['env']['numEnvs'] = num_actors

        # Create environment
        sim_device = 'cuda:0'
        graphics_device = 0
        headless = True  # Run without visualization for faster training

        self.env = PoppyHumanoid(
            cfg=task_cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device,
            headless=headless
        )

        # Create viewer if not headless
        if not headless:
            self.env.create_viewer()

        return self.env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def get_number_of_agents(self):
        return self.env.num_envs

    def get_env_info(self):
        info = {
            'action_space': self.env.num_actions,
            'observation_space': self.env.num_obs,
            'agents': self.env.num_envs
        }
        return info


class RLGPUAlgoObserver:
    """Callback for rl_games training events"""

    def __init__(self):
        pass

    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        pass


def create_rlgpu_env(**kwargs):
    """Factory function for creating environment"""
    return RLGPUEnv(**kwargs)


def register_env():
    """Register environment with rl_games"""
    env_configurations.register(
        'isaacgym',
        {
            'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
            'vecenv_type': 'RLGPU'
        }
    )


def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Poppy Humanoid with PPO')
    parser.add_argument('--task', type=str, default='PoppyHumanoid', help='Task name')
    parser.add_argument('--num_envs', type=int, default=None, help='Number of environments (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--max_epochs', type=int, default=None, help='Maximum training epochs')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment name for logging')

    args = parser.parse_args()

    # Register environment
    register_env()

    # Load training config
    train_cfg_path = f"cfg/train/{args.task}PPO.yaml"
    config = load_config(train_cfg_path)

    # Override config with command line arguments
    if args.num_envs is not None:
        config['params']['config']['num_actors'] = args.num_envs

    if args.max_epochs is not None:
        config['params']['config']['max_epochs'] = args.max_epochs

    if args.checkpoint is not None:
        config['params']['load_checkpoint'] = True
        config['params']['load_path'] = args.checkpoint

    # Set experiment name
    if args.experiment is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.task}_{timestamp}"
    else:
        experiment_name = args.experiment

    # Create output directory
    output_dir = f"runs/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    print("=" * 80)
    print(f"Training Poppy Humanoid")
    print(f"Task: {args.task}")
    print(f"Num Environments: {config['params']['config']['num_actors']}")
    print(f"Max Epochs: {config['params']['config']['max_epochs']}")
    print(f"Output Directory: {output_dir}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    # Create runner
    runner = Runner()
    runner.load(config)
    runner.reset()

    # Create and initialize environment
    env = create_rlgpu_env(
        config_name=args.task,
        num_actors=config['params']['config']['num_actors']
    )
    env.initialize(args.task, config['params']['config']['num_actors'])

    # Set environment in runner
    runner.env = env

    # Start training
    print("\nStarting training...")
    print("Press Ctrl+C to stop training and save checkpoint\n")

    try:
        runner.run({
            'train': True,
            'play': False,
            'checkpoint': args.checkpoint,
            'sigma': None
        })
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    print(f"\nTraining completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()