"""
Evaluation script for trained Poppy Humanoid models
Run trained agent with visualization
"""

import os
import yaml
import torch
import argparse
import numpy as np
from rl_games.common import env_configurations, vecenv
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import model_builder

from tasks.poppy_humanoid import PoppyHumanoid


class RLGPUEnv(vecenv.IVecEnv):
    """Wrapper for Isaac Gym environment"""

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = None
        self.config_name = config_name

    def initialize(self, config_name, num_actors, headless=False):
        """Initialize environment"""
        task_cfg_path = f"cfg/task/{config_name}.yaml"
        with open(task_cfg_path, 'r') as f:
            task_cfg = yaml.safe_load(f)

        task_cfg['env']['numEnvs'] = num_actors

        sim_device = 'cuda:0'
        graphics_device = 0

        self.env = PoppyHumanoid(
            cfg=task_cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device,
            headless=headless
        )

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
        return {
            'action_space': self.env.num_actions,
            'observation_space': self.env.num_obs,
            'agents': self.env.num_envs
        }


def load_checkpoint(checkpoint_path):
    """Load trained model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch_ext.load_checkpoint(checkpoint_path)
    return checkpoint


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained Poppy Humanoid')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--task', type=str, default='PoppyHumanoid', help='Task name')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of environments for evaluation')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to run')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions (no exploration)')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Evaluating Poppy Humanoid")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Num Environments: {args.num_envs}")
    print(f"Headless: {args.headless}")
    print("=" * 80)

    # Load config
    train_cfg_path = f"cfg/train/{args.task}PPO.yaml"
    with open(train_cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create environment
    env = RLGPUEnv(config_name=args.task, num_actors=args.num_envs)
    env.initialize(args.task, args.num_envs, headless=args.headless)

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)

    # Create model
    network_config = config['params']['network']
    model_builder_instance = model_builder.ModelBuilder()
    network = model_builder_instance.load(network_config)

    # Load weights
    network.load_state_dict(checkpoint['model'])
    network.eval()
    network.cuda()

    print("Model loaded successfully\n")

    # Evaluation loop
    obs = env.reset()
    episode_rewards = []
    episode_lengths = []
    current_rewards = torch.zeros(args.num_envs, device='cuda:0')
    current_lengths = torch.zeros(args.num_envs, device='cuda:0')

    completed_episodes = 0
    step = 0

    print(f"Running {args.num_episodes} episodes...")
    print("Press Ctrl+C to stop\n")

    try:
        while completed_episodes < args.num_episodes:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = obs.cuda()
                action = network.act_inference(obs_tensor)

                if args.deterministic:
                    # Use mean action (no exploration)
                    action = action['mus']
                else:
                    # Sample from distribution
                    action = action['actions']

            # Step environment
            obs, rewards, dones, infos = env.step(action)

            current_rewards += rewards
            current_lengths += 1
            step += 1

            # Check for completed episodes
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_indices) > 0:
                for idx in done_indices:
                    episode_rewards.append(current_rewards[idx].item())
                    episode_lengths.append(current_lengths[idx].item())
                    completed_episodes += 1

                    print(f"Episode {completed_episodes}/{args.num_episodes}: "
                          f"Reward = {current_rewards[idx].item():.2f}, "
                          f"Length = {int(current_lengths[idx].item())}")

                    current_rewards[idx] = 0
                    current_lengths[idx] = 0

                    if completed_episodes >= args.num_episodes:
                        break

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")

    # Print statistics
    if len(episode_rewards) > 0:
        print("\n" + "=" * 80)
        print("Evaluation Results:")
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Min/Max reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
        print("=" * 80)
    else:
        print("\nNo episodes completed")


if __name__ == "__main__":
    main()