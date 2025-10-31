import gymnasium as gym
import numpy as np


class MuJoCoHumanoidEnv(gym.Wrapper):

    def __init__(self, render_mode=None, terminate_when_unhealthy=True,
                 healthy_z_range=(1.0, 2.0), reset_noise_scale=1e-2,
                 use_custom_reward=True, reward_weights=None, **kwargs):

        env = gym.make(
            "Humanoid-v5",
            render_mode=render_mode,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            **kwargs
        )

        super().__init__(env)

        self.use_custom_reward = use_custom_reward

        # Default reward weights (can be tuned)
        self.reward_weights = reward_weights or {
            'forward': 1.0,        # Encourage forward movement
            'upright': 0.5,        # Encourage staying upright
            'energy': 0.01,        # Penalize energy consumption
            'smooth': 0.05,        # Penalize jerky movements
            'alive': 1.0,          # Reward for staying alive
        }

        # Track previous action for smoothness penalty
        self.prev_action = None

    def reset(self, seed=None, options=None):
        self.prev_action = None
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        # Apply custom reward shaping if enabled
        if self.use_custom_reward:
            reward = self._compute_custom_reward(observation, action, reward, info)

        self.prev_action = action.copy()
        return observation, reward, terminated, truncated, info

    def _compute_custom_reward(self, observation, action, original_reward, info):
        # pour changer la recompense en fonction des cas

        # 1. Forward velocity reward (encourage moving forward)
        forward_velocity = observation[22]  # x-velocity
        r_forward = self.reward_weights['forward'] * forward_velocity

        # 2. Upright reward (encourage staying upright)
        # z-position should be in healthy range (1.0 to 2.0)
        z_pos = observation[0]
        upright_deviation = abs(z_pos - 1.5)  # Target center of healthy range
        r_upright = self.reward_weights['upright'] * np.exp(-upright_deviation)

        # 3. Energy penalty (discourage excessive joint torques)
        energy_cost = np.sum(np.square(action))
        r_energy = -self.reward_weights['energy'] * energy_cost

        # 4. Smoothness reward (discourage jerky movements)
        r_smooth = 0.0
        if self.prev_action is not None:
            action_diff = np.sum(np.square(action - self.prev_action))
            r_smooth = -self.reward_weights['smooth'] * action_diff

        # 5. Alive bonus (encourage survival)
        r_alive = self.reward_weights['alive']

        # Combine all rewards
        total_reward = r_forward + r_upright + r_energy + r_smooth + r_alive

        # Optional: Store detailed reward info for debugging
        info['reward_breakdown'] = {
            'original': original_reward,
            'custom_total': total_reward,
            'forward': r_forward,
            'upright': r_upright,
            'energy': r_energy,
            'smooth': r_smooth,
            'alive': r_alive,
        }

        return total_reward

    def set_reward_weights(self, **weights):
        """Update reward weights dynamically"""
        self.reward_weights.update(weights)

    def get_observation_info(self):
        return {
            "shape": self.observation_space.shape,
            "low": self.observation_space.low,
            "high": self.observation_space.high,
            "dim": self.observation_space.shape[0]
        }

    def get_action_info(self):
        return {
            "shape": self.action_space.shape,
            "low": self.action_space.low,
            "high": self.action_space.high,
            "dim": self.action_space.shape[0]
        }

    def __repr__(self):
        return f"MuJoCoHumanoidEnv(render_mode={self.env.render_mode}, custom_reward={self.use_custom_reward})"


def make_humanoid_env(render_mode=None, use_custom_reward=True, **kwargs):
    """
    Helper to create humanoid environment

    Args:
        render_mode: None, 'human', or 'rgb_array'
        use_custom_reward: Whether to use custom reward shaping (default: True)
        **kwargs: Additional arguments for environment
    """
    return MuJoCoHumanoidEnv(render_mode=render_mode, use_custom_reward=use_custom_reward, **kwargs)


if __name__ == "__main__":
    print("Testing MuJoCoHumanoidEnv wrapper with custom rewards\n")

    # Test with custom rewards enabled
    env = make_humanoid_env(render_mode="human", use_custom_reward=True)

    print("Info:")
    print(f"   Observations: {env.get_observation_info()['dim']} dimensions")
    print(f"   Actions: {env.get_action_info()['dim']} dimensions")
    print(f"   Custom reward: {env.use_custom_reward}")
    print(f"   Reward weights: {env.reward_weights}\n")

    print("Running 100 random steps with reward breakdown...")
    obs, info = env.reset()
    total_reward = 0

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print reward breakdown every 20 steps
        if i % 20 == 0 and 'reward_breakdown' in info:
            print(f"\nStep {i}:")
            print(f"  Total reward: {reward:.3f}")
            breakdown = info['reward_breakdown']
            print(f"  Forward: {breakdown['forward']:.3f}")
            print(f"  Upright: {breakdown['upright']:.3f}")
            print(f"  Energy: {breakdown['energy']:.3f}")
            print(f"  Smooth: {breakdown['smooth']:.3f}")

        if terminated or truncated:
            print(f"\nEpisode terminated after {i + 1} steps")
            print(f"Total cumulative reward: {total_reward:.2f}")
            break

    env.close()
    print("\n[OK] Test complete!")
