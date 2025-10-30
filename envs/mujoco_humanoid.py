import gymnasium as gym
import numpy as np


class MuJoCoHumanoidEnv(gym.Wrapper):
    # Wrapper for Gymnasium Humanoid-v5 - allows future customization

    def __init__(self, render_mode=None, terminate_when_unhealthy=True,
                 healthy_z_range=(1.0, 2.0), reset_noise_scale=1e-2, **kwargs):

        env = gym.make(
            "Humanoid-v5",
            render_mode=render_mode,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            reset_noise_scale=reset_noise_scale,
            **kwargs
        )

        super().__init__(env)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        # TODO (Phase 2): Add custom reward shaping here
        # reward = self._compute_custom_reward(observation, action, reward, info)

        return observation, reward, terminated, truncated, info

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

    def _compute_custom_reward(self, observation, action, original_reward, info):
        # Placeholder for Phase 2 reward shaping
        # Potential: r_forward, r_upright, r_energy, r_smooth, r_symmetry, p_fall, p_limits
        velocity = observation[22:25]
        forward_reward = np.linalg.norm(velocity)
        energy_cost = np.sum(action ** 2)
        return original_reward + forward_reward - 0.01 * energy_cost

    def __repr__(self):
        return f"MuJoCoHumanoidEnv(render_mode={self.env.render_mode})"


def make_humanoid_env(render_mode=None, **kwargs):
    return MuJoCoHumanoidEnv(render_mode=render_mode, **kwargs)


if __name__ == "__main__":
    print("Testing MuJoCoHumanoidEnv wrapper\n")

    env = make_humanoid_env(render_mode="human")

    print("Info:")
    print(f"   Observations: {env.get_observation_info()['dim']} dimensions")
    print(f"   Actions: {env.get_action_info()['dim']} dimensions")
    print(f"   Environment: {env}\n")

    print("Running 100 random steps...")
    obs, info = env.reset()

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"   Episode terminated after {i + 1} steps")
            break

    env.close()
    print("\n[OK] Test complete!")
