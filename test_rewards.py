# Test script to compare original vs custom rewards

import numpy as np
from envs import make_humanoid_env


def test_reward_comparison(n_steps=100):
    """Compare original vs custom reward over random actions"""

    print("=" * 70)
    print("REWARD COMPARISON: Original vs Custom")
    print("=" * 70 + "\n")

    # Create both environments
    env_original = make_humanoid_env(use_custom_reward=False)
    env_custom = make_humanoid_env(use_custom_reward=True)

    # Reset with same seed
    obs_orig, _ = env_original.reset(seed=42)
    obs_custom, _ = env_custom.reset(seed=42)

    rewards_original = []
    rewards_custom = []
    breakdown_history = []

    print(f"Running {n_steps} steps with random actions...\n")

    for step in range(n_steps):
        # Use same action for both
        action = env_original.action_space.sample()

        # Step original
        obs_orig, reward_orig, term_orig, trunc_orig, info_orig = env_original.step(action)
        rewards_original.append(reward_orig)

        # Step custom
        obs_custom, reward_custom, term_custom, trunc_custom, info_custom = env_custom.step(action)
        rewards_custom.append(reward_custom)

        if 'reward_breakdown' in info_custom:
            breakdown_history.append(info_custom['reward_breakdown'])

        # Print comparison every 20 steps
        if step % 20 == 0:
            print(f"Step {step:3d} | Original: {reward_orig:6.2f} | Custom: {reward_custom:6.2f}")

        # Check if terminated
        if term_orig or trunc_orig:
            print(f"\nOriginal env terminated at step {step}")
            break
        if term_custom or trunc_custom:
            print(f"\nCustom env terminated at step {step}")
            break

    # Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70 + "\n")

    print(f"Original Reward:")
    print(f"   Mean:   {np.mean(rewards_original):8.3f}")
    print(f"   Std:    {np.std(rewards_original):8.3f}")
    print(f"   Min:    {np.min(rewards_original):8.3f}")
    print(f"   Max:    {np.max(rewards_original):8.3f}")
    print(f"   Total:  {np.sum(rewards_original):8.3f}\n")

    print(f"Custom Reward:")
    print(f"   Mean:   {np.mean(rewards_custom):8.3f}")
    print(f"   Std:    {np.std(rewards_custom):8.3f}")
    print(f"   Min:    {np.min(rewards_custom):8.3f}")
    print(f"   Max:    {np.max(rewards_custom):8.3f}")
    print(f"   Total:  {np.sum(rewards_custom):8.3f}\n")

    # Custom reward breakdown averages
    if breakdown_history:
        print("=" * 70)
        print("CUSTOM REWARD BREAKDOWN (Averages)")
        print("=" * 70 + "\n")

        avg_forward = np.mean([b['forward'] for b in breakdown_history])
        avg_upright = np.mean([b['upright'] for b in breakdown_history])
        avg_energy = np.mean([b['energy'] for b in breakdown_history])
        avg_smooth = np.mean([b['smooth'] for b in breakdown_history])
        avg_alive = np.mean([b['alive'] for b in breakdown_history])

        print(f"Forward velocity:  {avg_forward:8.3f}")
        print(f"Upright posture:   {avg_upright:8.3f}")
        print(f"Energy penalty:    {avg_energy:8.3f}")
        print(f"Smoothness:        {avg_smooth:8.3f}")
        print(f"Alive bonus:       {avg_alive:8.3f}")

    env_original.close()
    env_custom.close()

    print("\n[OK] Test complete!\n")


if __name__ == "__main__":
    test_reward_comparison(n_steps=100)
