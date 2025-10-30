# test_humanoid.py - Script de découverte Humanoid-v5

import gymnasium as gym
import numpy as np
import time


def test_humanoid_environment(n_episodes=5, render=True):
    """
    Test l'environnement Humanoid-v5 pour comprendre son fonctionnement.

    Args:
        n_episodes: Nombre d'épisodes à tester
        render: Afficher la visualisation 3D
    """

    # === 1. CRÉATION DE L'ENVIRONNEMENT ===
    print("🤖 Création de l'environnement Humanoid-v5...")

    env = gym.make(
        "Humanoid-v5",
        render_mode="human" if render else None,  # Mode visuel
        # Ces paramètres sont dans ton yaml
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    )

    print(f"✅ Environnement créé!\n")

    # === 2. INSPECTION DE L'ESPACE ===
    print("=" * 60)
    print("📊 INFORMATIONS SUR L'ENVIRONNEMENT")
    print("=" * 60)

    # Espace d'observation (ce que le robot perçoit)
    obs_space = env.observation_space
    print(f"\n🔍 OBSERVATIONS:")
    print(f"   - Type: {type(obs_space)}")
    print(f"   - Shape: {obs_space.shape}")
    print(f"   - Min/Max: [{obs_space.low[0]:.2f}, {obs_space.high[0]:.2f}]")
    print(f"   → Le robot reçoit {obs_space.shape[0]} valeurs à chaque step")

    # Espace d'action (ce qu'on peut contrôler)
    action_space = env.action_space
    print(f"\n🎮 ACTIONS:")
    print(f"   - Type: {type(action_space)}")
    print(f"   - Shape: {action_space.shape}")
    print(f"   - Min/Max: [{action_space.low[0]:.2f}, {action_space.high[0]:.2f}]")
    print(f"   → On contrôle {action_space.shape[0]} articulations")

    # === 3. TEST SUR PLUSIEURS ÉPISODES ===
    print("\n" + "=" * 60)
    print("🏃 LANCEMENT DES TESTS")
    print("=" * 60)

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        print(f"\n--- Épisode {episode + 1}/{n_episodes} ---")

        # Réinitialiser l'environnement
        observation, info = env.reset()

        total_reward = 0
        step = 0
        done = False

        # Boucle de l'épisode
        while not done:
            # ACTION ALÉATOIRE (pas d'IA pour l'instant)
            # Dans ton entraînement, c'est le réseau de neurones qui choisira
            action = env.action_space.sample()

            # STEP: appliquer l'action et observer le résultat
            observation, reward, terminated, truncated, info = env.step(action)

            # Accumuler les métriques
            total_reward += reward
            step += 1

            # L'épisode est fini si terminated (tombé) ou truncated (timeout)
            done = terminated or truncated

            # Petit ralentissement pour voir l'animation
            if render:
                time.sleep(0.01)

        # Statistiques de l'épisode
        episode_rewards.append(total_reward)
        episode_lengths.append(step)

        print(f"   ✅ Terminé après {step} steps")
        print(f"   💰 Reward total: {total_reward:.2f}")
        print(f"   ❌ Raison: {'Tombé (unhealthy)' if terminated else 'Timeout (1000 steps)'}")

    # === 4. STATISTIQUES GLOBALES ===
    print("\n" + "=" * 60)
    print("📈 RÉSULTATS GLOBAUX (Actions Aléatoires)")
    print("=" * 60)

    print(f"\n🎯 Rewards:")
    print(f"   - Moyenne: {np.mean(episode_rewards):.2f}")
    print(f"   - Min: {np.min(episode_rewards):.2f}")
    print(f"   - Max: {np.max(episode_rewards):.2f}")
    print(f"   - Écart-type: {np.std(episode_rewards):.2f}")

    print(f"\n⏱️  Durées des épisodes:")
    print(f"   - Moyenne: {np.mean(episode_lengths):.1f} steps")
    print(f"   - Min: {np.min(episode_lengths)} steps")
    print(f"   - Max: {np.max(episode_lengths)} steps")

    # === 5. INTERPRÉTATION ===
    print("\n" + "=" * 60)
    print("🧠 INTERPRÉTATION")
    print("=" * 60)

    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    print(f"\nAvec des actions ALÉATOIRES:")
    print(f"   → Reward moyen: {avg_reward:.1f}")
    print(f"   → Le robot tombe en ~{avg_length:.0f} steps")

    print(f"\nOBJECTIF de l'entraînement RL:")
    print(f"   → Reward > 5000+ (robot qui marche bien)")
    print(f"   → Durée maximale (1000 steps sans tomber)")
    print(f"   → Donc environ 50x mieux qu'aléatoire! 🚀")

    env.close()

    return episode_rewards, episode_lengths


# === POINT D'ENTRÉE ===
if __name__ == "__main__":
    print("\n" + "🎯" * 30)
    print("TEST HUMANOID-V5 - DÉCOUVERTE")
    print("🎯" * 30 + "\n")

    print("📌 Ce script va:")
    print("   1. Créer l'environnement Humanoid-v5")
    print("   2. Tester avec des actions ALÉATOIRES")
    print("   3. Te montrer les observations, actions, rewards")
    print("   4. Te donner une baseline pour comparer ton RL après\n")

    input("Appuie sur ENTER pour commencer...")

    # Lancer le test
    rewards, lengths = test_humanoid_environment(
        n_episodes=15,
        render=True  # Change à False si tu veux juste les chiffres
    )

    print("\n✅ Test terminé! Tu peux maintenant passer à l'entraînement RL.")