import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import time


def evaluate_and_render(
    model_path,
    vec_normalize_path=None,
    n_episodes=5,
    seed=42,
    fps=50,
):
    """
    Évalue et visualise un modèle PPO entraîné.
    """
    print(f"\n{'='*60}")
    print(f"Évaluation et Visualisation du Modèle")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vec_normalize_path if vec_normalize_path else 'None'}")
    print(f"Épisodes: {n_episodes}")
    print(f"FPS: {fps}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    
    env = gym.make(
        "Humanoid-v5",
        render_mode="human",
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    )
    env = Monitor(env)
    env.reset(seed=seed)
    
    env = DummyVecEnv([lambda: env])
    
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"✓ Chargement des statistiques de normalisation...")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("✓ Statistiques chargées\n")
    else:
        if vec_normalize_path:
            print(f"⚠ ATTENTION: {vec_normalize_path} introuvable!")
        print("⚠ Pas de normalisation appliquée (performances possiblement dégradées)\n")
    
    print("Chargement du modèle...")
    model = PPO.load(model_path)
    print("✓ Modèle chargé\n")

    episode_rewards = []
    episode_lengths = []
    
    sleep_time = 1.0 / fps if fps > 0 else 0

    print("Démarrage de l'évaluation avec visualisation...\n")
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        steps = 0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += float(reward[0])
            steps += 1
            
            if sleep_time > 0:
                time.sleep(sleep_time)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"  Épisode {ep+1}/{n_episodes} terminé:")
        print(f"    → Reward: {total_reward:.2f}")
        print(f"    → Steps: {steps}")
        print()
        
        if ep < n_episodes - 1:
            print("  Appuyez sur Entrée pour l'épisode suivant...")
            input()

    print("\nÉvaluation terminée!")
    print(f"\nRésultats:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min reward:  {np.min(episode_rewards):.2f}")
    print(f"  Max reward:  {np.max(episode_rewards):.2f}")
    print(f"  Mean steps:  {np.mean(episode_lengths):.1f}")
    print(f"\nAppuyez sur Entrée pour fermer...")
    input()
    
    env.close()
    
    return episode_rewards, episode_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évaluer et visualiser un modèle PPO Humanoid-v5 entraîné"
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Chemin vers le fichier .zip du modèle (ex: ./ppo_logs/2024-01-15_10-30-45/ppo_humanoid_final.zip)"
    )
    
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Chemin vers le fichier vec_normalize.pkl (optionnel, mais recommandé)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Nombre d'épisodes à évaluer (défaut: 5)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed aléatoire (défaut: 42)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Vitesse d'affichage en FPS (défaut: 50, 0 = max speed)"
    )
    
    args = parser.parse_args()
    
    vec_normalize_path = args.vec_normalize
    if vec_normalize_path is None:
        model_dir = os.path.dirname(args.model_path)
        potential_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_path):
            vec_normalize_path = potential_path
            print(f"✓ vec_normalize.pkl trouvé automatiquement: {potential_path}\n")
    
    evaluate_and_render(
        model_path=args.model_path,
        vec_normalize_path=vec_normalize_path,
        n_episodes=args.episodes,
        seed=args.seed,
        fps=args.fps,
    )