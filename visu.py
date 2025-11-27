import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import time


def evaluate_model(
    model_path,
    vec_normalize_path=None,
    n_episodes=5,
    seed=42,
    render=True,
    fps=50,
):
    """
    Évalue un modèle PPO entraîné avec ou sans visualisation.
    
    Args:
        model_path: Chemin vers le modèle .zip
        vec_normalize_path: Chemin vers vec_normalize.pkl (optionnel)
        n_episodes: Nombre d'épisodes à évaluer
        seed: Seed aléatoire
        render: Si True, affiche la visualisation
        fps: Vitesse d'affichage en FPS (seulement si render=True)
    
    Returns:
        tuple: (episode_rewards, episode_lengths)
    """
    print(f"\n{'='*60}")
    print(f"Évaluation du Modèle {'avec Visualisation' if render else 'sans Visualisation'}")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vec_normalize_path if vec_normalize_path else 'None'}")
    print(f"Épisodes: {n_episodes}")
    print(f"Mode: {'RENDER (FPS={})'.format(fps) if render else 'NO RENDER (max speed)'}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    
    # Créer l'environnement
    env = gym.make(
        "Humanoid-v5",
        render_mode="human" if render else None,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
    )
    env = Monitor(env)
    env.reset(seed=seed)
    
    env = DummyVecEnv([lambda: env])
    
    # Charger les statistiques de normalisation
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"✓ Chargement des statistiques de normalisation...")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False  # Pour voir les vraies récompenses
        print("✓ Statistiques chargées\n")
    else:
        if vec_normalize_path:
            print(f"⚠ ATTENTION: {vec_normalize_path} introuvable!")
        print("⚠ Pas de normalisation appliquée (performances possiblement dégradées)\n")
    
    # Charger le modèle
    print("Chargement du modèle...")
    model = PPO.load(model_path)
    print("✓ Modèle chargé\n")

    episode_rewards = []
    episode_lengths = []
    
    sleep_time = 1.0 / fps if (render and fps > 0) else 0

    print(f"Démarrage de l'évaluation {'avec visualisation' if render else '(mode rapide)'}...\n")
    
    start_time = time.time()
    
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
        
        print(f"  Épisode {ep+1}/{n_episodes}:")
        print(f"    → Reward: {total_reward:>8.2f}")
        print(f"    → Steps:  {steps:>4d}")
        
        # En mode render, pause entre les épisodes
        if render and ep < n_episodes - 1:
            print("    → Appuyez sur Entrée pour continuer...")
            input()
        print()

    eval_time = time.time() - start_time

    # Afficher les statistiques finales
    print(f"\n{'='*60}")
    print("RÉSULTATS DE L'ÉVALUATION")
    print(f"{'='*60}")
    print(f"Temps total:     {eval_time:.1f}s")
    print(f"Temps/épisode:   {eval_time/n_episodes:.1f}s")
    print(f"\nRécompenses:")
    print(f"  Mean:  {np.mean(episode_rewards):>8.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min:   {np.min(episode_rewards):>8.2f}")
    print(f"  Max:   {np.max(episode_rewards):>8.2f}")
    print(f"\nLongueur des épisodes:")
    print(f"  Mean:  {np.mean(episode_lengths):>8.1f} steps")
    print(f"  Min:   {np.min(episode_lengths):>8d} steps")
    print(f"  Max:   {np.max(episode_lengths):>8d} steps")
    print(f"{'='*60}\n")
    
    if render:
        print("Appuyez sur Entrée pour fermer...")
        input()
    
    env.close()
    
    return episode_rewards, episode_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évaluer un modèle PPO Humanoid-v5 entraîné",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Évaluation avec visualisation (par défaut)
  python evaluate_render.py models/ppo_humanoid_final.zip
  
  # Évaluation rapide sans visualisation
  python evaluate_render.py models/ppo_humanoid_final.zip --no-render
  
  # Évaluation avec vec_normalize spécifique
  python evaluate_render.py models/ppo_humanoid_final.zip --vec-normalize models/vec_normalize.pkl
  
  # Évaluation de 20 épisodes à 30 FPS
  python evaluate_render.py models/ppo_humanoid_final.zip --episodes 20 --fps 30
  
  # Évaluation rapide de 100 épisodes sans visualisation
  python evaluate_render.py models/ppo_humanoid_final.zip --no-render --episodes 100
        """
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Chemin vers le fichier .zip du modèle"
    )
    
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Chemin vers vec_normalize.pkl (auto-détecté si dans le même dossier)"
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
        "--no-render",
        action="store_true",
        help="Désactiver la visualisation (évaluation rapide)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Vitesse d'affichage en FPS si --render (défaut: 50, 0 = max speed)"
    )
    
    args = parser.parse_args()
    
    # Auto-détection de vec_normalize.pkl
    vec_normalize_path = args.vec_normalize
    if vec_normalize_path is None:
        model_dir = os.path.dirname(args.model_path)
        potential_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_path):
            vec_normalize_path = potential_path
            print(f"✓ vec_normalize.pkl trouvé automatiquement: {potential_path}\n")
    
    # Lancer l'évaluation
    rewards, lengths = evaluate_model(
        model_path=args.model_path,
        vec_normalize_path=vec_normalize_path,
        n_episodes=args.episodes,
        seed=args.seed,
        render=not args.no_render,  # Inverser le flag
        fps=args.fps,
    )
    
    print("✓ Évaluation terminée!")