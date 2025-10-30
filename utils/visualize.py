"""
Module de visualisation pour Poppy RL

Visualise un modèle entraîné en temps réel et peut enregistrer des vidéos.
"""

import numpy as np
import time
import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from envs import make_humanoid_env


def visualize(
        model_path,
        n_episodes=3,
        deterministic=True,
        save_video=False,
        video_folder="./videos",
        video_length=1000,
        fps=30
):
    """
    Visualise un modèle PPO entraîné en temps réel

    Args:
        model_path: Chemin vers le modèle .zip
        n_episodes: Nombre d'épisodes à visualiser (défaut: 3)
        deterministic: Si True, utilise la moyenne de la policy
        save_video: Si True, enregistre des vidéos MP4
        video_folder: Dossier où sauvegarder les vidéos
        video_length: Durée max d'une vidéo en steps
        fps: Frames par seconde de la vidéo

    Returns:
        None
    """

    print("\n" + "" * 35)
    print("VISUALISATION DU MODÈLE")
    print("" * 35 + "\n")

    print(f" Configuration:")
    print(f"   Modèle: {model_path}")
    print(f"   Nombre d'épisodes: {n_episodes}")
    print(f"   Mode: {'Déterministe' if deterministic else 'Stochastique'}")
    print(f"   Enregistrement vidéo: {'OUI' if save_video else 'NON'}")
    if save_video:
        print(f"   Dossier vidéos: {video_folder}")
        print(f"   FPS: {fps}\n")
    else:
        print()

    # ========================================================================
    # 1. CHARGER LE MODÈLE
    # ========================================================================

    print(" Chargement du modèle...")
    try:
        model = PPO.load(model_path)
        print("    Modèle chargé avec succès\n")
    except Exception as e:
        print(f"    Erreur lors du chargement: {e}")
        return

    # ========================================================================
    # 2. CRÉER L'ENVIRONNEMENT
    # ========================================================================

    if save_video:
        print(" Configuration de l'enregistrement vidéo...")

        # Créer le dossier vidéo
        Path(video_folder).mkdir(parents=True, exist_ok=True)

        # Créer un environnement avec render_mode="rgb_array" pour enregistrer
        def make_env():
            env = make_humanoid_env(render_mode="rgb_array")
            return env

        env = DummyVecEnv([make_env])

        # Wrapper pour enregistrer les vidéos
        env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == 0,  # Enregistre chaque épisode
            video_length=video_length,
            name_prefix="humanoid"
        )
        print(f"    Vidéos seront sauvées dans: {video_folder}\n")

    else:
        print(" Création de l'environnement de visualisation...")
        env = make_humanoid_env(render_mode="human")
        print("    Environnement créé\n")

    # ========================================================================
    # 3. VISUALISATION DES ÉPISODES
    # ========================================================================

    print("=" * 70)
    print(f" DÉBUT DE LA VISUALISATION ({n_episodes} épisodes)")
    print("=" * 70 + "\n")

    if not save_video:
        print(" Conseil: Regarde la fenêtre MuJoCo qui s'est ouverte!")
        print("   (Tu peux la repositionner et zoomer/dézoomer)\n")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        print(f"\n{'' * 70}")
        print(f" Episode {episode + 1}/{n_episodes}")
        print(f"{'' * 70}")

        if save_video:
            obs = env.reset()
        else:
            obs, info = env.reset()

        done = False
        step = 0
        episode_reward = 0

        print("   En cours...", end=" ", flush=True)

        while not done:
            # Prédire l'action
            if save_video:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                done = done[0]  # VecEnv retourne un array
                reward = reward[0]
            else:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_reward += reward
            step += 1

            # Ralentir pour visualisation en temps réel
            if not save_video:
                time.sleep(0.01)  # ~100 FPS (MuJoCo tourne à 500Hz normalement)

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        print(f"")
        print(f"   Reward total: {episode_reward:.2f}")
        print(f"   Durée: {step} steps")
        print(f"   Résultat: {' Succès (1000 steps)' if step >= 1000 else ' Tombé'}")

    # ========================================================================
    # 4. RÉSUMÉ
    # ========================================================================

    print(f"\n{'=' * 70}")
    print(" RÉSUMÉ DE LA VISUALISATION")
    print(f"{'=' * 70}\n")

    print(f"   Reward moyen: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Durée moyenne: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"   Succès: {sum(1 for l in episode_lengths if l >= 1000)}/{n_episodes} épisodes")

    if save_video:
        print(f"\n    Vidéos sauvegardées dans: {video_folder}")
        print(f"    Format: MP4, {fps} FPS")

        # Lister les vidéos créées
        video_files = list(Path(video_folder).glob("*.mp4"))
        if video_files:
            print(f"    Fichiers créés:")
            for vf in video_files[-n_episodes:]:  # Seulement les dernières
                print(f"      - {vf.name}")

    print(f"\n{'=' * 70}\n")

    # Nettoyer
    env.close()

    print(" Visualisation terminée!\n")


def visualize_comparison(model_path, n_episodes=2):
    """
    Visualise côte à côte: modèle entraîné vs actions aléatoires
    (Note: nécessite deux fenêtres, peut être complexe)

    Args:
        model_path: Chemin vers le modèle
        n_episodes: Nombre d'épisodes à comparer
    """
    print("\n  Fonction de comparaison visuelle pas encore implémentée")
    print("   Pour comparer, lance deux fois visualize() manuellement:")
    print(f"   1. python -c \"from utils.visualize import visualize; visualize('{model_path}')\"")
    print("   2. Lance test_humanoid.py (actions aléatoires)")
    print()


# ========================================================================
# FONCTION HELPER: CRÉER UN GIF
# ========================================================================

def create_gif_from_video(video_path, gif_path=None, fps=30):
    """
    Convertit une vidéo MP4 en GIF (nécessite imageio et imageio-ffmpeg)

    Args:
        video_path: Chemin vers la vidéo MP4
        gif_path: Chemin de sortie du GIF (None = même nom avec .gif)
        fps: FPS du GIF
    """
    try:
        import imageio

        if gif_path is None:
            gif_path = video_path.replace('.mp4', '.gif')

        print(f"  Conversion en GIF: {video_path} → {gif_path}")

        reader = imageio.get_reader(video_path)

        frames = []
        for frame in reader:
            frames.append(frame)

        imageio.mimsave(gif_path, frames, fps=fps)

        print(f"    GIF créé: {gif_path}")

    except ImportError:
        print("    imageio non installé. Installe avec: pip install imageio imageio-ffmpeg")
    except Exception as e:
        print(f"    Erreur: {e}")


# ========================================================================
# TEST
# ========================================================================

if __name__ == "__main__":
    print(" Test du module visualize\n")
    print("  Pour tester, tu dois avoir un modèle entraîné!")
    print("   Exemple: python utils/visualize.py\n")

    # Test avec un modèle (si existant)
    test_model = "./models/ppo_humanoid_final.zip"

    if os.path.exists(test_model):
        print(f" Modèle trouvé: {test_model}\n")

        choice = input("Veux-tu enregistrer une vidéo? (o/n): ").lower()
        save_vid = choice == 'o'

        visualize(
            test_model,
            n_episodes=2,
            save_video=save_vid,
            video_folder="./videos"
        )
    else:
        print(f" Aucun modèle trouvé à: {test_model}")
        print("   Lance d'abord un entraînement avec train.py")