"""
Module d'valuation pour Poppy RL

value un modle entran sur plusieurs pisodes et calcule des mtriques
dtailles pour analyser la performance et amliorer le reward engineering.
"""

import numpy as np
import time
from stable_baselines3 import PPO

from envs import make_humanoid_env


class EvaluationMetrics:
    """
    Classe pour stocker et calculer les mtriques d'valuation
    """

    def __init__(self):
        # Mtriques de base
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0  # Nombre d'pisodes qui atteignent 1000 steps

        # Mtriques pour reward engineering
        self.forward_velocities = []  # Vitesse vers l'avant moyenne par pisode
        self.energy_costs = []  # Consommation d'nergie par pisode
        self.vertical_stability = []  # Stabilit du torse (angle par rapport  vertical)
        self.smoothness_scores = []  # Fluidit du mouvement (variation des actions)

        # Mtriques de dfaillance
        self.fall_steps = []  #  quel step le robot tombe
        self.termination_reasons = {"fallen": 0, "timeout": 0}

    def add_episode(self, reward, length, forward_vel, energy, stability, smoothness, terminated):
        """Ajoute les donnes d'un pisode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        if length >= 1000:  # Succs = atteint la fin sans tomber
            self.success_count += 1

        self.forward_velocities.append(forward_vel)
        self.energy_costs.append(energy)
        self.vertical_stability.append(stability)
        self.smoothness_scores.append(smoothness)

        if terminated:
            self.fall_steps.append(length)
            self.termination_reasons["fallen"] += 1
        else:
            self.termination_reasons["timeout"] += 1

    def compute_statistics(self):
        """Calcule les statistiques finales"""
        n_episodes = len(self.episode_rewards)

        return {
            # Performance globale
            "n_episodes": n_episodes,
            "reward_mean": np.mean(self.episode_rewards),
            "reward_std": np.std(self.episode_rewards),
            "reward_min": np.min(self.episode_rewards),
            "reward_max": np.max(self.episode_rewards),

            "length_mean": np.mean(self.episode_lengths),
            "length_std": np.std(self.episode_lengths),
            "length_min": np.min(self.episode_lengths),
            "length_max": np.max(self.episode_lengths),

            "success_rate": self.success_count / n_episodes * 100,

            # Mtriques pour reward engineering
            "forward_velocity_mean": np.mean(self.forward_velocities),
            "energy_cost_mean": np.mean(self.energy_costs),
            "vertical_stability_mean": np.mean(self.vertical_stability),
            "smoothness_mean": np.mean(self.smoothness_scores),

            # Dfaillances
            "fall_step_mean": np.mean(self.fall_steps) if self.fall_steps else None,
            "termination_reasons": self.termination_reasons,
        }

    def print_summary(self, stats):
        """Affiche un rsum format des statistiques"""
        print("\n" + "=" * 70)
        print(" RSULTATS DE L'VALUATION")
        print("=" * 70)

        # Performance globale
        print(f"\n PERFORMANCE GLOBALE ({stats['n_episodes']} pisodes)")
        print(f"   Reward moyen:    {stats['reward_mean']:>10.2f}  {stats['reward_std']:.2f}")
        print(f"   Reward min/max:  {stats['reward_min']:>10.2f} / {stats['reward_max']:.2f}")
        print(f"   Dure moyenne:   {stats['length_mean']:>10.1f}  {stats['length_std']:.1f} steps")
        print(f"   Dure min/max:   {stats['length_min']:>10.0f} / {stats['length_max']:.0f} steps")
        print(f"   Taux de succs:  {stats['success_rate']:>10.1f}% (atteint 1000 steps)")

        # Mtriques pour reward engineering
        print(f"\n  MTRIQUES POUR REWARD ENGINEERING")
        print(f"   Vitesse avant moyenne:    {stats['forward_velocity_mean']:>8.3f} m/s")
        print(f"   Consommation nergie:     {stats['energy_cost_mean']:>8.3f}")
        print(f"   Stabilit verticale:      {stats['vertical_stability_mean']:>8.3f} rad")
        print(f"   Fluidit (smoothness):    {stats['smoothness_mean']:>8.3f}")

        # Analyse des dfaillances
        print(f"\n ANALYSE DES DFAILLANCES")
        print(f"   Tomb:        {stats['termination_reasons']['fallen']:>4d} pisodes")
        print(f"   Timeout:      {stats['termination_reasons']['timeout']:>4d} pisodes")
        if stats['fall_step_mean']:
            print(f"   Step moyen de chute: {stats['fall_step_mean']:>6.1f}")

        # Recommandations
        print(f"\n RECOMMANDATIONS POUR AMLIORER")
        self._print_recommendations(stats)

        print("\n" + "=" * 70 + "\n")

    def _print_recommendations(self, stats):
        """Analyse les mtriques et suggre des amliorations"""
        recommendations = []

        # Analyse vitesse
        if stats['forward_velocity_mean'] < 0.3:
            recommendations.append("  Vitesse faible  Augmenter reward pour avancer (r_forward)")

        # Analyse nergie
        if stats['energy_cost_mean'] > 1.0:
            recommendations.append("  Consommation leve  Augmenter pnalit nergie (r_energy)")

        # Analyse stabilit
        if stats['vertical_stability_mean'] > 0.3:
            recommendations.append("  Instabilit verticale  Ajouter reward pour rester droit (r_upright)")

        # Analyse succs
        if stats['success_rate'] < 50:
            recommendations.append("  Taux de succs faible  Rduire pnalits ou augmenter rewards positifs")

        # Analyse chutes
        if stats['fall_step_mean'] and stats['fall_step_mean'] < 200:
            recommendations.append("  Chutes prcoces  Vrifier curriculum learning ou simplifier task")

        if not recommendations:
            recommendations.append(" Performance satisfaisante! Continuer l'entranement ou passer  Phase 2")

        for rec in recommendations:
            print(f"   {rec}")


def evaluate(
        model_path,
        n_episodes=20,
        render=False,
        deterministic=True
):
    """
    value un modle PPO entran sur plusieurs pisodes

    Args:
        model_path: Chemin vers le modle .zip
        n_episodes: Nombre d'pisodes de test (dfaut: 20 pour stats robustes)
        render: Si True, affiche la visualisation
        deterministic: Si True, utilise la moyenne de la policy (pas de stochastique)

    Returns:
        dict: Dictionnaire avec toutes les statistiques
    """

    print("\n" + "" * 35)
    print("VALUATION DU MODLE")
    print("" * 35 + "\n")

    print(f" Configuration:")
    print(f"   Modle: {model_path}")
    print(f"   Nombre d'pisodes: {n_episodes}")
    print(f"   Mode: {'Dterministe' if deterministic else 'Stochastique'}")
    print(f"   Rendu: {'Oui' if render else 'Non'}\n")

    # ========================================================================
    # 1. CHARGER LE MODLE
    # ========================================================================

    print(" Chargement du modle...")
    try:
        model = PPO.load(model_path)
        print("    Modle charg avec succs\n")
    except Exception as e:
        print(f"    Erreur lors du chargement: {e}")
        return None

    # ========================================================================
    # 2. CRER L'ENVIRONNEMENT
    # ========================================================================

    print(" Cration de l'environnement de test...")
    env = make_humanoid_env(render_mode="human" if render else None)
    print("    Environnement cr\n")

    # ========================================================================
    # 3. VALUATION SUR N PISODES
    # ========================================================================

    print("=" * 70)
    print(f" DBUT DE L'VALUATION ({n_episodes} pisodes)")
    print("=" * 70 + "\n")

    metrics = EvaluationMetrics()

    start_time = time.time()

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}...", end=" ")

        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0

        # Mtriques dtailles pour cet pisode
        forward_velocities = []
        actions_history = []
        torse_angles = []

        while not done:
            # Prdire l'action avec le modle
            action, _states = model.predict(obs, deterministic=deterministic)

            # Stocker l'action pour analyse de smoothness
            actions_history.append(action.copy())

            # Step dans l'environnement
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step += 1
            done = terminated or truncated

            # Extraire des infos de l'observation pour mtriques avances
            # obs[22:25] = vitesse linaire du centre de masse (x, y, z)
            # obs[0:2] = orientation du torse (approximation)
            if len(obs) >= 25:
                forward_vel = obs[22]  # Vitesse en x (avant)
                forward_velocities.append(forward_vel)

                # Angle du torse (approximation avec quaternion)
                torse_angle = abs(obs[1])  # Simplification
                torse_angles.append(torse_angle)

            if render:
                time.sleep(0.01)  # Ralentir pour voir

        # Calculer les mtriques de l'pisode
        avg_forward_vel = np.mean(forward_velocities) if forward_velocities else 0
        avg_torse_angle = np.mean(torse_angles) if torse_angles else 0

        # Consommation d'nergie (somme des carrs des actions)
        energy_cost = sum(np.sum(a ** 2) for a in actions_history)

        # Smoothness (variance des changements d'actions)
        if len(actions_history) > 1:
            action_changes = [np.linalg.norm(actions_history[i + 1] - actions_history[i])
                              for i in range(len(actions_history) - 1)]
            smoothness = np.std(action_changes) if action_changes else 0
        else:
            smoothness = 0

        # Ajouter  la collection de mtriques
        metrics.add_episode(
            reward=episode_reward,
            length=step,
            forward_vel=avg_forward_vel,
            energy=energy_cost,
            stability=avg_torse_angle,
            smoothness=smoothness,
            terminated=terminated
        )

        print(f"Reward: {episode_reward:>8.2f}, Steps: {step:>4d}, "
              f"Raison: {'Tomb' if terminated else 'Timeout'}")

    eval_time = time.time() - start_time

    print(f"\n  Temps d'valuation: {eval_time:.1f}s")

    # ========================================================================
    # 4. CALCULER ET AFFICHER LES STATISTIQUES
    # ========================================================================

    stats = metrics.compute_statistics()
    metrics.print_summary(stats)

    # Nettoyer
    env.close()

    print(" valuation termine!\n")

    return stats


# ========================================================================
# FONCTION HELPER: COMPARAISON AVEC BASELINE
# ========================================================================

def compare_with_baseline(model_path, n_episodes=20):
    """
    Compare le modle entran avec des actions alatoires (baseline)

    Args:
        model_path: Chemin vers le modle entran
        n_episodes: Nombre d'pisodes pour chaque test
    """
    print("\n" + "" * 35)
    print("COMPARAISON: MODLE vs BASELINE ALATOIRE")
    print("" * 35 + "\n")

    # valuer le modle entran
    print("1  valuation du MODLE ENTRAN:")
    model_stats = evaluate(model_path, n_episodes=n_episodes, render=False)

    if model_stats is None:
        return

    # valuer baseline (actions alatoires)
    print("\n2  valuation de la BASELINE (actions alatoires):")

    env = make_humanoid_env()
    baseline_rewards = []
    baseline_lengths = []

    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}...", end=" ")
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action = env.action_space.sample()  # Action alatoire
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated

        baseline_rewards.append(episode_reward)
        baseline_lengths.append(step)
        print(f"Reward: {episode_reward:>8.2f}, Steps: {step:>4d}")

    env.close()

    # Comparer
    print("\n" + "=" * 70)
    print(" COMPARAISON DES RSULTATS")
    print("=" * 70 + "\n")

    baseline_reward_mean = np.mean(baseline_rewards)
    baseline_length_mean = np.mean(baseline_lengths)

    improvement_reward = (model_stats['reward_mean'] / baseline_reward_mean - 1) * 100
    improvement_length = (model_stats['length_mean'] / baseline_length_mean - 1) * 100

    print(f"{'Mtrique':<25} {'Baseline':>15} {'Modle':>15} {'Amlioration':>15}")
    print("-" * 70)
    print(
        f"{'Reward moyen':<25} {baseline_reward_mean:>15.2f} {model_stats['reward_mean']:>15.2f} {improvement_reward:>14.1f}%")
    print(
        f"{'Dure moyenne':<25} {baseline_length_mean:>15.1f} {model_stats['length_mean']:>15.1f} {improvement_length:>14.1f}%")
    print(f"{'Taux de succs':<25} {'0.0%':>15} {model_stats['success_rate']:>14.1f}% {'-':>15}")

    print("\n" + "=" * 70 + "\n")

    if improvement_reward > 1000:
        print(" EXCELLENT! Le modle est >10x meilleur que alatoire!")
    elif improvement_reward > 500:
        print(" TRS BON! Le modle est >5x meilleur que alatoire!")
    elif improvement_reward > 100:
        print(" BON! Le modle a bien appris, continue l'entranement!")
    else:
        print("  Le modle n'a pas encore beaucoup appris, continue l'entranement!")

    print()


# ========================================================================
# TEST
# ========================================================================

if __name__ == "__main__":
    print(" Test du module evaluate\n")
    print("  Pour tester, tu dois avoir un modle entran!")
    print("   Exemple: python utils/evaluate.py\n")

    # Test avec un modle (si existant)
    import os

    test_model = "./models/ppo_humanoid_final.zip"

    if os.path.exists(test_model):
        print(f" Modle trouv: {test_model}\n")
        evaluate(test_model, n_episodes=5, render=False)
    else:
        print(f" Aucun modle trouv : {test_model}")
        print("   Lance d'abord un entranement avec train.py")