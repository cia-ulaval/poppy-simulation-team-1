# Poppy Simulation 🤖

## Fiche d'Identité

*   **Type de projet :** Projet Club
*   **Team Lead :** Baptiste Bonin
*   **Partenaire Académique/Industriel :** Vooban
*   **Effectif recherché :** 5 membres
*   **Profils recherchés :** Reinforcement Learning, Computer Vision, Simulation robotique, Développement Python

## Description du Projet

Développer une intelligence artificielle pour apprendre à un robot humanoïde Poppy à marcher en utilisant l'apprentissage par renforcement dans un environnement de simulation, en préparation d'un déploiement sur le robot physique.

## Objectifs & Livrables

*   **Objectif Principal :** Développer un modèle de Reinforcement Learning capable de faire marcher le robot Poppy en simulation de manière robuste.
*   **Livrables attendus :**
    *   Environnement de simulation fonctionnel
    *   Modèle de RL entraîné pour la marche
    *   Documentation des algorithmes implémentés
    *   Pipeline de transfert simulation→réel (préparation)

## Timeline Prévisionnelle de la Session

| Semaine | Activité/Phase |
| :-----: | :------------- |
|  **1-2**  | **Préliminaires** - Survol projet et revue littérature |
|  **3-4**  | **Hello World** - Familiarisation librairies RL |
|  **5-6**  | **Simulation** - Prise en main environnements simplifiés |
|  **7-8**  | **Implémentation** - Développement algorithmes RL |
|  **9-10** | **TODO** |
|  **11-14**| **TODO** |

## Technologies & Compétences Visées

*   **Logiciels :** Python, PyTorch, OpenAI Gym, Librairies RL
*   **Matériels :** GPU (accès optionnel en fin de session)
*   **Compétences :** Reinforcement Learning, simulation robotique, optimisation ML, Python avancé

## Pourquoi rejoindre ce projet ?

Tu vas aimer ce projet si :
*   Tu veux travailler sur l'application concrète du RL en robotique
*   Tu es passionné par l'interface simulation→monde réel (Sim2Real)
*   Tu souhaites contribuer à faire marcher un vrai robot humanoïde

## Contact & Liens Utiles
*   **Référence :** [Poppy Project](https://www.poppy-project.org/)

## Démarrer

Tout passe par Docker. La référence complète est [docs/DOCKER.md](docs/DOCKER.md).

```bash
docker compose build train
docker compose --profile dev run --rm dev python -m pytest
docker compose --profile dev run --rm dev ruff check .
```

Entraîner, puis évaluer le modèle obtenu :

```bash
docker compose --profile train up
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_21-29-14/best_model.zip --episodes 10
```

**Première fois sur le projet ?** Suivez [docs/DEMARRAGE.md](docs/DEMARRAGE.md) :
un parcours guidé d'une heure, de la construction de l'image jusqu'à l'entraînement
de votre premier modèle.

Le guide technique complet — entraîner, visualiser, évaluer, toutes les commandes —
est dans [docs/TECHNIQUE.md](docs/TECHNIQUE.md). L'architecture et les diagrammes sont
dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), les règles de code dans
[CONTRIBUTING.md](CONTRIBUTING.md).
