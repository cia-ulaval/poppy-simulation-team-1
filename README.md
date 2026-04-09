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

## Lancer la Simulation ROS 2

### Build de l'image Docker

À la racine
```bash
docker build -t poppy-rolling .
```

### Lancer `run_robot`


```bash
docker run --rm -it --network host --ipc host \
  -e ROS_DOMAIN_ID=0 \
  -e ROS_LOCALHOST_ONLY=0 \
  -v "$PWD:/workspace" \
  -w /workspace \
  poppy-rolling \
  bash -lc 'source /opt/ros/rolling/setup.bash && xvfb-run -a python scripts/run_robot.py --model <best_model_path.zip> --vec-normalize <best_vec_norm_path>'
```

Par exemple:
```bash
docker run --rm -it --network host --ipc host \
  -e ROS_DOMAIN_ID=0 \
  -e ROS_LOCALHOST_ONLY=0 \
  -v "$PWD:/workspace" \
  -w /workspace \
  poppy-rolling \
  bash -lc 'source /opt/ros/rolling/setup.bash && xvfb-run -a python scripts/run_robot.py --model logs/poppy/2026-04-08_17-58-25/best_model/best_model.zip --vec-normalize logs/poppy/2026-04-08_17-58-25/best_model/vec_normalize.pkl'
```


### Vérifier le topic ROS

Avec un autre terminal sur la même machine :

Pour lsiter
```bash
docker run --rm --network host --ipc host \
  -e ROS_DOMAIN_ID=0 \
  -e ROS_LOCALHOST_ONLY=0 \
  ros:rolling-ros-core \
  bash -lc 'source /opt/ros/rolling/setup.bash && ros2 topic list | grep /poppy_motor_state'
```

Pour echo et inspecter le contenu des messages
```bash
docker run --rm --network host --ipc host \
  -e ROS_DOMAIN_ID=0 \
  -e ROS_LOCALHOST_ONLY=0 \
  ros:rolling-ros-core \
  bash -lc 'source /opt/ros/rolling/setup.bash && ros2 topic echo --once /poppy_motor_state'
```

Depuis une autre machine ROS 2 sur le même réseau :
> L'autre machine peut être dans un docker tant qu'elle a --network host et que les deux machines sont sur le même réseau, elles devraient découvrir les topics. Firewall UDP ouvert 

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
source /opt/ros/rolling/setup.bash
ros2 topic list
ros2 topic echo /poppy_motor_state
```
