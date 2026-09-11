# Docker — lancer le projet

Ce document est la référence unique pour démarrer quoi que ce soit sur le
projet. Si une commande n'est pas ici, c'est qu'elle n'est pas encore
stabilisée : ajoutez-la plutôt que de la garder dans votre historique shell.

---

## 1. Pourquoi trois images et pas une

Le dépôt mélange trois métiers qui n'ont ni les mêmes dépendances, ni la même
machine cible, ni le même rythme :

| Métier | Ce que ça fait | Où ça tourne |
|---|---|---|
| **Entraînement** | apprendre la marche en simulation MuJoCo | machine puissante, longtemps |
| **Pont robot** | envoyer les commandes au vrai Poppy | portable, à côté du robot |
| **Vision** | estimer la profondeur depuis une image | machine avec du GPU |

Une image unique obligerait tout le monde à télécharger `transformers` et
MuJoCo pour brancher un câble sur le robot. Trois images, un socle commun :

```
base            python 3.11 + torch + Stable-Baselines3
├── train       + MuJoCo hors écran + TensorBoard
├── bridge      + roslibpy
└── vision      + transformers + websockets

rosbridge       ROS 2 Humble + rosbridge_suite   (faux robot, à part)
```

C'est **un seul fichier**, [`docker/Dockerfile`](../docker/Dockerfile), avec
plusieurs cibles (`target:`). Les dépendances communes n'y sont écrites qu'une
fois : pas de dérive possible entre les images.

### Ce qu'on ne met pas dans Docker

`scripts/viewer.py` et `visu.py` ouvrent une fenêtre MuJoCo en OpenGL. Faire
sortir une fenêtre OpenGL d'un conteneur sous Windows demande un serveur X et
une demi-journée de réglages, pour zéro bénéfice. **Le viewer se lance en
natif**, voir § 8.

---

## 2. Prérequis

- **Docker Desktop** (Windows : backend WSL 2 activé).
- ~8 Go d'espace disque pour les images.
- Pour la variante GPU seulement : pilote NVIDIA récent + NVIDIA Container
  Toolkit dans WSL.

Vérifier que le démon tourne — si cette commande échoue, Docker Desktop
n'est pas démarré et rien d'autre ne marchera :

```bash
docker info --format '{{.ServerVersion}}'
```

---

## 3. Construire les images

Première fois, tout construire (comptez 10–20 min selon la connexion) :

```bash
docker compose --profile train --profile robot --profile vision build
```

Ou une seule cible :

```bash
docker compose build train
docker compose build bridge
docker compose build rosbridge
docker compose build vision
```

Vérifier le résultat :

```bash
docker images --filter reference='poppy-*'
```

---

## 4. Lancer

**Aucun service ne démarre par défaut.** Chacun est derrière un profil, pour
qu'un `docker compose up` distrait ne lance pas un entraînement de 10 millions
de pas. Il faut toujours nommer le profil.

### Entraînement + TensorBoard

```bash
docker compose --profile train up
```

TensorBoard écoute alors sur <http://localhost:6006>. Les logs sont écrits
dans `./logs/` sur votre machine, pas dans le conteneur : ils survivent.

Passer des options au script :

```bash
docker compose --profile train run --rm train python scripts/train_poppy.py --timesteps 100000 --n-envs 4
```

Vérification rapide que l'image est saine, sans rien entraîner :

```bash
docker compose --profile train run --rm train python scripts/train_poppy.py --help
```

Smoke test concret (2 048 pas, 1 env) :

```bash
docker compose --profile train run --rm train python scripts/train_poppy.py --config configs/poppy_robust.yaml --timesteps 2048 --n-envs 1 --seed 0 --log-dir logs/smoke
```

À la fin du run, un sous-dossier horodaté `logs/smoke/YYYY-MM-DD_HH-MM-SS/`
contient `poppy_ppo_final.zip` et `vec_normalize_final.pkl`.

### Évaluer un modèle entraîné

Répond à « est-ce que ce modèle marche ? » sans écran ni robot :

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model logs/smoke/<date>/poppy_ppo_final.zip --episodes 10
```

Les statistiques de normalisation (`vec_normalize_final.pkl`) sont cherchées à
côté du modèle. **Ne sautez pas cette étape** : un modèle entraîné avec
normalisation et évalué sans reçoit des observations sur une autre échelle et
paraît bien pire qu'il n'est, sans qu'aucune erreur ne soit levée.

La sortie donne récompense, durée, distance parcourue, verticalité, pourcentage
d'épisodes sans chute, puis **la décomposition des huit termes de récompense**.
C'est cette dernière partie qui sert : une politique qui tient la pose et une
politique qui marche obtiennent des totaux proches, seul le détail les sépare.
Si `healthy_reward` et `uprightness` dominent pendant que `gait_reward` reste
près de zéro, le robot ne marche pas.

Enregistrer une vidéo du premier épisode (rendu logiciel, aucun écran requis) :

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model logs/smoke/<date>/poppy_ppo_final.zip --episodes 1 --video logs/eval.mp4
```

Options utiles : `--floor-noise` mesure la robustesse au lieu de la performance
nominale (et cesse d'être reproductible), `--stochastic` échantillonne les
actions au lieu de prendre la moyenne, `--algorithm` force la classe si la
détection automatique échoue.

### Faux robot seul

Un rosbridge en conteneur, sans physique. Il accepte les connexions et relaie
les messages : de quoi vérifier que le pont publie ce qu'on croit, **sans
mobiliser le vrai robot**.

```bash
docker compose --profile mock up
```

Il écoute sur `ws://localhost:9090` (`POPPY_ROSBRIDGE_HOST` vaut `rosbridge`,
`POPPY_ROSBRIDGE_PORT` vaut `9090` dans la stack). Une fois le pont connecté,
le topic est visible et consommable :

```bash
# liste des topics
docker compose exec rosbridge bash -lc "source /opt/ros/humble/setup.bash && ros2 topic list"
# lecture des commandes publiées par le pont
docker compose exec rosbridge bash -lc "source /opt/ros/humble/setup.bash && ros2 topic echo /poppy_motor_state"
```

Cette validation a eu lieu contre le faux rosbridge uniquement ; le vrai
robot n'a pas été connecté.

### Pont robot (+ faux robot)

Le service `bridge` a pour commande par défaut `run_robot.py --help` : il
affiche l'aide et sort. **C'est voulu** — rien ne doit partir vers un robot
parce que quelqu'un a tapé `up`. Un `docker compose --profile robot up` verra
donc le pont s'arrêter aussitôt : ce n'est pas une panne.

L'image lit sa cible dans les variables d'environnement
`POPPY_ROSBRIDGE_HOST` (défaut `rosbridge`) et `POPPY_ROSBRIDGE_PORT` (défaut
`9090`), et le port est validé. Deux paramètres optionnels complètent le
comportement du pont :

- `POPPY_ROSBRIDGE_TIMEOUT_S` (défaut `10.0`) : timeout de connexion au pont
  rosbridge, en secondes. Doit être strictement positif.
- `POPPY_CONTROL_PERIOD_S` (défaut `5.0`) : période entre deux envois de
  commande dans l'adaptateur simulation, en secondes. Peut valoir `0` pour
  désactiver l'attente.

`./logs` est monté en lecture seule ; le pont recharge un modèle PPO et sa
normalisation via `--model` et `--vec-normalize`.

Pour valider contre le faux robot, en supposant un smoke test existant :

```bash
docker compose --profile mock up -d
docker compose --profile robot run --rm bridge python scripts/run_robot.py \
  --model /workspace/logs/smoke/<horodatage>/poppy_ppo_final.zip \
  --vec-normalize /workspace/logs/smoke/<horodatage>/vec_normalize_final.pkl
```

Où `<horodatage>` correspond au dossier horodaté produit par le smoke test.

Pour viser le **vrai** robot, redéfinir les variables à l'exécution :

```bash
docker compose --profile robot run --rm --no-deps \
  -e POPPY_ROSBRIDGE_HOST=<ip> \
  -e POPPY_ROSBRIDGE_PORT=<port> \
  bridge python scripts/run_robot.py --model <chemin> [--vec-normalize <chemin>]
```

Aucun essai sur robot réel n'a eu lieu à ce jour.

### Serveur de vision

```bash
docker compose --profile vision up
```

Il écoute sur le port 8000. Le client (`src/sensors/depth_client.py`) se
lance en natif, parce qu'il ouvre une fenêtre `cv2.imshow`.

### Tout arrêter

```bash
docker compose --profile train --profile robot --profile vision --profile mock down
```

---

## 5. Tests et analyse statique

L'image `train` sert aussi d'image de développement : c'est la seule à posséder
MuJoCo, donc la seule où les tests de l'environnement peuvent tourner. `pytest`
et `ruff` y sont installés.

```bash
docker compose --profile dev run --rm dev python -m pytest
docker compose --profile dev run --rm dev ruff check .
docker compose --profile dev run --rm dev ruff check . --fix
```

Le service `dev` est le seul à monter le dépôt **entier en écriture**. Les deux
lui sont nécessaires : entier parce que `visu.py` et `all_baseline.py` sont à
la racine et échappaient sinon au linter, en écriture parce que `--fix` doit
pouvoir corriger.

Attendu : `3 passed, 1 xfailed` et `All checks passed!`.

### Le test en échec attendu

`test_action_space_is_normalised` est marqué `xfail`. Il documente une
incohérence réelle : `_action_to_torque` (`poppy_humanoid_env.py:230`) calcule
`target = init + action × range`, ce qui **suppose** des actions dans `[-1, 1]`,
alors que `action_space` est hérité de `MujocoEnv` et calé sur
`actuator_ctrlrange` (±3,1 à ±7,3 selon l'articulation).

Stable-Baselines3 borne les actions à `action_space`, pas à `[-1, 1]` : une
politique peut donc émettre 2,5 et viser 2,5 fois au-delà de la limite
mécanique. Corriger `action_space` rendra **incompatibles tous les modèles déjà
entraînés** — c'est une décision d'équipe. Le marqueur est `strict` : le jour où
quelqu'un corrige l'environnement, le test passe au rouge pour signaler qu'il
faut retirer le marqueur.

### Ruff est vert, et ça se mérite

554 erreurs au premier passage. 566 corrections mécaniques ont été appliquées
(espaces, imports inutilisés, tri des imports, f-strings vides). Le reste est
listé explicitement dans `pyproject.toml` avec sa raison : 102 occurrences de
modernisation d'annotations reportées à un lot dédié, et 10 cas demandant un
jugement.

Un linter rouge en permanence est un linter ignoré. Celui-ci est vert, donc il
sert : il refusera toute nouvelle erreur dans du code neuf.

### Sans Docker

```bash
pip install -r requirements.txt
python -m pytest
ruff check .
```

---

## 6. Figer les versions

Les `requirements/*.txt` déclarent des plages (`mujoco>=3.0.0`). Deux personnes
qui construisent à une semaine d'écart n'obtiennent donc pas les mêmes paquets,
et un entraînement qui diverge devient impossible à attribuer : votre
modification, ou une version de MuJoCo qui a bougé ?

Les `requirements/*.lock` sont le relevé exact d'une image construite, fermeture
transitive comprise. Construire avec :

```bash
REQUIREMENTS=lock docker compose build train bridge vision
```

Vérifier qu'une image ne dérive pas de son lock :

```bash
docker run --rm poppy-bridge:latest python -m pip freeze | grep -viE "^(pip|setuptools|wheel)==" | sort -f > /tmp/actuel.txt
grep -vE "^#|^--|^$" requirements/bridge.lock | sort -f | diff - /tmp/actuel.txt
```

Pas de sortie : l'image correspond au lock.

### Régénérer un lock

Après avoir ajouté une dépendance dans un `.txt`, en deux temps — le `.txt` dit
**ce dont on dépend**, le `.lock` dit **ce qui est installé** :

```bash
docker compose build bridge
docker run --rm poppy-bridge:latest python -m pip freeze | grep -viE "^(pip|setuptools|wheel)==" | sort -f
```

Coller le résultat sous l'en-tête du `.lock` existant, en conservant la ligne
`--extra-index-url`.

**Pourquoi `txt` reste le défaut.** Un lock épingle `torch==…+cpu`. Construire
la variante GPU avec ce lock installerait la version CPU sur une machine à GPU.
Les deux options coexistent donc : `txt` pour développer et pour le GPU, `lock`
quand on veut qu'une image soit exactement reproductible.

---

## 7. Variante GPU

Par défaut les images installent **torch CPU** : ça marche sur toutes les
machines de l'équipe, sans pilote. MuJoCo tourne de toute façon sur CPU ; le
GPU ne sert qu'au réseau de neurones, et sur des politiques de cette taille le
gain est modeste.

Quand vous en voulez quand même, on **superpose** `compose.gpu.yaml` au
fichier principal. Pas de second service à maintenir : même image, même
volumes, seuls l'index torch et l'accès au matériel changent.

```bash
docker compose -f compose.yaml -f compose.gpu.yaml --profile train build
docker compose -f compose.yaml -f compose.gpu.yaml --profile train up
```

Vérifier que le conteneur voit bien la carte :

```bash
docker compose -f compose.yaml -f compose.gpu.yaml --profile train run --rm train python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Attention : la même étiquette `poppy-train:latest` sert aux deux variantes.
Après un build GPU, un build CPU l'écrase, et inversement. Si vous alternez,
reconstruisez.

Si ça affiche `False`, le problème est dans WSL/NVIDIA Container Toolkit, pas
dans nos images.

---

## 8. Sans Docker (viewer, client vision)

Tout ce qui ouvre une fenêtre se lance en natif. Installation :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installe l'ensemble complet, outils de développement
compris. Les images Docker, elles, utilisent les fichiers ciblés de
`requirements/`.

Deux fenêtres différentes, à ne pas confondre :

```bash
# Regarder le robot : articulations, limites, pose de départ.
# Aucun modèle entraîné nécessaire — le bon premier réflexe.
python scripts/viewer.py

# Regarder une politique entraînée se dérouler.
python visu.py logs/poppy/<date>/poppy_ppo_final.zip --episodes 5
```

Pour mesurer plutôt que regarder, l'évaluation tourne sans écran, en
conteneur : voir § 4.

---

## 9. Limites connues

Ce qui n'est pas encore stabilisé :

1. **La vision est mise de côté — décision d'équipe, septembre 2026.**
   `src/sensors/depth_server.py:9` importe `depth_anything_3`, qui n'est pas
   sur PyPI et n'a jamais figuré dans `requirements.txt`. L'image se construit
   — `torch`, `cv2`, `transformers` et `websockets` y sont — mais le serveur
   plante à l'import.

   Le chantier est **suspendu**, pas abandonné : la vision est un confort, la
   locomotion est l'objectif de la session. On le rouvrira quand quelqu'un
   aura retrouvé d'où venait ce paquet, ou quand on aura décidé de réécrire le
   serveur autour de `transformers` seul.

2. **Le transfert sur robot réel n'a pas été testé.**
   L'adresse n'est plus codée en dur : `src/robot/ros_publisher.py` lit
   `POPPY_ROSBRIDGE_HOST` et `POPPY_ROSBRIDGE_PORT`, avec `rosbridge` et
   `9090` par défaut. Toute la validation du pont a eu lieu contre le faux
   rosbridge.

   **Question ouverte, à trancher sur le robot lui-même** : quelle
   distribution ROS 2 y tourne, et `rosbridge_server` y est-il installé ?
   Tout le socle actuel suppose que oui. Une commande sur le robot répond :

   ```bash
   echo $ROS_DISTRO && ros2 pkg list | grep rosbridge
   ```

   Si `rosbridge_server` manque : `sudo apt install ros-$ROS_DISTRO-rosbridge-suite`.
   Sans accès shell, tester le port depuis le réseau suffit — une réponse
   `101 Switching Protocols` sur `http://<ip>:9090/` prouve qu'il écoute.

   L'ancienne connexion (`origin/feat/docker`) passait par `rclpy` natif avec
   `--network host`, impossible depuis Docker Desktop sous Windows. Les deux
   approches sont incompatibles : celle-ci est un pari tant que la question
   n'est pas tranchée.

3. **L'espace d'action contredit le contrat annoncé.**
   `_action_to_torque` (`poppy_humanoid_env.py:230`) suppose des actions dans
   `[-1, 1]`, alors qu'`action_space` vaut `actuator_ctrlrange` (±3,1 à ±7,3).
   Documenté par le test `xfail` `test_action_space_is_normalised`, voir § 5.
   Corriger invaliderait tous les modèles entraînés : décision d'équipe.

Par ailleurs, `logs/`, `ppo_logs/` et `baseline_logs/` pèsent ~691 Mo suivis
par git. Le `.dockerignore` les tient hors des images, et le `.gitignore` hors
des futurs commits, mais l'historique les garde. Décision reportée.

---

## 10. Dépannage

| Symptôme | Cause probable |
|---|---|
| `failed to connect to the docker API` | Docker Desktop n'est pas démarré |
| `Permission denied` sur `./logs` | l'utilisateur du conteneur est `poppy` (uid 1000) ; vérifier les droits du dossier hôte |
| Build très lent sur `torch` | c'est normal la première fois ; les couches sont ensuite en cache |
| `no such service: train` | profil oublié : `--profile train` |
| MuJoCo `Failed to initialize OpenGL` | vous lancez un rendu dans l'image `train` sans `MUJOCO_GL=osmesa` |

---

## 11. Comment on modifie tout ça

Le code est **monté en volume** dans les conteneurs (`./src`, `./scripts`,
`./configs`). Modifier un fichier Python ne demande donc pas de reconstruire :
relancez simplement le service.

Reconstruire n'est nécessaire que si vous touchez :

- un fichier de `requirements/`
- `docker/Dockerfile`

```bash
docker compose build --no-cache train
```
