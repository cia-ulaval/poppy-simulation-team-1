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
natif**, voir § 6.

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

### Faux robot seul

Un rosbridge en conteneur, sans physique. Il accepte les connexions et relaie
les messages : de quoi vérifier que le pont publie ce qu'on croit, **sans
mobiliser le vrai robot**.

```bash
docker compose --profile mock up
```

Il écoute sur `ws://localhost:9090`. Test depuis l'hôte :

```bash
docker compose --profile mock run --rm rosbridge bash -lc "source /opt/ros/humble/setup.bash && ros2 topic list"
```

### Pont robot (+ faux robot)

Le service `bridge` a pour commande par défaut `run_robot.py --help` : il
affiche l'aide et sort. **C'est voulu** — rien ne doit partir vers un robot
parce que quelqu'un a tapé `up`. Un `docker compose --profile robot up` verra
donc le pont s'arrêter aussitôt : ce n'est pas une panne.

Pour travailler vraiment, on démarre le faux robot en fond puis on lance le
pont avec sa commande :

```bash
docker compose --profile mock up -d
docker compose --profile robot run --rm bridge python scripts/run_robot.py --help
```

Pour viser le **vrai** robot au lieu du faux, il faudra d'abord câbler
l'adresse — voir § 7, ce n'est pas encore fait.

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

## 5. Variante GPU

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

## 6. Sans Docker (viewer, client vision)

Pour tout ce qui ouvre une fenêtre :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/viewer.py
```

`requirements.txt` installe l'ensemble complet. Les images Docker, elles,
utilisent les fichiers ciblés de `requirements/`.

---

## 7. Limites connues

Trois choses sont cassées ou incomplètes. Elles sont listées ici plutôt que
découvertes une par une :

1. **L'adresse du robot est codée en dur.**
   `src/robot/ros_publisher.py:26` contient `10.242.180.129`. Les variables
   `POPPY_ROSBRIDGE_HOST` / `POPPY_ROSBRIDGE_PORT` existent dans l'image et
   dans `compose.yaml`, mais **le code ne les lit pas encore**. C'est la
   première tâche de code à faire.

2. **Le serveur de vision ne démarre pas.**
   `src/sensors/depth_server.py:9` importe `depth_anything_3`, qui n'est pas
   sur PyPI et n'a jamais figuré dans `requirements.txt`. L'image se construit,
   le serveur plante à l'import. Il faut retrouver d'où venait ce paquet.

3. **`roslibpy` manquait de `requirements.txt`.**
   Corrigé dans `requirements/bridge.txt`. Mentionné parce que ça explique
   pourquoi le pont ne s'installait chez personne.

Par ailleurs, `logs/`, `ppo_logs/` et `baseline_logs/` pèsent ~691 Mo suivis
par git. Le `.dockerignore` les tient hors des images, et le `.gitignore` hors
des futurs commits, mais l'historique les garde. Décision reportée.

---

## 8. Dépannage

| Symptôme | Cause probable |
|---|---|
| `failed to connect to the docker API` | Docker Desktop n'est pas démarré |
| `Permission denied` sur `./logs` | l'utilisateur du conteneur est `poppy` (uid 1000) ; vérifier les droits du dossier hôte |
| Build très lent sur `torch` | c'est normal la première fois ; les couches sont ensuite en cache |
| `no such service: train` | profil oublié : `--profile train` |
| MuJoCo `Failed to initialize OpenGL` | vous lancez un rendu dans l'image `train` sans `MUJOCO_GL=osmesa` |

---

## 9. Comment on modifie tout ça

Le code est **monté en volume** dans les conteneurs (`./src`, `./scripts`,
`./configs`). Modifier un fichier Python ne demande donc pas de reconstruire :
relancez simplement le service.

Reconstruire n'est nécessaire que si vous touchez :

- un fichier de `requirements/`
- `docker/Dockerfile`

```bash
docker compose build --no-cache train
```
