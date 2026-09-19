# Guide technique

Pour faire tourner le projet. Le `README.md` de la racine présente le projet ;
celui-ci explique comment on s'en sert.

- Détail de Docker (GPU, versions figées, dépannage) : [`DOCKER.md`](DOCKER.md)
- Comment c'est construit, diagrammes C4 : [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Règles de code et de contribution : [`../CONTRIBUTING.md`](../CONTRIBUTING.md)

---

## 1. Carte du dépôt

| Dossier | Contenu |
|---|---|
| `src/environments/` | L'environnement Gymnasium du Poppy : physique, récompense, randomisation. **Le cœur.** |
| `src/config/` | Lecture des YAML de `configs/` et objets de configuration |
| `src/robot/` | Pont vers le robot réel, par rosbridge websocket |
| `src/sensors/` | Perception de profondeur — **en pause**, voir son [README](../src/sensors/README.md) |
| `scripts/` | Les cinq points d'entrée : `train_poppy`, `evaluate`, `visualize`, `viewer`, `run_robot` |
| `configs/` | Hyperparamètres en YAML. Rien de réglable n'est dans le code. |
| `assets/poppy_humanoid/` | Le modèle 3D : MJCF, URDF, 52 maillages STL |
| `models/` | Onze politiques entraînées, [classées ici](../models/README.md) |
| `tests/` | Les tests. `pytest` les lance tous. |
| `docker/`, `compose*.yaml` | Les images et les services |
| `logs/`, `figs/` | Sorties d'exécution. **Jamais versionnées.** |

## 2. Démarrer en trois commandes

```bash
docker compose build train
```

```bash
docker compose --profile train run --rm train python scripts/train_poppy.py --config configs/poppy_robust.yaml --timesteps 2048 --n-envs 1 --seed 0 --log-dir logs/smoke
```

Trente secondes. Si ça produit un `.zip` dans `logs/smoke/<date>/`, l'installation est
bonne. Puis, **en natif** (pas dans Docker, voir §4) :

```bash
python scripts/viewer.py
```

Le robot apparaît dans une fenêtre MuJoCo. Aucun modèle entraîné nécessaire.

## 3. Entraîner

```bash
docker compose --profile train up
```

Lance l'entraînement **et** TensorBoard sur <http://localhost:6006>. Les sorties vont
dans `logs/poppy/<date>/`.

| Option | Effet |
|---|---|
| `--config <fichier>` | Défaut `configs/poppy_robust.yaml` (ou `configs/humanoid_baseline.yaml` avec `--baseline`) |
| `--algorithm PPO\|SAC\|TD3\|A2C` | Écrase l'algorithme du YAML |
| `--timesteps <N>` | Écrase la durée |
| `--n-envs <N>` | Écrase le nombre d'environnements parallèles |
| `--seed <N>` | Graine |
| `--no-floor-noise` | Coupe la randomisation du sol, pour mesurer son effet |
| `--baseline` | Entraîne sur `Humanoid-v5` au lieu de Poppy — valide le pipeline |
| `--log-dir <chemin>` | Où écrire |

**Le piège du nombre d'environnements.** `configs/poppy_robust.yaml` demande
`n_envs: 64`, calibré « 32 cœurs + RTX 4090 ». Sur une machine ordinaire,
`SubprocVecEnv` n'arrive pas à lancer 64 processus MuJoCo et l'entraînement meurt sur
un `BrokenPipeError`. Le service compose met 8 par défaut. Pour relever :

```bash
POPPY_N_ENVS=32 docker compose --profile train up
```

```powershell
$env:POPPY_N_ENVS = "32"; docker compose --profile train up
```

**Un entraînement complet fait 10 millions de pas** et dure des heures. C'est pour ça
qu'aucun service compose ne démarre par défaut : tous sont derrière un profil, pour
qu'un `docker compose up` distrait ne lance rien.

## 4. Visualiser

Trois façons, par ordre de facilité.

### 4.1 TensorBoard — les courbes, pendant l'entraînement

```bash
docker compose --profile train up tensorboard
```

Puis <http://localhost:6006>. Récompense, longueur d'épisode, pertes. Rien à
installer, le service lit `logs/`.

### 4.2 Une vidéo — sans écran, sans installation

C'est le chemin le plus sûr, et le seul qui marche partout : le rendu se fait hors
écran dans le conteneur, en `osmesa`.

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_21-29-14/best_model.zip --episodes 1 --video logs/marche.mp4
```

**La caméra suit le robot.** C'est `--camera suivi`, le défaut. Sans ce suivi, une
politique qui marche 1,40 m sort simplement du cadre et la vidéo ne montre plus
qu'un sol vide.

| `--camera` | Ce que ça montre |
|---|---|
| `suivi` | Trois-quarts, accompagne le robot. **Le défaut**, le plus lisible. |
| `cote` | De profil : longueur de foulée, hauteur de pied. |
| `face` | De face : la dérive latérale, celle que la récompense pénalise. |
| `dessus` | De dessus : la trajectoire au sol. |
| `large` | Plan fixe et large : le robot traverse le champ, on juge la distance réelle. |

### 4.3 Le viewer interactif — à la souris, en natif

Pour tourner autour du robot, zoomer, inspecter une articulation. Ces deux scripts
ouvrent une fenêtre OpenGL, ce qu'un conteneur ne sait pas faire sous Windows sans un
serveur X et une demi-journée de réglages. **Ils demandent donc un Python installé sur
la machine.**

Une fois par poste — Python 3.12, un environnement isolé, les dépendances :

```powershell
winget install --id Python.Python.3.12 --exact --source winget --scope user
```

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/train.txt
```

Le torch CPU suffit : on ne fait ici que regarder, pas entraîner. Compter ~1,5 Go et
une dizaine de minutes. Sous bash, remplacer `.\.venv\Scripts\python.exe` par
`.venv/bin/python`.

Ensuite :

```powershell
.\.venv\Scripts\python.exe scripts/viewer.py
```

Le robot seul, sans politique. Articulations, limites, pose de départ. Aucun modèle
entraîné nécessaire : c'est le premier réflexe utile quand on découvre le projet.

```powershell
.\.venv\Scripts\python.exe scripts/visualize.py models/2026-04-08_21-29-14/best_model.zip --episodes 5
```

Une politique qui se déroule, à l'écran. `--camera` choisit le cadrage de départ, la
souris prend le relais ensuite. `--fps 30` ralentit, `--no-render` désactive la
fenêtre pour ne mesurer que les récompenses.

> `python` tout court ne fonctionne pas sous Windows tant que rien n'est installé :
> la commande est interceptée par un raccourci Microsoft Store qui ne fait rien.
> D'où le chemin explicite vers l'interpréteur de l'environnement.

## 5. Évaluer

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_21-29-14/best_model.zip --episodes 10
```

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --random --episodes 10
```

Le second donne le **plancher** : ce que vaut un robot qui agite ses membres au
hasard. Sans ce repère, une récompense de 2000 ne veut rien dire.

### Lire le résultat — la seule chose à vraiment comprendre

**Ne jamais juger sur la récompense seule.** Regarder d'abord trois lignes :
« Déplacement net », « Épisodes sans chute », et la répartition entre x et y.

`evaluate.py` sépare délibérément le déplacement total de sa composante en x, parce
que la récompense ne mesure que x. Un robot qui parcourt 5,70 m dont 5,68 en y **s'est
déplacé** ; la récompense, elle, le note comme immobile. Le script le dit explicitement
quand ça se produit.

La décomposition des huit termes vient ensuite. Si `healthy_reward` domine, que
`capped_vel` est proche de zéro **et** que le déplacement net est proche de zéro, alors
seulement le robot tient la pose sans marcher.

| Option | Effet |
|---|---|
| `--model <fichier.zip>` | Le modèle à évaluer |
| `--random` | Politique aléatoire à la place — le plancher |
| `--episodes <N>` | Défaut 10 |
| `--vec-normalize <fichier>` | Normalisation explicite. Trouvée toute seule si elle est à côté du modèle. |
| `--floor-noise` | Réactive la randomisation : mesure la robustesse, pas la performance nominale. Les résultats cessent d'être reproductibles. |
| `--stochastic` | Échantillonne les actions au lieu de prendre la moyenne |
| `--video sortie.mp4` | Enregistre le premier épisode, sans écran |
| `--camera suivi\|cote\|face\|dessus\|large` | Cadrage de la vidéo. Défaut `suivi` : la caméra accompagne le robot (voir §4.2). |

## 6. Tests et analyse statique

```bash
docker compose --profile dev run --rm dev python -m pytest
```

```bash
docker compose --profile dev run --rm dev ruff check .
```

**Le résultat attendu est `11 passed, 1 xfailed`.** Le `xfailed` n'est pas une panne :
c'est un test volontairement en échec attendu, qui documente une incohérence connue
entre le contrat annoncé (actions dans `[-1, 1]`) et `action_space`, hérité de
`actuator_ctrlrange`. Voir §8. `ruff check . --fix` corrige ce qui est mécanique.

## 7. Toutes les commandes

| Ce que ça fait | Commande |
|---|---|
| Construire l'image d'entraînement | `docker compose build train` |
| Entraîner + TensorBoard | `docker compose --profile train up` |
| TensorBoard seul | `docker compose --profile train up tensorboard` |
| Entraînement court de validation | `docker compose --profile train run --rm train python scripts/train_poppy.py --timesteps 2048 --n-envs 1 --log-dir logs/smoke` |
| Aide de l'entraînement | `docker compose --profile train run --rm train python scripts/train_poppy.py --help` |
| Évaluer un modèle | `docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/<date>/best_model.zip --episodes 10` |
| Plancher aléatoire | `docker compose --profile eval run --rm eval python scripts/evaluate.py --random --episodes 10` |
| Vidéo sans écran, caméra qui suit | `docker compose --profile eval run --rm eval python scripts/evaluate.py --model <modèle> --episodes 1 --video logs/eval.mp4` |
| Vidéo sous un autre angle | `docker compose --profile eval run --rm eval python scripts/evaluate.py --model <modèle> --episodes 1 --video logs/eval.mp4 --camera cote` |
| Tests | `docker compose --profile dev run --rm dev python -m pytest` |
| Analyse statique | `docker compose --profile dev run --rm dev ruff check .` |
| Corriger le style | `docker compose --profile dev run --rm dev ruff check . --fix` |
| Faux robot seul | `docker compose --profile mock up` |
| Écouter le faux robot | `docker compose exec rosbridge bash -lc "source /opt/ros/humble/setup.bash && ros2 topic echo /poppy_motor_state"` |
| Pont + faux robot | `docker compose --profile robot up` |
| Construire en CUDA | `docker compose -f compose.yaml -f compose.gpu.yaml --profile train build` |
| Vérifier le GPU | `docker compose -f compose.yaml -f compose.gpu.yaml --profile train run --rm train python -c "import torch; print(torch.cuda.is_available())"` |
| Versions figées | `REQUIREMENTS=lock docker compose build train` |
| Tout arrêter | `docker compose --profile train --profile robot --profile mock --profile vision down` |
| Le robot, en natif (voir §4.3) | `.\.venv\Scripts\python.exe scripts/viewer.py` |
| Une politique, en natif | `.\.venv\Scripts\python.exe scripts/visualize.py models/<date>/best_model.zip --episodes 5` |

## 8. Ce qui est cassé, et connu

Écrit pour que personne ne perde une journée à le redécouvrir.

**Le terme d'avance est projeté sur l'axe x du monde, pas sur le cap du robot.**
Conséquence mesurée : `2026-04-08_23-00-52` marche 5,70 m en 10 s à 0,57 m/s, en ligne
droite, avec 92 % d'appuis alternés et sans jamais tomber — mais à 98° de son propre
cap. C'est un pas chassé. La récompense ne voit presque rien de ce déplacement et le
pénalise même comme dérive latérale. Une politique peut donc marcher sans être
récompensée. Le seul modèle qui marche *droit*, `2026-04-08_21-29-14`, tombe au bout
de 3 secondes. Détail et pistes de correction dans
[`models/README.md`](../models/README.md).

**L'espace d'action ne correspond pas à son contrat.** La documentation de
`PoppyHumanoidEnv` et `_action_to_torque` annoncent des actions dans `[-1, 1]`, mais
`action_space` est hérité de `MujocoEnv`, donc calé sur `actuator_ctrlrange` (±1,8 à
±7,3 selon l'articulation). Stable-Baselines3 borne les actions à `action_space`, pas
à `[-1, 1]` : une politique peut émettre 2,5 et viser 2,5 fois au-delà de la limite
mécanique, saturant les actionneurs. Corriger **invalidera tous les modèles déjà
entraînés** : c'est une décision d'équipe. Le test `test_action_space_is_normalised`
est en `xfail(strict=True)` pour que le jour où c'est corrigé, il passe au rouge et
force à retirer le marqueur.

**La randomisation du rebond ne fait pas ce qu'elle annonce.**
`poppy_humanoid_env.py:454` écrit la restitution dans `geom_solimp[floor_id, 4]`. Cet
indice est l'exposant `power` de la fonction de solveur, pas le rebond — la
restitution vit dans `solref`. Le sol ne rebondit donc jamais, quelle que soit la
plage configurée.

**`prev_action` est pénalisé mais pas observé.** La récompense pénalise l'écart entre
deux actions consécutives, or l'action précédente ne fait pas partie des 63 dimensions
de l'observation. La politique est punie pour une variable qu'elle ne peut pas voir.

**Le pont ROS n'a jamais parlé à un vrai robot.** Il est testé contre le faux
`rosbridge` de la pile compose, et rien d'autre.

**La vision ne démarre pas.** `depth_anything_3` n'existe pas sur PyPI. Détail complet
dans [`../src/sensors/README.md`](../src/sensors/README.md).

**La variante GPU n'a jamais été testée.** `compose.gpu.yaml` est écrit, jamais
exécuté faute de machine.

## 9. Où on en est

État au 18 septembre 2026.

| Sujet | État |
|---|---|
| Socle Docker | ✅ 4 images, entraînement et pont validés contre le faux robot |
| Pont `ros_publisher` configurable | ✅ `POPPY_ROSBRIDGE_HOST/PORT/TIMEOUT_S`, bornes validées, aucune IP en dur |
| Évaluation sans écran | ✅ `scripts/evaluate.py`, service `eval`, vidéo `osmesa` |
| Outillage qualité | ✅ `ruff` vert, 11 tests + 1 `xfail` |
| Versions figées | ✅ `requirements/*.lock`, via `REQUIREMENTS=lock` |
| GPU | ⏸️ écrit, jamais testé — à faire seulement si une mesure le justifie |
| Vision | ⏸️ mise de côté, décision d'équipe de septembre 2026. La locomotion passe avant. |
| ROS du robot réel | ⏳ en attente du robot |
| Espace d'action normalisé | ⏳ décision d'équipe : corriger invalide tous les modèles |

**Deux questions ouvertes.**

*Quelle ROS tourne sur le Poppy ?* Tout le socle suppose un `rosbridge_server` en
websocket sur le port 9090. Une ancienne connexion réussie passait au contraire par
`rclpy` natif avec `--network host`, impossible depuis Docker Desktop sous Windows.
Les deux approches sont incompatibles. Réponse le jour où le robot est là :

```bash
echo $ROS_DISTRO && ros2 pkg list | grep rosbridge
```

*D'où venait `depth_anything_3` ?* Suspendue avec le chantier vision. Deux pistes si
elle se repose : `git log --all -S "depth_anything"` et l'auteur de `src/sensors/`.
