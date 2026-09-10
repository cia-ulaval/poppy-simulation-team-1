# État des lieux — Poppy Simulation

> Document vivant. Démarré le 2026-08-11 par le team lead, à la reprise du projet
> après ~4 mois d'inactivité (dernier commit : 2026-04-09).
>
> But : donner à toute personne qui arrive sur le projet une vision exacte de ce
> qui existe, de ce qui marche, et de ce qui reste à faire — sans avoir à
> archéologuer 12 branches.

---

## 1. Où est le code (résolu)

**La branche de référence est `model-for-presentation`.** Son nom est trompeur :
malgré l'apparence d'une branche de démo jetable, c'est elle qui contient
l'intégralité du travail.

Vérifié le 2026-08-11 :

| Comparaison | Résultat |
|---|---|
| vs `origin/main` | 0 en retard, 17 en avance → **superset strict**, merge en fast-forward |
| vs `add-randomization` | contient les 17 mêmes commits (rebasés) + 49 de plus |
| `poppy_humanoid_env.py` | **identique** à `add-randomization` (diff vide) |
| Code exclusif | +991 lignes : `src/sensors/`, `src/robot/`, `run_robot.py`, `viewer.py`, `solidenv_modifier/` |

**Conséquence** : la crainte d'une « scission » entre le travail RL et le travail
vision/ROS est levée. Les deux moitiés ont bien été réunifiées — simplement sur
une branche que personne n'a promue en `main`.

---

## 2. Inventaire des branches

Référence de comparaison : `origin/model-for-presentation`.

| Branche | Commits uniques | Contenu | Action |
|---|---|---|---|
| `main` | 0 | ancêtre direct | **fast-forward** |
| `obstacle_scene` | 0 | intégré | supprimer |
| `roslibpy` | 0 | intégré | supprimer |
| `xml-modifier` | 0 | intégré | supprimer |
| `Mujoco-humanoid1` | 0 | intégré (nov. 2025) | supprimer |
| `add-randomization` | 17 | rebasés dans la ref, env identique | supprimer |
| `Mujoco-humanoid` | 1 | commit `1d90f24` | supprimer |
| `mujoco-humanoid` | 1 | **le même commit `1d90f24`** | supprimer |
| `Mujuco-humanoid` | 4 | faute de frappe, nov. 2025 | supprimer |
| `feat/docker` | 1 | **Dockerfile + .dockerignore + README** | ⚠️ cherry-pick |
| `Issac` | 2 | **Isaac Lab** : `tasks/poppy_humanoid.py`, `train.py` | ⚠️ archiver (tag) |

### Deux pièges à ne pas manquer

1. **`feat/docker` porte le seul Dockerfile du projet** — la seule réponse
   existante au problème de reproductibilité. Mais elle dérive d'un tronc
   antérieur : elle *supprime* les 178 lignes d'amélioration de l'environnement.
   La merger telle quelle **ferait régresser la simulation**. → cherry-pick
   ciblé du Docker uniquement.
2. **`Mujoco-humanoid` et `mujoco-humanoid` ne diffèrent que par la casse.**
   Sur Windows (poste du lead), les manipuler localement peut mettre git dans un
   état incohérent. → supprimer côté distant uniquement.

---

## 3. Ce qui existe techniquement

### Environnement MuJoCo — `src/environments/poppy_humanoid_env.py` (476 l.)

Sérieusement écrit, au-dessus de la moyenne d'un projet club.

- 25 DoF, ~83 cm, ~3,5 kg — converti depuis l'URDF officiel Poppy
- Observation 63-dim : `qpos[2:]` (30) + `qvel` (31) + forces de contact pieds (2)
- Action 25-dim, **contrôle PD en position** (kp=8, kd=0,5) → émule les
  Dynamixel réels. Bon choix pour le sim2real.
- Pose de repos corrigée par grid-search (l'URDF a les bras croisés à qpos=0) ;
  tient ~700 pas en passif, sans action
- Reward à 8 termes : vitesse avant plafonnée, alive, verticalité, alternance
  des appuis, pénalités latérale / contrôle / lissage / vitesse articulaire
- Domain randomization : friction, « restitution » (voir bug #1), masses ±15 %,
  poussées externes 2 % des pas, yaw initial ±15°

### Architecture logicielle

Propre et découplée : `core/` (ABC + Protocols), `algorithms/` (PPO, SAC, TD3,
A2C), `env_factory`, `trainer`, `evaluator`, configs YAML.

### Vision — `src/sensors/`

`depth_server` / `depth_client` en float16, `mj_multiRay` → 64 rayons réduits en
grille 8×8. **Pas encore branché sur la politique RL.**

### Sim2real — `src/robot/`

`roslibpy` publie sur `/poppy_motor_state`. **Prototype de démo, pas un pont
utilisable** : IP `10.242.180.129` codée en dur, et un `sleep(5)` en pleine
boucle `step()`.

---

## 4. Les trous

- **Zéro test, zéro CI.** Aucun garde-fou contre les régressions.
- **Aucun environnement reproductible.** Les deux venvs du dépôt sont cassés
  (`.venv` = Python 3.14 sans dépendances ; `venv` pointe vers un Python 3.11
  disparu). `requirements.txt` : 11 lignes non épinglées. Le Dockerfile existe
  mais dort sur une branche non mergée.
- **Aucun résultat consigné.** Des dizaines de checkpoints et de logs
  TensorBoard, mais aucun document ne dit *est-ce que le robot marche, sur
  quelle distance, avec quel modèle*. Les messages de commit (`best model`,
  `new best model with higher Z`, `add final best model`) sont la seule trace.
- **Dépôt de 1,1 Go.** Checkpoints `.zip` (6,4 Mo × ~100), `frames.zip` (27 Mo),
  44 `.npy` de 762 Ko — tout dans l'historique git.
- **Signal d'alarme** : `reverse x` ×4 et `inverse x & y` ×2 dans l'historique.
  Quelqu'un s'est battu avec le sens de l'axe d'avance sans jamais documenter la
  conclusion. À réétablir et à écrire noir sur blanc.

---

## 5. Bugs probables repérés à la lecture

> Non confirmés par exécution — le code n'a pas encore pu tourner (venvs cassés).

1. **La randomisation de restitution ne fait pas ce qu'elle croit.**
   `poppy_humanoid_env.py:457` écrit dans `geom_solimp[floor_id, 4]`. Dans
   MuJoCo, `solimp` = `(d0, dmax, width, midpoint, power)` : l'indice 4 est
   l'**exposant `power`**, pas le rebond. La restitution se règle via `solref`.
   → on randomise un paramètre de solveur au hasard en croyant varier
   l'élasticité du sol.
2. **Observation incomplète vs reward.** Le reward pénalise
   `action - prev_action`, mais `prev_action` n'est pas dans l'observation → la
   politique est pénalisée sur une quantité qu'elle ne peut pas observer
   (POMDP partiel). Correction standard : concaténer l'action précédente à
   l'observation.

---

## 6. Journal des décisions

| Date | Décision | Raison |
|---|---|---|
| 2026-08-11 | `model-for-presentation` devient la base de travail | superset strict de toutes les autres branches |
| 2026-08-11 | *(en attente)* fast-forward de `main` | rendre le dépôt lisible avant l'arrivée de l'équipe |

---

## 7. Questions ouvertes

- **Objectif de session ?** (a) consolider et *prouver* la marche en simulation,
  (b) pousser le sim2real vers le robot physique, (c) intégrer la vision pour la
  navigation. Ce choix conditionne tout le reste du plan.
- **Purge de l'historique ?** `git filter-repo` réglerait les 1,1 Go mais
  réécrit les SHA et casse les clones existants. À faire une seule fois, annoncé
  — idéalement maintenant, pendant que l'équipe est en reconstitution.
  Alternative douce : `.gitignore` + Git LFS pour la suite, et on vit avec le
  passé.
- **Le robot marche-t-il aujourd'hui ?** Aucune preuve dans le dépôt. À établir
  en priorité : c'est la mesure de référence dont dépend toute évaluation
  future.
