# Poppy Simulation — matériel pour CV

> Base : dépôt `poppy-simulation-team-1`, branche `model-for-presentation`.
> Tout ce qui est listé ici existe réellement dans le code.

---

## 1. Intitulé (ligne de CV)

**Apprentissage par renforcement pour la marche d'un robot humanoïde (Poppy)**
Club d'Intelligence Artificielle — Université Laval · Partenaire : Vooban · Python, MuJoCo, PyTorch, ROS
`2025 – 2026`

Variantes d'intitulé selon le poste visé :
- *ML / RL* → « RL appliqué à la locomotion humanoïde — simulation MuJoCo, PPO/SAC/TD3 »
- *Robotique* → « Contrôle d'un humanoïde 25 DoF en simulation, pipeline Sim2Real (ROS) »
- *Vision* → « Perception de profondeur monoculaire pour la navigation d'un robot humanoïde »

---

## 2. Description courte (2–3 lignes, format CV classique)

> Développement d'une IA capable de faire marcher un robot humanoïde open source
> Poppy (25 degrés de liberté) par apprentissage par renforcement dans un
> simulateur physique MuJoCo, avec pipeline de transfert vers le robot réel
> (Sim2Real) et perception de l'environnement par estimation de profondeur.

## 3. Description longue (LinkedIn / portfolio / lettre de motivation)

> Projet d'équipe visant à apprendre la marche à un humanoïde imprimé en 3D
> (Poppy, 25 moteurs, 83 cm, 3,5 kg) sans écrire manuellement le contrôleur.
> Le robot a été porté depuis son URDF officiel vers un environnement Gymnasium /
> MuJoCo sur mesure : observation 63-dim (posture, vitesses articulaires, forces
> de contact aux pieds), action 25-dim en **contrôle PD en position** pour
> reproduire le comportement des servomoteurs Dynamixel réels et faciliter le
> transfert vers le matériel. Une fonction de récompense multi-critères
> (8 termes : vitesse d'avance, survie, verticalité, alternance des appuis,
> pénalités de contrôle et de lissage) et de la **domain randomization**
> (friction du sol, masses ±15 %, poussées externes, orientation initiale)
> entraînent une politique robuste aux variations du monde réel.
> L'entraînement s'appuie sur Stable-Baselines3 (PPO, SAC, TD3, A2C) avec
> vectorisation sur 64 environnements parallèles, suivi TensorBoard et
> configuration entièrement pilotée par YAML. Le projet couvre aussi la
> perception (estimation de profondeur monoculaire par modèle de vision,
> lancer de rayons dans le simulateur) et la communication robot–simulation
> via ROS (roslibpy).

---

## 4. Ce qui a été réalisé, par domaine

### 🤖 IA / Apprentissage par renforcement

- Environnement Gymnasium **sur mesure** pour Poppy (476 lignes), conforme à l'API `Humanoid-v5`,
  converti depuis l'URDF officiel — 25 DoF, observation 63-dim, action 25-dim continue.
- **Contrôle PD en position** (kp = 8, kd = 0,5) plutôt qu'en couple : émule les servos Dynamixel
  réels, choix motivé par le Sim2Real.
- **Fonction de récompense à 8 termes** conçue et itérée : vitesse d'avance plafonnée, bonus de
  survie, verticalité du buste, alternance des appuis (rythme de marche), pénalités de dérive
  latérale, d'effort de contrôle, de non-lissage de l'action, de vitesse articulaire.
- **Domain randomization** au reset : friction du sol (0,4–2,0), masses des corps ±15 %,
  poussées externes aléatoires sur 2 % des pas, orientation initiale (yaw) ±15°.
- **4 algorithmes** intégrés derrière une interface commune : PPO, SAC, TD3, A2C
  (Stable-Baselines3), interchangeables par une seule ligne de configuration.
- Entraînement **vectorisé sur 64 environnements parallèles**, jusqu'à 10 M de pas par run,
  normalisation des observations et des récompenses, checkpoints périodiques, évaluation en ligne.
- **Baseline de référence** sur `Humanoid-v5` avec hyperparamètres tunés (rl-baselines3-zoo)
  pour valider le pipeline indépendamment du modèle robot.
- Pipeline d'évaluation et de visualisation : lecture des logs TensorBoard, courbes
  d'apprentissage, rendu vidéo des politiques entraînées.
- Correction de la **pose de repos** du modèle par recherche systématique (l'URDF plaçait les bras
  croisés à qpos = 0) : le robot tient ~700 pas en passif, ce qui rend l'apprentissage possible.

### 👁️ Vision par ordinateur

- **Estimation de profondeur monoculaire** avec *Depth Anything 3* (`da3metric-large`,
  PyTorch/CUDA) : une image RGB de la caméra → carte de profondeur métrique.
- Architecture **client/serveur temps réel** en WebSockets : le serveur héberge le modèle sur GPU,
  le client (la simulation) envoie les frames encodées et reçoit la profondeur brute en
  **float16** — choix de format pour réduire la bande passante.
- **Analyse et réduction de la carte de profondeur** : découpage en régions, agrégation en une
  représentation compacte exploitable comme observation par la politique RL.
- **Lancer de rayons dans MuJoCo** (`mj_multiRay`) : 64 rayons horizontaux sur un champ de vision
  de 90° depuis la tête, réduits en grille 8×8 — capteur de distance léger, alternative « sans
  image » à la vision.
- Outillage de visualisation : overlay de la carte de profondeur sur le flux vidéo pour le débogage.

### 🔌 ROS / Sim2Real / Robotique

- **Pont simulation → robot physique** via `roslibpy` (bridge ROS WebSocket) : les positions
  articulaires calculées par la politique sont publiées sur le topic `/poppy_motor_state`,
  avec les noms de joints, pour être consommées par le robot réel.
- **Couche d'adaptation** (`SimulationAdaptater`) isolant le format simulateur du format robot :
  la même politique peut piloter la simulation ou le matériel.
- Intégration **pypot** (bibliothèque officielle de contrôle des moteurs Poppy) pour la cible réelle.
- **Conversion URDF → MJCF** du modèle Poppy officiel (meshes STL, limites articulaires, inerties),
  plus un générateur de scènes MuJoCo (terrain, obstacles) pour tester la robustesse.
- Exploration d'un portage vers **NVIDIA Isaac Lab / Isaac Gym** (configs de tâche et
  d'entraînement PPO) pour l'entraînement massivement parallèle sur GPU.

### ⚙️ Ingénierie logicielle & MLOps

- Architecture **découplée et extensible** : `core/` (classes abstraites + Protocols),
  `algorithms/`, `environments/` (factory), `training/`, `evaluation/`, `visualization/`,
  `sensors/`, `robot/`.
- **Configuration déclarative en YAML** : algorithme, hyperparamètres, randomisation, réseau,
  budget d'entraînement — aucun paramètre codé en dur, expériences reproductibles.
- Suivi d'expériences **TensorBoard**, gestion des checkpoints, scripts CLI (`train`, `viewer`,
  `run_robot`).
- **Audit technique du dépôt** : inventaire de 12 branches, identification de la branche de
  référence, repérage de deux bugs de fond (randomisation de la restitution écrivant dans le
  mauvais champ MuJoCo — `solimp` au lieu de `solref` ; action précédente pénalisée par la
  récompense mais absente de l'observation → POMDP), rédaction de la documentation d'onboarding.

---

## 5. Stack / mots-clés à faire apparaître

`Python` · `PyTorch` · `MuJoCo` · `Gymnasium` · `Stable-Baselines3` · `PPO` · `SAC` · `TD3` · `A2C`
`Reinforcement Learning` · `Domain Randomization` · `Sim2Real` · `Robotique humanoïde` · `Contrôle PD`
`ROS / roslibpy` · `pypot` · `Dynamixel` · `URDF / MJCF` · `Computer Vision` · `Depth Anything`
`OpenCV` · `WebSockets` · `TensorBoard` · `NumPy` · `YAML` · `Git` · `Isaac Lab`

---

## 6. Version anglaise (courte)

> **Reinforcement Learning for Humanoid Locomotion — Poppy Robot**
> AI Club, Université Laval (partner: Vooban) — Python, MuJoCo, PyTorch, ROS
> Built a custom Gymnasium/MuJoCo environment for the 25-DoF Poppy humanoid (63-dim observation,
> position-based PD control emulating Dynamixel servos), designed an 8-term reward function and
> domain randomization (friction, ±15 % masses, external pushes) to train robust walking policies
> with PPO/SAC/TD3/A2C across 64 parallel environments. Added monocular depth perception
> (Depth Anything 3 served over WebSockets on GPU, plus MuJoCo ray casting) and a ROS bridge
> publishing joint targets to the physical robot for Sim2Real transfer.

---

## 7. À savoir avant l'entretien (honnêteté / pièges)

- **Ne pas écrire « le robot marche pour de vrai ».** La politique produit une marche *en
  simulation* ; le transfert sur le robot physique est un prototype (le pont ROS existe mais n'a
  pas encore piloté le vrai Poppy). Formule sûre : *« politique de marche obtenue en simulation,
  pipeline Sim2Real en place »*.
- **Pas de métriques chiffrées consignées** (distance parcourue, vitesse, taux de chute). Évite
  d'annoncer un chiffre que tu ne pourrais pas justifier. Si tu en veux un, lance une évaluation
  et note le résultat d'abord.
- **La vision n'est pas encore branchée sur la politique RL.** Elle fonctionne de façon autonome :
  dis « module de perception développé », pas « politique guidée par la vision ».
- **Projet d'équipe.** Sois prêt à préciser ta part. Tes contributions traçables dans git :
  mise en place de l'environnement MuJoCo humanoïde, pipeline d'entraînement PPO et système de
  configuration, import de l'URDF Poppy, campagnes d'entraînement (équilibre puis marche),
  script d'évaluation, configs Isaac Gym.
- **Questions à préparer** : pourquoi PPO plutôt que SAC ici ? Pourquoi du contrôle en position
  et pas en couple ? À quoi sert la domain randomization ? Pourquoi le reward shaping est-il la
  partie difficile ?
