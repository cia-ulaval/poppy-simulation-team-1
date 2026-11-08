# Ce qu'il faut corriger avant de mettre un modèle sur le robot

> **Statut : liste de travail pour une prochaine session.** Rien n'est fait. Pour
> l'instant, on entraîne d'abord un modèle en simulation ; ce document garde la trace
> de tout ce qui empêche de le transférer, et de la façon de le corriger.
> Rédigé le 8 novembre 2026, après la piste 3 (marche de référence Placo,
> `scripts/generate_reference_walk.py` et `scripts/check_reference_walk.py`).

## Vue d'ensemble

| # | Problème | Il faut le robot ? | Change la physique ? |
| --- | --- | :---: | :---: |
| 1 | Conventions des articulations jamais vérifiées sur le robot | oui | non |
| 2 | Bras du MJCF tournés autrement que dans l'URDF | après 1 | **oui** |
| 3 | Masse simulée 2,61 kg, robot annoncé à 3,5 kg | oui (balance) | **oui** |
| 4 | Limites de couple : valeurs 14,8 V, pas de courbe couple-vitesse | oui (lire les moteurs) | **oui** |
| 5 | Moteurs : kp = 8 inventé, inertie de rotor, frottement, retard non mesurés | oui | — |
| 6 | Modèle d'actionneur dans l'environnement | non (après 5) | **oui** |
| 7 | La politique observe des grandeurs que le robot ne mesure pas | non | contrat obs |
| 8 | Randomisation trop pauvre pour le transfert | non | entraînement |
| 9 | Le pont : 5 s par commande, boucle ouverte, sans genou corrigé | non | non |
| 10 | Pas de mesure de l'écart simulation / réel | oui | non |
| 11 | Marche de référence à régénérer après 2 à 6 | non | non |

**Règle de regroupement.** Les tâches 2, 3, 4, 6 et 7 changent la physique ou le
contrat d'observation : chacune invalide les modèles déjà entraînés. On les fait **en un
seul lot** (« MJCF v2 »), décidé en équipe, puis on réentraîne une fois. Poser un tag git
avant, comme proposé dans [`recompense-et-paliers.md`](recompense-et-paliers.md).

**Règle de sécurité.** Tout essai sur le robot suit [`CONTRIBUTING.md`](../CONTRIBUTING.md)
§ Sécurité robot : lot séparé, humain présent, main sur l'alimentation, robot suspendu
tant qu'il n'a pas prouvé qu'il tient.

---

## Tâche 1 — Vérifier les conventions sur le vrai robot

### Objectif

Savoir si un angle envoyé par la simulation donne la même posture sur le robot.

### À savoir

- Le genou gauche du MJCF/URDF plie à l'envers ; `environment.fix_left_knee_axis`
  le corrige en simulation. Sur le vrai robot, le sens n'a jamais été vérifié.
- Les bras du MJCF ne sont pas ceux de l'URDF (tâche 2) : il faut savoir lequel des
  deux le robot suit.
- Les butées de `hip_y` sont volontairement plus serrées en simulation (±68,8°).

### Où regarder

| Fichier | Ce qu'il contient |
| --- | --- |
| `assets/poppy_humanoid/poppy_humanoid.urdf` | URDF officiel de poppy-project |
| `assets/poppy_humanoid/poppy_humanoid.xml` | MJCF simulé |
| `docs/TECHNIQUE.md` §8 | genou gauche, audit de symétrie |

### À faire

1. Robot suspendu, tous les moteurs à 0 : photographier de face et de profil, comparer
   à la simulation à `qpos[7:] = 0` (MJCF) et à l'URDF.
2. Pour chaque articulation : +10° sur le robot et en simulation, noter si le sens
   concorde. Tableau de 25 lignes.
3. Noter les références des moteurs (MX-28, MX-64, autre) et la tension
   d'alimentation mesurée.

### Ne pas faire

Poser le robot au sol. Envoyer une politique.

### Terminé quand

Un tableau articulation par articulation (sens, décalage du zéro) est dans ce dossier.

---

## Tâche 2 — Corriger les bras du MJCF

### Objectif

Que les bras simulés soient placés comme ceux du robot, aux mêmes angles.

### À savoir

- `l_shoulder_motor` et `r_shoulder_motor` (MJCF l. 291 et 338) portent
  `euler="3.14159265 0 ±1.5707963"`. MuJoCo lit ces angles en convention
  intrinsèque ; l'URDF donne les mêmes nombres en roulis-tangage-lacet extrinsèque.
- Mesuré le 8 novembre 2026 :

| Angles | MJCF | URDF |
| --- | --- | --- |
| tous à 0 | bras horizontaux vers l'intérieur, à travers la poitrine (« bras croisés ») | horizontaux vers l'extérieur |
| pose debout de l'env | coude 12 cm derrière et 9 cm sous l'épaule | coude 10 cm devant et 10 cm au-dessus |

- La pose debout (`_STANDING_POSE_OVERRIDES`, `poppy_humanoid_env.py` l. 93) a été
  cherchée pour compenser ce défaut. Envoyée à un robot qui suit l'URDF, elle lèverait
  les bras devant : ~0,6 kg sur 2,6 kg, de quoi déplacer le centre de masse de
  plusieurs centimètres vers l'avant.
- La copie pour Placo (`build_placo_urdf`) recopie déjà la cinématique du MJCF : elle
  suivra automatiquement la correction.

### À faire

1. Selon la tâche 1, corriger les `euler` du MJCF (ou `eulerseq`) pour retrouver
   l'URDF, ou traduire les angles des bras dans le pont.
2. Rechercher une nouvelle pose debout (bras le long du corps), et garder les coudes
   loin de leur butée : ils sont aujourd'hui à 1° de la limite.
3. Ajouter un test qui compare les positions des corps MJCF et URDF à angles égaux.

### Ne pas faire

Corriger seul : c'est un changement de physique (règle de regroupement).

### Terminé quand

MJCF et URDF coïncident (test), la nouvelle pose debout tient à action nulle.

---

## Tâche 3 — Masses et centre de masse

### Objectif

Que la simulation pèse ce que pèse le robot, au même endroit.

### À savoir

Le MJCF fait 2,61 kg ; la documentation annonce 3,5 kg. Les couples dus au poids
seraient ~1,3 fois plus forts que ceux calculés pour la marche de référence.

### À faire

1. Peser le robot entier ; si possible, une jambe démontée.
2. Centre de masse : robot debout en appui sur deux balances (avant / arrière, puis
   gauche / droite), en déduire la position du CoM.
3. Répartir l'écart de masse dans le MJCF (moteurs, câbles, électronique), au plus près
   de la réalité ; vérifier que le CoM simulé tombe au même endroit.

### Terminé quand

Masse totale à ±3 %, CoM à ±1 cm, mesures notées ici.

---

## Tâche 4 — Limites de couple réelles

### Objectif

Que la simulation n'autorise pas plus de couple que les moteurs n'en donnent.

### À savoir

- `ctrlrange` (MJCF l. 388 à 417) vaut 3,1 N·m et 7,3 N·m ; l'URDF donne 7,0 et
  8,2 rad/s. D'après les fiches techniques (à vérifier), ce sont les valeurs des MX-28
  et MX-64 **en 14,8 V**. En 12 V : environ 2,5 N·m et 6 N·m.
- Ce sont des couples de **blocage** : le couple disponible baisse avec la vitesse,
  jusqu'à zéro à la vitesse maximale. La simulation l'ignore. La marche de référence
  demande 2,17 N·m au genou, surtout pendant le vol, quand il tourne vite.
- Un servo qui tient longtemps près de son couple de blocage chauffe.

### À faire

1. Avec la tâche 1 : modèles de moteurs et tension réelle.
2. Dans l'environnement, borner le couple selon la vitesse :
   `τ_max(ω) = τ_blocage · (1 − |ω| / ω_max)`, par type de moteur.
3. Relancer `scripts/check_reference_walk.py` et regarder les marges.

### Terminé quand

Limites par type de moteur dans le MJCF ou la config, courbe couple-vitesse appliquée,
option désactivée par défaut tant que l'équipe n'a pas tranché.

---

## Tâche 5 — Identifier les moteurs

### Objectif

Remplacer le régulateur inventé de la simulation par des valeurs mesurées.

### À savoir

- `_KP = 8.0` et `_KD = 0.5` (`poppy_humanoid_env.py` l. 115) ont été choisis « pour que
  ça marche ». Avec eux, la marche de référence exige 16° d'erreur de suivi au genou, et
  elle tombe en 1,2 s rejouée en boucle ouverte. Un Dynamixel en position est
  probablement bien plus raide.
- Le registre de gain P du Dynamixel ne se convertit pas de façon fiable en N·m/rad :
  on mesure.
- L'essentiel du couple du genou dans la marche de référence vient de l'`armature`
  du MJCF (0,01 kg·m²), jamais mesurée.
- C'est ce qu'ont fait Disney (BDX) et Open Duck : modèle d'actionneur identifié par
  expériences, paramètres randomisés dans la plage observée.

### À faire

1. **Raideur statique.** Robot suspendu, segment à l'horizontale, consigne fixe ; masse
   connue m à distance connue d ; lire l'écart Δθ à l'encodeur. `kp = m·g·d / Δθ`.
   Deux ou trois masses pour vérifier la linéarité.
2. **Réponse dynamique.** Échelons et balayage sinusoïdal sur une articulation chargée ;
   enregistrer consigne et position au plus vite (≥ 100 Hz).
3. **Ajustement.** Rejouer le même essai dans MuJoCo (même masse, même consigne) et
   ajuster kp, kd, `armature`, `frictionloss`, `damping` et le retard par moindres
   carrés (`scipy.optimize.least_squares`) jusqu'à superposer les courbes.
4. Une fois par **type** de moteur, sur deux exemplaires pour voir la dispersion.

### Terminé quand

Paramètres par type de moteur, avec l'erreur d'ajustement et les courbes, notés ici ;
script d'ajustement dans `scripts/`.

---

## Tâche 6 — Modèle d'actionneur dans l'environnement

### Objectif

Que chaque moteur simulé se comporte comme le vrai.

### À faire

Option de configuration, désactivée par défaut, qui remplace `_action_to_torque` :
kp et kd par type de moteur (tâche 5), couple borné selon la vitesse (tâche 4), retard
de commande, `armature` et frottement mesurés. Test : un échelon simulé reproduit la
courbe mesurée.

### Terminé quand

`poppy_robust.yaml` donne toujours la même évaluation du modèle du 19 septembre ; la
nouvelle option reproduit les essais de la tâche 5.

---

## Tâche 7 — N'observer que ce que le robot mesure

### Objectif

Que la politique embarquée ne dépende que de ses capteurs.

### À savoir

| Dans l'observation | Mesurable sur le robot ? |
| --- | --- |
| angles et vitesses des 25 moteurs | oui, encodeurs |
| inclinaison et rotation du bassin | oui, IMU avec gyroscope |
| contacts des pieds (en N) | seulement avec des capteurs de pression |
| hauteur du bassin | **non** |
| vitesse d'avance | **non**, estimation imprécise |
| cap absolu | dérive |
| phase de l'horloge | oui, horloge de bord |

### À faire

1. **Acteur / critique asymétrique** : l'acteur (embarqué) ne reçoit que les lignes
   « oui » ; le critique (entraînement seulement) reçoit tout le simulateur. Open Duck
   et Disney font ainsi. Change le contrat d'observation.
2. Contacts en **tout-ou-rien** (pied posé ou non) plutôt qu'en newtons.
3. Bruit et retard sur les capteurs simulés : biais et bruit de l'IMU, quantification
   des encodeurs, quelques millisecondes de retard.

### Terminé quand

L'acteur n'a plus aucune entrée marquée « non » ; le choix est écrit dans
`docs/ARCHITECTURE.md`.

---

## Tâche 8 — Randomisation pour le transfert

### Objectif

Une politique qui marche sur mille robots légèrement différents, donc sur le vôtre.

### À faire

Ajouter, à chaque épisode : kp et kd (±10 à 20 %), couple maximal (tension de
batterie), retard de commande (0 à 2 pas), jeu d'engrenage (±0,5°), frottement des
articulations, décalage du zéro des encodeurs (±0,02 rad), position du CoM du tronc.
Existe déjà : friction du sol, masses ±15 %, orientation initiale, poussées.
Référence : Open Duck randomise kp ±10 %, masses ±10 %, CoM du tronc ±5 cm, retard 0 à
2 pas.

---

## Tâche 9 — Le pont vers le robot

### À savoir

`src/robot/simulation_adapter.py` envoie ce que la simulation a obtenu, pas ce que la
politique a demandé ; une commande toutes les 5 s (`POPPY_CONTROL_PERIOD_S`) ; rien
n'est relu ; l'environnement y est construit sans `fix_left_knee_axis` (l. 51).

### À faire

1. Boucle fermée à 50–100 Hz : lire encodeurs, IMU et contacts, construire
   l'observation, appeler la politique, envoyer les consignes.
2. Exporter la politique en ONNX, normalisation des observations incluse.
3. Construire l'environnement et la phase exactement comme à l'entraînement (genou,
   horloge).
4. Butées logicielles, limite d'amplitude par pas, chien de garde, arrêt propre.

---

## Tâche 10 — Mesurer l'écart simulation / réel

### Objectif

Un chiffre qui dit si la simulation ressemble au robot, à refaire après chaque
correction.

### À faire

Même suite de consignes (sinus lent sur chaque jambe, robot suspendu), envoyée au robot
et à la simulation ; comparer les trajectoires articulaires (erreur moyenne et
maximale, retard). Puis essais progressifs : debout suspendu, debout au sol avec
harnais, quelques pas avec harnais, marche.

---

## Tâche 11 — Régénérer la marche de référence

### À savoir

`assets/poppy_humanoid/reference_walk.npz` a été produite sur le MJCF actuel. Ses
incertitudes, mesurées le 8 novembre 2026 :

| Point | Valeur | Ce qui le corrige |
| --- | --- | --- |
| Couple max du genou | 2,17 N·m, 70 % de 3,1 (87 % de 2,5 si 12 V) | tâches 4 et 5 |
| Part due à l'`armature` | l'essentiel du couple du genou | tâche 5 |
| Masse | 2,61 kg simulés | tâche 3 |
| Centre de pression | sort de 7 % de la demi-semelle 10 ms par transfert d'appui | accepté |
| Rejeu en boucle ouverte | tombe à 1,2 s | tâches 5 et 6, puis la politique |
| Démarche | accroupie (genoux 35° à 60°), légèrement asymétrique | tâche 2 (pose des bras) |
| Validité | sol plat, `fix_left_knee_axis`, cycle de 0,7 s | — |

### À faire

Après le lot MJCF v2 : `scripts/generate_reference_walk.py` (Placo, voir
`requirements/reference.txt`), puis `scripts/check_reference_walk.py` ; mettre à jour le
tableau ci-dessus et `tests/test_reference_walk.py` le signalera si l'URDF pour Placo
n'est plus à jour.
