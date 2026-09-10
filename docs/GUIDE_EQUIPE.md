# Guide de l'équipe — Poppy Simulation

> **Pour qui ?** Tous les membres de l'équipe, y compris — et surtout — ceux qui n'ont
> jamais fait de robotique ni d'intelligence artificielle. Aucun prérequis n'est
> supposé au-delà de savoir programmer en Python.
>
> **Comment l'utiliser ?** Lis les parties 1 à 4 une fois (~30 min). Ensuite, va
> directement à la fiche de la tâche que tu as prise. Chaque fiche est autonome : elle
> contient le contexte, ce qu'il faut faire, où c'est dans le code, et comment savoir
> que c'est terminé.

---

## Sommaire

1. [Le projet en une page](#1-le-projet-en-une-page)
2. [Le vocabulaire, expliqué simplement](#2-le-vocabulaire-expliqué-simplement)
3. [Démarrer : installation et premiers pas](#3-démarrer--installation-et-premiers-pas)
4. [Comment on travaille ensemble](#4-comment-on-travaille-ensemble)
5. [Les tâches du sprint 1](#5-les-tâches-du-sprint-1--semaines-1-2)
6. [Les tâches du sprint 2](#6-les-tâches-du-sprint-2--semaines-3-4)
7. [La suite : sprints 3 à 6](#7-la-suite--sprints-3-à-6)
8. [Ce qu'on ne fait pas cette session](#8-ce-quon-ne-fait-pas-cette-session)

---

## 1. Le projet en une page

**On apprend à un robot humanoïde à marcher.**

Le robot s'appelle **Poppy**. C'est un humanoïde open source imprimé en 3D : environ
83 cm, 3,5 kg, et **25 moteurs** — un par articulation mobile (hanches, genoux,
chevilles, épaules, coudes, nuque, tronc).

Faire marcher un humanoïde est difficile parce que c'est un système en équilibre
instable : il tombe dès qu'on ne fait rien. Personne ne sait écrire à la main le code
qui coordonne 25 moteurs pour garder l'équilibre en avançant.

**Donc on ne l'écrit pas. On le fait apprendre.**

L'approche :

1. On construit une **copie virtuelle** du robot dans un simulateur physique.
2. On laisse un programme **essayer au hasard**, des millions de fois, en le
   récompensant quand il avance et en le pénalisant quand il tombe.
3. Petit à petit, il découvre tout seul une façon de marcher.
4. On transfère ce qu'il a appris **sur le vrai robot**.

L'étape 4 est la plus difficile, et c'est le cœur de cette session.

### Où on en est

Le projet existe depuis un an. Le socle est solide : la copie virtuelle du robot est
faite, l'environnement d'apprentissage fonctionne, et **on a un modèle qui marche en
simulation**. Le code est propre et bien organisé.

Ce qui manque, ce n'est pas du code — c'est de la **preuve** et du **passage au réel**.
Personne n'a encore fait tourner la politique apprise sur le robot physique.

### L'objectif de cette session (12 semaines)

> **La politique tourne sur le vrai Poppy, suspendu puis en appui au sol, et produit
> des mouvements de marche cohérents.**

Ce n'est volontairement *pas* « Poppy marche tout seul ». Avec le temps dont on
dispose, viser ça mènerait à un échec. L'objectif ci-dessus est ambitieux mais
atteignable, et il laisse la session suivante démarrer sur une base solide.

---

## 2. Le vocabulaire, expliqué simplement

Cette section est le glossaire du projet. Reviens-y quand un mot te bloque.

### Le simulateur

**MuJoCo** — Un moteur physique. Comme le moteur d'un jeu vidéo, mais précis : il
calcule les forces, les contacts et les mouvements. Chez nous, il recalcule l'état du
robot **500 fois par seconde**.

**Le modèle du robot (fichier `.xml`)** — La description de Poppy pour le simulateur :
la forme de chaque pièce, sa masse, comment les pièces s'emboîtent, quels moteurs
agissent où. C'est `assets/poppy_humanoid/poppy_humanoid.xml`.

**Degré de liberté (DoF)** — Un mouvement indépendant possible. Chaque moteur de Poppy
en donne un ; il y en a **25**. S'y ajoutent 6 degrés « libres » qui décrivent la
position et l'orientation du robot dans l'espace, et qu'aucun moteur ne commande
directement — c'est justement pour ça qu'il peut tomber.

**Environnement (`env`)** — Le programme qui fait le lien entre le simulateur et
l'apprentissage. Il sait démarrer un essai, appliquer une action, calculer la
récompense, et dire si le robot est tombé. C'est
`src/environments/poppy_humanoid_env.py`, le fichier le plus important du projet.

### L'apprentissage

**Apprentissage par renforcement (RL)** — La méthode. On ne dit jamais au programme
*comment* marcher. On lui donne seulement une **note** après chaque instant. Il essaie
des choses au hasard, garde ce qui augmente la note, abandonne le reste. Répété des
millions de fois, ça produit un comportement.

**Politique (*policy*)** — Le « cerveau » : un réseau de neurones. Il reçoit des
nombres décrivant la situation, il produit des nombres commandant les moteurs. C'est
ce qu'on entraîne, et c'est le seul fichier qu'on installera sur le vrai robot.

**Observation** — Ce que le cerveau reçoit à chaque instant. Aujourd'hui **63
nombres** : angles des articulations, vitesses, orientation du buste, contacts sous
les pieds.

**Action** — Ce que le cerveau produit : **25 nombres**, un par moteur. Chez nous
c'est un **angle cible** (« mets le genou à 30° »), pas une force. C'est important
pour la suite : les vrais moteurs de Poppy acceptent exactement ce format.

**Récompense (*reward*)** — La note. Chez nous c'est une somme de **8 termes** :
des bonus (avancer, rester debout, rester vertical, alterner les appuis) et des
pénalités (partir de travers, forcer inutilement, bouger de façon saccadée, agiter
les articulations trop vite).

**Pas (*step*)** — Une décision. Le cerveau décide **100 fois par seconde**.

**Épisode** — Un essai complet, du départ jusqu'à la chute ou la fin du temps imparti.

**PPO, SAC, TD3, A2C** — Les recettes d'apprentissage : les algorithmes qui décident
comment modifier le cerveau au vu des notes obtenues. On utilise **PPO** par défaut.

**Stable-Baselines3 (SB3)** — La bibliothèque Python qui fournit ces algorithmes.

**Checkpoint** — Une sauvegarde du cerveau à un instant de l'entraînement. Un fichier
`.zip`.

**`vec_normalize.pkl`** — ⚠️ **À retenir absolument.** Les observations sont mises à
l'échelle avant d'entrer dans le cerveau, et ce fichier contient l'échelle utilisée.
**Un modèle ne fonctionne qu'avec le `vec_normalize.pkl` du même dossier.** Si tu les
mélanges, le modèle reçoit des valeurs aberrantes et paraît complètement cassé. C'est
la cause n°1 des fausses alertes « le modèle ne marche plus ».

**Acteur et critique** — Il y a **deux** réseaux pendant l'entraînement, pas un seul.
L'**acteur** est la politique : il décide des actions. Le **critique** ne décide rien —
il juge les situations (« celle-ci est bonne / mauvaise ») et cette note sert à
corriger l'acteur. **À la fin de l'entraînement, on jette le critique.** Seul l'acteur
part sur le robot. On s'appuiera beaucoup sur ce fait au sprint 2.

### Le passage au réel

**Sim2real** — Le transfert de la simulation vers le vrai robot.

**Écart de réalité (*reality gap*)** — La simulation n'est jamais exacte : frottements,
jeu mécanique, masses réelles, retards de communication. Une politique parfaite en
simulation peut échouer en vrai à cause de ces écarts.

**Domain randomization** — La parade. Au lieu de simuler *un* monde, on en simule des
milliers légèrement différents (sol plus ou moins glissant, masses variées, poussées
aléatoires). La politique apprend alors quelque chose de robuste plutôt qu'un tour
optimisé pour un monde précis.

**Dynamixel** — La marque des moteurs de Poppy. Ils sont « intelligents » : on leur
envoie un angle cible et ils renvoient leur position réelle, leur vitesse, leur charge
et leur température.

**pypot** — La bibliothèque Python qui pilote les Dynamixel.

**Palier (0 à 4)** — Notre progression de mise en service, du plus sûr au plus risqué.
Voir la partie 6. On ne saute jamais un palier.

---

## 3. Démarrer : installation et premiers pas

### Installation

L'environnement de référence est **Docker** : il garantit que le code tourne pareil
chez tout le monde. La procédure exacte est dans le `README.md` à la racine.

> **Deux chemins assumés.** Docker sert à **entraîner** (sans affichage). Pour
> **visualiser** le robot en 3D, utilise une installation locale : l'affichage
> graphique depuis un conteneur est un nid à problèmes qu'on a décidé de ne pas
> ouvrir. Si tu perds plus d'une heure là-dessus, arrête-toi et demande.

### La carte du dépôt

| Chemin | Contenu |
|---|---|
| `assets/poppy_humanoid/` | Le modèle 3D du robot (XML + pièces) |
| `src/environments/poppy_humanoid_env.py` | **Le fichier central** : observation, action, récompense |
| `src/algorithms/` | Les algorithmes (PPO, SAC, TD3, A2C) |
| `src/training/` | La boucle d'entraînement |
| `src/evaluation/` | L'évaluation des modèles |
| `src/sensors/` | La perception par profondeur (pas utilisée cette session) |
| `src/robot/` | Le pont vers le vrai robot (ROS, pypot) |
| `scripts/` | Les points d'entrée : entraîner, visualiser, piloter le robot |
| `configs/` | Les réglages d'entraînement en YAML |
| `logs/poppy/<date>/` | Les résultats d'un entraînement passé |
| `docs/` | Ce guide et les autres documents |

### Les modèles déjà entraînés

Ils sont rangés par date d'entraînement :

```
logs/poppy/2026-04-08_20-41-01/best_model/
    ├── best_model.zip      ← le cerveau
    └── vec_normalize.pkl   ← son échelle, indissociable
```

**Prends toujours les deux fichiers du même dossier.**

### Ta première demi-heure

1. Installer, puis lancer la visualisation d'un modèle existant.
2. Regarder Poppy marcher.
3. Ouvrir `src/environments/poppy_humanoid_env.py` et lire la fonction `step()` : c'est
   là que tout se passe, une fois par centième de seconde.

---

## 4. Comment on travaille ensemble

### Deux équipes

| Équipe | Effectif | Domaine |
|---|---|---|
| **SIM** | 3 personnes | Tout ce qui tourne sur ordinateur : simulation, apprentissage, mesure |
| **ROBOT** | 2 personnes | Le robot physique : moteurs, sécurité, communication, paliers |

Une personne bascule de SIM vers ROBOT au sprint 4, quand le robot devient le gros du
travail.

### Les rituels

- **Un point de 30 minutes par semaine**, même créneau, non négociable.
- **Cinq lignes écrites à la fin de chaque séance** : ce qui a été fait, ce qui est
  cassé, quoi faire ensuite.

Ces cinq lignes sont la chose la plus importante de cette section. Beaucoup d'entre
nous manqueront des séances — c'est normal pour du bénévolat. Quelqu'un qui revient
après trois semaines lit six comptes-rendus et il est à jour, **sans réunion de
rattrapage**.

### Les cinq règles

1. **Aucune tâche n'a un propriétaire unique.** Deux personnes savent faire chaque
   chose critique. Une absence ne bloque alors rien.
2. **Aucun ticket ne dépasse 3 heures**, soit une séance. Une tâche qui déborde sera
   reprise à froid, et la remise en contexte mangera la séance suivante.
3. **Une réserve de tickets indépendants existe en permanence.** Si tu arrives et que
   ton binôme est absent, tu dois avoir quelque chose d'utile à faire tout de suite.
4. **Le jalon prime sur le planning.** Si le jalon d'un sprint n'est pas atteint, on ne
   démarre pas le suivant — on réduit le périmètre.
5. **En cas de doute, demande.** Trois heures perdues seul valent moins qu'un message
   de deux minutes. Personne ici n'est censé déjà savoir.

### Git

Une branche par ticket, nommée d'après lui : `s1-sim-2-callback-reward`. Une *pull
request* quand c'est prêt, relue par une personne de l'autre binôme. On ne pousse
jamais directement sur `main`.

### Comment lire une fiche de tâche

Chaque fiche suit le même format :

- **Difficulté** — 🟢 accessible sans prérequis · 🟡 demande de lire un peu de doc ·
  🔴 techniquement délicat, à faire en binôme
- **Contexte** — pourquoi cette tâche existe et à quoi elle sert
- **À faire** — les étapes concrètes
- **Où** — les fichiers concernés
- **Terminé quand** — le critère vérifiable, sans ambiguïté

---

## 5. Les tâches du sprint 1 — semaines 1-2

> **Objectif du sprint :** transformer des impressions en chiffres, et rendre le robot
> inoffensif.
>
> ⚠️ **Aucun entraînement long ne démarre ce sprint.** L'interface du cerveau va
> changer au sprint 2, et tout ce qui serait entraîné maintenant deviendrait
> inutilisable. C'est contre-intuitif quand on a envie de « faire de l'IA », mais
> lancer un entraînement de 10 millions de pas cette semaine, c'est jeter une semaine
> de calcul.

### 🟢 S1-SIM-1 — Reproduire la mesure de référence

**Difficulté :** 🟢 · **Temps :** 3 h · **Prérequis :** aucun

**Contexte.** Le projet contient des dizaines de modèles entraînés, et **aucun document
ne dit ce qu'ils valent**. On ne sait pas sur quelle distance Poppy marche, ni combien
de temps il tient. Sans ce chiffre de départ, on ne pourra jamais dire si nos
modifications améliorent ou dégradent quoi que ce soit.

C'est aussi ton parcours d'initiation : en faisant cette tâche, tu vérifies que ton
installation fonctionne et tu découvres la codebase en t'en servant.

**À faire.**
1. Repérer le dossier de `logs/poppy/` qui contient le meilleur modèle. Demande au
   lead lequel — sinon, prends le plus récent.
2. Charger le modèle **et son `vec_normalize.pkl`** (même dossier — relis le glossaire
   si tu ne sais plus pourquoi).
3. Lancer **20 épisodes** en mode déterministe.
4. Relever cinq chiffres : distance parcourue en x (moyenne et écart-type), vitesse
   moyenne, durée d'épisode (en pas **et** en secondes), taux de chute (pourcentage
   d'épisodes terminés par une chute plutôt que par la fin du temps).
5. Enregistrer une vidéo d'un épisode.
6. Écrire tout ça dans `docs/BASELINE.md`.

**Où.** `src/evaluation/`, `scripts/`, et le nouveau `docs/BASELINE.md`.

**Attention.** Un épisode dure 1000 pas, soit seulement **10 secondes** de temps
simulé (100 décisions par seconde). Ne confonds pas les deux unités quand tu écris tes
résultats.

**Terminé quand.** `docs/BASELINE.md` existe, contient les cinq chiffres, et une autre
personne de l'équipe obtient les mêmes valeurs de son côté.

---

### 🟡 S1-SIM-2 — Journaliser les 8 termes de la récompense

**Difficulté :** 🟡 · **Temps :** 6 h · **Prérequis :** avoir fait tourner un
entraînement une fois

**Contexte — le plus important du sprint.** La récompense est une somme de 8 termes.
Quand un entraînement se passe mal, la courbe de la récompense *totale* ne dit rien :
elle monte, mais on ignore quel terme monte.

Le mode d'échec classique de l'humanoïde : le robot apprend à **rester planté debout
sans avancer**. Les termes « rester en vie » et « rester vertical » rapportent
tranquillement, le terme « avancer » est abandonné, et **la récompense totale monte
quand même**. Sans décomposition, le problème est invisible pendant des semaines.

Bonne nouvelle : les 8 valeurs sont **déjà calculées** à chaque pas et rangées dans un
dictionnaire nommé `info`
([`poppy_humanoid_env.py`, fonction `step()`](../src/environments/poppy_humanoid_env.py)).
Les clés existantes sont :

```
forward_vel · capped_vel · lateral_vel · uprightness · healthy_reward
gait_reward · ctrl_cost · action_rate_cost · joint_vel_cost
```

Le problème : **Stable-Baselines3 ne les enregistre pas**. Elles sont calculées puis
jetées à chaque pas.

**À faire.**
1. Écrire un *callback* SB3 qui récupère ces clés à chaque pas et les moyenne.
2. Les écrire dans TensorBoard, une courbe par terme.
3. Le brancher dans la boucle d'entraînement.
4. Lancer un entraînement court et vérifier que les 9 courbes apparaissent.
5. Documenter en trois lignes dans le README comment les regarder.

**Où.** Un nouveau fichier dans `src/training/`, branché depuis `scripts/train_poppy.py`.

**Piste.** Cherche `BaseCallback` dans la documentation de Stable-Baselines3. Les
valeurs `info` sont accessibles via `self.locals["infos"]`, qui est une liste — une
entrée par environnement parallèle, et il y en a 64.

**Terminé quand.** Un entraînement produit 9 courbes séparées dans TensorBoard, et
quelqu'un d'autre sait les ouvrir en suivant ta doc.

---

### 🟡 S1-SIM-3 — Corriger le bug de rebond du sol

**Difficulté :** 🟡 · **Temps :** 2 h · **Prérequis :** aucun — bon premier ticket
pour découvrir MuJoCo

**Contexte.** À chaque nouvel essai, on tire au hasard les propriétés du sol pour que
la politique s'adapte à des surfaces variées. On tire deux choses : le **frottement**
(correct) et le **rebond** (cassé).

Dans MuJoCo, chaque surface a un vecteur `solimp` de **cinq** valeurs, qui décrit
comment la résistance du contact augmente quand deux objets s'interpénètrent :

```
solimp = (dmin, dmax, width, midpoint, power)
           [0]   [1]    [2]      [3]     [4]
```

La case `[4]` est **`power`** : l'exposant de cette courbe de raideur. Sa valeur par
défaut est **2**, et elle est censée rester ≥ 1.

Or le code écrit le « rebond » — un nombre entre 0 et 0,5 — **dans cette case**
([`_randomize_floor()`](../src/environments/poppy_humanoid_env.py)) :

```python
restitution = self.np_random.uniform(0.0, 0.5)
self.model.geom_solimp[floor_id, 4] = restitution   # ← écrit dans « power »
```

Deux problèmes cumulés : `power` n'a rien à voir avec le rebond, et la plage 0–0,5 est
hors du domaine prévu. **On randomise donc un paramètre de solveur au hasard en
croyant faire varier l'élasticité du sol.**

Et ça se propage : la fonction `get_floor_randomization_info()` relit la même case et
la journalise sous le nom `floor_restitution`. Les logs affichent une « restitution »
qui n'en a jamais été une.

Le vrai réglage du rebond est **`solref`**, un vecteur de deux valeurs :

```
solref = (timeconst, dampratio)
            [0]         [1]
```

`dampratio = 1` donne un contact amorti sans rebond ; en dessous de 1, ça rebondit.

**À faire.**
1. Remplacer l'écriture dans `geom_solimp[floor_id, 4]` par une écriture dans
   `geom_solref[floor_id, 1]`.
2. Adapter la plage de tirage : quelque chose comme 0,7 à 1,0 (à justifier).
3. Corriger aussi `get_floor_randomization_info()`, qui relit la mauvaise case.
4. Renommer le paramètre dans `configs/poppy_robust.yaml` si le nom devient trompeur.
5. Vérifier visuellement : avec un `dampratio` bas, une balle lâchée sur le sol doit
   rebondir davantage.

**Où.** `src/environments/poppy_humanoid_env.py`, fonctions `_randomize_floor()` et
`get_floor_randomization_info()` ; `configs/poppy_robust.yaml`.

**Lecture.** La page « Contact parameters » de la documentation MuJoCo — 15 minutes,
et tu comprendras `solref` et `solimp` mieux que la plupart des gens qui utilisent
l'outil.

**Terminé quand.** Le rebond est réellement piloté, les logs affichent la bonne
grandeur, et ta *pull request* explique en trois phrases ce qui était faux.

---

### 🟢 S1-SIM-4 — Écrire les trois premiers tests

**Difficulté :** 🟢 · **Temps :** 4 h · **Prérequis :** aucun

**Contexte.** Le projet n'a **aucun test**. Rien n'empêche quelqu'un de casser
l'environnement sans s'en apercevoir — et au sprint 2, on va justement le modifier en
profondeur. Ces trois tests sont le filet de sécurité de tout le reste de la session.

On n'en écrit que trois : mieux vaut trois tests qui tournent qu'une suite ambitieuse
jamais terminée.

**À faire.**
1. `test_env_creation` — l'environnement se crée, et son espace d'observation annonce
   bien 63 dimensions et son espace d'action 25.
2. `test_step_shapes` — après une action au hasard, l'observation renvoyée a la bonne
   taille et la récompense est un nombre fini (ni `NaN`, ni infini).
3. `test_reset_deterministic` — deux `reset()` avec la même graine donnent exactement
   le même état de départ.
4. Documenter la commande pour les lancer dans le README.

**Où.** Un nouveau dossier `tests/`, avec `pytest`.

**Pourquoi le troisième compte.** Sans reproductibilité, on ne peut pas comparer deux
entraînements : on ne saurait jamais si une différence vient de notre modification ou
du hasard.

**Terminé quand.** `pytest` passe au vert, et la commande est dans le README.

---

### 🟢 S1-SIM-5 — Trancher la question de l'axe d'avance

**Difficulté :** 🟢 · **Temps :** 2 h · **Prérequis :** aucun

**Contexte.** L'historique du projet contient six commits successifs intitulés
`reverse x` et `inverse x & y`. Quelqu'un s'est manifestement battu avec le sens de
l'axe d'avance — et **n'a jamais écrit la conclusion**. Résultat : personne ne sait
aujourd'hui si « avancer » veut dire x positif ou x négatif, et le prochain qui
touchera à la récompense refera la même bataille.

Deux heures de travail mettent fin à des mois de confusion.

**À faire.**
1. Charger un modèle entraîné, le faire tourner, et relever la position x au fil de
   l'épisode.
2. Vérifier visuellement dans quelle direction le robot avance, et si c'est bien la
   direction que la récompense encourage.
3. Vérifier la cohérence entre `forward_vel` dans `step()` et l'orientation initiale
   du robot.
4. **L'écrire** : une section courte et sans ambiguïté dans le README, avec un schéma
   si nécessaire.

**Où.** `src/environments/poppy_humanoid_env.py` (fonction `step()`), et le README.

**Terminé quand.** Le README répond en une phrase à « dans quelle direction Poppy
est-il censé avancer, et comment le vérifier ».

---

### 🟢 S1-ROB-1 — Inventorier les moteurs

**Difficulté :** 🟢 · **Temps :** 2 h · **Prérequis :** accès au robot

**Contexte.** Le robot a passé plusieurs mois sans servir. Avant tout, il faut savoir
ce qui répond encore. Un moteur mort ou mal adressé découvert maintenant coûte deux
heures ; découvert au sprint 4, il coûte une séance entière de confusion.

**À faire.**
1. Brancher, alimenter, et scanner le bus avec pypot (`DxlIO.scan()`).
2. Vérifier que les **25 moteurs** répondent. Noter les identifiants trouvés.
3. Relever pour chacun : modèle (MX-28, MX-64 ou AX-12), température, tension.
4. Noter le débit du bus (*baudrate*) actuellement configuré.
5. Écrire l'inventaire dans `docs/ROBOT.md`.

**Attention.** Il ne s'agit encore de faire bouger **aucun** moteur. On lit, c'est
tout.

**Terminé quand.** `docs/ROBOT.md` liste les 25 moteurs, leur état, et signale toute
anomalie.

---

### 🔴 S1-ROB-2 — L'arrêt d'urgence

**Difficulté :** 🔴 · **Temps :** 2 h · **Prérequis :** S1-ROB-1 · **À faire en
binôme**

**Contexte.** **Ce ticket passe avant tout mouvement du robot.** On ne fait pas bouger
une machine qu'on ne sait pas arrêter. Un humanoïde de 3,5 kg avec 25 moteurs sous
tension peut se blesser lui-même — plier une pièce imprimée, forcer sur une butée,
faire chauffer un servo jusqu'à la panne.

**À faire.**
1. **Arrêt logiciel** : une fonction qui met tous les moteurs en mode relâché
   (`compliant = True`), appelable à tout instant, y compris pendant qu'un script
   tourne.
2. **Arrêt physique** : identifier l'interrupteur d'alimentation et s'assurer qu'il
   est toujours à portée de main pendant les essais.
3. **Filet automatique** : une surveillance qui relâche tout si la température dépasse
   un seuil ou si la charge d'un moteur reste anormalement haute.
4. Écrire la procédure dans `docs/ROBOT.md` : que faire, dans quel ordre, si ça se
   passe mal.
5. **Tester réellement** l'arrêt, robot suspendu, avant de valider le ticket.

**Où.** Nouveau fichier dans `src/robot/`.

**Règle d'équipe.** À partir de maintenant, **personne ne fait bouger le robot sans
qu'une deuxième personne soit présente**, la main près de l'interrupteur.

**Terminé quand.** L'arrêt logiciel et l'arrêt physique ont été testés en conditions
réelles, et la procédure est écrite.

---

### 🟡 S1-ROB-3 — Palier 0 : les premiers mouvements

**Difficulté :** 🟡 · **Temps :** 3 h · **Prérequis :** S1-ROB-1, **S1-ROB-2**

**Contexte.** Premier mouvement du projet sur du matériel réel. On ne valide qu'une
chose : **la communication fonctionne**. Aucune intelligence, aucune politique — juste
des angles écrits à la main.

Le robot est **suspendu**, pieds dans le vide. Il ne peut ni tomber ni se blesser.

**À faire.**
1. Suspendre le robot au portique.
2. Écrire un script qui bouge **une seule** articulation, lentement, sur une petite
   amplitude.
3. L'étendre à un mouvement de tous les moteurs vers une pose de repos.
4. Vérifier que les positions atteintes correspondent aux positions demandées.
5. Filmer.

**Où.** `scripts/`, en s'appuyant sur `src/robot/`.

**Terminé quand.** Le robot suspendu rejoint une pose de repos de façon fluide et
répétable, et l'arrêt d'urgence l'interrompt correctement en cours de mouvement.

---

### 🟡 S1-ROB-4 — L'enregistreur de session

**Difficulté :** 🟡 · **Temps :** 3 h · **Prérequis :** S1-ROB-1

**Contexte.** Sans enregistrement, toute la mise au point des trois mois suivants se
fera à l'œil nu — « on dirait qu'il tremble », « ça a l'air mieux ». Avec, elle se fera
avec des chiffres.

Ces données serviront à trois choses : régler les gains du contrôleur, mesurer la
fréquence réellement tenue, et **rejouer en simulation les commandes envoyées au vrai
robot** pour comparer les deux mondes.

**À faire.**
1. Écrire un enregistreur qui, à chaque pas, note dans un CSV : horodatage précis,
   position **commandée**, position **réelle**, vitesse, charge, température — pour
   les 25 moteurs.
2. Le rendre activable depuis n'importe quel script de pilotage.
3. Vérifier que l'enregistrement ne ralentit pas la boucle de contrôle (écrire en
   mémoire, sauvegarder à la fin).
4. Faire un petit script qui trace commandé vs réel pour une articulation.

**Où.** Nouveau fichier dans `src/robot/`.

**Terminé quand.** Un essai de palier 0 produit un CSV exploitable, et le graphe
commandé/réel s'affiche.

---

### 🔴 S1-ROB-5 — Mesurer la fréquence réelle

**Difficulté :** 🔴 · **Temps :** 2 h · **Prérequis :** S1-ROB-1

**Contexte — le livrable bloquant du sprint.** En simulation, le cerveau décide **100
fois par seconde**. Sur le vrai robot, cette cadence dépend de la vitesse à laquelle
on peut parler aux 25 moteurs sur un bus série — et c'est probablement plus lent.

**Si l'écart existe, la politique ne pourra pas être exécutée telle quelle.** Toute la
décision du sprint 2 dépend de ce chiffre. C'est pour ça que ce ticket est prioritaire.

**À faire.**
1. Boucler 1000 fois sur la **lecture** des positions des 25 moteurs. Chronométrer,
   en déduire la fréquence.
2. Recommencer en **lecture + écriture** — c'est ce chiffre-là qui compte, parce que
   c'est ce que fera la vraie boucle de contrôle.
3. Relever aussi la régularité : est-ce stable, ou y a-t-il des à-coups ?
4. **Tester un débit de bus plus élevé.** Le réglage par défaut est souvent 57 600
   bauds alors que les moteurs acceptent bien plus. Ça peut doubler la fréquence à lui
   seul — et donc changer la décision du sprint 2.
5. Écrire les résultats dans `docs/ROBOT.md`.

**Attention.** Changer le débit se fait moteur par moteur et doit être fait sur les
25, sinon certains deviennent injoignables. Lis la doc pypot avant, et fais-le en
binôme.

**Terminé quand.** `docs/ROBOT.md` annonce une fréquence en hertz, avec la méthode de
mesure et le débit utilisé.

---

## 6. Les tâches du sprint 2 — semaines 3-4

> **Objectif du sprint :** décider ce que la politique reçoit, ce qu'elle produit, et à
> quel rythme — puis ne plus y toucher. **C'est le sprint le plus important de la
> session.**

### Pourquoi ce sprint existe

Voici le problème central du projet, et il vaut la peine d'être compris par tout le
monde.

Notre politique reçoit 63 nombres. On les a croisés avec ce que le vrai Poppy peut
mesurer :

| Contenu de l'observation | Dimensions | Mesurable en vrai ? |
|---|---|---|
| Positions des 25 articulations | 25 | ✅ les moteurs les renvoient |
| Vitesses des 25 articulations | 25 | ✅ les moteurs les renvoient |
| Hauteur du bassin | 1 | ❌ |
| **Orientation du buste** | **4** | ❌ pas de centrale inertielle |
| **Vitesse du bassin** | **3** | ❌ |
| **Rotation du bassin** | **3** | ❌ |
| **Appui sous les pieds** | **2** | ❌ pas de capteurs de force |

**13 dimensions sur 63 n'existent pas sur le robot réel — et ce sont exactement celles
qui portent l'équilibre.**

Notre modèle marche en simulation *parce qu'il connaît son orientation et sa vitesse
absolues*. Sur le robot, on ne peut littéralement pas les lui fournir.

**La solution n'est pas de les supprimer** — ce sont les plus utiles pour apprendre.
La solution vient du fait qu'il y a **deux réseaux** :

- L'**acteur** décide des actions. C'est lui qui partira sur le robot. Il ne doit donc
  voir que ce qui est mesurable.
- Le **critique** ne décide rien : il juge les situations pour corriger l'acteur, et
  **on le jette à la fin de l'entraînement**. Rien ne l'empêche de tout voir.

> **L'image :** un athlète qui doit concourir les yeux bandés. C'est la contrainte, on
> ne peut pas y échapper. Mais rien n'oblige son **entraîneur** à porter un bandeau
> aussi — et le jour de la compétition, seul l'athlète entre sur le terrain.

C'est ce qu'on appelle un **acteur-critique asymétrique**, et c'est ce que le sprint 2
met en place.

---

### 🟡 S2-SIM-1 — Séparer l'observation de l'acteur et du critique

**Difficulté :** 🟡 · **Temps :** 5 h · **Prérequis :** avoir lu l'encadré ci-dessus

**Contexte.** Aujourd'hui, `_get_obs()` renvoie un seul bloc de 63 nombres. On veut
qu'elle renvoie **deux blocs** : un pour l'acteur (seulement le mesurable) et un pour
le critique (tout).

Pour ça, il faut savoir ce qu'il y a à quel endroit dans les données MuJoCo :

| Tranche | Contenu | Mesurable ? |
|---|---|---|
| `qpos[0:3]` | position x, y, z du bassin | ❌ |
| `qpos[3:7]` | orientation du buste (quaternion) | ❌ |
| **`qpos[7:]`** | **25 positions articulaires** | ✅ |
| `qvel[0:3]` | vitesse du bassin | ❌ |
| `qvel[3:6]` | rotation du bassin | ❌ |
| **`qvel[6:]`** | **25 vitesses articulaires** | ✅ |

Les deux lignes en gras sont exactement ce que les Dynamixel renvoient.

L'acteur étant privé de son orientation, on lui donne en compensation **la mémoire des
instants précédents** : avec les dernières positions et les dernières actions
envoyées, il peut *déduire* qu'il bascule — comme toi les yeux fermés, tu sens que tu
perds l'équilibre sans avoir besoin de te voir.

**À faire.**
1. Ajouter un historique (`collections.deque(maxlen=3)`) rempli au `reset()` avec la
   première trame répétée trois fois.
2. Réécrire `_get_obs()` pour renvoyer un dictionnaire :

```python
def _get_obs(self):
    # --- privilégié : inchangé, pour le critique ---
    qpos = self.data.qpos[2:].copy()             # 30
    qvel = self.data.qvel.copy()                 # 31
    foot_contacts = self._get_foot_contacts()    # 2
    critic_obs = np.concatenate([qpos, qvel, foot_contacts])   # 63

    # --- déployable : pour l'acteur ---
    joint_pos = self.data.qpos[7:].copy()        # 25
    joint_vel = self.data.qvel[6:].copy()        # 25
    prev = self._prev_action if self._prev_action is not None else np.zeros(25)
    frame = np.concatenate([joint_pos, joint_vel, prev])       # 75

    self._history.append(frame)

    return {
        "actor":  np.concatenate(list(self._history)),   # 225
        "critic": critic_obs,                            # 63
    }
```

3. Changer `observation_space` en `gym.spaces.Dict`.
4. Adapter les tests de S1-SIM-4.

**Bonus important.** Mettre `prev_action` dans l'observation corrige au passage un
second bug : la récompense pénalise depuis toujours l'écart entre l'action courante et
la précédente, **sans que la politique puisse observer cette dernière**. On la
pénalisait donc sur une information qu'elle n'avait pas.

**Où.** `src/environments/poppy_humanoid_env.py`.

**Piège.** Si tu oublies de remplir l'historique au `reset()`, le premier pas de chaque
épisode plantera sur une taille incorrecte.

**Terminé quand.** L'environnement renvoie le dictionnaire, les tests passent, et un
`reset()` suivi de trois `step()` fonctionne sans erreur.

---

### 🔴 S2-SIM-2 — La politique asymétrique dans Stable-Baselines3

**Difficulté :** 🔴 — **le code le plus délicat de la session** · **Temps :** 8 h ·
**Prérequis :** S2-SIM-1 · **Obligatoirement en binôme**

**Contexte.** SB3 suppose que l'acteur et le critique lisent la même observation. Il
faut lui apprendre le contraire : envoyer `obs["actor"]` au réseau de politique et
`obs["critic"]` au réseau de valeur.

**À faire.**
1. Sous-classer `ActorCriticPolicy` et redéfinir l'extraction des caractéristiques
   pour router les deux clés vers les bons réseaux.
2. Brancher cette politique dans la configuration d'entraînement.
3. Vérifier les tailles : l'acteur doit voir 225 entrées, le critique 63.
4. Lancer un entraînement très court et vérifier que **la perte du critique diminue** —
   c'est le signe que le câblage est bon.

**Où.** Nouveau fichier dans `src/algorithms/`.

**Avertissements.**
- L'API de SB3 a changé entre les versions 1.x et 2.x. **Un exemple trouvé sur un
  forum ne fonctionnera probablement pas tel quel** — vérifie toujours contre la
  version installée dans notre Docker.
- Prévois que ce ticket déborde sur le sprint 3. Ce n'est pas un échec, c'est attendu.

**Terminé quand.** Un entraînement de 500 000 pas tourne sans erreur et sa courbe de
récompense monte. On ne cherche pas la performance ici — on cherche la preuve que
l'apprentissage fonctionne encore.

---

### 🟡 S2-SIM-3 — Ajuster la fréquence de contrôle

**Difficulté :** 🟡 · **Temps :** 3 h · **Prérequis :** **S1-ROB-5** (le chiffre du
robot)

**Contexte.** Aujourd'hui le cerveau décide 100 fois par seconde :

```
timestep (dans le XML) × frame_skip = 0.002 × 5 = 0.01 s → 100 Hz
```

Si l'équipe robot mesure 50 Hz, il faut aligner la simulation :

```
0.002 × 10 = 0.02 s → 50 Hz
```

**On change `frame_skip`, jamais `timestep`** : ce dernier gouverne la précision du
calcul physique.

**À faire.**
1. Récupérer la fréquence mesurée auprès de l'équipe robot.
2. Adapter `frame_skip` dans `configs/poppy_robust.yaml`.
3. Vérifier la durée d'épisode : 1000 pas passent de 10 à 20 secondes simulées. Est-ce
   toujours pertinent ?
4. **Signaler le déséquilibre de récompense au ticket S2-SIM-4** (voir ci-dessous).
5. Documenter le choix dans `docs/INTERFACE.md`.

**Le piège à ne pas rater.** Le bonus « rester en vie » est versé **par pas**. À 50 Hz,
le robot en reçoit deux fois moins par seconde, alors que la distance parcourue par
seconde, elle, ne change pas. **Le rapport entre « rester debout » et « avancer »
bascule tout seul, sans que personne n'ait touché aux coefficients.** Si on ne le
corrige pas, on peut passer un sprint entier à se demander pourquoi le robot n'avance
plus.

**Terminé quand.** La fréquence de simulation correspond à celle du robot, et le choix
est écrit avec sa justification.

---

### 🟡 S2-SIM-4 — Rééquilibrer les poids de la récompense

**Difficulté :** 🟡 · **Temps :** 3 h · **Prérequis :** S2-SIM-3, S1-SIM-2

**Contexte.** Conséquence directe du ticket précédent : si la fréquence change, les
poids relatifs des 8 termes changent aussi. Il faut les recalculer pour que l'équilibre
voulu soit préservé.

**À faire.**
1. Lister les termes versés **par pas** (bonus de survie, verticalité, alternance des
   appuis) et ceux qui dépendent du **temps réel** (vitesse d'avance).
2. Recalculer les coefficients pour retrouver l'équilibre d'origine **par seconde**.
3. Vérifier avec le tableau de bord de S1-SIM-2 que la répartition ressemble à celle
   d'avant.
4. Documenter dans `docs/INTERFACE.md`.

**Terminé quand.** Sur un entraînement court, la contribution relative de chaque terme
est comparable à celle mesurée avant le changement de fréquence.

---

### 🟡 S2-ROB-1 — Palier 1 : le robot marionnette

**Difficulté :** 🟡 · **Temps :** 5 h · **Prérequis :** S1-ROB-3, S1-ROB-4

**Contexte.** Premier lien entre la politique et le vrai robot. **La politique tourne
en simulation** et lit les observations **de la simulation** ; le robot se contente de
recopier les positions produites. Il est une marionnette.

C'est pour ça qu'on peut utiliser **le modèle actuel**, celui qui existe déjà, sans
attendre le nouveau : puisque rien ne lit les vrais capteurs, le fait que la politique
ait besoin de 13 dimensions non mesurables n'a aucune importance ici — la simulation
les lui fournit.

Ce qu'on valide n'a rien à voir avec l'intelligence de la politique. On valide **le
tuyau**.

**À faire.**
1. Robot suspendu. Lancer la politique existante en simulation.
2. Envoyer les positions produites aux moteurs en temps réel.
3. Enregistrer avec l'outil de S1-ROB-4.
4. Vérifier : la fréquence est-elle tenue ? Les commandes arrivent-elles dans l'ordre ?
   Aucune valeur aberrante ne passe-t-elle ?
5. Mettre en place un garde-fou qui refuse toute commande hors des limites
   articulaires.

**Où.** `scripts/run_robot.py` et `src/robot/`.

**Terminé quand.** Le robot suspendu reproduit les mouvements de la simulation pendant
60 secondes sans commande aberrante ni décrochage.

---

### 🟡 S2-ROB-2 — Mesurer la souplesse des servos

**Difficulté :** 🟡 · **Temps :** 3 h · **Prérequis :** S2-ROB-1

**Contexte.** En simulation, quand on demande 30° à une articulation, un contrôleur
interne calcule la force nécessaire avec deux réglages appelés `kp` et `kd`. Sur le
vrai robot, ce travail est fait par l'électronique du Dynamixel, avec ses propres
réglages.

**Si les deux ne se ressemblent pas, le robot réel ne bougera pas comme en
simulation** — même avec une politique parfaite. Il faut donc mesurer le comportement
réel pour pouvoir accorder la simulation dessus.

**À faire.**
1. Robot suspendu, envoyer un échelon : demander brusquement +10° à une articulation.
2. Enregistrer la position réelle au fil du temps.
3. Mesurer le temps de réponse, le dépassement, et l'erreur résiduelle.
4. Répéter sur trois articulations représentatives (hanche, genou, cheville) et à deux
   charges différentes.
5. Écrire les résultats dans `docs/ROBOT.md` et les transmettre à l'équipe SIM.

**Terminé quand.** Les courbes de réponse sont enregistrées et l'équipe SIM dispose
des chiffres pour ajuster `kp` et `kd`.

---

### 🟢 S2-ROB-3 — Nettoyer le pont ROS

**Difficulté :** 🟢 · **Temps :** 3 h · **Prérequis :** aucun

**Contexte.** Le pont vers le robot existe mais il a été écrit pour une démonstration,
pas pour un usage réel. Deux défauts le rendent inutilisable en l'état :

- une **adresse IP écrite en dur** dans le code (`10.242.180.129`) — elle ne
  correspondra à rien chez nous ;
- un **`sleep(5)` en pleine boucle de contrôle** — cinq secondes d'attente entre deux
  pas, alors qu'on en voudrait cent par seconde.

**À faire.**
1. Sortir l'adresse IP et le port dans un fichier de configuration.
2. Supprimer le `sleep(5)` et le remplacer par une vraie cadence régulée.
3. Ajouter une reconnexion propre en cas de coupure.
4. Gérer les erreurs : que se passe-t-il si un moteur ne répond plus ?

**Où.** `src/robot/ros_publisher.py`, `src/robot/SimulationAdaptater.py`.

**Terminé quand.** Le pont se configure sans modifier le code et tient la cadence
mesurée à S1-ROB-5.

---

### 🔴 S2-ROB-4 — Les limites de sécurité

**Difficulté :** 🔴 · **Temps :** 3 h · **Prérequis :** S1-ROB-2 · **En binôme**

**Contexte.** Au sprint 4, la politique pilotera le robot en continu. Une politique mal
entraînée peut demander un mouvement violent. Il faut une couche de protection **entre
la politique et les moteurs**, qui ne dépende jamais de la qualité de l'apprentissage.

**À faire.**
1. Bornes articulaires : refuser toute commande hors des limites physiques.
2. Limite de vitesse : refuser un saut trop grand entre deux commandes successives.
3. Surveillance de température : relâcher tout au-delà d'un seuil.
4. Surveillance de charge : détecter un moteur qui force contre un obstacle.
5. Journaliser chaque déclenchement — c'est une information de diagnostic précieuse.

**Où.** Nouveau fichier dans `src/robot/`, traversé par toutes les commandes.

**Terminé quand.** Une commande volontairement aberrante est bloquée et journalisée,
sans que le robot ne bouge.

---

## 7. La suite : sprints 3 à 6

Ces sprints seront détaillés **au jalon du sprint 2**, quand on connaîtra trois
chiffres qu'on n'a pas encore : la fréquence réelle du robot, la perte de performance
due au changement d'observation, et l'accès effectif au matériel. Les détailler
maintenant serait de la fausse précision.

**Sprint 3 — Ça réapprend (semaines 5-6).** Réentraîner sur la nouvelle observation,
avec **au moins trois graines aléatoires** — un seul entraînement ne prouve rien, le
RL est bruité. Côté robot : palier 2, la politique lit les vrais capteurs, robot
toujours suspendu.
*Jalon : atteindre au moins 60 % de la performance de référence.*

**Sprint 4 — Boucle fermée (semaines 7-8).** Une personne passe de SIM à ROBOT. La
politique pilote le robot en continu, suspendu.
*Jalon : 60 secondes sans commande aberrante.*

**Sprint 5 — Appuis au sol (semaines 9-10).** Palier 3 : mise au sol progressive dans
un harnais qui décharge d'abord 80 % du poids, puis 50 %, puis 20 %.
*Jalon : le robot prend appui et produit des mouvements de marche reconnaissables.*

**Sprint 6 — Passation (semaines 11-12).** Documentation, vidéo, backlog de la session
suivante. **C'est aussi le sprint tampon** : avec des bénévoles, un jalon glisse
toujours.

### Quand ça ne marche pas sur le vrai robot

Garde ce tableau sous la main à partir du sprint 3. **Le réflexe naturel — « changeons
la récompense » — est presque toujours le mauvais**, parce qu'il impose de tout
réentraîner, soit plusieurs semaines.

| Symptôme | Cause quasi certaine | Quoi faire | Coût |
|---|---|---|---|
| Le robot **vibre** | Réglages trop raides face à la souplesse réelle | Baisser `kp`, monter `kd` | minutes |
| Mouvements **en retard, saccadés** | Fréquence non tenue | Baisser la cadence | minutes |
| Mouvement **aberrant** | Un capteur ne transporte pas ce qu'on croit : signe inversé, degrés au lieu de radians, articulations dans le désordre | **C'est un bug de correspondance**, pas d'apprentissage | heures |
| Mouvement **plausible mais il tombe** | Écart de physique réel/simulé | Élargir la randomisation, réentraîner | jours |
| Mouvement correct mais **trop violent** | La simulation est trop permissive | **Là seulement : la récompense** | semaines |

**L'ordre : bug de correspondance → réglage → randomisation → récompense.** Le
troisième cas est de loin le plus fréquent lors d'une première mise en service, et
c'est aussi celui qui ressemble le plus à un « échec de l'IA » alors que c'est une
inversion de signe.

---

## 8. Ce qu'on ne fait pas cette session

Ces quatre lignes protègent notre temps. Elles ne sont pas négociables sans discussion
d'équipe.

- **Pas de migration vers MJX** (la version GPU de MuJoCo). Techniquement faisable,
  mais ça coûterait la totalité du budget de la session pour zéro progrès sur la
  marche réelle. À reconsidérer la prochaine fois.
- **Pas d'intégration de la vision.** Le code de perception par profondeur existe dans
  `src/sensors/` et fonctionne, mais le brancher sur la politique est un chantier
  entier.
- **Pas de navigation avec obstacles.**
- **Pas de marche autonome sans appui.** C'est l'objectif de la session suivante.

**Pourquoi c'est écrit noir sur blanc :** on a environ 120 heures de travail effectif
au total, réparties entre six personnes sur trois mois. Chacun de ces quatre chantiers
consommerait à lui seul une bonne part du budget. Dire non maintenant, c'est pouvoir
dire oui à ce qui compte.

---

## Où trouver quoi

| Document | Contenu |
|---|---|
| `docs/GUIDE_EQUIPE.md` | Ce guide — contexte, vocabulaire, tâches |
| `docs/ETAT_DES_LIEUX.md` | L'inventaire technique du projet à la reprise |
| `docs/BASELINE.md` | Ce que fait le meilleur modèle *(créé en S1-SIM-1)* |
| `docs/INTERFACE.md` | Le contrat gelé du cerveau *(créé au sprint 2)* |
| `docs/ROBOT.md` | Le matériel, les mesures, les procédures *(créé en S1-ROB-1)* |
| `README.md` | Installation et commandes |

---

*Document vivant. Si quelque chose n'est pas clair, ce n'est pas toi qui as mal lu —
c'est le document qui doit être corrigé. Signale-le, ou corrige-le directement.*
