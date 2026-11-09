# Journal d'expériences — palier 1 « marcher »

Tenu sur la branche locale `exp/palier1-recompense`. Ce journal est écrit pour
être relu par quelqu'un qui n'a pas suivi : un membre de l'équipe ou un autre
modèle d'IA. Il sépare **ce qui est mesuré**, **ce qui est supposé** et **ce
qui reste à décider**. Les nombres viennent de `scripts/evaluate.py`
(5 épisodes de 10 s, graine 1, sauf mention contraire).

Dernière mise à jour : 8 novembre 2026, 08:10. **Pour aller vite : §4.2, la synthèse.**

---

## 1. Le problème

Poppy Humanoid : un humanoïde de 25 articulations, de 2,6 kg dans le modèle
(3,5 kg annoncés pour le vrai robot), simulé dans MuJoCo
(`src/environments/poppy_humanoid_env.py`) et entraîné avec PPO
(Stable-Baselines3). La politique sort une cible de position par
articulation, toutes les 10 ms.

**But du palier 1** : une marche à vitesse **lente et constante**
(~0,25 m/s), **tronc droit**, sans gestes brusques, transférable au vrai
robot. On ne veut pas qu'il coure, ni qu'il se jette en avant pour se
rattraper. Contrainte de l'équipe : la politique ne doit **observer** que ce
que le robot mesure (IMU avec gyroscope, encodeurs, capteurs de pression sous
les pieds, horloge de l'ordinateur de bord). La récompense, elle, peut
utiliser des grandeurs de simulation, puisqu'elle ne sert qu'à
l'entraînement.

Pièges du dépôt à connaître (voir `CLAUDE.md`) :
- L'avant de Poppy est son **−y local** et sa gauche son **+x**. La vitesse est
  projetée sur le cap du bassin, pas sur un axe du monde.
- Le genou gauche du MJCF tourne à l'envers. Toutes les expériences depuis le
  6 nov. l'activent avec `fix_left_knee_axis: true`.
- Trois défauts connus sont documentés et volontairement non corrigés
  (`docs/TECHNIQUE.md` §8) : l'espace d'action non normalisé, la restitution
  du sol écrite au mauvais indice, et `prev_action` pénalisé sans être
  observé.

## 2. Comment on mesure

`evaluate.py` sort, pour un modèle :
- **tient** : durée de l'épisode en pas de 10 ms (1000 = les 10 s complètes) ;
- **avance** : déplacement vers l'avant en 10 s ;
- **sans chute** : part des épisodes qui vont au bout ;
- **cap** : rotation du bassin sur l'épisode ;
- **appui D/G** : part du temps en appui simple, sur chaque pied ;
- **cadence** : nombre de poses de pied par seconde ;
- **tronc** : verticalité de la poitrine, le cosinus de son inclinaison
  (1 = droit, 0,985 ≈ 10°, 0,905 ≈ 25°, 0,82 ≈ 35°) ;
- **décomposition de la récompense**, terme par terme.

### Critères de « marche », fixés avant les expériences de la nuit du 8 nov.

On les mesure sur **10 épisodes** (`--episodes 10 --seed 1`), au modèle final
et au meilleur modèle :

| Critère | Seuil |
|---|---|
| M1 sans chute | ≥ 90 % |
| M2 vitesse vers l'avant | 0,20 à 0,35 m/s |
| M3 tronc | ≥ 0,985 (moins de ~10° d'inclinaison) |
| M4 dérive de cap | moins de 15° en valeur absolue |
| M5 symétrie d'appui | écart D−G de moins de 10 points |
| M6 dérive latérale | moins de 0,3 m en 10 s |

Un run « marche » s'il passe les six. Une hypothèse n'est confirmée que si
le résultat **se reproduit avec une deuxième graine**.

Indicateurs pour le robot réel, mesurés sans seuil pour l'instant : la part
du temps où un moteur est à sa limite de couple, le couple maximal rapporté
à la limite, et la part des cibles demandées hors de la course de
l'articulation.

### Règles de méthode

Elles ont été adoptées après les erreurs de la section 5.
1. **Une seule variable** change par rapport à un témoin qui marche.
2. **Au moins 8 à 10 M de pas** avant de conclure. La marche est apparue vers
   8 M chez robust-genou, vers 7 M chez B, vers 0,5 M chez A.
3. On juge sur les **mesures**, pas sur la courbe de récompense.
4. L'hypothèse et le résultat attendu sont **écrits ici avant** le lancement.

## 3. Chronologie et résultats

### 3.1 Avant le 6 novembre

| Run | Ce que c'est | Résultat |
|---|---|---|
| **Référence du 19 sept.** (`models/2026-09-19_recompense-corrigee`) | Ancienne récompense (vitesse brute ×5 plafonnée, socle, prime d'appui), genou gauche à l'envers | Tient 835 ± 330 pas, 5,40 m (0,65 m/s), 80 % sans chute, cap **+47°**, appui 32/57 %, cadence 4,8, tronc 0,914. Marche rapide, boiteuse, qui tourne. |

### 3.2 Palier 1 de la spec, 6-7 nov.

Récompense de `spec/recompense-et-paliers.md`, plus un nouveau **cadre
d'entraînement** introduit en même temps :
- action normalisée à [-1, 1] ;
- PPO avec lr 3e-4 décroissant linéairement jusqu'à 0, 5 époques, clip 0,2,
  entropie 0,005 ;
- perte de symétrie miroir.

| Run | Pas | Résultat |
|---|---|---|
| palier1-zero, palier1-reprise (portable) | — | Arrêtés : genou gauche à l'envers. Docker Desktop est ensuite tombé. |
| palier1-v2, de zéro (Hetzner) | 10 M | **Plongeon** : chute après ~1,1 s, 0 % sans chute, plateau dès 1,5 M. Le terme de vitesse faisait ~70 % de la récompense (5 × 0,8 m/s, contre ~1 de socle par pas). |
| palier1-reprise-v2, repris du modèle du 19 sept. | 3,5 M (arrêté) | Même plongeon : 106 pas, 0 % sans chute. |
| **robust-genou** : ancienne récompense, ancien cadre, genou corrigé | 10 M | 5,6 M : tombe à 2,3 s et **tourne sur lui-même** (+225°). Meilleur modèle à **8,3 M** : 1000 pas, 5,90 m (0,59 m/s), 100 % sans chute, appui 40/43 %, mais **tronc 0,905 (~25° penché)** et cap +87°. Durée d'épisode : 125 pas à 1 M, 160 à 2 M, 180 à 3 M, 224 à 4,7 M, 288 à 7 M, 1000 à 8,3 M. |

### 3.3 Pilotes « marche », 7-8 nov.

Tous dans le **nouveau cadre**, tous de 2 M de pas sauf mention contraire.
Chaque pilote **ajoute** un changement au précédent : c'est l'erreur n° 3 de
la section 5. Configs : `configs/palier1_marche*.yaml`, commit `ae3d514`.

| Pilote | Ce qui change | Tient | Avance | Sans chute | Remarque |
|---|---|---|---|---|---|
| 1 `palier1_marche` | Vitesse brute remplacée par un suivi en cloche autour de 0,25 m/s, sans filtre | 1000 | 0,07 m | 100 % | Se balance d'avant en arrière et touche la cible à chaque oscillation |
| 2 | + vitesse filtrée sur 1 s (τ = 1), socle 0,5 | 197 | 0,46 m | 0 % | Bascule d'un bloc vers l'avant, sans lever les pieds |
| 3 `_appui` | + prime d'appui simple 0,3 | 207 | 0,50 m | 0 % | À 4 M : 1000 pas, 0,11 m, **piétine** (cadence 3,1) |
| 4 `_posture` | + pénalité d'inclinaison du tronc ×5 | 1000 | 0,07 m | 100 % | Immobile, tronc 0,999 |
| 5 `_standard` | + hauteur du bassin, roulis et tangage (gyroscope), glissement des pieds (legged_gym) | 819 | 0,24 m | 60 % | Petits pas |
| 6 `_horloge` | + horloge de marche (sin/cos de la phase observés, 65 dimensions), prime de contact en phase (style Unitree G1) | 854 | 0,42 m | 80 % | **Pas alternés, tronc droit, mais n'avance pas.** Run de 10 M : meilleur modèle 837 pas, 0,50 m, 80 %, appui 32/29 %. |
| 6b `_horloge_vitesse` | Suivi de vitesse ×2 (3,0) | 585 | 0,35 m | 40 % | Pire |
| 7 `_survie` | Socle et primes de verticalité baissés | 154 | 0,51 m | 0 % | Retour du plongeon |

### 3.4 Expériences contrôlées A et B, 8 nov.

**Question posée** : les pilotes échouent-ils à cause de la récompense, ou du
cadre d'entraînement introduit en même temps ?

- **A** = ancien cadre (celui de robust-genou : action non normalisée, lr
  2,5e-5 constant, 20 époques, clip 0,3, sans entropie ni miroir,
  8 environnements) + récompense du pilote 6. Seule exception : effort ×0,1
  et à-coups ×0,05 au lieu de 0,5 et 0,5, les valeurs d'origine pour une
  action non normalisée. Config :
  `configs/exp_a_ancien_cadre_nouvelle_recompense.yaml`.
- **B** = cadre du palier 1 + ancienne récompense (celle de robust-genou),
  sans horloge. Config : `configs/exp_b_nouveau_cadre_ancienne_recompense.yaml`.
- Graine 42 dans les deux cas, 10 M de pas, sur Hetzner (A sur un CCX43, B
  sur un CPX62).

| | Pas | Tient | Avance | Sans chute | Cap | Appui D/G | Tronc |
|---|---|---|---|---|---|---|---|
| A | 0,4 M | 862 (éval.) | — | — | — | — | — |
| A | 2 M | 894 ± 211 | 2,37 m | 80 % | −0,1° | 33/47 | 0,991 |
| A | 5 M | 1000 | 2,87 m | 100 % | +4,9° | 40/51 | 0,993 |
| A | 6 M | 1000 | 2,66 m (0,27 m/s) | 100 % | +3,7° | 43/48 | 0,994 |
| B | 3 M | 179 ± 20 | 0,60 m | 0 % | −48,7° | 14/21 | 0,896 |
| B | 7 M | 1000 | 2,47 m | 100 % | −6,8° | 33/44 | **0,824 (~35°)** |
| B, final (10 M), 10 épisodes | 10 M | 1000 | 3,35 m (0,34 m/s), latéral 0,52 m | 100 % | **−57,5°** | 46/40 | 0,847 |
| B, meilleur (8,9 M), 10 épisodes | 8,9 M | 1000 | 4,06 m (0,41 m/s), latéral 0,70 m | 100 % | −19,7° | 41/48 | 0,836 |

B, aux critères : il passe M1 et M5. Il **échoue** à M3 (tronc penché d'environ
32°), M4 (cap) et M6 (dérive latérale). M2 est passé par le modèle final et
manqué par le meilleur, qui va trop vite. Vidéos :
`logs/videos/2026-11-08/exp-b-final.mp4` et `exp-b-final-profil.mp4`.

**A, mesure finale** (fini à 02:42), avec 10 épisodes par checkpoint, sur les
critères M1 à M6 (`criteres.py`, dérive latérale et cap en valeur absolue
épisode par épisode) :

| Checkpoint | M1 sans chute | M2 vitesse | M3 tronc | M4 \|cap\| | M5 appui D−G | M6 \|latéral\| | Score |
|---|---|---|---|---|---|---|---|
| 3 M | 100 % | 0,277 | 0,990 | 5,2° | 17,2 ✗ | 1,24 ✗ | 4/6 |
| 5 M | 100 % | 0,260 | 0,993 | 4,6° | 11,3 ✗ | 1,20 ✗ | 4/6 |
| 6 M | 100 % | 0,265 | 0,994 | 3,8° | 5,8 | 0,43 ✗ | 5/6 |
| **6,15 M (`best_model`)** | 90 % | 0,265 | 0,992 | 5,6° | 11,0 ✗ | **0,24** | 5/6 |
| 7 M | 100 % | 0,288 | 0,994 | 5,1° | 0,5 | 0,30 ✗ (max 0,80) | 5/6 |
| 8 M | 70 % ✗ | 0,308 | 0,993 | 14,8° | 0,9 | 0,49 ✗ | 4/6 |
| 9 M | 100 % | 0,250 | 0,997 | 7,3° | 6,2 | 0,46 ✗ | 5/6 |
| **final (10 M)** | **50 %** ✗ | 0,338 | 0,979 ✗ | 33,5° ✗ | 1,3 | 0,47 ✗ | **2/6** |

- **Aucun checkpoint ne passe les six critères.** Les meilleurs (6 M,
  6,15 M, 7 M, 9 M) en passent cinq ; le critère manqué est la dérive
  latérale, ou la symétrie d'appui pour `best_model`.
- **La fin du run se dégrade** : le modèle final tombe dans la moitié des
  épisodes. Ça confirme l'instabilité que montrait le KL (jusqu'à 3). Le
  modèle final de A **ne doit pas** servir ; c'est l'argument de
  l'expérience `target_kl`.
- **Vidéo** (`exp-a-best-6M-profil.mp4`, images à 1, 3, 5, 7 et 9 s) : le
  robot marche droit, mais **un bras levé au-dessus de la tête pendant tout
  l'épisode**, sans doute comme balancier. C'est probablement ce qui donne
  l'impression d'une marche « statique ». Sur un vrai robot, c'est à
  proscrire : rien dans la récompense de A ne tient le haut du corps. La
  récompense d'imitation de I le tient (`imitation_upper_weight`). B, au
  contraire, garde les bras le long du corps, mais marche voûté
  (`exp-b-final-profil.mp4`).

Autres observations :
- A dérive latéralement de **0,45 m pour 2,66 m d'avance** à 6 M : il marche
  en crabe, à ~10° de son cap. **Il échoue au critère M6.**
- Le `approx_kl` de A monte de ~0,07 au début à 0,3–1,0 vers 6,5 M, et la
  durée moyenne en entraînement est passée de 957 à 845, puis remontée à 917.
  Instabilité possible en fin de run, à surveiller. Le meilleur modèle est
  gardé de toute façon.

**Ce qu'on peut conclure**, avec une seule graine chacune :
1. La récompense « marche + horloge » **peut** apprendre une marche lente,
   droite et stable. Elle le fait dans l'ancien cadre, et vite (0,5 M contre
   8 M pour l'ancienne récompense).
2. Le nouveau cadre **n'empêche pas** la marche : B marche à 7 M avec
   l'ancienne récompense, à peu près au même rythme que robust-genou (8,3 M).
3. Ce qui bloquait les pilotes est donc une **interaction** entre la
   nouvelle récompense et le nouveau cadre. Le cadre seul, ou la récompense
   seule, n'expliquent pas l'échec. Ma première lecture (« c'est le cadre »),
   donnée à 3 M de B, était prématurée : elle violait la règle n° 2.

### 3.5 Mesure du 8 nov. vers 02:10 : A à 6 M contre pilote 6 au meilleur de ses 10 M

Somme par épisode, 5 épisodes (A : 1000 pas ; pilote 6 : 837 pas en moyenne).

| Terme | A | Pilote 6 |
|---|---|---|
| suivi de vitesse | 1387,6 | 451,7 |
| contact en phase (horloge) | 466,6 | 346,5 |
| effort (`ctrl_cost`) | 115,9 | 44,8 |
| à-coups (`action_rate_cost`) | 26,5 | 1,5 |
| vitesse articulaire | 123,1 | 5,8 |
| **butées (`joint_limit_cost`)** | **5,0** | **125,0** |

Première lecture, **réfutée** à 02:35 par une relecture externe puis
vérifiée dans le code. Elle est barrée et gardée pour la trace :
- ~~Le pilote 6 est poussé aux butées par un bruit d'exploration plus fort en
  action normalisée (0,14 = 14 % de la course, contre 0,14 rad).~~ **Faux.**
  `_action_to_torque` (`poppy_humanoid_env.py`) convertit l'action en cible
  de la **même façon** dans les deux cadres :
  `cible = pose debout + action × course`. `normalize_action_space` ne change
  que les **bornes** de l'espace d'action, `[-1, 1]` au lieu de ±3,1 à
  ±7,3. Le même écart-type donne donc le même bruit en radians. Ce qui
  change vraiment, c'est que **A peut demander des cibles hors course**,
  jusqu'à 3 fois la course : le PD sature alors le couple et pousse
  l'articulation en butée. D teste exactement ce bornage.
- ~~Les butées du pilote 6 viennent de l'exploration.~~ **Faux.** La relecture
  attribue 74 % de sa pénalité de butée aux **coudes**. Vérifié : à la pose
  debout, `l_elbow_y`, `r_elbow_y` et les deux genoux sont **déjà** dans la
  zone pénalisée (au-delà de 95 % de la course). Une politique qui garde les
  bras à la pose initiale paie environ 0,15 par pas sans rien faire de mal.
  C'est un défaut de calibrage du terme, pas un comportement.
- ~~Les poids ×5 et ×10 du pilote 6 compensaient une action ~10 fois plus
  petite.~~ **Faux** pour la même raison : l'échelle est identique. La
  différence mesurée sur palier1-v2 venait de politiques différentes, pas de
  l'échelle. Entre A et le pilote 6, l'effort (`ctrl_cost`, la somme des
  carrés de l'action, donc l'écart à la pose debout, pas l'énergie) est
  **vraiment 5 fois** plus pénalisé, et les à-coups **10 fois**. **A et B ne
  séparent donc pas parfaitement récompense et cadre.**

Lecture corrigée :
- Le pilote 6 bouge peu (vitesse articulaire 20 fois plus faible) et paie peu
  de pénalités de douceur, **parce que** ces pénalités sont 5 à 10 fois plus
  lourdes que chez A. L'hypothèse « pénalités trop fortes : la politique
  préfère les petits gestes » redevient la **principale**. Elle peut se
  combiner avec les mises à jour PPO plus agressives du nouveau cadre.
- Le **miroir** est un autre suspect crédible. Dans `MirrorPPO.train()`
  (`src/training/mirror_ppo.py`), la perte de symétrie fait une passe de
  gradient complète **après** la mise à jour PPO, sans le garde-fou du
  clipping. B l'avait pourtant, et a marché.
- La relecture a évalué A à 6 M avec ses actions bornées à `[-1, 1]`, en
  évaluation seulement : résultats identiques, 10/10 sans chute. A se sert
  donc peu des cibles hors course une fois entraîné. C'est encourageant pour
  D, sans rien prouver sur l'entraînement.

## 4. Plan de la nuit du 8 nov.

Plafond de budget validé par l'utilisateur : **15 $** (environ 7 $ prévus).
Ressources : un CCX43 (16 cœurs dédiés), un CPX62 (16 cœurs partagés) et le
portable (12 threads, Docker Desktop). Le compte Hetzner est limité à ces
deux serveurs.

### Vague 1 (lancée vers 02:20–02:45, révisée à 02:40)

| Où | Run | Hypothèse | Ce qui la confirme |
|---|---|---|---|
| CPX62 | **D** = A avec `normalize_action_space: true`, seule variable changée (`configs/exp_d_ancien_cadre_action_normalisee.yaml`) | Borner les cibles à la course, ce qu'il faut pour le robot, n'empêche pas d'apprendre | D passe M1 à M5 avant 10 M |
| CCX43 | **I** = piste 3 (imitation Placo) **dans le cadre de A** (`exp_i_imitation_cadre_a.yaml`, construite sur le code de la branche `claude/piste3-placo-4cdd4e`, non fusionnée) | À cadre égal, imiter une référence faisable donne une marche plus propre que la récompense de A | Elle passe M1 à M6, avec un appui plus symétrique que A |
| Portable | **A, graine 7** (`--seed 7`) | A est reproductible | Il passe M1 à M5 comme A-42 |

Notes :
- **Révision de 02:40** : la piste 3 devait tourner telle que sa session
  l'avait conçue, dans le nouveau cadre. La relecture externe fait remarquer
  qu'on retomberait dans la confusion de variables (erreur n° 2). I la place
  donc dans le seul cadre validé : action non normalisée, PPO de A, sans
  miroir, 8 environnements. Sa récompense et son environnement (départ dans
  la pose de la référence, horloge) restent ceux de la session Placo.
- Une seule graine par run, pour l'instant.

### Règles de décision pour la vague 2 (révisées à 02:40, avant tout résultat de D ou I)

Ordre de priorité des runs de la vague 2, chacun à **une variable de A** :
1. **E** = A avec les poids d'effort et d'à-coups du pilote 6 (`ctrl_weight`
   0,5 et `action_rate_weight` 0,5). Teste l'hypothèse principale : si E ne
   marche plus, ce sont ces pénalités qui bloquaient les pilotes.
2. **A + miroir** (`symmetry_loss_weight: 1.0`). Le miroir aide-t-il, ou
   gêne-t-il ?
3. **A + `target_kl`** (0,03 environ, à calibrer). Teste si l'on peut
   supprimer l'instabilité de fin de run de A (KL jusqu'à 3).

Décisions :
- **D marche** : D devient le candidat pour le robot, puis on le relance avec
  une deuxième graine.
- **D ne marche pas et A-7 si** : le bornage gêne l'apprentissage. On
  regarde comment A se sert des cibles hors course pendant l'entraînement
  avant de tester autre chose. Le test `log_std_init: -3`, prévu à
  l'origine, est **abandonné** : sa justification était fausse.
- **A-7 ne marche pas** : A-42 était peut-être un coup de chance. On le
  relance avec une troisième graine avant toute autre conclusion.
- **I marche** : on la compare à A sur M1 à M6, sur la symétrie et sur les
  indicateurs pour le robot, puis on la relance avec une deuxième graine.
- **I échoue** : on lit sa décomposition (erreur d'imitation par
  articulation) avant toute relance.

**Choix des modèles** : `best_model` est choisi sur la récompense, pas sur
les critères. Pour chaque run fini, on mesure aussi plusieurs checkpoints sur
M1 à M6, et la dérive latérale **en valeur absolue, épisode par épisode** :
une moyenne signée peut cacher des dérives opposées.

### Vague 3 et fin de nuit

On confirme le meilleur candidat avec deux graines, puis on produit les
vidéos de face et de profil, un tableau final face à la référence du
19 sept. et à robust-genou, et les indicateurs pour le robot. Ensuite, on
**supprime tous les serveurs** (au plus tard vers 08:00), et aucun serveur ne
reste plus de 30 min sans run.

### 4.1 Résultats de la vague 1

**I** (imitation Placo dans le cadre de A), fini à 04:25. 10 épisodes par
checkpoint, `criteres.py` :

| Checkpoint | M1 | M2 vitesse | M3 tronc | M4 \|cap\| | M5 appui D−G | M6 \|latéral\| (max) | Score |
|---|---|---|---|---|---|---|---|
| 3 M | 100 % | 0,285 | 0,995 | 10,2° | 2,4 | 0,47 ✗ (0,50) | 5/6 |
| 5 M | 100 % | 0,266 | 0,996 | 11,3° | 3,5 | 0,91 ✗ (0,94) | 5/6 |
| 7 M | 100 % | 0,262 | 0,997 | 12,8° | 6,5 | 1,57 ✗ (1,61) | 5/6 |
| 8 M | 100 % | 0,265 | 0,996 | 11,9° | 0,5 | 1,50 ✗ (1,55) | 5/6 |
| final (10 M) = `best_model` | 100 % | 0,263 | 0,997 | 9,2° | 6,5 | 1,35 ✗ (1,37) | 5/6 |

- **I est le run le plus régulier** : 5/6 à **tous** les checkpoints mesurés,
  100 % sans chute, le tronc le plus droit (0,997). Il ne se dégrade pas en fin
  de run, bien que son `approx_kl` ait atteint **64** à la dernière mise à jour.
  Contrairement à A, son modèle final est utilisable.
- **Sa dérive latérale est systématique** : la moyenne et le maximum sur
  10 épisodes sont presque égaux, et elle part toujours du même côté (signe
  négatif). Le robot ne vacille pas : il marche **en crabe, dans une direction
  fixe**. Elle grandit de 0,47 m à 3 M jusqu'à ~1,5 m vers 7–8 M. C'est
  l'argument le plus net pour l'hypothèse d'une **récompense latérale trop
  faible** (voir la ligne de 03:56 dans le journal de la nuit).
- Vidéos : `exp-i-final.mp4` (face), `exp-i-final-profil.mp4`.

**D** (A avec actions bornées à `[-1, 1]`), fini à 04:57 :

| Checkpoint | M1 | M2 vitesse | M3 tronc | M4 \|cap\| | M5 appui D−G | M6 \|latéral\| (max) | Score |
|---|---|---|---|---|---|---|---|
| 3 M | 90 % | 0,222 | 0,989 | 24,3° ✗ | 1,4 | **0,15** (0,18) | 5/6 |
| 5 M | 100 % | 0,264 | 0,992 | 23,7° ✗ | 6,4 | 0,93 ✗ | 4/6 |
| 6 M | 80 % ✗ | 0,244 | 0,990 | 14,7° | 14,0 ✗ | 0,88 ✗ | 3/6 |
| 7 M | 100 % | 0,236 | 0,993 | 6,0° | 8,1 | 0,81 ✗ | 5/6 |
| 8 M | 100 % | 0,261 | 0,994 | 2,8° | 7,0 | 0,91 ✗ | 5/6 |
| 9 M | 100 % | 0,244 | 0,993 | 2,0° | 5,9 | 0,59 ✗ | 5/6 |
| **final (10 M)** | 100 % | 0,225 | 0,994 | 2,8° | 4,7 | 0,54 ✗ (0,73) | **5/6** |
| `best_model` (9,75 M) | 100 % | 0,242 | 0,993 | 3,2° | 5,2 | 0,71 ✗ | 5/6 |

- **Borner les actions n'empêche pas d'apprendre la marche** : D atteint le
  même niveau que A (5/6), avec un cap mieux tenu en fin de run (2 à 3°).
  L'hypothèse de la vague 1 est confirmée, pour cette graine.
- **Son modèle final est bon**, contrairement à celui de A, bien que son KL
  soit monté jusqu'à 1,2.
- **Posture** (`exp-d-final-profil.mp4`) : droit, les bras le long du corps.
  C'est la posture la plus « robot réel » des runs de la nuit : ni bras levé
  (A), ni bras tendus vers l'avant (I), ni dos voûté (B).
- **D est donc le meilleur candidat pour le robot à ce stade** : ses cibles
  restent dans la course des articulations, sa fin de run est stable et sa
  posture est sobre. Il reste à corriger la dérive latérale (M6), le défaut
  commun à tous les runs.

**A, graine 7** (même config que A, `--seed 7`, portable), fini à 05:42 :

| Checkpoint | M1 | M2 vitesse | M3 tronc | M4 \|cap\| | M5 appui D−G | M6 \|latéral\| (max) | Score |
|---|---|---|---|---|---|---|---|
| 5 M | 90 % | 0,229 | 0,990 | 7,8° | 6,0 | 0,97 ✗ | 5/6 |
| 7 M | 100 % | 0,261 | 0,995 | 14,4° | 0,5 | 0,52 ✗ | 5/6 |
| **8 M** | 100 % | 0,237 | 0,991 | 13,1° | 0,1 | **0,19** (0,27) | **6/6** |
| **9 M** | 100 % | 0,220 | 0,991 | 8,3° | 2,7 | **0,26** (0,35) | **6/6** |
| **`best_model` (8,05 M)** | 100 % | 0,243 | 0,990 | 13,0° | 2,4 | **0,25** (0,36) | **6/6** |
| final (10 M) | 100 % | 0,229 | 0,985 ✗ (à l'arrondi) | 5,0° | 0,5 | 0,16 | 5/6 |

- **Premiers modèles à passer les six critères** (8 M, 9 M, `best_model`).
  Le cap reste proche de la limite (13° pour 15°) : à surveiller avec des
  épisodes plus longs.
- **A est reproductible** sur la marche elle-même : 100 % sans chute, vitesse à
  la cible, tronc droit, pour les deux graines. Le modèle final de la graine 7
  se dégrade moins que celui de la graine 42.
- **Le bras levé de A-42 était propre à cette graine** : A-7 marche les bras
  tenus vers l'avant, comme I (`exp-a-graine7-best-profil.mp4`). Les deux
  graines montrent que rien dans la récompense de A ne tient la posture des
  bras : chaque graine trouve la sienne.
- La dérive latérale varie beaucoup d'une graine à l'autre (A-42 : 0,24 à
  1,24 m selon le checkpoint ; A-7 : 0,16 à 0,97 m). C'est un défaut de la
  récompense, avec beaucoup de variance, pas une propriété stable.

**E** (A avec les pénalités d'effort et d'à-coups du pilote 6), fini à 05:55 :

| Checkpoint | M1 | M2 vitesse | M3 tronc | M4 \|cap\| | M5 appui D−G | M6 \|latéral\| (max) | Score |
|---|---|---|---|---|---|---|---|
| 5 M | 100 % | 0,191 ✗ | 0,995 | 8,0° | 7,3 | 1,12 ✗ | 4/6 |
| 7 M | 80 % ✗ | 0,249 | 0,990 | 5,7° | 11,9 ✗ | 1,22 ✗ | 3/6 |
| 8 M | 100 % | 0,250 | 0,991 | 3,5° | 14,3 ✗ | 1,12 ✗ | 4/6 |
| 9 M | 100 % | 0,262 | 0,989 | 2,7° | 13,4 ✗ | 1,24 ✗ | 4/6 |
| final (10 M) | 90 % | 0,268 | 0,992 | 6,9° | 14,9 ✗ | 0,85 ✗ | 4/6 |
| `best_model` (8,25 M) | 100 % | 0,251 | 0,990 | 2,7° | 12,0 ✗ | 1,26 ✗ | 4/6 |

- **Verdict** (une graine) : les pénalités du pilote 6 **n'empêchent pas** la
  marche dans le cadre de A (M1 à M4 tenus à partir de 8 M). L'hypothèse
  « elles bloquaient les pilotes » est **réfutée**.
- Elles **dégradent pourtant la qualité** : l'appui devient asymétrique (12 à
  15 points d'écart, contre 0 à 6 pour A), la dérive latérale dépasse 1 m, et
  la vitesse cible n'est atteinte qu'après 7 M. Plafond à 4/6.
- Ce qui bloquait les pilotes reste donc à trouver parmi le **miroir** (F, en
  cours) et les **réglages PPO** du nouveau cadre.
- Vidéos : `exp-e-best.mp4`, `exp-e-best-profil.mp4`.

**F** (A avec la perte de symétrie miroir), fini à 07:28 :

| Checkpoint | M1 | M2 vitesse | M3 tronc | M4 \|cap\| | M5 appui D−G | M6 \|latéral\| (max) | Score |
|---|---|---|---|---|---|---|---|
| **5 M** | 100 % | 0,260 | 0,997 | 6,2° | 1,0 | **0,02** (0,05) | **6/6** |
| 6 M | 100 % | 0,276 | 0,994 | 1,3° | 9,6 | 0,43 ✗ | 5/6 |
| **7 M** | 100 % | 0,256 | 0,994 | 3,2° | 2,6 | 0,16 (0,21) | **6/6** |
| **8 M** | 100 % | 0,218 | 0,990 | 3,6° | 4,7 | 0,11 (0,17) | **6/6** |
| **9 M** | 100 % | 0,226 | 0,987 | 0,9° | 3,1 | 0,14 (0,18) | **6/6** |
| final (10 M) | 100 % | 0,166 ✗ | 0,988 | 6,1° | 9,4 | 0,14 | 5/6 |
| **`best_model` (6,5 M)** | 100 % | 0,247 | 0,997 | **0,9°** | 2,6 | 0,21 (0,22) | **6/6** |

- **Le meilleur run de la nuit** : 6/6 à quatre checkpoints sur cinq entre 5
  et 9 M, plus `best_model`. Cap quasi nul, dérive latérale faible et stable.
- **Le miroir aide**, au lieu de gêner. C'est l'inverse de ce qui était
  soupçonné : il corrige justement le défaut commun des autres runs, la
  dérive latérale. Une graine seulement.
- **La fin de run ralentit** (0,166 m/s au modèle final) : même tendance
  qu'A-42 et E, une dégradation en fin d'entraînement, plus douce ici.
- Posture (`exp-f-best-profil.mp4`) : droit, les bras le long du corps,
  légèrement vers l'avant.

**G** (A avec `ppo.target_kl: 0.03`, graine 42 comme A), fini à 08:02 sur le
portable :

| Checkpoint | M1 | M2 vitesse | M3 tronc | M4 \|cap\| | M5 appui D−G | M6 \|latéral\| (max) | Score |
|---|---|---|---|---|---|---|---|
| 5 M | 100 % | 0,226 | 0,984 ✗ | 4,4° | 5,2 | 0,66 ✗ | 4/6 |
| **6 M** | 100 % | 0,243 | 0,991 | 0,6° | 3,2 | 0,23 (0,29) | **6/6** |
| **7 M** | 100 % | 0,304 | 0,988 | 3,0° | 6,2 | 0,14 (0,20) | **6/6** |
| 8 M | 100 % | 0,294 | 0,989 | 3,8° | 1,5 | 0,67 ✗ | 5/6 |
| **9 M** | 100 % | 0,292 | 0,996 | 7,0° | 5,8 | 0,18 (0,22) | **6/6** |
| **final (10 M)** | 100 % | 0,285 | 0,990 | 3,5° | 7,0 | 0,20 (0,24) | **6/6** |
| `best_model` (8,85 M) | 100 % | 0,283 | 0,997 | 6,2° | 1,8 | 0,36 ✗ | 5/6 |

- **`target_kl` supprime la dégradation de fin de run** : le KL ne dépasse
  jamais 0,10 (contre 3 pour A, 64 pour I), et le **modèle final passe les
  six critères**. C'est le seul run de la nuit dont on peut prendre le
  dernier modèle les yeux fermés. Le portable a aussi tourné plus vite
  (~1 150 pas/s contre ~800), parce que les époques s'arrêtent tôt.
- **Mais il lève le bras au-dessus de la tête, comme A-42**
  (`exp-g-final-profil.mp4`). Même graine, même posture : c'est la graine
  qui choisit la posture des bras, pas `target_kl`. **Les critères M1 à M6 ne
  voient pas ce défaut**, alors qu'il disqualifie le modèle pour le robot.

## 4.2 Synthèse de la nuit du 8 novembre

| Run | Ce qui change par rapport à A | Meilleur score M1–M6 | Modèle final | Posture |
|---|---|---|---|---|
| A (graine 42) | — | 5/6 | **2/6**, dégradé | bras levé |
| A (graine 7) | graine | **6/6** (8 M, 9 M, best) | 5/6 | bras vers l'avant |
| B | nouveau cadre + ancienne récompense | 3/6 | 3/6 | voûté (~32°) |
| D | actions bornées | 5/6 | 5/6 | **bras le long du corps** |
| I | imitation Placo | 5/6 | 5/6 | genoux fléchis, bras vers l'avant |
| E | pénalités du pilote 6 | 4/6 | 4/6 | — |
| **F** | **miroir** | **6/6** (5, 7, 8, 9 M, best) | 5/6 (ralentit) | droit, bras bas |
| **G** | **`target_kl`** | **6/6** (6, 7, 9 M, final) | **6/6** | bras levé (graine 42) |

**Ce qu'on sait maintenant** (une graine par variante, sauf A) :
1. **La récompense « marche + horloge » dans le cadre de A apprend une marche
   lente, droite et stable**, reproductible sur deux graines.
2. **Rien de ce qu'on a testé ne bloque** : ni les actions bornées (D), ni les
   pénalités lourdes (E), ni le miroir (F). Ce qui faisait échouer les
   pilotes se trouve donc dans les **réglages PPO** du nouveau cadre (lr 3e-4
   décroissant, 5 époques, clip 0,2, entropie 0,005), non testés un par un.
   B prouve que ce cadre peut apprendre avec l'ancienne récompense : c'est une
   interaction.
3. **Trois ingrédients améliorent A, chacun seul** : le miroir (dérive
   latérale corrigée), `target_kl` (fin de run stable), les actions bornées
   (cibles dans la course, bras sobres). **Leur combinaison n'a pas été
   testée.**
4. **La dérive latérale** est le défaut principal sans miroir. Elle est
   systématique (I : toujours du même côté), et le terme `lateral_weight` est
   trop faible pour la retenir.
5. **La posture du haut du corps n'est tenue par rien** dans la récompense de
   A : une graine sur deux lève un bras au-dessus de la tête. Il faut **un
   critère M7** (bras près de la pose debout, ou amplitude des épaules) et,
   probablement, un terme de récompense, comme `imitation_upper_weight` chez
   I.

**Prochaines expériences proposées**, à décider avec l'équipe :
- **H = A + miroir + `target_kl` + actions bornées**, avec deux graines (42 et
  7). C'est le candidat pour le robot. Chaque ingrédient a été validé seul ;
  on teste ici leur combinaison.
- **Ajouter un critère M7 de posture du haut du corps** au banc de mesure,
  puis, si H lève encore un bras, un terme qui tient les bras.
- Pour comprendre l'échec des pilotes (intérêt scientifique, pas pratique) :
  A avec le lr du nouveau cadre, puis avec ses 5 époques, chacun séparément.
- **Évaluation plus dure** du candidat : épisodes de 30 s, poussées
  (`--push`), friction variée, et mesures pour le robot (saturation des
  moteurs, couples maximaux).
- **Décision d'équipe** : retirer de l'observation la hauteur et la vitesse
  linéaire exactes du bassin, que le robot ne mesure pas (voir §6).

### 4.3 Test sur sol accidenté (8 nov., après la nuit, sans entraînement)

Les quatre meilleurs modèles ont été mesurés avec M1–M6 (10 épisodes,
graine 1) sur le sol du palier 2 (`terrain` de `configs/palier2.yaml`),
à quatre difficultés. Aucun n'a jamais été entraîné sur ce sol.

| Modèle | Difficulté 0 (sol plat, en champ de hauteur) | 0,3 | 0,6 | 1,0 |
|---|---|---|---|---|
| F, meilleur | **6/6**, 90 % sans chute | 1/6, 30 % | 2/6, 0 % | 2/6, 0 % |
| G, final | 5/6, 70 % | 4/6, 20 % | 2/6, 0 % | 2/6, 0 % |
| A graine 7, meilleur | **2/6**, 40 %, tronc 0,907 | 1/6, 0 % | 3/6, 0 % | 4/6, 0 % |
| D, final | 4/6, 80 % | 3/6, **60 %** | 3/6, 10 % | 3/6, 0 % |

- **Aucun modèle ne tient sur le relief** : 0 % sans chute à partir de 0,6.
  C'est attendu, puisqu'aucun n'a vu de relief.
- **Résultat inattendu à la difficulté 0.** Le terrain y est parfaitement
  plat (toutes les amplitudes sont multipliées par la difficulté), avec la
  même friction et le même `condim` que le plan. Seule la géométrie de
  contact change : un champ de hauteur au lieu d'un plan. A graine 7 passe
  pourtant de 6/6 à 2/6. Les politiques dépendent donc des détails du modèle
  de contact, ce qui est un mauvais signe pour le transfert au robot. Cause
  probable, non vérifiée : MuJoCo génère les contacts pied/champ de hauteur
  autrement que pied/plan (nombre de points, pénétration).
- F (miroir) est le plus robuste à ce changement, D le plus robuste au
  relief faible (60 % à 0,3).
- **Pour le palier 2** : `configs/palier2.yaml` utilise le nouveau cadre
  PPO, celui qui a cassé les pilotes. Il faudra le réécrire dans le cadre
  de A avant de s'en servir.

### 4.4 Bras du MJCF corrigés avant la vague 1 (8 nov., après-midi)

Décision du lead modèle : corriger les bras seulement, avant de relancer,
puisque la posture des bras était l'un des défauts relevés ; le reste du lot
« MJCF v2 » attend des mesures sur le robot.

- **Cause** : le MJCF a recopié le `rpy` URDF des deux `*_shoulder_motor`
  dans `euler`, que MuJoCo lit en convention intrinsèque x-y-z, alors que
  l'URDF est extrinsèque. Les deux repères sont tournés de 180° ; tous les
  autres corps du MJCF coïncident avec l'URDF (vérifié corps par corps).
- **Correction** : option `environment.fix_arm_frames`, fausse par défaut,
  appliquée au chargement comme `fix_left_knee_axis` (`assets/` inchangé).
  Elle prend une pose debout bras le long du corps, symétrique au miroir.
  Commit ac9dc72, test `tests/test_bras_urdf.py`.
- **Pose debout** : tronc et chevilles à 0 après recherche sur grille. À
  action nulle, le robot tient 480 pas en moyenne, contre 291 avec la pose
  d'origine.
- **Trouvé au passage** : la pose d'origine met les deux `ankle_y` à 0,026.
  Leurs axes sont en miroir, donc les deux pieds y penchent en sens opposés.
  Ce n'est pas corrigé pour les anciennes configs.
- **Piste Placo** : référence régénérée sur le modèle corrigé
  (`reference_walk_bras_urdf.npz`, commit 550eaa1 sur
  `claude/piste3-placo-4cdd4e`). Elle va toujours à 0,25 m/s avec des pas de
  8,75 cm, mais le genou monte à **80 % de son couple maximal** (69 % avant)
  et elle tient 1,29 s en boucle ouverte (1,45 s avant). À surveiller pour
  le robot, surtout s'il est alimenté en 12 V.
- **Hypothèse non vérifiée** : le robot suit l'URDF officiel. Une photo du
  robot à angles nuls la confirmera (tâche 1 de `spec/sim2real-a-corriger.md`).

**Reste du lot MJCF v2, pour plus tard (il faut le robot)** :
- masse (2,61 kg simulés, 3,5 kg annoncés) et centre de masse : pesée ;
- couples maximaux selon la tension réelle, et perte de couple avec la
  vitesse : références des moteurs et tension de l'alimentation ;
- identification des moteurs (kp, kd, inertie, frottement, retard), puis un
  modèle d'actionneur dans l'environnement ;
- observation limitée aux capteurs du robot (BNO055, 4 FSR, encodeurs) :
  acteur/critique asymétrique, décision d'équipe ;
- randomisation pour le transfert, pont en boucle fermée, mesure de l'écart
  simulation/réel.

Le détail est dans `spec/sim2real-a-corriger.md` (branche Placo).

### 4.5 Vague 1 de l'après-midi : H et H-Placo (8 nov., 13:04–14:45)

Deux variantes, deux graines chacune (42 et 7), 10 M de pas, sur le cadre de
A :
- **H** : A + les trois correctifs (`normalize_action_space`, perte miroir
  `symmetry_loss_weight: 1.0`, `target_kl: 0.03`) + bras corrigés
  (`fix_arm_frames`). Config `configs/exp_h_a_trois_correctifs.yaml`.
- **H-Placo** : la même chose sur I (imitation de la marche Placo), avec la
  référence régénérée sur les bras corrigés. Code sur
  `claude/piste3-placo-4cdd4e`.

Serveurs : CCX43 pour H-Placo (~2 200 pas/s par run), CPX62 pour H
(~1 700 pas/s par run), 4 threads par run. Fin vers 14:20 et 14:40. Logs
vérifiés par empreinte (94 fichiers sur 94 par serveur), serveurs supprimés.

**M7, fixé avant de lire les résultats** : en moyenne sur les épisodes, chaque
coude reste au moins 10 cm sous son épaule (axe z du monde ; la pose debout
donne ~12 cm). Validé sur deux modèles déjà connus : F-best (bras bas en
vidéo) a 14,5 / 14,5 cm ✓, G-final (bras levé en vidéo) a −12,5 / 5,9 cm ✗.

Mesure : M1–M6 + M7, 10 épisodes, graine 1, sol plat.

| Run | 2,5 M | 5 M | 7,5 M | best | final | M7 |
|---|---|---|---|---|---|---|
| H-Placo, 42 | 5/6 | **6/6** | 5/6 (lat. 0,30) | **6/6** | **6/6** | ✓ partout |
| H-Placo, 7 | **6/6** | **6/6** | **6/6** | **6/6** | **6/6** | ✓ partout |
| H, 42 | 5/6 (0,41 m/s) | **6/6** | 5/6 (lat. 0,51) | 5/6 (lat. 0,40) | 5/6 (lat. 0,33) | ✗ au best (G 5 cm) et au final (bras levés) |
| H, 7 | 5/6 (0,15 m/s) | 4/6 | 5/6 (lat. 0,76) | 5/6 (lat. 0,34) | 5/6 (lat. 0,62) | ✓ partout |

H-Placo au meilleur modèle : graine 7 à 0,273 m/s, tronc 0,998, cap 1,1°,
appui 0,4 pt, latéral 0,04 m ; graine 42 à 0,249 m/s, tronc 0,998, cap 1,4°,
appui 0,7 pt, latéral 0,12 m.

**Conclusions** :
- **H-Placo est le premier candidat qui passe 6/6 + M7 sur les deux graines**,
  et presque à chaque checkpoint (8 points sur 10 : la graine 42 rate 2,5 M et 7,5 M de peu). C'est la variante la plus
  reproductible mesurée jusqu'ici ; la référence n'est plus une graine
  chanceuse.
- **H tient debout partout mais dérive de côté** (M6 raté à 8 points sur 10),
  et une graine sur deux lève encore les bras en fin de run, même avec les
  bras corrigés. **La correction des bras ne suffit donc pas** à fixer la
  posture sans imitation ; la référence Placo, elle, la fixe.
- `target_kl` a tenu : aucun effondrement, épisodes de 1000 pas dès 1 à 3 M
  sur les quatre runs.
- Les récompenses de H (~3 000) et de H-Placo (~7 200) ne se comparent pas : la
  seconde contient le terme d'imitation.

**Sol en champ de hauteur** (même protocole qu'au §4.3, meilleur modèle) :

| Modèle | plat (sol normal) | hfield 0 (plat) | hfield 0,3 |
|---|---|---|---|
| H-Placo, 7 | 6/6, 100 % | 6/6, 90 % | 4/6, 80 % |
| H-Placo, 42 | 6/6, 100 % | 5/6, 80 % | 4/6, 70 % |
| H, 7 | 5/6, 100 % | 6/6, 100 % | 4/6, 80 % |
| F-best (§4.3) | 6/6, 100 % | 6/6, 90 % | 30 % |

La sensibilité au modèle de contact demeure : sur un hfield pourtant plat,
H-Placo tombe une à deux fois sur dix. Ces modèles résistent mieux au relief
que ceux de la nuit (70 à 80 % à 0,3 contre 30 à 60 %), mais **aucun n'est prêt
pour le palier 2 sans un entraînement sur terrain**.

**Vidéos** (`logs/videos/2026-11-08/exp-h*-profil.mp4`) :
- H-Placo, deux graines : tronc droit, bras le long du corps, genoux fléchis
  comme la référence Placo, petits pas.
- H graine 7 : jambes plus tendues, bras bas, tronc droit.
- H graine 42 au final : **les deux bras levés au-dessus de la tête**, comme
  A-42 et G (M7 à −6,2 / −2,0 cm).

**Caméras** : depuis la correction des axes, la caméra `cote` (azimut 90°)
filme le robot **de face** et `face` (azimut 180°) **de profil**. Les vidéos
« -profil » de cette section ont été faites avec `--camera face`. Les noms dans
`CAMERAS` ne sont pas corrigés : c'est à décider avec l'équipe. Les vidéos
`*-profil.mp4` de la nuit sont à revoir avec ça en tête.

### 4.6 Palier 2 sur H-Placo : sol accidenté (8-9 nov., 19:14–00:25)

`configs/palier2_hplaco.yaml` : la config de H-Placo, plus le terrain (champ de
hauteur régénéré à chaque épisode) avec un curriculum de relief de 0,1 à 1,0 sur
14 M de pas, puis 6 M au relief maximal. 20 M de pas, `--init-from` le meilleur
modèle H-Placo de chaque graine (7 et 42). CCX43, ~1 070 pas/s par run (le terrain
coûte ~50 % du débit), ~2,75 $. Logs vérifiés (174 fichiers sur 174), serveur
supprimé.

**Pendant l'entraînement** : épisodes entre 940 et 1000 pas pendant la montée du
relief, récompense en légère baisse (~7 000 → ~6 400), puis les deux remontent
après 14 M, au relief maximal.

**Survie au relief maximal** (20 épisodes, graine 1, criteres.py) :

| Checkpoint | Graine 7 | Graine 42 |
|---|---|---|
| départ (H-Placo palier 1) | 30 % (2/6) | — |
| 3 M | 35 % | 40 % |
| 6 M | 60 % | 65 % |
| 11,5 M | 75 % | 80 % |
| 14 M | 80 % | 80 % |
| 16 M | 90 % | 85 % |
| 17 à 19 M | 90 % | 90 % |
| **final (20 M)** | **95 %, 5/6 + M7** | **90 %, 5/6 + M7** |

Au relief maximal, seul **M6 manque** : dérive latérale de 0,33 à 0,41 m (limite
0,30). Vitesse 0,24 à 0,31 m/s, tronc ≥ 0,992, cap ≤ 6°, appui équilibré, bras bas.

**Sur sol plat, aucune régression** (config du palier 1, 10 épisodes) : les deux
modèles finaux font **6/6 + M7**, avec une dérive plus faible qu'au palier 1
(0,24 et 0,28 m).

**Vidéos** (`logs/videos/2026-11-08/` et `2026-11-09/`) : sur le terrain où le
modèle du palier 1 tombe à 4,8 s, le modèle du palier 2 tient les 10 s. Sur 8
terrains tirés au hasard, le palier 1 tombe 5 fois, le palier 2 à 10 M jamais.

**Modèles de départ du palier 3**, selon la règle fixée avant la mesure
(meilleure survie au relief maximal, puis le plus de critères, puis le plus
récent) : **final s7** et **final s42**.

**Résistance aux poussées de ces modèles** (avant tout entraînement aux
poussées ; 20 épisodes, 10 N pendant 0,15 s à 3 s) :

| | Graine 7 | Graine 42 |
|---|---|---|
| sol plat, 10 N | 90 % | 100 % |
| relief 0,5, 10 N | 70 % | 80 % |

Le critère du palier 3 sur sol plat (≥ 80 % à 10 N) est donc déjà atteint au
départ ; sur relief 0,5, il manque 10 points à la graine 7. Les modèles ont
déjà subi des « poussées » pendant tout l'entraînement : l'ancien code les
écrit comme un couple sur le bassin, pas comme une force.

**GPU** : le terrain ne passe pas sur MuJoCo Warp (6 contacts terrain–robot
contre 24 pour MuJoCo C ; les modèles du palier 2 y survivent à 40-50 % au
relief 0,3 contre 90 % sur CPU). Paliers 2 et 3 restent sur CPU.

## 5. Erreurs de méthode, et ce qu'on en a appris

1. **Des pilotes de 2 M de pas** ne détectent pas la marche, qui apparaît
   après 5 à 8 M avec l'ancienne récompense. En plus, le lr linéaire est
   déjà à 80 % de sa décroissance à 2 M dans une config de 10 M, et
   l'apprentissage est figé dans une config de 2 M.
2. **Confusion de variables** : le nouveau cadre et la nouvelle récompense
   ont été introduits ensemble. Il a fallu A et B pour les séparer.
3. **Changements empilés** d'un pilote à l'autre, sans ablation, avec une
   seule graine. On ne sait pas lequel des ajouts a aidé ou nui.
4. **Conclusion trop tôt** : « c'est le cadre » a été dit à 3 M de B, et
   démenti à 7 M.
5. **Plan changé sans accord** : un pilote a été remplacé par un autre, alors
   que l'équipe avait demandé un run de 10 M.
6. **Coût** : le prix du CCX63 avait d'abord été sous-estimé (0,60 $/h annoncé
   pour 1,61 $/h réels), puis corrigé.

## 6. Remarques et questions ouvertes

**Sur la marche obtenue**
- Selon l'utilisateur, en voyant les courbes, A paraît « trop statique » et
  B « cherche trop la vitesse » : un compromis donnerait le meilleur des deux.
  Les mesures nuancent : A avance à 0,27 m/s, ce qui est la cible, et B à
  ~0,25 m/s, donc B ne va pas plus vite. La vraie différence est le
  **tronc** : A est droit, B penché de ~35°. **À trancher sur les vidéos de
  A et B**, à produire.
- A marche en crabe (M6). Le terme `lateral_weight: 0.5` ne suffit pas.
- Appui de A : 43/48 %, presque symétrique à 6 M.

**Sur le modèle de simulation**, signalé par la session Placo
(`spec/sim2real-a-corriger.md`, branche `claude/piste3-placo-4cdd4e`)
- Le modèle pèse 2,61 kg, contre 3,5 kg annoncés : les couples dus à la
  gravité sont sous-estimés de ~1,34 fois.
- L'inertie de rotor est de 0,01 kg·m² par moteur. Elle domine le couple du
  genou en vol et n'est pas vérifiée sur le vrai moteur.
- Les bras du MJCF ne sont pas placés comme dans l'URDF : une rotation des
  `*_shoulder_motor` a été convertie dans la mauvaise convention. C'est la
  cause probable du « bug des bras croisés » que compense
  `_STANDING_POSE_OVERRIDES`.
- Les caméras `cote` et `face` sont inversées depuis la correction des axes.
- La marche Placo jouée en boucle ouverte tombe en 1,2 s, avec ~16° d'erreur
  au genou, ce qui donne une idée de l'écart que la politique doit combler.

**Sur le transfert au robot**, signalé par la relecture externe et vérifié :
`_get_obs()` donne à la politique la **hauteur** et la **vitesse linéaire
exactes** du bassin, lues dans MuJoCo. Aucun capteur du robot ne les mesure
directement : la contrainte « seulement ce que le robot perçoit » **n'est pas
encore tenue**. Il faudra les retirer ou les remplacer par une estimation
embarquée (IMU et cinématique), avec son bruit et son retard. Comme cela
change le contrat d'observation, c'est une **décision d'équipe**. Le pont
actuel (`src/robot/simulation_adapter.py`) rejoue des positions simulées : il
ne fait pas encore de boucle fermée sur les capteurs réels.
- Le terme `ctrl_cost` s'appelle « effort », mais c'est l'écart à la pose
  debout, pas l'énergie. Le zéro de la pénalité de butée met les coudes et
  les genoux déjà dans la zone pénalisée à la pose debout.

**Sur la méthode** : les nouveaux termes (suivi de vitesse filtré, horloge,
tronc) n'ont pas été testés un par un dans le cadre qui marche. Une fois le
candidat choisi, une ablation dirait lesquels sont utiles.

**À faire, noté le 8 nov. au soir** (rien de tout cela n'est fait)
- **Décisions d'équipe**
  - Acteur limité aux capteurs du robot **et** mémoire courte (historique de 3 à
    5 pas) : à décider ensemble, de préférence avant le palier 3. Détail dans
    `spec/sim2real-a-corriger.md`, tâche 7.
  - Renommer les caméras `cote` et `face`, inversées depuis la correction des
    axes (`--camera face` donne le profil).
  - Masse simulée 2,61 kg gardée (le robot ne sera pas pesé pour l'instant) :
    les forces de poussée du palier 3 s'entendent pour 2,61 kg.
- **Outils**
  - Mettre `criteres.py` (M1 à M7, relief, poussée) dans le dépôt : il vit
    encore dans un dossier temporaire. M7 n'est pas dans `evaluate.py`.
  - Choisir le meilleur modèle sur terrain pendant l'entraînement : au
    palier 2, `best_model` est choisi sur sol plat et ne sert à rien.
  - `--init-from` exige un `vec_normalize.pkl` à côté du modèle : les
    checkpoints doivent être copiés et renommés à la main.
- **Simulation**
  - Sensibilité au modèle de contact : sur un champ de hauteur pourtant plat,
    les modèles entraînés sur sol plat tombent 1 à 2 fois sur 10.
  - Le terrain ne passe pas sur GPU : MuJoCo Warp fait 6 contacts terrain–robot
    là où MuJoCo C en fait 24 (survie 40-50 % contre 90 % au relief 0,3). Le
    calcul complet est un TODO de MuJoCo Warp, même en 4.0.0. Paliers 2 et 3 sur
    CPU ; à retester à chaque nouvelle version de MuJoCo Warp.
  - Fusionner la branche GPU (`claude/blissful-jennings-3d0967`) dans `main`
    une fois relue : validée sur sol plat seulement.
  - Kit RunPod prêt (`gpu/runpod/`), utile seulement pour des lots de runs sur
    sol plat.

**Questions pour un relecteur**
1. Le mécanisme proposé en 3.5 (exploration plus bruitée en action normalisée,
   pénalité de butée, lr fort) est-il plausible ? Quelle expérience le
   réfuterait le plus vite ?
2. Pour le robot réel, vaut-il mieux garder l'action non normalisée de A et
   ajouter une saturation des cibles, ou faire marcher l'action normalisée
   (D) ?
3. Faut-il garder la perte de symétrie miroir, alors que le seul run qui
   marche vite (A) n'en a pas ?
4. Les critères M1 à M6 sont-ils suffisants pour juger d'une marche
   transférable ? Que manque-t-il : robustesse aux poussées, variation de
   masse, latence ?

## 7. Coûts Hetzner

Prix hors taxes : CCX43 0,52 $/h, CPX62 0,245 $/h. Un serveur éteint reste
facturé : il faut le supprimer.

| Période | Coût |
|---|---|
| 7-8 nov., jusqu'à 01:40 | ~3,80 $ (CCX43 zero-v2 1 h 27, CPX62 4 h 37, CCX43 2 h 55) |
| Nuit du 8 nov. | prévu ~7 $, plafond 15 $ |
| 8 nov. après-midi (vague H / H-Placo) | ~1,55 $ (CCX43 ~2 h, CPX62 ~2 h) |
| 8-9 nov. soir (palier 2 sur H-Placo) | ~2,75 $ (CCX43 ~5 h 15) |

---

## Journal de la nuit (mis à jour au fil des contrôles)

- **02:10** — A à 6,65 M (1000 pas, 2,66 m, tronc 0,994, KL en hausse) ; B à
  8,6 M. Configs A, B et D committées (`bbeefcb`). Images de la nuit en
  construction sur les deux serveurs et sur le portable.
- **02:14** — A, graine 7, lancé sur le portable (~780 pas/s, fin vers 05:50).
- **02:28** — B fini (10 M, `best_model/`, modèle final et normalisation
  présents). **D lancé** sur le CPX62 (~1 260 pas/s au départ). A à 8,8 M ;
  son `approx_kl` atteint 2,9–3,2 et sa durée moyenne en entraînement est
  retombée à 845 : signe d'instabilité en fin de run, à vérifier sur la mesure
  finale. A, graine 7 : 0,78 M, durée moyenne de 639 pas en entraînement, déjà
  bien au-dessus de B et de robust-genou au même stade. Coût cumulé ~4,60 $.
- **02:40** — Relecture externe reçue. Vérifiée dans le code : la conversion
  action → cible est la même dans les deux cadres, et les coudes sont déjà
  dans la zone de butée pénalisée à la pose debout. Le mécanisme de §3.5 est
  réfuté, et §3.5 et §4 sont réécrits. La piste 3 est replacée dans le cadre
  de A (I) ; E (pénalités du pilote 6) passe en tête de la vague 2. I est
  programmé pour démarrer automatiquement sur le CCX43 dès la fin de A.
- **03:00** — A fini et rapatrié (`best_model` à 6,15 M, modèle final).
  Mesure par checkpoint : 5/6 au mieux, le modèle final dégradé (2/6), et un
  bras levé en permanence (vidéo). D à 1,85 M (durée moyenne de 765 pas en
  entraînement, à ~1 120 pas/s, fin vers 04:55) ; I à 1,43 M (878 pas, fin
  vers 04:30) ; A, graine 7, à 2,16 M (803 pas). Coût cumulé ~5,40 $.
- **03:06** — Vague 2 préparée et testée (2 048 pas chacune, sans erreur) :
  E (`5f763f2`), F et G (`55f0723`), chacune à une variable de A. Les
  enchaînements sont automatiques : E démarre sur le CCX43 à la fin de I, F sur
  le CPX62 à la fin de D, G sur le portable à la fin de A graine 7. Fins
  prévues : E ~06:15, F ~07:30, G ~09:30 (portable, sans coût). Coût estimé
  à la suppression des serveurs : ~8 $.
- **03:16** — À 3 M de pas, **les trois runs de la vague 1 marchent déjà** (5
  épisodes, sans conclusion avant 8 M) :
  - D : 1000 pas, 2,15 m, 100 % sans chute, latéral 0,16 m, cap −22°, appui
    40/41, tronc 0,988 ;
  - I : 1000 pas, 2,85 m, 100 %, latéral 0,48 m, cap −10°, appui 45/42, tronc
    0,995 ;
  - A graine 7 : 839 pas, 1,81 m, 80 %, cap −6°, appui 45/43, tronc 0,971.

  Vidéos de profil à 3 M (`exp-d-3M-profil.mp4`, `exp-i-3M-profil.mp4`) :
  **D garde les bras le long du corps**. Ses cibles bornées l'empêchent sans
  doute de lever un bras au-dessus de la tête comme A. I marche genoux
  fléchis, comme la référence Placo, avec les bras tenus vers l'avant. Coût
  cumulé ~5,40 $.
- **03:36** — D à 4,5 M (1000 pas, 2,60 m, latéral 0,58 m, cap −26°, appui
  43/41, tronc 0,992) ; I à 5 M (1000 pas, 2,65 m, **latéral −0,91 m**, cap
  −11°, appui 41/44, tronc 0,996) ; A graine 7 à 3,5 M (1000 pas, 2,34 m,
  100 %, latéral 0,51 m, appui 39/52, tronc 0,992). Les trois tiennent les
  10 s. **La dérive latérale reste le défaut commun**, ainsi que le cap pour D.
  Pas de plantage ; `approx_kl` de D et de I entre 0,26 et 0,29, à
  surveiller.
- **03:56** — D à 5,9 M (1000 pas, 2,64 m, latéral 0,93 m, cap −15°, appui
  45/44, tronc 0,993) ; I à 7,2 M (1000 pas, 2,61 m, **latéral −1,58 m**, cap
  −13°, appui 39/46, tronc 0,997), dont le **KL monte à 1,4–1,7**, même schéma
  que A avant sa dégradation ; A graine 7 à 5 M (875 pas, 80 %, latéral
  0,74 m, appui 46/46).

  **Observation** : dans les trois runs, la dérive latérale **augmente** au
  fil de l'entraînement (I : 0,48 → 0,91 → 1,58 m ; D : 0,16 → 0,58 →
  0,93 m), alors que la vitesse vers l'avant reste à la cible. Une dérive qui
  croît de façon systématique ressemble plus à un défaut de récompense qu'à du
  bruit. Le terme `lateral_weight: 0.5 × |v_lat|` coûte au plus ~0,05 par pas
  à 0,1 m/s de côté, contre ~1,5 pour le suivi de vitesse : marcher en crabe
  ne coûte presque rien. **Expérience candidate pour plus tard**, hors du plan
  de la nuit et à décider avec l'équipe : A avec `lateral_weight` relevé,
  ou un suivi de vitesse latérale nulle en cloche, à une variable.
- **04:16** — D à 7,2 M (1000 pas, 2,37 m, latéral 0,82 m, cap −6°, appui
  48/41 ; KL ~0,55) ; I à 9,2 M (1000 pas, 2,63 m, latéral −1,33 m, cap −10°,
  appui 40/46, tronc 0,997 ; **KL jusqu'à 4**) ; A graine 7 à 6 M (850 pas,
  80 %, latéral 0,55 m, cap −12°, appui 50/40). Le KL monte dans les trois
  runs du cadre A, d'autant plus vite que le run est avancé. Fin de I vers
  04:25, puis démarrage automatique de E. Coût ~5,90 $.
- **04:36** — I fini et rapatrié ; mesure ci-dessus (§4.1). **E a démarré
  automatiquement** sur le CCX43 (1,2 M de pas à 04:36, durée moyenne de 641
  pas en entraînement, ~1 810 pas/s, fin vers 06:05). D à 8,6 M (KL ~0,6–0,7),
  A graine 7 à 7,1 M (KL 1,3). Coût cumulé ~6,10 $.
- **04:57** — D fini, mesuré (§4.1) et vidéos faites. **F (A + miroir) a
  démarré automatiquement** sur le CPX62 (fin vers ~07:15). E à 3,5 M :
  **1000 pas, 100 % sans chute, 2,37 m**, latéral 1,21 m, cap +8°, appui
  52/40, tronc 0,989. Avec les pénalités de douceur du pilote 6, **A marche
  toujours**. L'hypothèse « ces pénalités bloquaient les pilotes » est donc
  en mauvaise posture, à confirmer à 8 M. A graine 7 à 8 M : 1000 pas, 100 %,
  2,42 m, **latéral 0,20 m**, cap −14°, appui 47/46, tronc 0,991 : **A est
  reproductible** sur une deuxième graine. Coût cumulé ~6,60 $.
- **05:16** — E à 5,9 M (1000 pas, 100 %, **1,81 m soit 0,18 m/s**, latéral
  1,33 m, cap +6°, appui 49/41, tronc 0,995) : il marche, mais **plus
  lentement** que A, sous la vitesse cible ; les pénalités lourdes réduisent
  l'amplitude sans empêcher la marche. F à 1,25 M (1000 pas, 0,52 m, appui
  31/22). A graine 7 à 8,9 M (1000 pas, 2,41 m, latéral −0,04 m, cap −14°,
  appui 47/44). Prévu : supprimer le CCX43 à la fin de E (~06:05), rien d'autre
  n'y est planifié avant 08:00. Coût cumulé ~6,60 $.
- **05:36** — E à 8 M : **1000 pas, 100 %, 2,49 m**, latéral 1,12 m, cap +3°,
  appui 54/39, tronc 0,991. À 8 M, E marche : **les pénalités de douceur du
  pilote 6 n'empêchent pas la marche dans le cadre de A.** Ce n'est pas ce qui
  bloquait les pilotes. Restent suspects : le miroir (F) et les réglages PPO
  du nouveau cadre (lr 3e-4 linéaire, 5 époques, clip 0,2, entropie 0,005),
  pas encore testés un par un. F à 2,5 M : 872 pas, 60 %, 2,05 m, appui
  43/37. A graine 7 à 9,96 M, presque fini. Coût cumulé ~6,90 $.
- **05:42** — A graine 7 fini et mesuré (§4.1) : **6/6 à 8 M, 9 M et au
  meilleur modèle**. G (A + `target_kl` 0,03) démarré sur le portable.
- **05:58** — E fini, logs vérifiés fichier par fichier (141 sur 141) puis
  **CCX43 supprimé** à 05:57 (7 h 09 allumé, 8 h dues, ~4,16 $). F à 4 M :
  **1000 pas, 100 %, 2,82 m, latéral −0,16 m, appui 46/45**, tronc 0,993. G à
  1,1 M sur le portable, KL ~0,03–0,04 comme voulu. Coût cumulé ~7,40 $ ; seul
  le CPX62 reste, jusqu'à la fin de F (~07:25).
- **06:16** — F à 5,2 M : 1000 pas, 100 %, 2,59 m, **latéral −0,02 m**, cap
  −7°, appui 46/45, tronc 0,997 : la marche la plus droite et la plus
  symétrique de la nuit à ce stade. G à 2,3 M (portable, ~1 020 pas/s grâce aux
  époques écourtées, fin vers 08:25) : 1000 pas, 0,88 m, appui 17/44, KL tenu à
  ~0,04. Coût cumulé ~7,40 $.
- **06:36** — F à 6,6 M : 1000 pas, 100 %, 2,47 m, latéral −0,21 m, **cap
  +0,7°**, appui 45/47, tronc 0,997. G à 3,8 M : 1000 pas, 100 %, 2,52 m,
  latéral 0,49 m, cap +12°, appui 50/39, KL ~0,045. Fin de F vers 07:20, fin de
  G vers 08:05. Coût cumulé ~7,60 $.
- **06:56** — F à 7,9 M : 1000 pas, 100 %, 2,18 m, **latéral −0,04 m**, cap
  +8°, appui 47/44, tronc 0,994. G à 5,3 M : 1000 pas, 100 %, 2,28 m, latéral
  0,67 m, cap +4°, appui 45/40, KL ~0,02. Fin de F vers 07:28, puis suppression
  du CPX62. Coût cumulé ~7,60 $.
- **07:16** — F à 9,2 M : 1000 pas, 100 %, 2,26 m, latéral −0,13 m, cap
  +0,4°, appui 47/44, tronc 0,987. G à 6,8 M : 1000 pas, 100 %, 2,98 m, latéral
  −0,24 m, cap +1°, **appui 46/47**, tronc 0,990, KL ~0,04. Coût cumulé
  ~7,60 $.
- **07:36** — F fini, mesuré : **6/6** (§4.1). Logs du CPX62 vérifiés
  (248 fichiers sur 248), puis **CPX62 supprimé à 07:37. Plus aucun serveur.**
  Coût final de Hetzner : **~7,90 $** (zero-v2 1,04 + CCX43 8 h × 0,52 = 4,16 +
  CPX62 11 h × 0,245 = 2,70), sous le plafond de 15 $. G à 8,2 M sur le
  portable, fin vers 08:00.
- **08:02** — G fini sur le portable : **6/6 au modèle final**, KL au plus
  0,10, mais le bras est levé (même graine que A). Synthèse écrite (§4.2). Fin
  de la nuit : **plus aucun serveur, plus aucun run en cours**, surveillance
  arrêtée. Coût Hetzner total de la nuit et de la veille : ~7,90 $.
