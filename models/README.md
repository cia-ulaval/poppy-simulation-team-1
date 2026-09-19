# Modèles entraînés

Onze politiques Poppy, une par entraînement du 8 avril 2026. Chaque dossier contient
une paire **indissociable** :

```
models/<date>/
├── best_model.zip      la politique
└── vec_normalize.pkl   les statistiques de normalisation des observations
```

Charger l'une sans l'autre ne lève **aucune erreur** : les formes et les types
concordent, les observations sont simplement recentrées sur la mauvaise moyenne, et la
politique produit des actions absurdes. `scripts/evaluate.py` trouve le `.pkl` tout
seul tant que les deux restent côte à côte.

## Lequel est « le » modèle du projet

`2026-04-08_23-00-52`, celui que l'équipe avait poussé sur `model-for-presentation`
par le commit `7583141 add final best model` du 9 avril 2026. Le fichier ici est
**identique à l'octet près** (blob `e05c62a`), sa normalisation aussi (`42f560f`).

**C'est bien le bon modèle, et il marche.** 5,63 m vers l'avant en 10 s à 0,56 m/s,
sans jamais tomber, épaules perpendiculaires au trajet, 92 % d'appuis alternés.

## Classement

Mesuré le 19 septembre 2026 : 10 épisodes déterministes par modèle, graine 42,
`configs/poppy_robust.yaml`, randomisation du sol désactivée, **avec la récompense
corrigée** (voir plus bas). Reproduire une ligne :

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_23-00-52/best_model.zip --episodes 10
```

**Avance** et **dérive** sont mesurées dans le repère du robot : avancer, c'est aller
là où il regarde. Une avance négative signifie qu'il recule.

| Modèle | Récompense | Avance (m) | Dérive (m) | Pas | Debout | Verticalité |
|---|---:|---:|---:|---:|---:|---:|
| `2026-04-08_23-00-52` | **4565** | **+5,63** | −0,86 | 1000 | **100 %** | 0,998 |
| `2026-04-08_21-29-14` | 703 | +0,25 | +1,08 | 319 | 0 % | 0,963 |
| `2026-04-08_20-41-01` | 489 | +0,04 | +0,67 | 241 | 0 % | 0,961 |
| `2026-04-08_20-22-11` | 377 | +0,08 | +0,40 | 174 | 0 % | 0,933 |
| `2026-04-08_18-21-57` | 115 | +0,12 | +0,17 | 48 | 0 % | 0,706 |
| `2026-04-08_17-58-25` | 104 | +0,08 | +0,22 | 51 | 0 % | 0,657 |
| `2026-04-08_18-43-51` | 41 | −0,04 | +0,33 | 74 | 0 % | 0,516 |
| `2026-04-08_16-45-32` | 36 | −0,15 | +0,16 | 66 | 0 % | 0,720 |
| `2026-04-08_19-49-15` | 4 | −0,27 | +0,17 | 74 | 0 % | 0,873 |
| `2026-04-08_18-30-32` | −5 | −0,31 | +0,16 | 76 | 0 % | 0,832 |
| `2026-04-08_22-42-32` | −21 | **−0,94** | +0,01 | 209 | 0 % | 0,953 |
| *politique aléatoire* | *−74* | *−0,05* | *+0,14* | *106* | *0 %* | *0,918* |

Ce que le tableau dit :

- **Un seul modèle marche** : `23-00-52`, avec 22 fois plus d'avance que le suivant.
- `21-29-14`, `20-41-01` et `20-22-11` **font du pas chassé** : leur dérive dépasse
  leur avance. Ils se déplacent, mais de côté, et ils tombent.
- `22-42-32` **recule** de 94 cm, sur 209 pas. C'est cohérent : rien ne l'en
  empêchait avant la correction.
- Cinq modèles font **moins bien que la politique aléatoire** ou à peine mieux.
- Le plancher aléatoire (`--random`) est négatif : agiter les membres au hasard coûte
  plus que ça ne rapporte.

## Pourquoi ce classement a changé le 19 septembre

**La récompense mesurait l'avance sur le mauvais axe.**

L'URDF de Poppy place son axe sagittal sur **y** : les pieds sont écartés de 13,2 cm
selon **x**, et plier un genou déplace le pied selon **y**. L'environnement ayant été
écrit sur le modèle de `Humanoid-v5` — qui, lui, regarde vers +x — `forward_reward`
mesurait l'avance sur **x**, c'est-à-dire sur l'axe **gauche-droite** du robot, et
`lateral_cost` pénalisait **y**, son axe de marche.

**Les deux axes étaient croisés.** La récompense récompensait le pas chassé et
facturait la vraie marche au titre de la dérive.

Effet sur `23-00-52`, le même modèle, avant et après correction :

| | Avant | Après |
|---|---:|---:|
| Récompense | 2024 | **4565** |
| `capped_vel` (avance créditée) | 28 | **485** |
| `lateral_vel` (dérive facturée) | −568 | **−84** |
| « Vitesse avant » rapportée | 0,029 m/s | **0,563 m/s** |

Le modèle n'a pas bougé d'un octet : seule la mesure était fausse. Et le classement
était **inversé** — `21-29-14`, qui arrivait second, fait en réalité du pas chassé.

La correction projette la vitesse sur le cap du robot au lieu d'un axe fixe du monde,
ce qui la rend aussi insensible à la randomisation d'orientation initiale (±15°).
`tests/test_heading.py` verrouille cette convention : si le MJCF est un jour
régénéré avec d'autres axes, les tests le diront.

## Lequel prendre

| Pour | Prendre |
|---|---|
| Montrer le projet, tester le pont ROS, repartir pour un entraînement | `23-00-52` — le seul qui marche et ne tombe pas |
| Illustrer ce qu'est un pas chassé | `21-29-14` |
| Illustrer une marche arrière | `22-42-32` |

Vues utiles : `--camera dessus` montre la trajectoire au sol, `--camera cote` la
foulée. La trajectoire de `23-00-52` est une ligne droite ; celle de `21-29-14` part
en biais avec les épaules alignées sur le trajet.

## Ce qui reste à faire

Ces onze modèles ont été entraînés **avec la récompense fausse**. Ils restent
chargeables — l'espace d'observation n'a pas changé — et leurs récompenses ci-dessus
sont mesurées avec la bonne. Mais aucun n'a jamais été *entraîné* avec un signal
d'avance correct.

Le prochain entraînement, lui, le sera. Repartir de `23-00-52` plutôt que de zéro
conserve l'équilibre et la démarche déjà appris, et ne corrige que la direction.

## `_humanoid-v5-baseline/`

Ce n'est **pas** une politique Poppy. C'est un entraînement sur `Humanoid-v5`,
l'humanoïde générique de Gymnasium, servant à valider le pipeline
(`train_poppy.py --baseline`). Son observation fait 348 dimensions contre 63 : la
charger dans l'environnement Poppy échoue sur
`spaces must have the same shape: (348,) != (63,)`. Le préfixe `_` est là pour qu'on
ne la confonde pas.

## Les modèles plus anciens, et pourquoi ils ne sont pas là

L'historique contient d'autres modèles Poppy, de mars 2026 : deux runs `2026-03-29_*`
(dont un de 10,2 M de pas) et quatre `2026-03-14_*`. Ils avaient été retirés de la
branche **avant** le nettoyage de septembre, pas par lui.

Inutile de les ressortir : leur **observation fait 61 dimensions**. Les deux capteurs
de contact au pied ont été ajoutés après, portant l'observation à 63. Ils ne se
chargent pas dans l'environnement actuel. Même chose pour `configs/models/best_model.zip`,
qui est un Humanoid-v5 (348 dimensions).

Ils restent récupérables si besoin :

```bash
git show 36db868b:logs/poppy/2026-03-29_13-08-02/best_model/best_model.zip > ancien.zip
```

## Avertissement au chargement

Les modèles ont été entraînés avec une version plus ancienne de Stable-Baselines3. Au
chargement :

```
UserWarning: Could not deserialize object lr_schedule.
```

Sans conséquence pour l'évaluation : seul le calendrier de taux d'apprentissage, qui
ne sert qu'à poursuivre un entraînement, n'est pas reconstruit. Les poids de la
politique se chargent normalement.

## Pourquoi ces onze-là

Chaque entraînement produisait des dizaines de checkpoints intermédiaires
(`poppy_ppo_<N>_steps.zip`) en plus de son `best_model`, soit 660 Mo au total dans le
dépôt. Le `best_model` est déjà celui qu'`EvalCallback` a retenu comme meilleur du
run : garder les instantanés en plus n'apportait rien. Ils restent récupérables :

```bash
git show archive/avant-clean-2026-09-18:logs/poppy/2026-04-08_18-43-51/poppy_ppo_6600000_steps.zip > checkpoint.zip
```
