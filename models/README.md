# Modèles entraînés

Onze politiques Poppy, une par entraînement du 8 avril 2026. Chaque dossier
contient une paire **indissociable** :

```
models/<date>/
├── best_model.zip      la politique
└── vec_normalize.pkl   les statistiques de normalisation des observations
```

Charger l'une sans l'autre ne lève **aucune erreur** : les formes et les types
concordent, les observations sont simplement recentrées sur la mauvaise moyenne, et
la politique produit des actions absurdes. `scripts/evaluate.py` trouve le `.pkl` tout
seul tant que les deux restent côte à côte.

## Classement

Mesuré le 18 septembre 2026 : 10 épisodes déterministes par modèle, graine 42,
`configs/poppy_robust.yaml`, **randomisation du sol désactivée**. Reproduire une
ligne :

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_21-29-14/best_model.zip --episodes 10
```

| Modèle | Récompense | Pas | Avance (m) | Dérive (m) | Vitesse (m/s) | Verticalité | Debout |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2026-04-08_23-00-52` | **2008** | 1000 | 0,24 | **−5,70** | 0,024 | 0,998 | 100 % |
| `2026-04-08_21-29-14` | 1263 | 319 | **1,44** | 0,20 | **0,455** | 0,963 | 0 % |
| `2026-04-08_20-41-01` | 911 | 241 | 0,94 | −0,11 | 0,393 | 0,961 | 0 % |
| `2026-04-08_22-42-32` | 502 | 209 | 0,26 | 0,89 | 0,122 | 0,953 | 0 % |
| `2026-04-08_20-22-11` | 501 | 174 | 0,40 | −0,20 | 0,234 | 0,933 | 0 % |
| `2026-04-08_18-43-51` | 208 | 74 | 0,29 | 0,10 | 0,399 | 0,516 | 0 % |
| `2026-04-08_18-30-32` | 207 | 76 | 0,20 | 0,28 | 0,268 | 0,832 | 0 % |
| `2026-04-08_17-58-25` | 172 | 51 | 0,21 | 0,04 | 0,411 | 0,657 | 0 % |
| `2026-04-08_18-21-57` | 148 | 48 | 0,18 | 0,00 | 0,385 | 0,706 | 0 % |
| `2026-04-08_16-45-32` | 119 | 66 | 0,06 | 0,28 | 0,094 | 0,720 | 0 % |
| `2026-04-08_19-49-15` | 61 | 74 | −0,12 | 0,32 | −0,157 | 0,873 | 0 % |
| *politique aléatoire* | *8* | *106* | *0,13* | *0,03* | *0,110* | *0,918* | *0 %* |

La dernière ligne est le plancher : `scripts/evaluate.py --random`. Sans elle, on ne
sait pas ce que « 2008 » vaut.

## Lequel prendre ? Ça dépend de la question

**Il n'y a pas un meilleur modèle, il y en a deux, et ils ne répondent pas à la même
question.** C'est le constat le plus important de ce tableau.

### `2026-04-08_23-00-52` — la plus haute récompense, et il ne marche pas

2008 de récompense, seul à tenir les 1000 pas sans tomber. Et pourtant : **0,24 m
parcourus en 10 secondes**, soit 2,4 cm/s, pendant qu'il dérive de **5,70 m sur le
côté**. Il avance vingt-quatre fois moins qu'il ne glisse latéralement.

La décomposition explique tout. Sur un épisode il encaisse `healthy` ≈ 1000 et
`uprightness` ≈ 998 — deux termes qui ne récompensent que le fait de rester debout —
plus `gait` ≈ 283, et il paie `lateral_vel` ≈ −568. Rester planté et glisser lui
rapporte 2281 pour 568 de pénalité. Il a trouvé que le plus rentable était de ne pas
marcher. C'est un cas d'école de *reward hacking* : la politique optimise la fonction
de récompense, pas l'intention derrière.

Utile pour : une démonstration où le robot doit rester debout, ou un test du pont ROS
qui a besoin d'une politique qui ne tombe pas.

### `2026-04-08_21-29-14` — la meilleure démarche

1,44 m parcourus à 0,455 m/s, presque en ligne droite (0,20 m de dérive), verticalité
0,963. Il marche vraiment. Puis il tombe, au pas 319.

Sa récompense totale est plus basse uniquement parce que son épisode est trois fois
plus court. **Par pas, il rapporte 3,96 contre 2,01** au précédent — il est
deux fois meilleur à chaque instant.

Utile pour : tout ce qui concerne la marche. C'est de celui-là qu'il faut repartir.

### Ce que ça dit de la fonction de récompense

Un modèle qui ne marche pas gagne le concours. Les huit termes actuels ne
départagent pas « rester debout » de « avancer » ; `healthy` et `uprightness` pèsent
2000 par épisode complet quand `capped_vel` en rapporte 28. Rééquilibrer ces poids
est probablement le premier vrai levier sur la marche — et c'est une décision
d'équipe, parce qu'elle rend ces onze modèles incomparables aux suivants.

Ne jamais classer sur la récompense totale seule. `scripts/evaluate.py` affiche la
décomposition des huit termes exactement pour ça.

## `_humanoid-v5-baseline/`

Ce n'est **pas** une politique Poppy. C'est un entraînement sur `Humanoid-v5`,
l'humanoïde générique de Gymnasium, servant à valider le pipeline
(`train_poppy.py --baseline`). Son observation fait 348 dimensions contre 63 : la
charger dans l'environnement Poppy échoue sur
`spaces must have the same shape: (348,) != (63,)`. Le préfixe `_` est là pour qu'on
ne la confonde pas.

## Avertissement au chargement

Les modèles ont été entraînés avec une version plus ancienne de Stable-Baselines3.
Au chargement :

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
