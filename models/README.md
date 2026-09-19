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

`2026-04-08_23-00-52` est celui que l'équipe avait poussé sur la branche
`model-for-presentation`, par le commit `7583141 add final best model` du
9 avril 2026. Le fichier ici est **identique à l'octet près** à celui de ce commit
(`e05c62a`), sa normalisation aussi (`42f560f`). Rien n'a changé en chemin.

## Classement

Mesuré le 18 septembre 2026 : 10 épisodes déterministes par modèle, graine 42,
`configs/poppy_robust.yaml`, **randomisation du sol désactivée**. Reproduire une ligne :

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_21-29-14/best_model.zip --episodes 10
```

Classé sur le **déplacement réel**, pas sur la récompense. La raison est plus bas.

| Modèle | Déplacé (m) | v en x | v en y | Pas | Debout | Écart cap | Appuis alternés | Récompense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `2026-04-08_23-00-52` | **5,70** | +0,02 | **−0,57** | 1000 | **100 %** | **98°** | 92 % | 2008 |
| `2026-04-08_21-29-14` | 1,46 | **+0,46** | +0,06 | 319 | 0 % | 16° | 86 % | 1263 |
| `2026-04-08_20-41-01` | 0,97 | +0,39 | −0,05 | 241 | 0 % | 28° | 88 % | 911 |
| `2026-04-08_22-42-32` | 0,97 | +0,12 | +0,42 | 209 | 0 % | 90° | 78 % | 502 |
| `2026-04-08_20-22-11` | 0,45 | +0,23 | −0,12 | 174 | 0 % | 24° | 32 % | 501 |
| `2026-04-08_18-30-32` | 0,35 | +0,27 | +0,37 | 76 | 0 % | 61° | 89 % | 207 |
| `2026-04-08_19-49-15` | 0,34 | −0,16 | +0,44 | 74 | 0 % | 57° | 51 % | 61 |
| *politique aléatoire* | *0,33* | *+0,11* | *+0,02* | *106* | *0 %* | *61°* | *55 %* | *8* |
| `2026-04-08_18-43-51` | 0,32 | +0,40 | +0,14 | 74 | 0 % | 11° | 82 % | 208 |
| `2026-04-08_16-45-32` | 0,29 | +0,09 | +0,42 | 66 | 0 % | 50° | 59 % | 119 |
| `2026-04-08_17-58-25` | 0,21 | +0,41 | +0,08 | 51 | 0 % | 17° | 84 % | 172 |
| `2026-04-08_18-21-57` | 0,19 | +0,39 | +0,01 | 48 | 0 % | 34° | 79 % | 148 |

- **Déplacé** : distance entre le point de départ et le point d'arrivée, toutes
  directions confondues.
- **Écart cap** : angle entre la direction du torse et la direction du déplacement.
  0° = le robot va là où il regarde. 90° = il se déplace de côté.
- **Appuis alternés** : part des pas où un seul pied touche le sol. Un chiffre élevé
  signale une vraie démarche, pas un glissement.
- *La politique aléatoire est le plancher* : `scripts/evaluate.py --random`. Sans elle,
  « 2008 » ne veut rien dire.
- Ne pas lire une vitesse comme une performance sur les modèles qui tombent : une
  politique qui chute au bout de 0,5 s peut afficher une vitesse instantanée élevée
  sans aller nulle part. C'est la colonne **Déplacé** qui compte.

## Ce que le tableau raconte

### `2026-04-08_23-00-52` marche — mais de côté

**5,70 m en 10 secondes, à 0,57 m/s, sans jamais tomber, avec 92 % d'appuis alternés.**
C'est une vraie démarche, en ligne droite : le chemin parcouru (5,63 m) et le
déplacement net (5,70 m) coïncident, le robot ne titube pas.

Le problème est ailleurs : **l'écart entre son cap et sa direction est de 98°.** Il
avance perpendiculairement à son torse. C'est un pas chassé, pas une marche en avant.

La récompense ne mesure l'avance que sur l'axe **x**. Ce robot fait +0,02 m/s en x et
−0,57 m/s en y. Elle le crédite donc de 28 points d'avance sur un épisode entier, lui
en facture 568 de dérive latérale, et compense avec `healthy` (1000) et `uprightness`
(998). Résultat : la meilleure récompense du dépôt, pour un déplacement que la
fonction ne voit pas.

> **Attention à ce qu'on en dit.** Ce n'est pas un modèle qui ne marche pas : c'est un
> modèle qui marche dans une direction que la récompense ne mesure pas. La nuance
> change complètement la lecture, et ce que ça dit du travail restant.

Les trois commits qui l'ont produit s'appellent `reverse x`, `update best model`,
`add final best model`. Quelqu'un se battait déjà avec cette histoire d'axe.

### `2026-04-08_21-29-14` marche droit

1,46 m à 0,46 m/s, cap et déplacement à 16° l'un de l'autre : il va là où il regarde.
Puis il tombe, au pas 319. C'est la seule politique du lot qui fasse une marche en
avant au sens courant du terme.

### Lequel prendre

| Pour | Prendre | Pourquoi |
|---|---|---|
| Montrer un robot qui se déplace et ne tombe pas | `23-00-52` | 10 s debout, 5,70 m, jamais au sol |
| Travailler la marche en avant | `21-29-14` | le seul avec cap et déplacement alignés |
| Tester le pont ROS | `23-00-52` | il ne tombe pas, la séquence dure |

Vues utiles : `--camera dessus` sur `23-00-52` montre la trajectoire au sol d'un coup
d'œil, `--camera face` montre qu'il se déplace latéralement.

### Ce que ça dit de la fonction de récompense

Le terme d'avance est projeté sur l'axe **x du monde**, pas sur le cap du robot. Une
politique peut donc marcher parfaitement sans être récompensée, si elle n'est pas
orientée comme le repère. Deux pistes, à trancher en équipe :

1. Projeter l'avance sur le cap du robot plutôt que sur l'axe x du monde.
2. Récompenser explicitement l'alignement entre le cap et le déplacement.

Les deux rendent les onze modèles incomparables aux suivants. C'est pour ça que ce
n'est pas une correction à glisser au passage.

## `_humanoid-v5-baseline/`

Ce n'est **pas** une politique Poppy. C'est un entraînement sur `Humanoid-v5`,
l'humanoïde générique de Gymnasium, servant à valider le pipeline
(`train_poppy.py --baseline`). Son observation fait 348 dimensions contre 63 : la
charger dans l'environnement Poppy échoue sur
`spaces must have the same shape: (348,) != (63,)`. Le préfixe `_` est là pour qu'on ne
la confonde pas.

## Les modèles plus anciens, et pourquoi ils ne sont pas là

L'historique contient d'autres modèles Poppy, de mars 2026 : deux runs `2026-03-29_*`
(dont un de 10,2 M de pas) et quatre `2026-03-14_*`. Ils avaient été retirés de la
branche **avant** le nettoyage de septembre, pas par lui.

Inutile de les ressortir : leur **observation fait 61 dimensions**. Les deux capteurs
de contact au pied ont été ajoutés après, portant l'observation à 63. Ils ne se
chargent pas dans l'environnement actuel, et les faire tourner demanderait de revenir
à l'ancien environnement — donc de ne plus pouvoir les comparer aux onze ci-dessus.
Même chose pour `configs/models/best_model.zip`, qui est un Humanoid-v5
(348 dimensions).

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
