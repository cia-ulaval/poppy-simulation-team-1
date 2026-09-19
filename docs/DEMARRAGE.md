# Démarrage — votre première heure sur le projet

Un parcours guidé, à faire dans l'ordre. À la fin vous aurez construit l'image, vu le
robot en 3D, regardé la meilleure politique marcher, lu ses chiffres, entraîné votre
propre modèle et comparé les deux.

**Aucune connaissance du projet n'est supposée.** Chaque étape dit quoi taper, ce que
vous devez voir, et ce que ça veut dire.

Référence complète des commandes : [`TECHNIQUE.md`](TECHNIQUE.md).
Comment c'est construit : [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Étape 0 — Docker tourne ?

```bash
docker info --format "{{.ServerVersion}}"
```

Un numéro de version s'affiche → c'est bon. Une erreur de connexion → Docker Desktop
n'est pas démarré, et rien d'autre ne marchera.

## Étape 1 — Construire l'image

```bash
docker compose build train
```

**Dix à vingt minutes la première fois** (PyTorch pèse lourd), quelques secondes
ensuite. Les couches sont mises en cache.

Vérifier que tout est en place :

```bash
docker compose --profile dev run --rm dev python -m pytest
```

> **Attendu : `16 passed, 1 xfailed`.** Le `xfailed` n'est **pas** une panne : c'est un
> test volontairement en échec attendu, qui documente une incohérence connue. Il est
> expliqué dans [`TECHNIQUE.md` §8](TECHNIQUE.md).

## Étape 2 — Voir le robot

Deux façons. La première ne demande rien de plus, la seconde est plus agréable.

**Sans rien installer** — une image du robot dans sa pose de départ :

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --random --episodes 1 --video logs/robot.mp4 --camera cote
```

Ouvrez `logs/robot.mp4`. C'est une politique aléatoire : le robot s'écroule presque
aussitôt. C'est normal, et c'est justement le point de comparaison de l'étape 5.

**Avec le viewer interactif** — vous tournez autour à la souris, vous zoomez. Il faut
un Python installé sur la machine, une fois par poste :

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

Puis :

```powershell
.\.venv\Scripts\python.exe scripts/viewer.py
```

Une fenêtre MuJoCo s'ouvre avec Poppy. Clic gauche glissé pour tourner, molette pour
zoomer. **Aucun modèle entraîné nécessaire.** Regardez les articulations : il y en a
25, une par moteur.

> Sous Windows, `python` tout court ne marche pas tant que rien n'est installé — la
> commande est interceptée par un raccourci Microsoft Store qui ne fait rien. D'où le
> chemin explicite. Sous bash, c'est `.venv/bin/python`.

## Étape 3 — Regarder le meilleur modèle marcher

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_23-00-52/best_model.zip --episodes 1 --video logs/marche.mp4 --camera suivi
```

Ouvrez `logs/marche.mp4`. Le robot marche pendant 10 secondes sans tomber. La caméra
le suit — sans ça, il sortirait du cadre au bout de deux secondes.

Essayez les autres angles, `--camera` accepte `suivi`, `cote`, `face`, `dessus`,
`large`. **`dessus` est le plus instructif** : on voit la trajectoire au sol.

Si vous avez fait l'installation de l'étape 2, en interactif :

```powershell
.\.venv\Scripts\python.exe scripts/visualize.py models/2026-04-08_23-00-52/best_model.zip --episodes 2
```

## Étape 4 — Lire les chiffres

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_23-00-52/best_model.zip --episodes 5
```

Vous devez voir quelque chose comme :

```
Récompense totale                4550.5
Durée (pas)                        1000
Déplacement net (m)                5.69
  vers l avant (m)                 5.63
  derive laterale (m)             -0.84
Épisodes sans chute                100 %
```

**Les trois lignes qui comptent**, dans cet ordre :

1. **Épisodes sans chute** — 100 %, il ne tombe jamais sur 1000 pas (10 s simulées).
2. **Vers l'avant** — 5,63 m. C'est mesuré dans le repère du robot : avancer, c'est
   aller là où il regarde. Une valeur négative voudrait dire qu'il recule.
3. **Dérive latérale** — 0,86 m seulement, contre 5,63 m d'avance. Il marche droit.

La récompense vient **après**, pas avant. C'est un nombre arbitraire qui ne veut rien
dire tout seul — l'étape suivante explique à quoi le comparer.

En dessous, la décomposition des huit termes montre *d'où* vient la récompense :
`healthy_reward` pour être resté debout, `capped_vel` pour avoir avancé, `gait_reward`
pour avoir alterné les appuis, et les pénalités en négatif.

## Étape 5 — À quoi comparer : le plancher

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --random --episodes 5
```

Une politique qui agite les membres au hasard. Résultat : **récompense négative
(≈ −74), chute systématique en une seconde**.

Voilà le repère. « 4550 » ne voulait rien dire ; « 4550 contre −74 » veut dire quelque
chose. **Prenez ce réflexe** : tout nouveau modèle se compare d'abord au hasard.

## Étape 6 — Voir un modèle raté, et comprendre pourquoi

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model models/2026-04-08_21-29-14/best_model.zip --episodes 5 --video logs/rate.mp4 --camera dessus
```

Le script vous dit lui-même ce qui cloche :

```
Le robot se déplace de 1.43 m, mais de CÔTÉ : 1.03 m de dérive latérale
contre 0.33 m vers l avant. C est un pas chassé, pas une marche.
```

Regardez `logs/rate.mp4` vue de dessus : il part en biais, épaules dans l'axe du
trajet, et il tombe au bout de 3 secondes.

C'est le cœur du sujet du projet. [`models/README.md`](../models/README.md) classe les
onze modèles et raconte pourquoi celui-ci arrivait deuxième jusqu'à ce qu'on
découvre que la récompense mesurait le mauvais axe.

## Étape 7 — Entraîner votre propre modèle

Dans un premier terminal :

```bash
docker compose --profile train run --rm train python scripts/train_poppy.py --timesteps 500000 --n-envs 8 --seed 1 --log-dir logs/moi
```

**Environ 20 minutes.** Dans un second terminal, pendant que ça tourne :

```bash
docker compose --profile train up tensorboard
```

Puis <http://localhost:6006>. La courbe `rollout/ep_rew_mean` doit monter, et
`rollout/ep_len_mean` aussi — le robot apprend d'abord à ne pas tomber, avancer vient
après.

**Vous pouvez arrêter quand vous voulez avec Ctrl-C.** `best_model/` est sauvegardé
toutes les 10 000 étapes, il y a toujours un modèle utilisable dans
`logs/moi/<date>/best_model/`.

Quelques repères sur une machine à 12 cœurs :

| Pas | Durée | Ce qu'on voit |
|---|---|---|
| 500 000 | ~20 min | la courbe monte, le robot tient debout quelques secondes |
| 2 000 000 | ~1 h 30 | premiers pas |
| 10 000 000 | 5 à 7 h | le défaut du YAML, à lancer le soir |

`POPPY_N_ENVS=12 docker compose --profile train up` utilise les 12 cœurs. **Ne montez
pas au-delà de ce que la machine a** : `configs/poppy_robust.yaml` demande 64
environnements, calibré pour une station à 32 cœurs, et `SubprocVecEnv` meurt sur un
`BrokenPipeError` si vous en demandez trop.

## Étape 8 — Comparer votre modèle

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --model logs/moi/<date>/best_model/best_model.zip --episodes 5
```

Remplacez `<date>` par le dossier créé à l'étape 7.

Comparez aux trois repères que vous connaissez maintenant :

| | Récompense | Avance | Sans chute |
|---|---:|---:|---:|
| Politique aléatoire | −74 | −0,05 m | 0 % |
| **Votre modèle après 500 k** | ~470 | ~+0,4 m | 0 % |
| Le meilleur du dépôt (10 M) | 4565 | +5,63 m | 100 % |

Ces chiffres du milieu sont réels, mesurés sur un run de 580 000 pas. Vous serez
loin du meilleur — c'est vingt fois moins d'entraînement — et le robot tombera
encore au bout d'une seconde ou deux. Deux choses doivent être vraies :

1. **Nettement au-dessus du hasard.** ~470 contre −74.
2. **L'avance dépasse la dérive.** ~0,4 m d'avance pour 0,07 m de dérive : il part
   droit devant. C'est le signe que la récompense mesure le bon axe — dix des onze
   modèles du dépôt, entraînés avant la correction, font l'inverse.

Ce qui manquera encore à 500 k, c'est `gait_reward` : autour de 20 sur un maximum de
300, l'alternance des appuis n'est pas installée. C'est ce qui vient ensuite, et
c'est la partie lente.

## Étape 9 — Et après

- Toutes les commandes, en un tableau : [`TECHNIQUE.md` §7](TECHNIQUE.md)
- Ce qui est cassé et connu, à lire avant de s'étonner : [`TECHNIQUE.md` §8](TECHNIQUE.md)
- Le contrat observation → action, à lire avant de toucher à l'environnement :
  [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Les règles de contribution : [`../CONTRIBUTING.md`](../CONTRIBUTING.md)

Chaque script explique ses options lui-même :

```bash
docker compose --profile train run --rm train python scripts/train_poppy.py --help
```

```bash
docker compose --profile eval run --rm eval python scripts/evaluate.py --help
```

## Si ça coince

| Symptôme | Cause |
|---|---|
| `failed to connect to the docker API` | Docker Desktop n'est pas démarré |
| `no such service: train` | profil oublié : `--profile train` |
| `Python est introuvable` | voir l'étape 2 — utiliser le chemin explicite de `.venv` |
| `BrokenPipeError` pendant l'entraînement | trop d'environnements, baisser `--n-envs` |
| `spaces must have the same shape` | modèle incompatible avec l'environnement, voir [`models/README.md`](../models/README.md) |
| `Could not deserialize object lr_schedule` | avertissement sans conséquence, les poids se chargent |
| La commande colle deux lignes ensemble | la doc n'utilise pas d'antislash ; copiez la ligne entière |

Le dépannage détaillé est dans [`DOCKER.md` §10](DOCKER.md).
