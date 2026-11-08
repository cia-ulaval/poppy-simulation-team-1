# Entraînement GPU : mjlab, MuJoCo Warp et rsl_rl

Portage de l'entraînement Poppy sur GPU, **sans changer ce qu'on apprend** :
même MJCF, même contrôle PD, même observation, même récompense, même
randomisation, mêmes réglages PPO que le témoin A
(`configs/exp_a_ancien_cadre_nouvelle_recompense.yaml`). Le chemin CPU
(`scripts/train_poppy.py`) n'est pas touché.

> **État au 8 novembre 2026 : porte de parité passée sur la RTX 3050.** Le témoin A
> porté atteint 5/6 sur M1–M6 (7, 8 et 9 M de pas), comme A sur CPU, en 33 min
> au lieu de 2 à 3 h. Une seule graine (42) ; la RTX 4090 louée n'a pas été essayée.

## En bref

| | CPU (SB3, avant) | GPU (ce dossier) |
|---|---|---|
| Simulation | MuJoCo C 3.13, float64 | MuJoCo Warp 3.11, float32 |
| Environnement | `PoppyHumanoidEnv` | `gpu/poppy_env.py`, constantes **lues** sur `PoppyHumanoidEnv` |
| PPO | Stable-Baselines3 | rsl_rl 5.4.2, réglé et corrigé pour faire la même mise à jour (`gpu/ppo.py`) |
| Normalisation | `VecNormalize` | la même, recopiée en torch (`gpu/vecnormalize.py`) |
| Évaluation | `scripts/evaluate.py` | **la même** : le modèle GPU est exporté en `.zip` SB3 |

## Utilisation

Tout passe par l'image `gpu/Dockerfile` (Linux, CUDA). Sous Windows : Docker
Desktop en WSL 2, pilote NVIDIA récent.

```bash
docker build -f gpu/Dockerfile -t poppy-gpu .
```

Entraîner le témoin A (10 M de pas, graine 42, comme sur CPU) :

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace poppy-gpu python -m gpu.train --config gpu/configs/exp_a.yaml --num-envs 256 --steps-per-env 16
```

Exporter un checkpoint pour le banc CPU, puis mesurer M1–M6 dans l'image CPU :

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace poppy-gpu python -m gpu.export_sb3 logs/gpu/<run>/model_1220.pt
```

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace poppy-train:latest python gpu/criteres.py --model logs/gpu/<run>/model_1220.zip --episodes 10 --seed 1
```

Le `.zip` et son `_vecnormalize.pkl` se lisent aussi avec
`scripts/evaluate.py --model … --vec-normalize …` et `scripts/visualize.py`.

Contrôles (image GPU) :

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace poppy-gpu python -m pytest gpu/tests -q
```

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace poppy-gpu python -m gpu.parite
```

`gpu/tests` n'est pas dans les `testpaths` du dépôt : `pytest` à la racine
(image CPU, sans mjlab) ne le voit pas, et reste à `19 passed, 1 xfailed`.

## Fichiers

| Fichier | Rôle |
|---|---|
| `poppy_env.py` | Scène mjlab, action PD, observation, récompense, terminaison, reset et randomisation |
| `vecnormalize.py` | `VecNormalize` de SB3 en torch, dans l'enveloppe rsl_rl |
| `ppo.py` | PPO rsl_rl aligné sur SB3, mise à jour capturée en graphe CUDA |
| `symmetry.py` | Miroir pour l'extension de symétrie de rsl_rl, tables de `MirrorPPO` |
| `train.py` | Entraînement ; traduit `ppo:` et `network:` du YAML CPU |
| `export_sb3.py` | Checkpoint rsl_rl → `.zip` SB3 + `_vecnormalize.pkl` |
| `criteres.py` | M1–M6 du journal (§2) sur le banc CPU |
| `parite.py` | Parité modèle, dynamique et randomisation, CPU contre GPU |
| `configs/exp_a.yaml` | Le témoin A : désigne le YAML CPU, ne fixe que la répartition GPU |
| `tests/test_portage.py` | 8 tests (image GPU) |

## Ce qui est reproduit, et comment c'est vérifié

**Modèle.** La scène compilée par mjlab porte exactement la physique du
modèle CPU, champ par champ : masses, inerties, axes (genou gauche corrigé par
`fix_left_knee_axis`), butées, amortissements, frottements, `solref`/`solimp`,
moteurs, et les options du solveur. Piège évité : `MjSpec.attach` ne transmet
pas `<option>` à la scène, et `MujocoCfg` de mjlab n'accepte que Euler et
implicitfast. Sans correction, la scène tournerait en **Euler** au lieu de
**RK4**. `MjcfOptions` recopie toutes les options du modèle CPU. Test :
`test_compiled_model_matches_cpu`.

**Pas de temps.** Même ordre que `PoppyHumanoidEnv.step` :

1. poussée historique (2 % des pas, 1 à 5 unités écrites en couple dans
   `xfrc_applied[bassin, 3:5]`, défaut connu reproduit) ;
2. action bornée à l'espace d'action annoncé, ±3,1 à ±7,3, comme SB3 le fait
   (l'action non normalisée de TECHNIQUE §8 est reproduite, pas corrigée) ;
3. couple PD `kp (pose + action × course − q) − kd q̇`, calculé une fois et
   tenu sur les 5 sous-pas ;
4. `rne_postconstraint` après le dernier sous-pas, comme Gymnasium, pour
   `cfrc_ext`.

Les grandeurs dérivées (`xmat`, `xpos`, `cfrc_ext`) lues par la récompense
ont un sous-pas de retard sur `qpos`, **comme dans Gymnasium**. La position
« avant » des pieds est celle du pas précédent, avec le même retard. Les
contacts observés valent 0 juste après un reset, comme sur CPU.

**Récompense.** Les 18 termes, filtres de vitesse et de lacet, temps de vol,
action précédente et horloge, dans `compute_reward`. Ce code est vérifié
**sur des données MuJoCo C** : nourri des `MjData` d'un déroulé CPU de 300
pas, il rend l'`info` CPU terme par terme, à moins de 1e-3 près
(`test_reward_matches_cpu_on_mujoco_c_data`).

**Observation.** `qpos[2:]`, `qvel`, forces des deux pieds, sin et cos de la
phase : 65 dimensions, contrat inchangé. Écart avec le CPU au reset : 8e-8.

**Randomisation.** Friction du sol (glissement, et torsion et roulis
proportionnels) dans `[0,4 ; 2,0]` ; `solimp[sol, 4]` dans `[0 ; 0,5]` (la
« restitution » au mauvais indice, défaut §8 reproduit) ; masses ×[0,85 ;
1,15] sans recalcul des constantes, comme `_randomize_masses` ; lacet initial
±0,26 rad ; bruit ±0,01 sur `qpos` et `qvel`. Plages relevées sur 512 mondes
par `parite.py`.

**Terminaison.** Bassin hors de ]0,25 ; 0,70[ m ; fin de temps à 1000 pas
(10 s).

**PPO.** Traduction terme à terme des sections `ppo:` et `network:` du YAML
CPU (`gpu/train.py`, `runner_cfg`). rsl_rl est corrigé sur trois points qu'il
fait autrement que SB3 (`gpu/ppo.py`) :

- norme de gradient bornée **globalement** (acteur, critique et `log_std`) ;
- Adam à `eps = 1e-5` ;
- transitions rebattues **à chaque époque**.

S'y ajoutent les avantages normalisés par mini-lot, une perte de valeur non
bornée de poids 0,5, un taux constant, des réseaux séparés [256, 256] ReLU,
`log_std` initial à −2 et l'initialisation par défaut de torch. Pour A : 20
époques, mini-lots de 256, 4096 transitions par mise à jour, comme les 8 × 512
du CPU.

**Miroir.** `gpu/symmetry.py` fournit à l'extension de symétrie de rsl_rl
les tables **calculées** par `build_mirror_spec`, celles de `MirrorPPO` :
paires `r_`/`l_`, signes tirés des axes, horloge décalée d'un demi-cycle,
reflet sur l'observation brute. Test : `test_mirror_tables_match_mirror_ppo`.
Coupé pour A, qui n'en a pas.

**Export.** Le `.zip` SB3 rend la même action que l'acteur rsl_rl, à 1e-4 près,
normalisation comprise (`test_sb3_export_reproduces_policy`).

## Ce qui diffère, et ne peut pas être identique

1. **Les contacts pied-sol.** MuJoCo Warp génère ses contacts plan-maillage
   par la méthode de MJX, avec jusqu'à 4 points par paire. MuJoCo C en
   produit 3 par pied dans l'état initial : 6 contacts contre 8. Dès le
   premier sous-pas de 2 ms, les vitesses articulaires s'écartent d'environ
   0,1 rad/s. Après 100 pas, les deux trajectoires restent proches à 2 cm
   près (`parite.py`), mais ne sont pas identiques. Ce n'est ni le modèle (la
   scène mjlab simulée par MuJoCo C est identique au bit près au modèle
   d'origine) ni ce code : c'est le moteur. Le journal (§4.3) a montré que
   les politiques sont sensibles au modèle de contact. C'est pourquoi la
   porte de parité **évalue sur le banc CPU**, en MuJoCo C, des politiques
   entraînées en MuJoCo Warp.
2. **float32** sur GPU contre float64 sur CPU.
3. **Bootstrap en fin de temps** : rsl_rl ajoute `γ V(s_t)` à la récompense
   d'un épisode coupé à 1000 pas ; SB3 ajoute `γ V(s_{t+1})`. mjlab ne rend
   pas l'observation terminale après un reset automatique.
4. **Horizon des tranches** : 4096 transitions par mise à jour, mais réparties
   en `N` environnements × `T` pas au lieu de 8 × 512. L'avantage GAE est
   tronqué à `T` pas (au-delà de 16 pas, son poids est déjà de 0,94^16 ≈ 0,37).
5. **Adam fused** (graphe CUDA) au lieu de foreach : mêmes formules, arrondis
   différents.
6. **MuJoCo 3.11** (épinglé par mjlab 1.6) contre 3.13 dans l'image CPU.

Non portés, et refusés au lancement plutôt que simulés de travers : le sol
accidenté (`terrain`), `restitution_in_solref`, `push_as_force`, `target_kl`,
un taux d'apprentissage variable.

## Débit

La mise à jour PPO de A (780 000 pas de gradient sur des mini-lots de 256
pour 10 M de pas) coûte ~11 ms par pas en exécution normale sur la RTX 3050 :
presque tout en lancements de noyaux. Le GPU était alors *plus lent* que le
CPU (730 pas/s). Capturée en graphe CUDA, elle tombe à ~0,6 ms par pas, avec
le même calcul (test `test_cuda_graph_update_equals_eager_update`).

Simulation seule, RTX 3050 Laptop 4 Go (pas de 10 ms, observation et
récompense comprises, sans politique) :

| Environnements | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| pas/s | 5 200 | 12 300 | 14 800 | 17 100 |

Entraînement complet, un run seul sur la RTX 3050 Laptop (mesuré sur
409 600 pas, démarrage et compilation des noyaux compris ; mise à jour de A
inchangée : 4096 transitions, 16 × 256, 20 époques) :

| Répartition | Collecte / itération | Mise à jour / itération | Débit | 10 M de pas |
|---|---|---|---|---|
| 512 × 8 | 0,43 s | 0,19 s | 6 700 pas/s | ~25 min |
| **256 × 16** (retenu) | 0,52 s | 0,18 s | **5 000 pas/s** | **~33 min** |
| 128 × 32 | 0,99 s | 0,18 s | 3 300 pas/s | ~50 min |
| CPU, SB3, 8 environnements (journal) | | | 800 à 1 150 pas/s | 2 à 3 h |

Les deux runs de 10 M ci-dessous ont tourné **en même temps** sur la 3050 :
61,7 min (256 × 16) et 79,1 min (128 × 32).

La mise à jour PPO fixe un plancher : 2442 itérations × 0,18 s ≈ 7 min, quel
que soit le GPU, tant qu'on garde les mini-lots de 256 et les 20 époques de A.
Une RTX 4090 accélérerait surtout la collecte ; **estimation non mesurée** :
10 à 15 min pour 10 M. Descendre à quelques minutes demande des mini-lots plus
gros, donc un PPO différent de A, avec sa propre preuve de parité. Ce n'est pas
fait ici.

## Résultats

### Porte de parité

Critères M1–M6 du journal (§2), mesurés par `gpu/criteres.py` sur le banc
CPU : `make_eval_env` puis `run_episodes` de `scripts/evaluate.py`, MuJoCo C
3.13, sol plat (plan), sans randomisation, 10 épisodes, graine 1, actions
déterministes. Formules identiques au `criteres.py` utilisé la nuit du 8 nov.
Les politiques ont été **entraînées sur MuJoCo Warp et évaluées sur MuJoCo
C** : la porte couvre aussi l'écart de contacts décrit plus haut.

Témoin A, graine 42, 10 M de pas. Critères manqués entre parenthèses.

| Checkpoint | A sur CPU (journal §3.4) | GPU 256 × 16 | GPU 128 × 32 |
|---|---|---|---|
| 2 M | — | 4/6 (appui, latéral) | 4/6 (vitesse 0,16, latéral) |
| 3 M | 4/6 | 4/6 (appui, latéral) | 4/6 (appui, latéral) |
| 5 M | 4/6 | 3/6 (tronc, appui, latéral) | 4/6 (appui, latéral) |
| 6 M | 5/6 | 4/6 (appui, latéral) | 4/6 (appui, latéral) |
| 7 M | 5/6 | **5/6** (latéral 0,57 m) | 4/6 (appui, latéral) |
| 8 M | 4/6 | **5/6** (appui 10,4) | 4/6 (appui, latéral) |
| 9 M | 5/6 | **5/6** (appui 13,0) | 4/6 (appui, latéral) |
| final, 10 M | 2/6 | 3/6 (chute 20 %, appui, latéral) | **5/6** (latéral 0,77 m) |

Détail des checkpoints à 5/6, GPU 256 × 16 :

| Checkpoint | M1 | M2 vitesse | M3 tronc | M4 \|cap\| | M5 D−G | M6 \|latéral\| (max) |
|---|---|---|---|---|---|---|
| 7 M | 100 % | 0,236 | 0,996 | 1,5° | 2,9 | 0,57 ✗ (0,69) |
| 8 M | 100 % | 0,274 | 0,997 | 2,3° | 10,4 ✗ | 0,17 (0,31) |
| 9 M | 90 % | 0,299 | 0,996 | 3,5° | 13,0 ✗ | 0,17 (0,31) |

**Verdict : porte passée** (au moins 5/6, comme A sur CPU), pour les deux
répartitions. Ce qu'on retrouve de A :

- la marche apparaît tôt : dès 2 M, 100 % sans chute, 0,21 à 0,31 m/s,
  tronc ≥ 0,99, cap de quelques degrés ;
- les mêmes défauts, la symétrie d'appui (M5) et la dérive latérale (M6),
  avec une forte variance d'un checkpoint à l'autre, comme A-42 (0,24 à
  1,24 m de dérive) ;
- la même dégradation de fin de run pour 256 × 16 (modèle final 3/6, A-42
  final 2/6).

Ce qui n'est **pas** établi :

- **Une graine seulement.** Sur CPU, A-7 passe 6/6 là où A-42 plafonne à
  5/6 : la variance entre graines est du même ordre que l'écart entre CPU et
  GPU. Une deuxième graine (7) sur GPU est la suite logique.
- 128 × 32 passe moins souvent que 256 × 16 (5/6 une fois sur huit
  checkpoints, contre trois). Avec une graine, rien ne dit que c'est
  l'horizon plutôt que le hasard.
- Pas de vidéo de posture : M1–M6 ne voient pas un bras levé (journal §4.2).
  `scripts/visualize.py` lit les `.zip` exportés.
- Aucun essai sur la 4090 louée.

Les checkpoints, leurs exports et `criteres.json` sont dans
`logs/gpu/exp_a_256x16_s42/` et `logs/gpu/exp_a_128x32_s42/` (non versionnés).

### Choix de répartition

`gpu/configs/exp_a.yaml` reste à 128 × 32 par défaut, la répartition la plus
proche des 8 × 512 du CPU. La commande d'entraînement plus haut, celle qui a
passé la porte au plus grand nombre de checkpoints et qui est 1,5 fois plus
rapide, précise `--num-envs 256 --steps-per-env 16`.
