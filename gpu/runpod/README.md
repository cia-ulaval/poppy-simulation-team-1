# Kit RunPod : un run de test sur une carte louée

But : mesurer, en moins d'une heure de location, ce qu'apporte une RTX 4090
ou 5090 au portage GPU (`gpu/`), avant d'itérer en masse. On mesure le débit
d'un run, la montée en parallèle, la mémoire vidéo et, en option, le terrain
du palier 2.

**Rien n'est loué par ces scripts.**
- L'utilisateur crée le pod dans l'interface web de RunPod et y saisit
  lui-même ses identifiants.
- Le code part par `scp` sous forme d'archive git. Rien n'est poussé.
- Le pod est supprimé par l'utilisateur, ou par la session principale avec son
  accord, jamais par un script.

## Coût attendu

| Poste | Durée estimée | Coût, 4090 à ~0,34 $/h* |
|---|---|---|
| `setup.sh` : deux environnements Python, puis les tests GPU | 10 à 15 min | ~0,08 $ |
| `bench.sh` (a) à (c) | 20 à 30 min | ~0,15 $ |
| `TERRAIN=1` : (d) en plus | ~10 min | ~0,06 $ |
| **Total** | **≤ 1 h** | **≤ 0,35 $** |

\* Prix communautaire relevé en novembre 2026, à vérifier dans l'interface au
moment de la location. Une 5090 coûte plus cher. Les durées sont des
**estimations** : elles partent du débit mesuré sur la 3050 et seront
remplacées par les mesures du banc.

## Étapes

### 1. Sur le portable : préparer l'envoi

```bash
bash gpu/runpod/pack.sh --models
```

Le script produit, dans `logs/runpod/envoi/` (ignoré par git) :
- `poppy-<commit>.tar.gz`, l'archive git du commit courant ;
- `models.tar.gz`, les `best_model` H-Placo CPU des graines 7 et 42, pour
  `--init-from` au palier 2.

Il refuse de partir si du code n'est pas commité.

### 2. Dans l'interface RunPod (utilisateur)

- **Modèle :** PyTorch officiel (`runpod/pytorch`, Ubuntu, Python 3.10 à 3.12).
  Le torch du modèle n'est pas utilisé : `setup.sh` installe le sien dans un
  environnement à part.
- **Carte :** RTX 4090 (24 Go) ou RTX 5090 (32 Go, Blackwell sm_120).
- **CUDA du pilote :** prendre de préférence une machine en **CUDA ≥ 13.0**,
  filtre « CUDA Version » à la création. Sinon, `setup.sh` bascule de lui-même
  sur la roue Warp `+cu12`.
- **Accès :** SSH root par clé publique. Disque conteneur ≥ 30 Go.

### 3. Envoyer le code et installer

```bash
scp -P <port> logs/runpod/envoi/poppy-<commit>.tar.gz logs/runpod/envoi/models.tar.gz root@<ip>:~
```

```bash
ssh -p <port> root@<ip> 'tar -xzf poppy-*.tar.gz && tar -xzf models.tar.gz -C poppy && cd poppy && bash gpu/runpod/setup.sh'
```

`setup.sh` fait, dans l'ordre :
1. `nvidia-smi` et la version CUDA du pilote ;
2. `.venv-gpu` :
   - torch 2.11.0 cu128, la version de l'image validée, dont les roues
     couvrent sm_89 et sm_120 ;
   - puis `gpu/requirements.txt` (mjlab 1.6.0, MuJoCo Warp 3.11, rsl_rl 5.4.2) ;
3. **Warp :**
   - `warp-lang` 1.19.0 sur PyPI est construit avec le CUDA Toolkit 13 (vu sur
     la 3050 : « CUDA Toolkit 13.4 ») ;
   - si le pilote est en CUDA 12.x, le script installe la roue
     `warp_lang-1.19.0+cu12` publiée par NVIDIA sur la même version, après
     vérification de son SHA-256 ;
4. contrôle que torch a des noyaux pour l'architecture de la carte, et que
   Warp tourne dessus ;
5. `.venv-cpu`, le banc CPU (`requirements/train.lock`, MuJoCo C 3.13) ;
6. **porte d'entrée : `pytest gpu/tests`** (32 tests).
   - Si elle échoue, ne rien lancer d'autre : rapatrier `logs/runpod/` et
     supprimer le pod.
   - Les compilations de Warp de ce premier passage (plus d'une minute rien
     que pour le noyau de collision du terrain) restent en cache dans
     `~/.cache/warp` pour la suite.

### 4. Le banc

```bash
ssh -p <port> root@<ip> 'cd poppy && TERRAIN=1 bash gpu/runpod/bench.sh'
```

| Mesure | Quoi | Où |
|---|---|---|
| (a) | `nvidia-smi`, nom, mémoire, pilote, `compute_cap` | `carte.csv`, `nvidia-smi.txt` |
| (b) | H-Placo sur sol plat, 2 M de pas, 256 × 16 (répartition validée) | `b_plat_1x.txt` |
| (c) | 2 puis 4 runs H-Placo de 2 M en même temps : débit total, mémoire vidéo max | `c_plat_2x.txt`, `c_plat_4x.txt` |
| (d) | `TERRAIN=1` : palier 2 sur terrain, 1 M de pas, 256 × 16 puis 1024 × 4 | `d_terrain_*.txt` |
| (e) | tout le reste, et la mémoire vidéo échantillonnée chaque seconde | `logs/runpod/<carte>/`, archive `logs/runpod/<carte>.tar.gz` |

Le débit est celui que `gpu/train.py` écrit dans `throughput.json`. Il
compte la compilation des noyaux : c'est ce que coûte un vrai run. La
référence sur la 3050 :
- H-Placo, sol plat, 256 × 16 : 4 750 à 5 000 pas/s ;
- terrain, 256 environnements : ~330 pas/s ; 1024 environnements : ~1 000 pas/s.

### 5. Rapatrier, puis supprimer le pod

```bash
scp -P <port> root@<ip>:~/poppy/logs/runpod/<carte>.tar.gz logs/runpod/
```

Ouvrir l'archive et vérifier qu'elle contient les `*.txt` du banc. **Puis
supprimer le pod** (Terminate) : un pod arrêté sans être supprimé facture
encore son disque. Le script l'affiche à la fin, mais ne le fait pas.

## Mesurer sur le CPU du pod

Quand on itérera, les mesures M1–M7 sur le banc CPU deviendront le goulot,
avec 10 épisodes par modèle et par difficulté. `criteres_parallel.sh` les
répartit sur les cœurs du pod, un processus par cœur, à un seul fil
(`OMP_NUM_THREADS=1` : sans lui, chaque PyTorch prend tous les cœurs, comme
sur Hetzner).

```bash
bash gpu/runpod/criteres_parallel.sh configs/palier2_hplaco.yaml logs/runpod/crit_p2 runs/p2_s7/model_*.zip -- 0.3 0.6 1.0
```

- Les modèles GPU s'exportent d'abord avec `gpu/export_sb3.py`, dans `.venv-gpu`.
- Le tableau arrive dans `<sortie>/tableau.md`, une ligne par mesure.
- `JOBS` règle le nombre de processus.

## Ce que ce kit ne fait pas

- Il ne crée, n'arrête ni ne supprime aucun pod.
- Il ne lance pas le run de 20 M du palier 2. On le lancera à la main sur le
  pod, quand la session principale l'aura décidé :

  ```bash
  .venv-gpu/bin/python -m gpu.train --config gpu/configs/palier2_hplaco.yaml --seed 7 --init-from <best_model s7>
  ```
- Il n'a pas encore tourné sur une carte louée : le premier `setup.sh` le
  vérifiera.
