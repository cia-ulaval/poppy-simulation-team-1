# 🚀 GUIDE DES COMMANDES - POPPY RL

## 📋 Table des Matières
1. [Entraînement](#entraînement)
2. [Évaluation](#évaluation)
3. [Visualisation](#visualisation)
4. [Comparaison](#comparaison)
5. [Workflow Complet](#workflow-complet)

---

## 🎓 ENTRAÎNEMENT

### Commandes de Base

```bash
# Entraînement par défaut (10M steps, 8 envs)
python main.py train

# Test rapide (100k steps, 4 envs) - ~5 minutes
python main.py train --steps 100000 --envs 4 --name test_rapide

# Entraînement court (1M steps, 8 envs) - ~30 min
python main.py train --steps 1000000 --name baseline_1M

# Entraînement complet (10M steps, 16 envs) - ~1h avec RTX 4080
python main.py train --steps 10000000 --envs 16 --name baseline_complet
```

### Reprendre un Entraînement

```bash
# Reprendre depuis le dernier checkpoint
python main.py train --resume models/ppo_humanoid_5000000_steps.zip

# Reprendre et continuer jusqu'à 15M steps
python main.py train --resume models/ppo_humanoid_10000000_steps.zip --steps 15000000

# Reprendre avec une config différente
python main.py train --resume models/baseline_v1.zip --config configs/ppo_humanoid_custom.yaml --name baseline_v2
```

### Configurations Différentes

```bash
# Utiliser config de test (timesteps réduits)
python main.py train --config configs/ppo_humanoid_test.yaml

# Phase 2: Reward engineering
python main.py train --config configs/ppo_humanoid_custom.yaml --name custom_reward_v1
```

### Ajuster Performance / Vitesse

```bash
# PC Puissant (RTX 4080, i9) - Maximum speed
python main.py train --envs 16 --steps 10000000

# PC Moyen (RTX 3070, i7) - Équilibré
python main.py train --envs 8 --steps 5000000

# PC Faible (GTX 1660, i5) - Conservateur
python main.py train --envs 4 --steps 2000000

# Debug (1 seul env, erreurs plus claires)
python main.py train --envs 1 --steps 10000
```

---

## 📊 ÉVALUATION

### Commandes de Base

```bash
# Évaluer un modèle (20 épisodes par défaut)
python main.py evaluate models/ppo_humanoid_final.zip

# Évaluer sur plus d'épisodes (stats plus robustes)
python main.py evaluate models/ppo_humanoid_final.zip --episodes 50

# Évaluer sans rendu (plus rapide, juste les stats)
python main.py evaluate models/ppo_humanoid_final.zip --no-render

# Évaluer un checkpoint spécifique
python main.py evaluate models/ppo_humanoid_5000000_steps.zip --episodes 30
```

### Comparer Plusieurs Modèles

```bash
# Évaluer baseline
python main.py evaluate models/exp1_baseline.zip --episodes 50

# Évaluer avec plus d'envs
python main.py evaluate models/exp2_16envs.zip --episodes 50

# Évaluer reward custom
python main.py evaluate models/exp3_custom_reward.zip --episodes 50

# Comparer manuellement les résultats dans le terminal
```

---

## 🎬 VISUALISATION

### Commandes de Base

```bash
# Visualiser un modèle (3 épisodes par défaut)
python main.py visualize models/ppo_humanoid_final.zip

# Visualiser plus d'épisodes
python main.py visualize models/ppo_humanoid_final.zip --episodes 5

# Visualiser ET enregistrer vidéo MP4
python main.py visualize models/ppo_humanoid_final.zip --video

# Visualiser checkpoint intermédiaire
python main.py visualize models/ppo_humanoid_2000000_steps.zip --episodes 3 --video
# Visualiser checkpoint intermédiaire des meilleurs épisodes (5 meilleurs sur 200)
python main.py visualize-best configs/models/ppo_humanoid_final.zip --total 200 --best 5

```

### Vidéos

```bash
# Enregistrer 5 épisodes en vidéo
python main.py visualize models/ppo_humanoid_final.zip --episodes 5 --video

# Les vidéos seront dans ./videos/
# Format: humanoid-episode-0.mp4, humanoid-episode-1.mp4, etc.
```

---

## 📈 COMPARAISON AVEC BASELINE

```bash
# Comparer modèle entraîné vs actions aléatoires (20 épisodes chacun)
python main.py compare models/ppo_humanoid_final.zip

# Comparer sur plus d'épisodes
python main.py compare models/ppo_humanoid_final.zip --episodes 50

# Comparer checkpoint intermédiaire
python main.py compare models/ppo_humanoid_5000000_steps.zip --episodes 30
```

---

## 🔄 WORKFLOW COMPLET

### Phase 1: Test Rapide

```bash
# 1. Test que tout marche (5 minutes)
python main.py train --steps 100000 --envs 4 --name test

# 2. Évaluer
python main.py evaluate models/test.zip

# 3. Visualiser
python main.py visualize models/test.zip
```

### Phase 2: Baseline Complète

```bash
# 1. Entraînement complet (1-2 heures)
python main.py train --steps 10000000 --envs 16 --name baseline_10M

# 2. Évaluation robuste
python main.py evaluate models/baseline_10M.zip --episodes 50

# 3. Comparaison avec aléatoire
python main.py compare models/baseline_10M.zip --episodes 50

# 4. Visualisation + vidéo
python main.py visualize models/baseline_10M.zip --episodes 5 --video

# 5. Regarder TensorBoard
tensorboard --logdir=tensorboard_logs
```

### Phase 3: Amélioration Progressive

```bash
# 1. Reprendre baseline et continuer
python main.py train --resume models/baseline_10M.zip --steps 15000000 --name baseline_15M

# 2. Ou essayer reward custom
python main.py train --config configs/ppo_humanoid_custom.yaml --steps 10000000 --name custom_v1

# 3. Comparer les deux
python main.py evaluate models/baseline_15M.zip --episodes 50
python main.py evaluate models/custom_v1.zip --episodes 50
```

### Phase 4: Debugging

```bash
# Si entraînement plante ou comportement bizarre:

# 1. Utiliser 1 seul env pour voir erreurs clairement
python main.py train --envs 1 --steps 10000 --name debug

# 2. Réduire steps pour test rapide
python main.py train --steps 10000 --envs 2 --name quick_debug

# 3. Regarder les logs
cat logs/*.monitor.csv
tensorboard --logdir=tensorboard_logs
```

---

## 🎯 EXEMPLES PRATIQUES

### Scénario 1: Je débute, je veux tester

```bash
# Test ultra-rapide (5 min)
python main.py train --config configs/ppo_humanoid_test.yaml --name mon_premier_test

# Voir le résultat
python main.py visualize models/mon_premier_test.zip
```

### Scénario 2: Je veux un bon modèle baseline

```bash
# Entraînement optimal pour baseline
python main.py train --steps 10000000 --envs 16 --name baseline_final

# Attendre 1-2h...

# Évaluer
python main.py evaluate models/baseline_final.zip --episodes 50

# Si satisfait, garder. Sinon:
python main.py train --resume models/baseline_final.zip --steps 20000000 --name baseline_extended
```

### Scénario 3: Mon PC a planté pendant l'entraînement

```bash
# Trouver le dernier checkpoint
ls -lh models/

# Exemple: ppo_humanoid_7000000_steps.zip existe

# Reprendre depuis là
python main.py train --resume models/ppo_humanoid_7000000_steps.zip --steps 10000000
```

### Scénario 4: Je veux comparer plusieurs configs

```bash
# Expérience A: Baseline
python main.py train --steps 5000000 --name exp_A_baseline

# Expérience B: Plus d'envs
python main.py train --steps 5000000 --envs 16 --name exp_B_16envs

# Expérience C: Config custom
python main.py train --config configs/ppo_humanoid_custom.yaml --steps 5000000 --name exp_C_custom

# Comparer les 3
python main.py evaluate models/exp_A_baseline.zip --episodes 50
python main.py evaluate models/exp_B_16envs.zip --episodes 50
python main.py evaluate models/exp_C_custom.zip --episodes 50
```

---

## 💡 TIPS & ASTUCES

### Optimisation Vitesse

```bash
# Plus d'envs = plus rapide (si ton PC suit)
python main.py train --envs 16  # RTX 4080 → OK
python main.py train --envs 8   # RTX 3070 → OK
python main.py train --envs 4   # GTX 1660 → OK

# Désactiver eval pendant training (plus rapide)
# (modifier train.py: use_eval=False)
```

### Monitoring en Temps Réel

```bash
# Terminal 1: Lance l'entraînement
python main.py train --steps 10000000

# Terminal 2: Regarde TensorBoard
tensorboard --logdir=tensorboard_logs

# Ouvre http://localhost:6006 dans ton navigateur
```

### Sauvegarder tes Expériences

```bash
# Utilise des noms descriptifs
python main.py train --name YYYYMMDD_description --steps 5000000

# Exemple:
python main.py train --name 20250130_baseline_16envs_10M --envs 16 --steps 10000000
```

---

## ⚠️ ERREURS COMMUNES

### "Out of Memory"
```bash
# Réduire nombre d'envs
python main.py train --envs 4  # Au lieu de 16

# Ou réduire batch_size dans le YAML
```

### Entraînement trop lent
```bash
# Augmenter nombre d'envs (si RAM/GPU OK)
python main.py train --envs 16

# Ou réduire steps pour tester plus vite
python main.py train --steps 1000000
```

### Robot tombe immédiatement même après entraînement
```bash
# Entraîner plus longtemps
python main.py train --steps 20000000

# Ou vérifier que tu évalues le BON modèle
python main.py evaluate models/ppo_humanoid_final.zip  # Pas un checkpoint précoce
```

---

## 📖 RESSOURCES

- **Logs d'entraînement:** `logs/`
- **Modèles sauvegardés:** `models/`
- **TensorBoard:** `tensorboard_logs/`
- **Vidéos:** `videos/`
- **Config YAML:** `configs/`

**Pour plus d'aide:**
```bash
python main.py --help
python main.py train --help
python main.py evaluate --help
```

---

**Bon entraînement! 🚀**
