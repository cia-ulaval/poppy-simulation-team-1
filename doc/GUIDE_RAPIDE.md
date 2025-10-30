# 📖 GUIDE RAPIDE - POPPY RL

## 🎯 Commandes Essentielles

```bash
# Entraîner
python main.py train

# Évaluer
python main.py evaluate models/MODEL.zip

# Visualiser
python main.py visualize models/MODEL.zip

# Comparer avec baseline
python main.py compare models/MODEL.zip
```

---

## ⚙️ Paramètres d'Entraînement

### Table des Paramètres

| Paramètre | Description | Exemple | Défaut |
|-----------|-------------|---------|--------|
| `--steps` | Nombre total d'étapes d'entraînement | `--steps 1000000` | 10,000,000 |
| `--envs` | Nombre de robots en parallèle (vitesse) | `--envs 16` | 8 |
| `--config` | Fichier de configuration | `--config configs/test.yaml` | `configs/ppo_humanoid.yaml` |
| `--resume` | Reprendre depuis un checkpoint | `--resume models/checkpoint.zip` | None |
| `--name` | Nom du modèle final | `--name mon_modele` | `ppo_humanoid_final` |

### 📊 `--steps` : Durée d'Entraînement

| Steps | Durée (RTX 4080, 16 envs) | Usage |
|-------|---------------------------|-------|
| 100,000 | ~5 minutes | Test rapide |
| 1,000,000 | ~30 minutes | Baseline rapide |
| 10,000,000 | ~1-2 heures | Entraînement complet |
| 20,000,000 | ~3-4 heures | Performance maximale |

### 🚀 `--envs` : Vitesse d'Entraînement

| Carte Graphique | Recommandation | Commande |
|-----------------|----------------|----------|
| RTX 4080/4090 | 16 environnements | `--envs 16` |
| RTX 3080/3090 | 8-12 environnements | `--envs 8` |
| RTX 3060/3070 | 4-8 environnements | `--envs 4` |
| GTX 1660/2060 | 2-4 environnements | `--envs 2` |

**💡 Plus d'envs = entraînement plus rapide (8 envs → 8x plus rapide!)**

---

## 📝 Exemples de Commandes

### 1️⃣ Premier Test (5 minutes)

```bash
python main.py train --steps 100000 --envs 4 --name test_rapide
```

### 2️⃣ Entraînement Court (30 min)

```bash
python main.py train --steps 1000000 --envs 8 --name baseline_1M
```

### 3️⃣ Entraînement Complet (1-2h) - RECOMMANDÉ

```bash
python main.py train --steps 10000000 --envs 16 --name baseline_complet
```

### 4️⃣ Entraînement Long (3-4h)

```bash
python main.py train --steps 20000000 --envs 16 --name baseline_20M
```

---

## 🔄 Reprendre un Entraînement

### Pourquoi Reprendre?

- ✅ Ton PC a planté
- ✅ Tu veux entraîner plus longtemps
- ✅ Tu veux améliorer un modèle existant

### Comment ça Marche?

Pendant l'entraînement, des **checkpoints** sont automatiquement sauvegardés:

```
models/
├── ppo_humanoid_50000_steps.zip      # Checkpoint à 50k
├── ppo_humanoid_100000_steps.zip     # Checkpoint à 100k
├── ppo_humanoid_500000_steps.zip     # Checkpoint à 500k
└── ppo_humanoid_final.zip            # Modèle final
```

### Exemples de Reprise

#### Cas 1: PC Planté

```bash
# Tu avais lancé:
python main.py train --steps 5000000

# Ça s'est arrêté à 2M → Tu reprends:
python main.py train --resume models/ppo_humanoid_2000000_steps.zip --steps 5000000
```

#### Cas 2: Continuer l'Entraînement

```bash
# Tu as un modèle à 10M steps, tu veux aller à 15M:
python main.py train --resume models/ppo_humanoid_10000000_steps.zip --steps 15000000
```

#### Cas 3: Améliorer le Modèle Final

```bash
# Reprendre le modèle final et l'entraîner davantage:
python main.py train --resume models/ppo_humanoid_final.zip --steps 20000000 --name extended
```

#### Cas 4: Fine-Tuning

```bash
# Reprendre avec moins d'envs (plus stable):
python main.py train --resume models/baseline_complet.zip --envs 4 --steps 2000000 --name fine_tuned
```

---

## 📊 Évaluation & Visualisation

### Évaluer un Modèle

```bash
# Évaluation standard (20 épisodes)
python main.py evaluate models/ppo_humanoid_final.zip

# Évaluation robuste (50 épisodes)
python main.py evaluate models/ppo_humanoid_final.zip --episodes 50

# Sans visualisation (plus rapide)
python main.py evaluate models/ppo_humanoid_final.zip --no-render
```

### Visualiser le Robot

```bash
# Visualisation simple (3 épisodes)
python main.py visualize models/ppo_humanoid_final.zip

# Plus d'épisodes
python main.py visualize models/ppo_humanoid_final.zip --episodes 5

# Avec enregistrement vidéo MP4
python main.py visualize models/ppo_humanoid_final.zip --video --episodes 3
```

**Les vidéos seront sauvegardées dans:** `./videos/`

### Comparer avec Baseline Aléatoire

```bash
# Compare ton modèle vs actions aléatoires
python main.py compare models/ppo_humanoid_final.zip --episodes 50
```

---

## 🔧 Configurations Différentes

### Fichiers de Config Disponibles

| Fichier | Description | Usage |
|---------|-------------|-------|
| `configs/ppo_humanoid.yaml` | Config par défaut (10M steps) | Baseline complète |
| `configs/ppo_humanoid_test.yaml` | Config de test (100k steps) | Tests rapides |
| `configs/ppo_humanoid_custom.yaml` | Config custom (Phase 2) | Reward engineering |

### Utiliser une Config Différente

```bash
# Config de test (rapide)
python main.py train --config configs/ppo_humanoid_test.yaml

# Config custom (Phase 2 - plus tard)
python main.py train --config configs/ppo_humanoid_custom.yaml --name custom_v1
```

---

## 🎯 Workflow Recommandé

### Pour Débuter (Phase 1)

```bash
# 1. Test rapide (5 min) - Vérifier que tout marche
python main.py train --steps 100000 --envs 4 --name test

# 2. Visualiser
python main.py visualize models/test.zip

# 3. Si OK, lancer entraînement complet (1-2h)
python main.py train --steps 10000000 --envs 16 --name baseline_10M

# 4. Évaluer
python main.py evaluate models/baseline_10M.zip --episodes 50

# 5. Comparer avec aléatoire
python main.py compare models/baseline_10M.zip

# 6. Visualiser + vidéo
python main.py visualize models/baseline_10M.zip --video
```

### Si Performance Insuffisante

```bash
# Option A: Entraîner plus longtemps
python main.py train --resume models/baseline_10M.zip --steps 20000000 --name baseline_20M

# Option B: Fine-tuning
python main.py train --resume models/baseline_10M.zip --envs 4 --steps 2000000 --name fine_tuned
```

---

## 📈 Monitoring en Temps Réel

### TensorBoard

```bash
# Terminal 1: Lance l'entraînement
python main.py train --steps 10000000

# Terminal 2: Lance TensorBoard
tensorboard --logdir=tensorboard_logs

# Ouvre dans ton navigateur: http://localhost:6006
```

**Tu verras:**
- Reward moyen par épisode
- Durée des épisodes
- Loss du réseau
- Learning rate

---

## 💾 Organisation des Fichiers

```
poppy-simulation-team-1/
├── models/                          # Modèles entraînés
│   ├── ppo_humanoid_50000_steps.zip
│   ├── ppo_humanoid_100000_steps.zip
│   └── ppo_humanoid_final.zip
│
├── logs/                            # Logs d'entraînement
├── tensorboard_logs/                # Pour TensorBoard
├── videos/                          # Vidéos enregistrées
│
└── configs/                         # Configurations YAML
    ├── ppo_humanoid.yaml
    ├── ppo_humanoid_test.yaml
    └── ppo_humanoid_custom.yaml
```

---

## ⚠️ Problèmes Courants

### "Out of Memory"

```bash
# Réduire le nombre d'environnements
python main.py train --envs 4  # Au lieu de 16
```

### Entraînement Trop Lent

```bash
# Augmenter le nombre d'environnements (si ton PC suit)
python main.py train --envs 16
```

### Robot Tombe Immédiatement Après Entraînement

```bash
# Entraîner plus longtemps
python main.py train --steps 20000000

# Ou reprendre et continuer
python main.py train --resume models/ppo_humanoid_final.zip --steps 5000000
```

---

## 📋 Checklist Rapide

**Avant de Lancer un Entraînement:**

- [ ] Config choisie (`--config` ou défaut)
- [ ] Nombre de steps adapté à ton temps disponible
- [ ] Nombre d'envs adapté à ton PC
- [ ] Nom descriptif pour le modèle (`--name`)
- [ ] TensorBoard prêt à lancer (optionnel)

**Après l'Entraînement:**

- [ ] Évaluer le modèle (`evaluate`)
- [ ] Visualiser le comportement (`visualize`)
- [ ] Comparer avec baseline (`compare`)
- [ ] Sauvegarder les résultats
- [ ] Décider: continuer ou passer à Phase 2?

---

## 🚀 Commande Recommandée pour Toi (RTX 4080)

```bash
# Entraînement optimal:
python main.py train --steps 10000000 --envs 16 --name baseline_final

# Durée: ~1-2 heures
# Résultat attendu: Robot qui marche correctement!
```

---

## 📞 Aide Rapide

```bash
# Voir toutes les options disponibles
python main.py --help
python main.py train --help
python main.py evaluate --help
```

**Fichiers importants:**
- `configs/ppo_humanoid.yaml` - Hyperparamètres
- `COMMANDS.md` - Guide détaillé (si besoin)
- `README.md` - Vue d'ensemble du projet

---

**🎉 Bon entraînement!**
