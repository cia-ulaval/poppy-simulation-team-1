# Reprise propre : Docker, agents et revue locale

Date : 10 septembre 2026. Statut : cadrage uniquement, aucun code ni fichier Docker créé.

## 1. Base Git conservée

- Base : `model-for-presentation`, commit `7583141e0cc499a6c8a4e36adce6a7c4198d65d4`.
- Sauvegarde : `codex/sauvegarde-avant-clean-2026-09-10`. Elle conserve les changements initialement non suivis : les trois documents, les deux modèles de `configs/models`, les réglages `.idea` et `.claude`.
- Travail : `clean`, créée depuis la base puis enrichie des trois documents existants et de ce guide.
- Aucun push, aucune fusion distante, aucune suppression de branche. La sauvegarde contient des réglages personnels : elle sert d'archive locale.
- Les environnements et caches ignorés par Git ne sont pas archivés dans les commits ; ils ne sont pas supprimés. `clean` reprend le code existant, ce n'est pas un dépôt vide ni un nettoyage de l'historique.

Les anciens documents sont conservés tels quels. Leurs constats datés et leurs propositions ne prouvent pas qu'une fonctionnalité est déjà livrée. Ce guide décrit la prochaine étape.

Commandes de consultation, depuis la racine du dépôt :

```powershell
git status --short
git branch --list
git log --oneline model-for-presentation..clean
git diff --stat model-for-presentation clean
git show codex/sauvegarde-avant-clean-2026-09-10:docs/GUIDE_EQUIPE.md
```

Pour récupérer ultérieurement les deux modèles archivés, après vérification de l'état de travail :

```powershell
git restore --source codex/sauvegarde-avant-clean-2026-09-10 -- configs/models
```

Cette commande écrit les fichiers ; elle ne crée pas de commit. Ne pas supposer que ces modèles ont leur normalisation correspondante. Les modèles et normalisations déjà suivis sous `logs/` restent dans la base.

## 2. Ce qu'il faut isoler

Un environnement Python (`venv`) contient des paquets. Un fichier `.env` fournit des paramètres locaux. Un environnement Gymnasium est la simulation utilisée par l'algorithme. Une image Docker contient le système et ses dépendances ; un conteneur en est une exécution. Ces notions ne demandent pas chacune un Dockerfile.

Proposition : trois usages Compose, avec une base Python/MuJoCo commune à la simulation et à l'entraînement. Le nombre final de Dockerfiles découlera des dépendances, pas du nombre de personnes.

| Usage proposé | Contenu | Première validation |
| --- | --- | --- |
| `train` | MuJoCo, Gymnasium, SB3, PyTorch, TensorBoard ; CPU d'abord, variante GPU ensuite | Petit entraînement, sauvegarde, rechargement |
| `sim` | Évaluation d'un modèle, simulation quotidienne ; rendu sans écran d'abord | Charger un couple modèle/normalisation, dérouler un épisode |
| `robot` | Client robot et connexion au pont ROS ; serveur ROS séparé si nécessaire | Connexion et messages sur un pont de test sans moteurs |

L'usage quotidien signifie ici simulation/évaluation. L'utilisation quotidienne du robot physique relèvera du profil robot. Une interface graphique interactive est un lot séparé du fonctionnement sans écran.

Les profils Compose permettent de sélectionner des services. Les noms ci-dessus sont un contrat proposé, pas des services existants. Source : [Docker Compose, profils](https://docs.docker.com/compose/how-tos/profiles/).

### Constats du dépôt

- `requirements.txt` mélange simulation, entraînement et vision ; plusieurs versions sont seulement bornées par `>=`.
- `src/robot/ros_publisher.py` importe `roslibpy`, absent de ce fichier de dépendances. Il utilise un pont WebSocket avec une IP fixe et le port 9090. Les imports ROS natifs sont commentés : cela ne détermine pas la distribution ROS du serveur.
- `src/robot/SimulationAdaptater.py` impose un rendu interactif et contient `sleep(5)` dans `step`. Ces points devront être traités dans des lots explicites.
- `origin/feat/docker` contient déjà un Dockerfile consultable localement : ROS Rolling, PyTorch CPU, bibliothèques graphiques. C'est une référence à réexaminer, pas une base à fusionner automatiquement. La documentation historique signale des différences de simulation sur cette branche.
- Le moteur Docker n'était pas joignable lors de l'inspection. Le client indique 29.2.0 ; la lecture de sa configuration utilisateur est aussi restreinte dans cette session. Aucun build ou test conteneur n'a été réalisé.

### Décisions à prendre avant d'écrire Docker

1. Machines de l'équipe : Windows/WSL2 ou Linux, architecture CPU, GPU NVIDIA et mémoire disponibles.
2. Robot : OS et architecture, ROS 1 ou 2 et distribution exacte, emplacement de rosbridge et du pilote moteur, réseau ou USB.
3. Usage quotidien : vidéo enregistrée, fenêtre MuJoCo, ou commande du robot.
4. Couple modèle/normalisation choisi comme référence et emplacement des sorties.

Ne pas choisir ROS Rolling uniquement parce que l'ancien Dockerfile l'utilise. Ne pas imposer ROS à l'entraînement si ses imports n'en ont pas besoin. Commencer par une simulation CPU sans écran ; mesurer ensuite l'intérêt du GPU.

### Paramètres et données

Prévoir un exemple de configuration documenté avec seulement les paramètres nécessaires : adresse/port du pont, chemins de modèles et sorties. Garder les hyperparamètres dans les YAML existants. Fixer les versions de dépendances après une installation testée et conserver la résolution exacte.

Monter les modèles en lecture seule et les sorties sur un emplacement persistant. Exclure du contexte de build les environnements locaux, caches, réglages IDE, secrets et gros journaux. Ne pas inclure la vision dans le premier lot si elle n'est pas utilisée. Une connexion au robot ne doit jamais démarrer par défaut avec la simulation.

## 3. Choix de l'outil et des rôles

Recommandation : commencer avec Codex dans ce dépôt pour cadrer un seul lot, puis réaliser ce lot quand son périmètre est décidé. Utiliser une session distincte pour la revue. L'application propose des outils de revue ; voir la [documentation officielle de revue Codex](https://learn.chatgpt.com/docs/code-review).

Pour la phase actuelle, choisir le mode Plan quand disponible et donner explicitement la consigne « aucune modification de code ». Pour la réalisation, autoriser les fichiers du lot et les tests locaux nécessaires. Garder la publication distante hors du périmètre.

| Outil | Usage conseillé ici |
| --- | --- |
| Codex | Continuité du dépôt, petits lots, exécution des commandes, examen des diffs |
| Claude Code | Alternative pour réaliser un lot ou fournir une seconde revue ; les sous-agents ont un contexte et des outils configurables |
| OpenCode | Expérimenter des modèles différents par agent, avec permissions séparées |

Claude Code documente les [sous-agents personnalisés](https://code.claude.com/docs/en/sub-agents). OpenCode documente des agents Plan/Build, des modèles par agent et leurs permissions dans son [guide des agents](https://opencode.ai/docs/agents/). La disponibilité et la facturation des modèles dépendent des fournisseurs connectés ; la présence des exécutables ne prouve pas l'accès aux comptes.

Organisation cible :

1. **Planificateur** : inspecte le dépôt, fixe le périmètre et les critères d'acceptation.
2. **Développeur** : réalise un lot et fournit les commandes, sorties et limites.
3. **Relecteur** : examine le diff dans une nouvelle session, sans modifier les sources. Peut employer un autre modèle ou fournisseur.
4. **Validateur** : exécute les vérifications définies à l'avance et rapporte les résultats ; il ne réécrit pas les critères pour faire passer le lot.

Un seul agent écrit dans un même répertoire à la fois. Pour plusieurs développeurs simultanés, utiliser un worktree par branche. Changer de modèle peut diversifier la revue ; cela ne remplace pas l'exécution des tests ni la validation matérielle.

### Essayer OpenCode sans coder

OpenCode, Claude Code et Codex ont été trouvés dans le PATH. Leur authentification n'a pas été vérifiée. Commandes à lancer soi-même ; elles n'ont pas été exécutées pendant ce cadrage :

```powershell
opencode --version
opencode --help
opencode auth login
opencode models
opencode --agent plan
```

La connexion est interactive et n'est nécessaire que si le fournisseur voulu n'est pas déjà configuré. Choisir un identifiant exact dans `opencode models`, puis utiliser `opencode --agent plan --model "FOURNISSEUR/MODELE"` en remplaçant la valeur. Pour une revue, ouvrir une nouvelle session et demander une analyse sans édition. Plus tard, définir les profils développeur/relecteur via `opencode agent create`, avec modèles et permissions distincts. Vérifier l'aide de la version installée avant de générer sa configuration : les documentations V1 et V2 diffèrent.

Source des commandes : [CLI OpenCode](https://opencode.ai/docs/cli/).

Prompt de cadrage :

> Lis docs/DEMARRAGE_PROPRE.md et les fichiers du lot. Ne modifie aucun code. Propose le plus petit lot Docker CPU reproductible, ses dépendances, ses commandes de test et ses critères d'acceptation. Distingue les observations des hypothèses. Aucun push ni accès aux moteurs.

Prompt de revue :

> Examine le diff du lot par rapport à son commit de départ. Ne modifie pas les sources. Cherche les erreurs de dépendances, chemins, volumes, réseau et compatibilité Windows/Linux. Pour chaque problème, indique fichier, conséquence et vérification reproductible. Distingue les tests exécutés de ceux qui restent à faire.

## 4. Petits commits et validation

Après ce cadrage, proposer un lot à la fois :

1. Base CPU et dépendances reproductibles, avec exclusions du contexte Docker.
2. Entraînement court et persistance des sorties.
3. Évaluation quotidienne et rendu sans écran ; affichage interactif ensuite si nécessaire.
4. Configuration du client robot et pont de test, après identification du ROS réel.
5. GPU si le matériel et les mesures le justifient.

Pour chaque lot : relever le commit de départ, définir les tests, implémenter, examiner le diff, exécuter les tests, faire relire, corriger, puis enregistrer un commit cohérent. La revue doit identifier le commit exact ; toute correction ultérieure demande une vérification adaptée. Aucune PR distante n'est nécessaire pour cette phase locale.

```powershell
git rev-parse HEAD
git diff --check
git diff --stat
git diff
```

Après revue, ajouter explicitement les fichiers du lot avec `git add -- chemin`, examiner `git diff --cached`, puis créer le commit. Éviter l'ajout global des logs et modèles produits par les tests. Aucun push dans cette procédure.

Consigner pour chaque lot : commit, OS, versions Docker/Compose, image construite, commandes exactes, codes de sortie, résultats attendus/observés et tests non réalisés.

## 5. Tests Docker : terminal d'abord

Le terminal permet à l'agent de construire les images, démarrer les services, lire les logs et vérifier les résultats. Un MCP Docker n'est pas requis. Il pourra être ajouté si une intégration précise apporte un bénéfice ; aucun MCP n'a été installé ici.

Après avoir ouvert Docker Desktop, depuis le terminal qui servira au projet :

```powershell
docker version
docker info
docker context show
docker compose version
```

Attendre une réponse de la partie serveur. Si l'accès reste impossible, vérifier le contexte et les permissions du terminal. Ne pas commencer les builds tant que ce prérequis échoue.

### Contrat de commandes futur — non exécutable aujourd'hui

Les exemples suivants supposent qu'un fichier Compose a été créé avec les services `train`, `sim` et `robot`. Ils devront être validés puis remplacés par un guide d'utilisation effectivement testé.

```powershell
docker compose config --quiet
docker compose --profile train build
docker compose run --rm train python -m pip check
docker compose run --rm train python scripts/train_poppy.py --config configs/poppy_robust.yaml --timesteps 2048 --n-envs 1 --seed 0 --log-dir /outputs/smoke
docker compose --profile sim up -d
docker compose ps -a
docker compose logs --tail 100
docker compose down
```

Les options Python de cet exemple existent dans le script, mais leur exécution en conteneur n'a pas été testée. `/outputs` devra être monté vers un dossier persistant. `up -d` suppose que le service sim ait une commande par défaut adaptée. La commande exacte de lecture d'un modèle reste à définir avec le service ; ne pas inventer une interface de viewer.

| Vérification future | Critère d'acceptation |
| --- | --- |
| Installation | Build sur une machine sans venv du projet ; dépendances cohérentes |
| Simulation | Chargement des assets, reset et steps sans écran, valeurs finies et formes correctes |
| Entraînement | Exécution courte réussie, checkpoint et normalisation persistants |
| Rechargement | Couple modèle/normalisation rechargé dans un nouveau conteneur, épisode terminé sans erreur |
| Rendu | Fichier vidéo/image non vide et inspection visuelle, si inclus dans le lot |
| Robot simulé | Messages au format attendu, unités et ordre des articulations vérifiés, déconnexion traitée |
| Matériel réel | Essai supervisé distinct avec arrêt et limites validés ; jamais déduit du succès d'un test logiciel |

Un conteneur démarré ne prouve pas que le modèle fonctionne. Un entraînement court ne prouve pas que le robot apprend à marcher. Un test sur pont simulé ne prouve pas la compatibilité du robot physique.

## 6. État de cette livraison

Seuls Git et la documentation ont été modifiés. La vérification porte sur la filiation des branches, la conservation des fichiers et l'absence de modifications applicatives. Aucune revue indépendante par un autre modèle n'a été réalisée pendant ce cadrage. Les tests applicatifs, les builds et les essais robot restent à effectuer dans les lots futurs.
