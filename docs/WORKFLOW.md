# WORKFLOW — comment on travaille sur `clean`

Ce document décrit le processus par lot avec OpenCode : qui fait quoi, dans quel ordre, avec quelles commandes. Les règles de fond (Git, normes Python, sécurité robot) sont dans [`AGENTS.md`](../AGENTS.md). Les commandes Docker sont dans [`DOCKER.md`](DOCKER.md).

Statut : mis en place le 10 septembre 2026. Tout est local ; aucun push.

---

## 1. Point de départ

| Branche | Rôle |
|---|---|
| `model-for-presentation` | Base figée (`7583141`). On n'y touche plus. |
| `codex/sauvegarde-avant-clean-2026-09-10` | Archive des fichiers locaux d'avant (docs, modèles, réglages IDE). Lecture seule. |
| `clean` | **Branche de travail.** Base + documentation + lot Docker. Tous les lots suivants s'y font. |

Vérifier où on est :

```powershell
git branch --show-current          # doit afficher : clean
git status --short                 # doit être vide avant de commencer un lot
git log --oneline -10
```

---

## 2. Les agents OpenCode

La configuration est versionnée dans le dépôt : `opencode.json`, `.opencode/agent/`, `.opencode/command/`. Elle se charge automatiquement quand OpenCode démarre **depuis la racine du dépôt**. Après toute modification de ces fichiers, quitter et relancer OpenCode.

| Agent | Type | Modèle | Écrit ? | Comment l'appeler |
|---|---|---|---|---|
| `plan` | primaire (défaut) | `openrouter/fable-supervisor` (alias de Fable 5.1) | non | démarre par défaut ; `Tab` pour changer |
| `architect` | sous-agent, optionnel | `openai/gpt-5.6-sol` | non | `@architect <lot>` |
| `build` | primaire | `openrouter/moonshotai/kimi-k2.7-code` | **oui** | `Tab` jusqu'à `build`, ou `opencode --agent build` |
| `build-openai` | primaire, secours | `openai/gpt-6-astra` | oui | `opencode --agent build-openai` |
| `reviewer` | sous-agent | `openrouter/fable-supervisor` (alias de Fable 5.1) | non | `@reviewer ...` ou `/review [fichiers/diff]` |
| `validator` | sous-agent | `openrouter/fable-supervisor` (alias de Fable 5.1) | non | `@validator ...` ou `/validate [preuves]` |

Principe : **Kimi écrit, Fable supervise et vérifie.** OpenAI Sol intervient
ponctuellement comme architecte en lecture seule afin de proposer une stratégie
à Kimi. Astra ne code qu'en secours si Kimi reste bloqué. Le même modèle ne
code pas et ne relit pas le même lot.

Kimi et Fable utilisent `openrouter`. OpenAI Sol/Astra utilisent le fournisseur
`openai` connecté en OAuth. Le catalogue ne garantit pas à lui seul qu'une
requête réussira : l'accès doit être vérifié par une exécution réelle.
L'alias local `fable-supervisor` limite les réponses de Fable à 8192 tokens
pour maîtriser le coût maximal d'une revue.

Garde-fous configurés dans `opencode.json` et les fichiers des agents :

- `git push`, `git commit`, `git rebase`, `git merge`, `git reset --hard`, `git clean`, `git checkout --`, `git restore` et toute suppression de branche : **refusés** pour tous les agents. L'humain peut commiter depuis son propre terminal.
- Les agents codeurs ne peuvent ni `git add` ni `git commit` : c'est l'humain qui commite.
- `reviewer` et `validator` : édition et shell refusés. L'endpoint Fable
  d'OpenRouter ne supporte pas le tool `bash` ; le superviseur leur fournit les
  fichiers, diffs et sorties de tests à vérifier.
- `docker compose down -v`, `docker system prune` : refusés (protège volumes et images).

Vérifier que la configuration est valide :

```powershell
opencode agent list
```

---

## 3. Cycle d'un lot

Un lot = un périmètre petit, testable, qui tient en un à trois commits.

```
 plan (Fable)    architect (OpenAI)    build (Kimi)    reviewer    tests    validator    humain
 ───────────     ──────────────────    ────────────    ────────    ─────    ─────────    ──────
 1. cadrer  ───► 2. conseiller     ───► 3. coder  ───► 4. relire ─► 5. exécuter ─► 6. valider ─► 7. commiter
```

### Étape 1 — Cadrer (`plan`)

Dans une session `plan`, décrire le besoin. Le cadrage produit, dans la conversation (pas de fichier) :

- commit de départ (`git rev-parse HEAD`) ;
- fichiers à toucher, fichiers **interdits** ;
- critères d'acceptation, chacun avec sa commande de vérification exacte ;
- ce qui est hors périmètre.

Prompt type :

> Lis AGENTS.md et docs/DOCKER.md. Cadre le lot suivant : <besoin>. Donne le commit de départ, les fichiers touchés, les critères d'acceptation avec commandes de test, et ce qui est hors périmètre. Ne modifie rien.

### Étape 2 — Conseiller ponctuellement (`architect`, optionnel)

Si le lot comporte un choix d'architecture ou si Kimi a besoin d'une seconde
analyse, appeler `@architect` avec le cadrage. Transmettre ensuite sa note à
Kimi. Sauter cette étape pour un changement simple afin d'économiser le quota
OpenAI.

### Étape 3 — Implémenter (`build`)

Nouvelle session ou `Tab` vers `build` (Kimi). Coller le cadrage et, si elle
existe, la note de l'architecte OpenAI. Prompt type :

> Voici le cadrage du lot. Implémente exactement ce périmètre en respectant AGENTS.md. Ne touche pas aux fichiers hors périmètre. Ne commite pas. À la fin, liste les fichiers modifiés, les commandes exécutées avec leur code de sortie, et ce que tu n'as pas testé.

Le codeur peut lancer `python`, `pytest`, `docker compose build` et `docker compose run --rm` sans demander ; tout autre shell demande confirmation.

### Étape 4 — Relire (`reviewer`)

Depuis n'importe quelle session primaire :

Joindre le diff et les fichiers concernés dans la demande, puis lancer par
exemple `/review @requirements/train.txt`. Fable n'a pas accès au shell : un
simple SHA ne lui permet pas de lire le diff. Le relecteur rend un verdict
`ACCEPTER` / `ACCEPTER AVEC CORRECTIONS` / `REFUSER` avec fichier:ligne. Les
corrections repartent à l'étape 3, **chez le codeur**, pas chez le relecteur.

### Étape 5 — Exécuter les tests (superviseur)

Le superviseur exécute les commandes définies à l'étape 1 et conserve pour
chacune la commande exacte, le code de sortie et les lignes utiles. Il ne lance
jamais un service vers le robot réel.

### Étape 6 — Valider les preuves (`validator`)

```
/validate <coller les critères, commandes, codes de sortie et résultats>
```

Fable compare chaque preuve au critère prévu. Une commande ou une sortie
manquante rend le critère non vérifié ; Fable ne réinterprète pas les critères.

### Étape 7 — Commiter (humain)

```powershell
git diff --check
git add -- chemin/fichier1 chemin/fichier2
git diff --cached --stat
git commit
```

Message : Conventional Commits en français, sujet ≤ 72 caractères, corps sur le **pourquoi**, voir `AGENTS.md` §3. Un commit par sujet cohérent. Pas de push.

Ce que le journal doit garder pour chaque lot (dans le message de commit ou la conversation) : commit de départ, commandes exactes, codes de sortie, tests non réalisés.

---

## 4. Validation standard du lot Docker

Le superviseur exécute cette liste après tout changement de
`docker/Dockerfile`, `compose*.yaml` ou `requirements/`, puis fournit les
résultats à `/validate`.

| # | Critère | Commande | Attendu |
|---|---|---|---|
| 1 | Démon joignable | `docker info --format '{{.ServerVersion}}'` | une version, code 0 |
| 2 | Compose valide | `docker compose --profile train --profile robot --profile mock --profile vision config --quiet` | code 0 |
| 3 | Image train se construit | `docker compose build train` | code 0 |
| 4 | Dépendances cohérentes | `docker compose --profile train run --rm train python -m pip check` | `No broken requirements found.` |
| 5 | Imports du projet | `docker compose --profile train run --rm train python -c "import src.environments.poppy_humanoid_env"` | code 0 |
| 6 | Script accessible | `docker compose --profile train run --rm train python scripts/train_poppy.py --help` | aide affichée, code 0 |
| 7 | Entraînement court | `docker compose --profile train run --rm train python scripts/train_poppy.py --config configs/poppy_robust.yaml --timesteps 2048 --n-envs 1 --seed 0 --log-dir logs/smoke` | code 0 ; `poppy_ppo_final.zip` et `vec_normalize_final.pkl` sous un dossier horodaté de `logs/smoke/` |
| 8 | Faux robot démarre | `docker compose --profile mock up -d` puis `docker compose --profile mock ps` | `rosbridge` en `running` |
| 9 | Image bridge se construit | `docker compose build bridge` | code 0 |
| 10 | Bridge importe et charge son script | `docker compose --profile robot run --rm --no-deps bridge python scripts/run_robot.py --help` | aide affichée, code 0 |
| — | Nettoyage | `docker compose --profile mock down` | code 0 |

Les options exactes du critère 7 doivent être confirmées contre `scripts/train_poppy.py --help` : si une option n'existe pas, le critère est **rapporté en échec**, pas adapté à la volée.

Ce que cette liste **ne prouve pas** : que le modèle apprend, que le pont parle au vrai robot, que la vision fonctionne (dépendance `depth_anything_3` manquante, voir `DOCKER.md` §7).

---

## 5. Lots prévus

| # | Lot | Codeur | Dépend de |
|---|---|---|---|
| 0 | Config OpenCode, `AGENTS.md`, ce document | humain, cadré et relu par Fable | — |
| 1 | Validation du lot Docker commité ; corrections si échec | superviseur, `validator`, puis Kimi (`build`) | Docker Desktop démarré |
| 2 | `ros_publisher.py` : lire `POPPY_ROSBRIDGE_HOST/PORT`, refuser une cible réelle non explicite, arrêt propre | `build` | 1 |
| 3 | Évaluation sans écran : recharger modèle + `vec_normalize.pkl`, dérouler un épisode, vidéo `osmesa` | `build` | 1 |
| 4 | Outillage qualité : `ruff`, `pytest` minimal, `pyproject.toml` | `build` | 1 |
| 5 | Figer les versions (`requirements/*.lock` depuis `pip freeze` d'une image construite) | `build` | 1 |
| 6 | GPU (`compose.gpu.yaml`) si une mesure le justifie | `build` | 1, matériel |
| — | Vision : bloqué tant que `depth_anything_3` n'est pas identifié | — | question à l'ancienne équipe |
| — | Essai robot réel : lot séparé, humain présent, hors OpenCode | — | 2, réponse sur la ROS du robot |

Question ouverte à poser à l'ancienne équipe avant le lot 2 : **quelle distribution ROS 2 tourne sur le Poppy, et un `rosbridge_server` y est-il installé ?** L'ancienne connexion réussie (`origin/feat/docker`) passait par `rclpy` natif en ROS 2 Rolling avec `--network host`, ce qui n'est pas possible depuis Docker Desktop Windows ; le code actuel passe par websocket (`roslibpy`), ce qui suppose un rosbridge côté robot.

---

## 6. Dépannage OpenCode

| Symptôme | Cause / action |
|---|---|
| OpenCode refuse de démarrer après une modification de config | PowerShell : `$env:OPENCODE_DISABLE_PROJECT_CONFIG=1; opencode`, puis corriger `opencode.json` |
| Un agent n'apparaît pas dans `Tab` | fichier hors `.opencode/agent/`, ou `mode: subagent` (accessible par `@nom` seulement) |
| Le codeur ne trouve pas `AGENTS.md` | OpenCode lancé hors de la racine du dépôt |
| `model not found` | `opencode models openai` / `opencode models openrouter` pour l'identifiant exact ; `opencode auth list` pour les comptes |
| Une commande Docker est refusée à un agent | voulu : voir les listes de permissions dans `opencode.json` et `.opencode/agent/*.md` ; la lancer soi-même |
