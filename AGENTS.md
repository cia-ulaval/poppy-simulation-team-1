# AGENTS.md — contrat de travail pour les agents et les humains

Ce fichier est lu automatiquement par OpenCode (`instructions` dans `opencode.json`). Il s'applique à tout agent qui écrit, relit ou valide du code dans ce dépôt. Il est court volontairement : ce qui n'est pas ici se discute dans le cadrage d'un lot.

## 1. Contexte en trois lignes

- Projet : apprendre la marche à un humanoïde Poppy par RL (MuJoCo + Gymnasium + Stable-Baselines3), puis transférer sur le robot réel via un pont ROS (`roslibpy` → rosbridge websocket).
- Branche de travail : `clean` (base : `model-for-presentation`). Tout est local, **aucun push** pendant cette phase.
- Référence Docker : `docs/DOCKER.md`. Processus de travail : `docs/WORKFLOW.md`. Cadrage historique : `docs/DEMARRAGE_PROPRE.md`. Les autres documents de `docs/` (`GUIDE_EQUIPE.md`, `ETAT_DES_LIEUX.md`, `CV.md`) sont des brouillons de recherche : les lire avec prudence, ne pas les considérer comme des décisions.

## 2. Rôles

| Agent | Modèle | Écrit du code ? | Rôle |
|---|---|---|---|
| `plan` | Fable (Anthropic) | non | Cadre le lot : périmètre, fichiers touchés, critères d'acceptation, commandes de test |
| `architect` | OpenAI `gpt-5.6-sol` | non | Conseiller ponctuel : prépare une stratégie concise pour Kimi |
| `build` | Kimi `k2.7-code` | **oui** | Codeur principal : implémente le lot cadré, rien de plus |
| `build-openai` | OpenAI `gpt-6-astra` | oui | Codeur de secours, seulement si Kimi est bloqué |
| `reviewer` | Fable | non | Relit le diff, rapporte fichier:ligne, conséquence, vérification |
| `validator` | Fable | non | Vérifie les sorties des tests définis à l'avance et rapporte les écarts |
| humain | — | — | Tranche, indexe (`git add -- chemin`), commite |

Le flux normal est : Fable cadre, OpenAI conseille si nécessaire, Kimi code,
le superviseur exécute les tests, puis Fable relit et valide les preuves. Le codeur n'est jamais son propre relecteur. Le
relecteur et le validateur n'ont pas le droit d'écrire. Personne ne pousse.

## 3. Règles Git (obligatoires)

- **Interdit** : `git push`, `git rebase`, `git merge`, `git reset --hard`, `git clean`, `git checkout --`, `git restore` et suppression de branche. Ces commandes sont bloquées par `opencode.json` ; ne pas chercher à les contourner.
- Un lot = un ou plusieurs commits **cohérents et petits**. Un commit ne mélange pas config Docker, code Python et documentation, sauf si l'un ne fonctionne pas sans l'autre.
- Indexation explicite par chemin : `git add -- chemin/fichier`. Jamais `git add .` ni `git add -A` : le dépôt contient des logs et modèles lourds.
- Messages au format Conventional Commits, en français, impératif, sujet ≤ 72 caractères, corps expliquant le **pourquoi** :
  `feat:`, `fix:`, `build:` (Docker, dépendances), `docs:`, `refactor:`, `test:`, `chore:`.
- Avant tout commit : `git diff --check` (espaces/CRLF), `git diff --cached --stat`, relecture par `reviewer`.
- Fins de ligne : LF pour tout fichier livré en conteneur (`.gitattributes` fait foi).

## 4. Normes Python

- Python 3.11 (celui des images). Pas de syntaxe plus récente.
- PEP 8, 4 espaces, lignes ≤ 100 caractères, `from __future__ import annotations` en tête des modules qui en ont besoin.
- **Type hints** sur toutes les signatures publiques (paramètres et retour). `numpy.typing.NDArray` pour les tableaux.
- **Docstring** (style Google, en français ou anglais mais cohérent dans le fichier) sur chaque module, classe et fonction publique : ce que ça fait, unités physiques (rad, deg, m, s), forme des tableaux.
- Imports en tête de fichier, groupés stdlib / tiers / projet. Pas d'import dans une fonction sauf pour éviter une dépendance lourde optionnelle, et alors commenté.
- Une fonction fait une chose ; au-delà de ~50 lignes, découper.
- Pas de `print` dans `src/` : utiliser `logging.getLogger(__name__)`. Les `print` sont tolérés dans `scripts/` pour la sortie utilisateur.
- Pas de `except Exception: pass`. Attraper précis, journaliser, relancer ou échouer clairement.
- Pas de `time.sleep` dans une boucle de contrôle sans justification écrite en commentaire et sans paramètre.

## 5. Configuration, chemins, secrets

- **Aucune adresse IP, port, chemin absolu ou identifiant en dur dans le code.** Tout paramètre d'environnement se lit via une variable d'environnement préfixée `POPPY_` avec une valeur par défaut **sûre** (jamais l'adresse d'un robot réel) et documentée dans `docs/DOCKER.md`.
- Hyperparamètres : dans les YAML de `configs/`, pas dans le code.
- Chemins : `pathlib.Path`, relatifs à la racine du dépôt (`/workspace` en conteneur), jamais `C:\...` ni `/home/...`.
- Aucun secret, token ou `.env` commité. `.env.example` si un exemple est nécessaire.

## 6. Dépendances et Docker

- Une dépendance ajoutée va dans **le** fichier `requirements/<cible>.txt` de l'image qui en a besoin, pas dans tous. `requirements.txt` racine ne sert qu'à l'installation native complète.
- Dire pourquoi : chaque ligne de `requirements/*.txt` non évidente a un commentaire (quel module l'importe).
- Ne pas changer d'image de base, de distribution ROS ou d'index torch sans que le lot le prévoie explicitement.
- Un service compose ne démarre **jamais** par défaut vers un robot réel. La cible par défaut du pont est le faux robot `rosbridge`.

## 7. Sécurité robot

- Tout code susceptible d'envoyer une commande moteur doit : lire sa cible depuis la configuration, refuser de démarrer si la cible n'est pas explicitement fournie pour un robot réel, prévoir une limite d'amplitude et un arrêt propre (`finally` / gestionnaire de signal).
- Un test logiciel réussi ne vaut jamais autorisation d'essai sur le matériel. L'essai réel est un lot séparé, supervisé par un humain présent.

## 8. Ce qu'un agent doit produire à la fin d'un lot

1. Liste des fichiers touchés et pourquoi.
2. Commandes exactes exécutées, avec codes de sortie.
3. Ce qui a été testé, ce qui **n'a pas** été testé, et les hypothèses restantes.
4. Aucun commit : c'est l'humain qui commite après revue.

Distinguer toujours ce qui a été **constaté** de ce qui est **supposé**.
