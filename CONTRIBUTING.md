# Contribuer

Court volontairement. Ce qui n'est pas ici se discute avant d'être codé.

Pour faire tourner le projet : [`docs/TECHNIQUE.md`](docs/TECHNIQUE.md).
Pour comprendre comment il est construit : [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Le projet en trois lignes

Apprendre la marche à un humanoïde Poppy par renforcement (MuJoCo + Gymnasium +
Stable-Baselines3), puis transférer sur le robot réel via un pont ROS
(`roslibpy` → rosbridge websocket). Branche de travail : `clean`.

## Git

- Un lot = un ou plusieurs commits **cohérents et petits**. Un commit ne mélange pas
  configuration Docker, code Python et documentation, sauf si l'un ne fonctionne pas
  sans l'autre.
- **Indexation explicite par chemin** : `git add -- chemin/fichier`. Jamais
  `git add .` ni `git add -A` sans chemin.
- Messages au format [Conventional Commits](https://www.conventionalcommits.org/),
  **en français**, à l'impératif, sujet ≤ 72 caractères, corps expliquant le
  **pourquoi** : `feat:`, `fix:`, `build:` (Docker, dépendances), `docs:`,
  `refactor:`, `test:`, `chore:`.
- Avant tout commit : `git diff --check` (espaces, CRLF) et `git diff --cached --stat`.
- Fins de ligne : LF pour tout fichier livré en conteneur. `.gitattributes` fait foi —
  un fichier en CRLF dans une image Linux casse à l'exécution.

## Python

- **Python 3.11**, celui des images. Pas de syntaxe plus récente.
- PEP 8, 4 espaces, lignes ≤ 100 caractères. `from __future__ import annotations` en
  tête des modules qui en ont besoin.
- **Annotations de type** sur toutes les signatures publiques, paramètres et retour.
  `numpy.typing.NDArray` pour les tableaux.
- **Docstring** (style Google, en français) sur chaque module, classe et fonction
  publique : ce que ça fait, **les unités physiques** (rad, deg, m, s, N) et la
  **forme des tableaux**. Sur ce projet, une unité implicite est un bug en attente.
- Imports en tête de fichier, groupés stdlib / tiers / projet. Pas d'import dans une
  fonction, sauf pour éviter une dépendance lourde optionnelle — et alors commenté.
- Une fonction fait une chose. Au-delà de ~50 lignes, découper.
- Pas de `print` dans `src/` : `logging.getLogger(__name__)`. Les `print` sont tolérés
  dans `scripts/`, qui s'adressent à un humain.
- Pas de `except Exception: pass`. Attraper précis, journaliser, relancer ou échouer
  clairement.
- Pas de `time.sleep` dans une boucle de contrôle sans justification écrite et sans
  paramètre.
- **Rien à l'import.** Pas de lecture de fichier, pas de chargement de modèle, pas
  d'appel réseau au niveau module. Un `import` doit être gratuit et sans effet de
  bord ; sinon il échoue selon le répertoire courant, et le linter comme les tests
  deviennent imprévisibles.

Avant de proposer quoi que ce soit :

```bash
docker compose --profile dev run --rm dev ruff check .
docker compose --profile dev run --rm dev python -m pytest
```

Attendu : `All checks passed!` et `16 passed, 1 xfailed`.

## Tests

Une logique non triviale laisse **un** contrôle exécutable derrière elle : le plus
petit test qui échoue si la logique casse. Pas de framework, pas de fixture élaborée.

Trois tests qui tournent valent mieux qu'une suite ambitieuse jamais terminée.

Le seul mode de défaillance qui mérite toujours un test est le **silencieux** : celui
qui ne lève rien, ne rougit nulle part, et produit des chiffres faux. Une
normalisation mal appariée à son modèle, une observation dont les indices se décalent,
des angles désalignés de leurs moteurs.

## Configuration, chemins, secrets

- **Aucune adresse IP, port, chemin absolu ou identifiant en dur dans le code.** Tout
  paramètre d'environnement se lit dans une variable préfixée `POPPY_`, avec une
  valeur par défaut **sûre** — jamais l'adresse d'un robot réel — et documentée.
- Hyperparamètres : dans les YAML de `configs/`, jamais dans le code.
- Chemins : `pathlib.Path`, relatifs à la racine du dépôt (`/workspace` en conteneur).
  Jamais `C:\...` ni `/home/...`.
- Aucun secret, jeton ou `.env` commité. Un `.env.example` si un exemple est nécessaire.

## Ce qu'on ne commite pas

Le dépôt a déjà porté 1,2 Go de checkpoints d'entraînement. On ne recommence pas.

- Pas de modèles entraînés, sauf dans `models/`, et alors avec leur ligne dans
  [`models/README.md`](models/README.md) : à quoi ils servent et ce qu'ils valent.
- Pas de sorties d'exécution : `logs/`, `figs/`, `*.mp4`, fichiers TensorBoard.
- Pas de données de test lourdes. Si des données sont nécessaires, elles vivent hors
  du dépôt et un script les récupère.
- Pas de fichiers personnels : CV, notes, réglages d'IDE.

## Dépendances et Docker

- Une dépendance ajoutée va dans **le** `requirements/<cible>.txt` de l'image qui en a
  besoin, pas dans tous. Le `requirements.txt` de la racine ne sert qu'à
  l'installation native complète.
- **Dire pourquoi** : chaque ligne non évidente porte un commentaire nommant le module
  qui l'importe.
- Ne pas changer d'image de base, de distribution ROS ou d'index torch sans que ce
  soit le sujet du lot.
- Un service compose ne démarre **jamais** par défaut vers un robot réel.

## Sécurité robot

Cette section n'est pas négociable.

- Tout code susceptible d'envoyer une commande moteur doit : lire sa cible depuis la
  configuration, **refuser de démarrer** si la cible d'un robot réel n'est pas
  explicitement fournie, prévoir une limite d'amplitude, et un arrêt propre
  (`finally` ou gestionnaire de signal).
- **Un test logiciel réussi ne vaut jamais autorisation d'essai sur le matériel.**
  L'essai réel est un lot séparé, supervisé par un humain présent.
- Le pont n'a jamais parlé à un vrai robot. La première fois qu'il le fera, quelqu'un
  aura la main sur l'alimentation.

## Ce qu'on livre à la fin d'un lot

1. Les fichiers touchés, et pourquoi.
2. Les commandes exactes exécutées, avec leur résultat.
3. Ce qui a été testé, ce qui **ne l'a pas** été, et les hypothèses restantes.

Distinguer toujours ce qui a été **constaté** de ce qui est **supposé**.
