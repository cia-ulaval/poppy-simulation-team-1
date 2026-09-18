# Notes pour les agents

Les règles de fond sont dans [`CONTRIBUTING.md`](CONTRIBUTING.md) — normes Python,
règles Git, configuration, **sécurité robot**. Elles s'appliquent telles quelles. Ce
fichier n'ajoute que les pièges propres à ce dépôt.

## Avant de modifier quoi que ce soit

Lire [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), en particulier le contrat
observation → action. Tout passe par `src/environments/poppy_humanoid_env.py` :
entraînement, évaluation, visualisation et robot.

## Les pièges

**Changer l'espace d'observation ou d'action invalide les onze modèles de `models/`.**
Une politique chargée dans un espace différent échoue sur `spaces must have the same
shape`. Ce n'est pas une modification à glisser au passage : c'est une décision
d'équipe.

**Ne pas « corriger » les défauts connus sans qu'on le demande.** Trois d'entre eux
changeraient le comportement de l'entraînement et sont listés dans
[`docs/TECHNIQUE.md` §8](docs/TECHNIQUE.md) : l'espace d'action non normalisé (un
`xfail(strict=True)` le documente), la restitution du sol écrite au mauvais indice de
`geom_solimp`, et `prev_action` pénalisé sans être observé. Les signaler, pas les
réparer spontanément.

**La vision de `src/sensors/` est en pause et ne fonctionne pas.** Ne pas essayer de
la réparer. Son état exact, défaut par défaut, est dans
[`src/sensors/README.md`](src/sensors/README.md).

**Le dépôt a porté 1,2 Go de checkpoints.** Ne rien ajouter de lourd. Indexer par
chemin explicite, jamais `git add .`.

**Les deux `venv/` locaux sont cassés** (mauvaise version, dépendances absentes). Tout
passe par Docker. Pour un script Python jetable, `py` fonctionne, mais sans les
dépendances du projet.

## Vérifier

```bash
docker compose --profile dev run --rm dev ruff check .
docker compose --profile dev run --rm dev python -m pytest
```

Attendu : `All checks passed!` et `11 passed, 1 xfailed`. Le `xfailed` est voulu.

Preuve que le cœur n'a pas bougé, quand c'est ce qu'on veut montrer :

```bash
git diff archive/avant-clean-2026-09-18 -- src/environments/poppy_humanoid_env.py assets configs
```
