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

**Les axes du robot ne sont pas ceux d'Humanoid-v5.** L'avant de Poppy est son
**−y local**, sa gauche son **+x**. La récompense projette la vitesse sur le cap lu
dans l'orientation du bassin, jamais sur un axe du monde — mesurer sur x reviendrait à
récompenser le pas chassé, ce qui a réellement eu lieu jusqu'en septembre 2026.
`tests/test_heading.py` verrouille la convention ; si ces tests rougissent, c'est que
le MJCF a changé de repère et que la récompense, les caméras et le pont sont tous à
revoir.

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

**La caméra de rendu tient à un détail non évident.** Gymnasium cherche une caméra
MuJoCo nommée `track` dans le modèle ; le MJCF Poppy n'en a pas, donc
`OffScreenViewer.render()` remet `cam.type` à `mjCAMERA_FREE` à chaque image et
annule silencieusement `DEFAULT_CAMERA_CONFIG`. C'est
`self.mujoco_renderer.camera_id = None`, dans `PoppyHumanoidEnv.__init__`, qui rend le
suivi possible. Ne pas le retirer en croyant nettoyer.

**Tout passe par Docker**, sauf le viewer interactif. `.venv/` est un Python 3.12
avec les dépendances de rendu, installé pour `scripts/viewer.py` et
`scripts/visualize.py` uniquement — voir `docs/TECHNIQUE.md` §4.3. Le dossier `venv/`
à côté, lui, est un reliquat cassé qui pointe vers un interpréteur disparu ; ne pas
s'en servir. Sous Windows, `python` tout court est intercepté par un raccourci
Microsoft Store qui ne fait rien : utiliser le chemin explicite de l'environnement, ou
`py` pour un script jetable sans dépendances.

## Vérifier

```bash
docker compose --profile dev run --rm dev ruff check .
docker compose --profile dev run --rm dev python -m pytest
```

Attendu : `All checks passed!` et `16 passed, 1 xfailed`. Le `xfailed` est voulu.

Preuve que la physique et la récompense n'ont pas bougé :

```bash
git diff archive/avant-clean-2026-09-18 -- assets configs
```

Ce diff doit être **vide**. `src/environments/poppy_humanoid_env.py` n'y est plus :
il a reçu les réglages de caméra, qui ne touchent qu'au rendu. Un diff sur ce fichier
ne doit montrer que le bloc `CAMERAS`, l'argument `default_camera_config` et la ligne
`camera_id = None`.
