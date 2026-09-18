# Perception de profondeur — en pause

> **Statut : rien ici ne tourne aujourd'hui, et rien n'est branché sur la politique.**
> Le code est conservé parce qu'une caméra stéréo est envisagée. Ce fichier dit
> exactement où en est chaque morceau, pour que la reprise ne commence pas par une
> demi-journée d'archéologie.

La décision de mise en pause est de l'équipe (septembre 2026). Aucun module de ce
dossier n'est importé par l'entraînement, l'évaluation, le viewer ou le pont robot.

## Les quatre modules

| Fichier | Ce que ça fait | Ce qui bloque |
|---|---|---|
| `depth_server.py` | Serveur websocket (port 8000) : reçoit une image JPEG, renvoie une carte de profondeur estimée par un modèle monoculaire. | **Ne démarre pas.** Importe `depth_anything_3`, qui n'est pas publié sur PyPI. L'image Docker `vision` se construit mais le serveur s'arrête à l'import. |
| `depth_client.py` | Client : capture la webcam, envoie les images au serveur, affiche la profondeur et surligne les régions trop proches (`cv2.imshow`). | Dépend d'un serveur qui ne démarre pas. Importe aussi `scipy`, qui ne figurait dans aucun `requirements/`. |
| `depthmap_analysis.py` | Découpe une carte de profondeur en régions verticales et donne la distance minimale de chaque région. | Partiellement cassé, voir ci-dessous. |
| `depth_rays.py` | Lance 64 rayons horizontaux dans MuJoCo depuis la tête et renvoie la distance au plus proche obstacle par secteur. Pas de caméra du tout. | **Écrit pour Humanoid-v5, pas pour Poppy.** Voir ci-dessous. |

## Ce qu'une caméra stéréo changerait

Une stéréo remplace l'**estimation** de profondeur, pas son **exploitation** :

- `depth_server.py` et `depth_client.py` deviennent inutiles — une stéréo produit la
  profondeur par disparité, sans modèle d'apprentissage ni serveur d'inférence. C'est
  le morceau à jeter.
- `depthmap_analysis.py` reste pertinent : il consomme une carte de profondeur, peu
  importe d'où elle vient.
- `depth_rays.py` est orthogonal : il mesure dans le simulateur, pas dans le monde
  réel. C'est la source de vérité qui servira à valider ce que la caméra rapporte.

## Défauts connus, non corrigés

Relevés en lisant le code, **pas corrigés** : ils demandent de savoir ce que l'auteur
voulait, et deviner produirait du code faux qui a l'air juste.

**`depth_rays.py` — cherche un corps `torso` qui n'existe pas dans Poppy.**
Le modèle Poppy n'a pas de corps `torso` ; ses corps du haut sont `chest`,
`bust_motors`, `neck`, `head`. Appelé sur Poppy, `cast_horizontal_rays` lève une
`ValueError`. Le paramètre `body_name` permet de passer le bon nom, mais **lequel
est le bon reste à trancher** : `chest` est le plus proche d'un torse, `head` est
l'endroit où serait la caméra. Le décalage `head_offset` de 19 cm a été choisi pour
Humanoid-v5 et n'a aucune raison de valoir pour Poppy, qui fait 83 cm.

**`depth_rays.py` — les rayons touchent le robot lui-même.**
`mj_multiRay` reçoit `bodyexclude = body_id`, qui n'exclut que le corps de
référence. Les bras et les jambes bloquent donc les rayons. Mesuré sur le modèle
Poppy au repos, depuis `chest` : `[0.85, 1.09, 1.60, 3.24, 10, 10, 10, 10]` — les
quatre premiers secteurs voient le robot, pas le monde. Un capteur d'obstacles qui
rapporte 85 cm quand la pièce est vide est inutilisable tel quel.

**`depthmap_analysis.close_warning` — ne peut pas s'exécuter.**
Deux erreurs dans la même fonction : elle lit une variable globale `splitframe` qui
n'est définie nulle part (`NameError`), et elle fait `size = frame.size` puis
`size[0]` alors que `.size` est un entier (`TypeError`). Elle n'a jamais été appelée
que depuis un bloc `__main__` qui définissait `splitframe` par effet de bord. La
variable `regions` qu'elle calcule n'est utilisée nulle part non plus.

**`depthmap_analysis.loop` — `splitFrames` accumulé puis jeté.**
De même pour `lengths` dans `frame_splitting`.

**Constantes en dur.** Le `300` de `splitframe_to_1Ddepthmap` (`focal * min / 300`)
et le `3.2` du bloc `__main__` ne sont documentés nulle part. Sans savoir de quelle
caméra vient ce `300`, la distance calculée n'a pas d'unité vérifiable.

## Ce qui a été corrigé pendant le nettoyage

Uniquement ce qui empêchait le code de s'*importer*, sans toucher aux algorithmes :

- `depthmap_analysis.py` chargeait 44 fichiers `.npy` **au moment de l'import**,
  relativement au répertoire courant. Un simple `import` échouait ou réussissait selon
  l'endroit d'où Python était lancé, et coûtait 33 Mo de lecture disque. Le chargement
  est descendu dans `load_frames()`, avec un chemin absolu.
- `depth_client.py` faisait `import depthmap_analysis`, un import nu de module frère
  qui ne résolvait que si `src/sensors/` était lui-même dans `sys.path`. Devenu
  `from src.sensors import depthmap_analysis`.
- `__init__.py` réexportait `cast_horizontal_rays`, donc tirait `mujoco` dès qu'on
  touchait au paquet. L'image `vision` n'a pas MuJoCo : `import src.sensors` y
  échouait. Le fichier n'importe plus rien.
- `DepthRayConfig`, dans `src/config/settings.py`, dupliquait exactement les valeurs
  par défaut de `cast_horizontal_rays` et n'avait aucun utilisateur. Supprimée.

## Les captures de profondeur

`scripts/vision/frames/` contient 44 fichiers `.npy` (33 Mo) et `frames.zip` est leur
copie compressée (26 Mo) — la même charge stockée deux fois. Ce sont les seules
données d'exemple de `depthmap_analysis`.

Elles **ne sont plus suivies par git** : 59 Mo de données de test dans un dépôt de
code, pour un module en pause. Elles restent sur le poste où elles ont été produites,
et dans le tag `archive/avant-clean-2026-09-18` :

```bash
git show archive/avant-clean-2026-09-18:scripts/vision/frames.zip > frames.zip
```

## Reprendre

1. Trancher la caméra. Si c'est une stéréo, `depth_server.py` et `depth_client.py`
   sortent, et avec eux `requirements/vision.txt`, la cible Docker `vision` et le
   service compose.
2. Choisir le corps de référence de `depth_rays.py` pour Poppy, et remesurer
   `head_offset` sur le modèle.
3. Décider comment la profondeur entre dans la politique. C'est la vraie question :
   l'observation fait 63 dimensions aujourd'hui (`docs/ARCHITECTURE.md`), et
   l'élargir rend incompatibles tous les modèles déjà entraînés.
