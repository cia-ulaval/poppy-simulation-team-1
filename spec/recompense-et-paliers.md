# Récompense et entraînement par paliers — proposition

> **Statut : proposition, à discuter en équipe.** Rien n'est décidé. Ajoutez vos avis
> en bas du fichier ou en commentaire de l'issue.

## Le constat

L'entraînement **plafonne**. Sur 10 M de pas, il atteint 98 % de la récompense
maximale vers 2,8 M, puis ne gagne plus que 5 % sur les 7,2 M suivants. Ajouter des
pas ne sert plus à rien : c'est la récompense qui bloque, pas l'apprentissage.

Trois causes :

1. **La vitesse est plafonnée à 0,5 m/s**, alors que le meilleur modèle marche à
   0,65 m/s. Au-delà de 0,5, aller plus vite ne rapporte plus rien et coûte en
   pénalités.
2. **Rester debout rapporte presque autant qu'avancer.** `healthy` + `upright` =
   2000 points sur 4800 possibles par épisode (42 %), rien que pour ne pas tomber.
3. **Au plafond, il n'y a plus de pente.** Toutes les bonnes démarches valent environ
   4700 points : rien ne pousse vers une démarche plus naturelle ou plus robuste.

### La récompense actuelle, par pas de 10 ms

Code : `src/environments/poppy_humanoid_env.py`, méthode `step()` (l. 417).

| Terme | Calcul | Max / pas |
| --- | --- | --- |
| `forward_reward` | 5 × vitesse avant, plafonnée à 0,5 m/s | 2,5 |
| `healthy_reward` | 1 si le bassin est entre 25 et 70 cm | 1 |
| `upright_reward` | Verticalité du buste (1 = droit) | 1 |
| `gait_reward` | 0,3 sur un pied, 0,1 sur deux, 0 en l'air | 0,3 |
| `lateral_cost` | − 0,5 × vitesse de côté | pénalité |
| `ctrl_cost` | − 0,1 × Σ action² | pénalité |
| `action_rate_cost` | − 0,05 × Σ (action − action précédente)² | pénalité |
| `joint_vel_cost` | − 0,005 × Σ vitesse articulaire² | pénalité |

Les poids sont écrits en dur dans le constructeur (l. 232), pas dans
`configs/poppy_robust.yaml`.

---

## Les poids de la récompense dans le YAML

Aujourd'hui, essayer un autre réglage de la récompense oblige à modifier le code, et
rien ne garde la trace du réglage utilisé pour un modèle.

- [ ] Créer une section `reward:` dans `configs/poppy_robust.yaml`, avec tous les
      poids et `max_forward_vel`.
- [ ] Les faire lire par `src/config/loaders.py` et passer à `PoppyHumanoidEnv`.
- [ ] Garder une copie du YAML à côté de chaque modèle entraîné.

## Les réglages de l'apprentissage — à discuter

Section `ppo` de `configs/poppy_robust.yaml`. Deux réglages actuels freinent
l'apprentissage. Les autres sont listés pour comprendre ce qu'ils font.

| Paramètre | Ce qu'il fait, en simple | Aujourd'hui | Piste |
| --- | --- | --- | --- |
| `learning_rate` | La taille de chaque correction. Trop petite : le robot apprend lentement et reste coincé dans sa première démarche. Trop grande : l'apprentissage devient instable. | 2.5e-5 (environ 10 fois sous la valeur courante pour PPO) | 3e-4, qui diminue au fil de l'entraînement |
| `ent_coef` | Un bonus pour garder un peu de hasard dans les mouvements. Sans lui, le robot arrête d'essayer autre chose dès qu'il a trouvé une démarche qui marche. | 0.0 | 0.005 |
| `n_epochs` | Combien de fois on réapprend sur le même lot d'essais. Trop : le robot apprend « par cœur » ce lot au lieu de généraliser. | 20 | 5 à 10 |
| `clip_range` | Un garde-fou : la limite de changement de la politique à chaque mise à jour. | 0.3 | 0.2 |

À tester **un par un**, sur le banc d'essai, pour savoir lequel aide.

---

## Les anciens modèles vont-ils à la poubelle ?

**Non.** Ils restent la référence, à deux conditions.

**1. On compare sur des mesures physiques, pas sur la récompense.**
La récompense change d'une version à l'autre, donc ses chiffres ne se comparent pas.
Les mesures physiques, si : `scripts/evaluate.py` sort déjà la distance parcourue vers
l'avant, la vitesse, la dérive latérale, la verticalité et le taux d'épisodes sans
chute. C'est d'ailleurs déjà comme ça que `models/README.md` compare les modèles :

| | 19 septembre | 8 avril |
| --- | ---: | ---: |
| Avance | +6,58 m | +5,63 m |
| Vitesse | 0,65 m/s | 0,56 m/s |
| Dérive | −0,38 m | −0,86 m |
| Sans chute | 100 % | 100 % |
| Récompense | 4461 | 4565 ← trompeuse |

**2. On garde la version du code qui sait les faire tourner.**
La correction de l'espace d'action (voir la liste des défauts plus bas) change les
bornes des actions. Un ancien modèle ne se charge alors plus dans le nouvel
environnement. Solution : poser un **tag git** avant le premier changement (par exemple
`archive/recompense-v1`) ; les anciens modèles s'évaluent avec ce tag, les nouveaux avec
`main`, **sur le même banc d'essai**.

Et si on veut comparer « à armes égales », on peut **réentraîner** l'ancienne
configuration à partir du tag : le code, le YAML et les graines y sont.

### Le banc d'essai commun

Pour que les chiffres restent comparables d'une version à l'autre, toutes les
évaluations utilisent les mêmes conditions :

- sol plat, sans randomisation (`evaluate.py` sans `--floor-noise`) ;
- mêmes graines, même nombre d'épisodes (par exemple `--episodes 20 --seed 1`) ;
- mêmes mesures : avance, vitesse, dérive, verticalité, % sans chute ; à ajouter :
  dérive de cap, appui gauche / droite, cadence, hauteur des pieds (voir
  « Ce que montre la démarche du modèle du 19 septembre »).

Chaque palier ajoute ensuite **son** test (sol accidenté, poussées), mais le banc plat
reste toujours mesuré.

---

## Les trois paliers

Chaque palier ajoute **une** difficulté. On ne passe au suivant que quand le précédent
est réussi sur le banc d'essai. Idée : un palier repart du meilleur modèle du palier
précédent, plutôt que de zéro.

**Règle commune aux trois paliers : l'observation garde ses 63 dimensions.** Rien n'y
est ajouté. C'est ce qui permet à un palier de repartir du modèle du précédent. La
récompense, elle, peut évoluer librement : c'est un nombre calculé à côté, elle n'ajoute
aucune dimension.

**La fin d'un épisode ne change pas** : un essai s'arrête quand le bassin sort de 25 à
70 cm de hauteur. Cette règle ne sert qu'à l'entraînement.

### Palier 1 — Sol plat, récompense corrigée

**But :** marcher mieux et plus vite sur sol plat, sans plafond artificiel.

Changements proposés :

- [ ] **Lever le plafond de vitesse** — monter `max_forward_vel` (par exemple à
      0,8 m/s), ou récompenser le **suivi d'une vitesse cible** plutôt que la vitesse
      brute.
- [ ] **Réduire le socle** — baisser `healthy` et `upright`, pour qu'avancer pèse
      davantage.
- [ ] **Récompenser la qualité de la marche** — temps de vol des pieds, symétrie
      gauche / droite, énergie consommée. De préférence à partir de grandeurs que le
      vrai robot pourra mesurer (IMU, pression sous les pieds, moteurs) : ce que la
      politique optimise reste alors vérifiable sur Poppy.
- [ ] **Remettre de l'exploration** — `ent_coef` légèrement positif (aujourd'hui 0).
- [ ] **Sortir les poids dans le YAML** — pour essayer des réglages sans toucher au
      code.
- [ ] Corriger les défauts de `TECHNIQUE.md` §8 attribués au palier 1 (voir plus bas).
- [ ] Étudier les quatre propositions tirées des mesures du 9 octobre (voir
      « Ce que montre la démarche du modèle du 19 septembre ») : virage, boiterie,
      piétinement, postures.

**Réussi quand :** sur le banc plat, plus loin et plus vite que le modèle du
19 septembre, sans chute, et la courbe d'entraînement ne plafonne plus avant la fin.

### Palier 2 — Sol accidenté

**But :** marcher sur un sol qui n'est pas plat.

Changements proposés :

- [ ] Remplacer ou compléter le sol plat par un **sol à relief aléatoire**, régénéré à
      chaque épisode : bosses, creux, petites marches, pentes douces. Piste MuJoCo : un
      `hfield` (champ de hauteur), dont on modifie les hauteurs à chaque `reset()`.
- [ ] Monter la difficulté progressivement : relief faible au début, plus marqué
      ensuite.
- [ ] Corriger les défauts attribués au palier 2 (voir plus bas).

**Réussi quand :** sur sol accidenté, le robot tient X % des épisodes sans chute
*(seuil à fixer)*, et ne régresse pas sur le banc plat.

### Palier 3 — Poussées

**But :** encaisser des poussées et se rattraper, comme sur un vrai plancher avec des
vraies personnes autour.

Changements proposés :

- [ ] **Corriger les poussées actuelles, qui n'en sont pas.** Le code écrit dans
      `xfrc_applied[1, 3:5]`, qui est un **couple** (une torsion), pas une force. La
      force est dans `[1, 0:3]`. Vérifié sur MuJoCo 3.13. Ce défaut n'est pas encore
      dans `TECHNIQUE.md`.
- [ ] Donner aux poussées une durée et une intensité réalistes (aujourd'hui : un seul
      pas de 10 ms, intensité 1 à 5). L'équipe robot peut mesurer une vraie poussée sur
      Poppy.
- [ ] Monter l'intensité progressivement.

**Réussi quand :** le robot encaisse une poussée de X N *(à fixer)* sans tomber dans
Y % des cas, et ne régresse pas sur les paliers 1 et 2.

---

## Les défauts de `docs/TECHNIQUE.md` §8

| Défaut | Palier | Casse les anciens modèles ? |
| --- | --- | --- |
| **Espace d'action non normalisé** : la politique peut viser au-delà des limites mécaniques (±1,8 à ±7,3 au lieu de ±1). Le test `xfail(strict=True)` passera au rouge quand ce sera corrigé : retirer le marqueur. | 1 | **Oui** : l'espace d'action change. |
| **`prev_action` pénalisé mais pas observé** : la politique est punie pour un écart avec l'action précédente, qu'elle ne voit pas. **Proposition : garder la pénalité, ne rien ajouter à l'observation** (voir ci-dessous). | — | Non. |
| **Restitution (rebond) écrite au mauvais endroit** : `geom_solimp[floor, 4]` au lieu de `solref`. Le sol ne rebondit jamais. | 2 | Non, mais l'entraînement change. |
| **Poussées appliquées comme un couple** (pas encore dans §8, à y ajouter). | 3 | Non, mais l'entraînement change. |
| Les onze modèles d'avril ont la récompense aux axes faux. | — | Déjà corrigé dans le code ; c'est le modèle du 19 septembre qui sert de référence. |
| Le pont n'a jamais parlé à un vrai robot ; la vision ne démarre pas ; la variante GPU n'a jamais tourné. | — | Hors sujet ici (équipe robot, vision en pause, matériel). |

**Pourquoi garder la pénalité `prev_action` telle quelle :**

- L'ajouter à l'observation la ferait passer de 63 à 88 dimensions, contre la règle
  commune aux paliers.
- La politique n'est pas aveugle : une action est une position cible, et les positions
  et vitesses des articulations, qu'elle observe, en gardent la trace.
- La pénalité est utile pour le vrai robot : des commandes qui changent brutalement
  d'un pas à l'autre fatiguent les servomoteurs.
- Elle pèse peu : les quatre pénalités réunies coûtent environ 270 points par épisode
  au meilleur modèle (4730 gagnés, 4461 au total), soit 6 %.

À revoir si les vidéos montrent des tremblements, ou si la pénalité empêche la
démarche d'évoluer.

---

## Ce que montre la démarche du modèle du 19 septembre

> **Statut : propositions**, ajoutées le 9 octobre 2026. Rien n'est décidé : elles
> seront tranchées avec le reste au moment de figer la récompense.

Mesures sur `models/2026-09-19_recompense-corrigee/best_model.zip`, sol plat, 5 graines
(1 à 5), 1000 pas chacune. Les chiffres se répètent d'une graine à l'autre.

| Mesure | Valeur | Ce que ça veut dire |
| --- | --- | --- |
| Dérive de cap en 10 s | **+45° à +52°**, toujours vers la gauche | il marche en arc de cercle |
| Appui sur le pied droit seul / gauche seul | **35 % / 58 %** | il boite |
| Amplitude de hanche (tangage) droite / gauche | 25° / 44° | idem |
| Rotation moyenne de la hanche droite | −16,5° (gauche : −1,2°) | jambe droite tournée |
| Cadence | **4,4 pas/s**, pas de ~15 cm | il piétine |
| Appui simple | 94 % du temps | `gait_reward` payé presque à chaque pas de 10 ms |
| Hauteur des pieds en vol | 5,5 cm (droit), 7,9 cm (gauche) | correct |
| Coudes en butée | **~100 % du temps** | pose de départ à 1° de la limite |
| Tronc (`abs_y`, `bust_y`) | ~−13° chacun en moyenne | penché, et rien ne le voit |
| Moteurs à leur couple maximal | 3–4 % du temps | correct |
| Actions hors de [−1, 1] | quasi jamais (max 1,25) | correct |

### Proposition 1 — Arrêter de tourner

**Cause.** La récompense mesure l'avance *dans la direction où le robot regarde*. C'est
ce qui a corrigé le pas chassé, mais tourner ne coûte donc rien : ni
`forward_reward`, ni `lateral_cost` ne voient un virage.

**Proposition.** Pénaliser la vitesse de rotation autour de l'axe vertical
(`yaw_rate_cost = w × ω_z²`). Variante plus tard : un cap cible fixé au `reset()` et
une récompense d'avance dans *cette* direction. Elle demande d'ajouter une consigne à
l'observation, donc on l'écarte pour l'instant.

### Proposition 2 — Corriger la boiterie

**Proposition.** Au choix :
- une **récompense de symétrie** (temps d'appui et amplitudes gauche / droite proches) ;
- ou l'**augmentation miroir** : chaque transition est aussi apprise en version
  gauche-droite inversée. C'est souvent plus efficace qu'un terme de récompense. Il
  faut une table de correspondance gauche ↔ droite des articulations, des
  observations et des actions, mais pas de nouvelle dimension.

La boiterie et le virage sont probablement liés : corriger l'un peut corriger l'autre.

### Proposition 3 — Des pas plus longs

**Cause.** `gait_reward` donne 0,3 dès qu'un seul pied touche, à chaque pas de 10 ms.
Alterner très vite rapporte autant que marcher posément.

**Proposition.** Remplacer ce terme par une récompense de **temps de vol** : au moment
où un pied se repose, récompenser `(durée en l'air − 0,25 s)`. C'est le terme classique
des marches apprises (legged_gym, Isaac Lab). Le seuil de 0,25 s est à régler.

### Proposition 4 — Deux postures à surveiller

- **Butées articulaires** : une pénalité quand une articulation s'approche de ses
  limites. Les coudes du modèle actuel y restent en permanence ; sur le vrai robot,
  garder un moteur en butée le fait chauffer. À voir aussi : la pose de départ des
  coudes, à 1° de la limite.
- **Verticalité du haut du corps** : `upright_reward` se mesure sur le **bassin**. Le
  tronc peut se plier sans pénalité. Proposition : mesurer la verticalité sur la
  poitrine ou la tête, ou ajouter une pénalité sur les angles du tronc.

### Avec quels capteurs ?

**La récompense n'est pas une observation.** Elle n'est calculée qu'en simulation,
pendant l'entraînement ; le vrai robot ne la calcule jamais. Elle peut donc utiliser
tout ce que le simulateur sait, **sans ajouter de dimension**. Il faut seulement que la
politique puisse percevoir ce qu'on lui demande de corriger :

| Proposition | Ce que la politique doit percevoir | Capteur sur le robot | Nouvelle dimension ? |
| --- | --- | --- | --- |
| 1. Rotation | vitesse de rotation du bassin (déjà observée) | gyroscope de l'IMU | non |
| 2. Symétrie / miroir | angles des articulations | encodeurs des moteurs | non |
| 3. Temps de vol | contacts des pieds (déjà observés) | capteurs de pression sous les pieds | non |
| 4. Butées, tronc | angles des articulations, inclinaison du bassin | encodeurs + IMU | non |

**Les quatre propositions tiennent dans les 63 dimensions**, à condition d'avoir une IMU
(avec gyroscope) et des capteurs de contact sous les pieds.

**À noter pour le moment de figer le contrat d'observation.** Les 63 dimensions
actuelles (`qpos[2:]` + `qvel` + contacts) contiennent déjà des grandeurs que le vrai
robot ne mesure pas :

| Dans l'observation | Mesurable sur le robot ? |
| --- | --- |
| 25 angles articulaires et leurs vitesses | oui, encodeurs |
| Inclinaison du bassin (roulis, tangage) | oui, IMU |
| Vitesse de rotation du bassin | oui, gyroscope |
| Contacts des pieds | oui, si capteurs de pression |
| **Hauteur du bassin** (`qpos[2]`) | **non** |
| **Vitesse d'avance du bassin** (`qvel[0:3]`) | **non**, à estimer, imprécis avec une IMU seule |
| **Cap absolu** (dans le quaternion) | dérive avec le temps |

La réponse habituelle est une politique **asymétrique acteur / critique** : l'acteur,
qui tourne sur le robot, ne voit que ce que les capteurs mesurent ; le critique, qui
n'existe qu'à l'entraînement, voit tout le simulateur. Cela change l'espace
d'observation et invalide les modèles existants. Mieux vaut le décider **avant** de
retoucher la récompense, pour ne réentraîner qu'une fois.

---

## Questions pour l'équipe

1. Plafond de vitesse plus haut, ou suivi d'une vitesse cible ?
2. Quels termes de « qualité de marche » en premier ?
3. Un palier repart-il du modèle précédent, ou de zéro ?
4. Quels seuils de réussite pour les paliers 2 et 3 ?
5. Avons-nous la puissance de calcul pour plusieurs entraînements de 10 M de pas en
   parallèle (5 à 7 h chacun sur 12 cœurs) ?
6. Quels capteurs aura le robot ? Une IMU avec gyroscope et des capteurs de contact
   sous les pieds sont nécessaires aux quatre propositions ci-dessus.
7. Passe-t-on à une politique asymétrique acteur / critique avant de retoucher la
   récompense ?

## Avis

| Qui | Date | Avis |
| --- | --- | --- |
| | | |
