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
- mêmes mesures : avance, vitesse, dérive, verticalité, % sans chute.

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

## Questions pour l'équipe

1. Plafond de vitesse plus haut, ou suivi d'une vitesse cible ?
2. Quels termes de « qualité de marche » en premier ?
3. Un palier repart-il du modèle précédent, ou de zéro ?
4. Quels seuils de réussite pour les paliers 2 et 3 ?
5. Avons-nous la puissance de calcul pour plusieurs entraînements de 10 M de pas en
   parallèle (5 à 7 h chacun sur 12 cœurs) ?

## Avis

| Qui | Date | Avis |
| --- | --- | --- |
| | | |
