# Le Poppy réel : accès, calibration, pièges

État au 13 novembre 2026. Les 25 moteurs répondent et le **sens** de chaque
articulation est établi. Les **zéros** ne sont confirmés que pour les deux coudes et
`r_arm_z`. Aucune politique n'a encore tourné sur le robot.

## 1. Accès

Le robot est un Raspberry Pi joignable en SSH sur le réseau local. L'adresse change
d'un réseau à l'autre : on la lit sur le robot avec `hostname -I`. On se connecte par
clé, sans mot de passe. Côté poste, dans `~/.ssh/config` :

```
Host poppy
    HostName <adresse lue sur le robot>
    User poppy
```

Les moteurs sont répartis sur deux bus Dynamixel, deux USB2AX :

| Bus | Identifiants | Moteurs |
|---|---|---|
| jambes | 11-15, 21-25 | MX-28, sauf les hanches y en MX-64 |
| haut | 31-37, 41-44, 51-54 | MX-64 (abs_y, abs_x), MX-28, et deux AX-12 pour la tête (36, 37) |

Le protocole est Dynamixel 1.0 à 1 Mbit/s. Les noms `ttyACM0`/`ttyACM1` peuvent
s'échanger au redémarrage, il vaut donc mieux passer par `/dev/serial/by-id/`. Pour
lister les ports : `ls -l /dev/serial/by-id/`.

## 2. L'outil `scripts/robot_dxl.py`

Il tourne **sur le robot** et ne dépend que de `pyserial`, plus `pyyaml` pour
`pose-zero`. Les deux sont installés sur le Pi.

```bash
scp scripts/robot_dxl.py configs/robot_calibration.yaml poppy:/tmp/
ssh -t poppy
export POPPY_DXL_PORT_JAMBES=/dev/serial/by-id/usb-Xevelabs_USB2AX_<…>-if00
export POPPY_DXL_PORT_HAUT=/dev/serial/by-id/usb-Xevelabs_USB2AX_<…>-if00
cd /tmp && python3 robot_dxl.py lire
```

Il n'y a pas de port par défaut : sans ces variables, le script refuse de démarrer.
Il vérifie aussi que le moteur 11 répond sur le port « jambes » et le moteur 31 sur le
port « haut ». Des ports inversés sont donc refusés.

| Commande | Effet |
|---|---|
| `lire` | position, limites, tension, température des 25 moteurs. Lecture seule |
| `suivre [--ids …]` | positions en continu pendant qu'on manipule le robot à la main. Lecture seule |
| `tester-sens --confirme --groupe tronc\|bras\|jambes [--ids …]` | chaque articulation tourne de 30°, revient, puis passe la main à la suivante, 3 s plus tard |
| `pose-zero --confirme [--groupes …] [--maintien 30]` | amène le robot à l'angle 0 de la simulation, groupe par groupe, tient la pose, puis relâche le couple en douceur |
| `coucou --confirme` | redresse la tête et agite le bras droit |

Toute commande qui bouge le robot exige la présence de quelqu'un pour le surveiller,
une main sur l'alimentation (`CONTRIBUTING.md`, « Sécurité robot »).

## 3. Calibration

`configs/robot_calibration.yaml`, chargée par `src/robot/calibration.py` :

    ticks = zero + sens × angle_rad × K

`K` vaut 4096/2π pour les MX et 1024/radians(300) pour les AX. L'angle est celui de la
simulation **avec `fix_left_knee_axis` et `fix_arm_frames` activés**, donc l'angle URDF.

### Établir un sens

`tester-sens` fait tourner une articulation **vers les ticks croissants**. On observe
de quel côté elle part, puis on compare à la simulation : le même mouvement vers +,
rendu dans MuJoCo. Si les deux vont dans le même sens, `sens` vaut +1, sinon −1. Pour
les `arm_z`, l'outil plie d'abord le coude, sinon la rotation ne se voit pas.

En pratique, voici ce qui a marché :

- **filmer** le robot, de profil pour les axes y, de face pour les axes x et z ;
- **asseoir** le robot sur une chaise pour les hanches. Suspendu, il se balance et
  les jambes libres vrillent, ce qui fausse tout ;
- **lire les sources** du YAML : chaque sens y dit comment il a été établi, et avec
  quelle confiance.

### Vérifier un zéro (à faire)

Le zéro est la position moteur qui correspond à l'angle 0 de la simulation. Il se
vérifie en posant le robot à zéro (`pose-zero`), puis en comparant à un rendu de la
pose 0 de la simulation, articulation par articulation. Un écart constant sur une
articulation signifie que son `zero` est faux. On corrige alors la valeur en ticks et
on passe `zero_verifie` à `true`.

Ce qu'il faut savoir avant :

- la pose 0 de la simulation a les **bras à l'horizontale, en T**, pas le long du
  corps ;
- les limites EEPROM reproduisent les plages URDF avec un zéro à 2048 et un sens +1
  **partout**. Elles ont été écrites sans tenir compte du sens. Pour `abs_y` et
  `bust_y`, deux moteurs inversés, elles laissent penser à un zéro vers 1615 et 1592
  plutôt que 2048. Ce sont les deux zéros à regarder en premier ;
- le robot ne tient **pas** debout seul en pose 0. Quelqu'un doit le tenir.

## 4. Pièges Dynamixel constatés

Ils sont tous gérés par l'outil. Chacun a laissé au moins une fois un moteur sous
couple sans qu'on le veuille.

1. **Écrire la consigne, la vitesse ou le couple limite active le couple.** Couper le
   couple se fait donc **en dernier**, puis on relit le registre pour s'assurer qu'il
   vaut 0.
2. **Une consigne hors des limites EEPROM** renvoie l'erreur 0x02, mais le moteur
   rabat quand même la consigne sur la limite **et active le couple**. Un moteur qui
   repose hors de ses limites, comme `head_y` qui pend ou `abs_y` en position assise,
   n'est donc jamais écrit.
3. **Couple coupé, la consigne suit la position.** Écrire d'abord la vitesse revient
   à tenir la position actuelle. C'est la façon sûre de prendre le couple sur place.
4. **Le robot suspendu bouge.** Avant de démarrer, l'outil attend jusqu'à 30 s que
   plus rien ne bouge.
5. **`head_y` et `abs_y` reposent souvent hors limites** quand le robot est éteint
   ou assis. C'est normal.

## 5. Reste à faire avant un premier rejeu de politique

1. Vérifier les zéros, en commençant par le tronc (`abs_y`, `bust_y`) et les hanches.
2. Repasser sur les sens dont la confiance est « moyenne » si un zéro surprend.
3. Écrire le pont observation → action : lecture des positions, conversion par
   `load_calibration`, consigne. Il faut aussi définir ce qu'on donne à la politique
   pour les observations qu'aucun capteur ne fournit (voir
   `docs/ARCHITECTURE.md`).
4. Premier rejeu robot suspendu, couple bridé.
