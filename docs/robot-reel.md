# Le Poppy réel : accès, calibration, pièges

État au 13 novembre 2026. Les 25 moteurs répondent et le **sens** de chaque
articulation est établi. Les **zéros** sont confirmés pour les coudes, `r_arm_z`, les
hanches y, les genoux et les chevilles. Aucune politique n'a encore tourné sur le
robot.

Le code de l'équipe précédente est sur le robot, dans
`~/work/poppy-conception/src/poppy_motors/` : un coucou (`motor_node_test.py`), une
danse et un dab rejoués depuis des poses enregistrées (`dynamic_input.py`,
`positions_*.csv`). Il ne contient aucune calibration : tous les zéros y sont à 2048,
et seuls trois moteurs y sont inversés, à titre d'essai.

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
  **partout**. Elles ont été écrites sans tenir compte du sens, et ne prouvent rien ;
- le robot ne tient **pas** debout seul en pose 0. Quelqu'un doit le tenir ;
- **la bonne méthode est de coucher le robot sur le dos**, sur un sol plat. Debout et
  tenu par le torse, le bassin pivote librement entre le torse et les jambes, les
  hanches sortent de leurs limites, et l'outil les laisse libres : les jambes se
  croisent. Couché, le corps est déjà presque en pose 0 et le sol sert de règle.

Fait le 13 novembre 2026, robot couché : jambes tendues à plat, pieds à la verticale.
Les zéros des hanches y, des genoux et des chevilles sont confirmés. Pour le torse,
2048 est **supposé**, sans test dédié. Trois relevés indiquent pourtant qu'un torse
droit demande `abs_y` + `bust_y` environ 16 à 24° au-dessus de 2048 en cumulé (détail
dans l'en-tête de la calibration). Le test qui les départagerait : robot couché sur
le côté, comparer le torse à 2048 et à environ 2130/2240.

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

## 5. Démonstration

`scripts/robot_danse.py` rejoue le debout, le coucou et le floss de l'ancienne équipe
avec les garde-fous de `robot_dxl.py`. Il a tourné sur le robot le 13 novembre 2026.
Ses poses, `positions_11.csv` à `positions_14.csv`, sont copiées dans
`configs/robot_poses/`. Sous-commandes, ordres et options sont décrits en tête du
script.

```bash
scp scripts/robot_dxl.py scripts/robot_danse.py configs/robot_poses/*.csv poppy:/tmp/
```

Puis, sur le robot, avec les deux variables de port exportées :

```bash
cd /tmp && python3 robot_danse.py verifier
```

```bash
cd /tmp && python3 robot_danse.py spectacle --confirme --marge-deg 12
```

Les ordres s'écrivent depuis un second terminal, par exemple
`echo floss > /tmp/poppy_ordre`. `stop` relâche tout.

`echo 67 > /tmp/poppy_ordre` joue la danse « 67 » (bras seuls, 5 cycles ; `67 N` pour
N de 1 à 6). Le visage qui l'accompagne, `src/ui/poppy_visage_67.py`, a pour yeux un 6
et un 7 qui balancent au rythme des mains : il lit la cadence dans `/tmp/poppy_67`.
On le lance sur l'écran de la tête avec `DISPLAY=:0 python3 poppy_visage_67.py` ; Échap
le ferme. La pose a été vérifiée dans MuJoCo, pas encore sur le robot.

Trois coucous existent. `coucou`, celui de l'ancienne équipe, porte les bras vers
l'avant et l'arrière : il a fait tomber le robot. `coucou2` lève le bras droit sur le
côté et l'agite par `r_shoulder_x` : il tient l'équilibre, mais ressemble à un bras qui
se balance. `coucou3` lève le bras droit sur le côté, plie l'avant-bras vers le haut et
fait saluer la main par le coude, dans le plan du corps. Sa pose a été vérifiée dans
MuJoCo, pas encore sur le robot.

### Procédure

1. Coucher le robot **sur le dos**, sur un sol plat.
2. Lancer la commande : le couple est pris robot couché, puis le corps passe en pose
   debout.
3. Le relever **d'un bloc, comme une planche**, en le basculant sur ses pieds. Garder
   **les mains sur lui en permanence**, jusqu'au `stop` compris.

### Ce qu'on a constaté sur le torse

La meilleure pose debout observée est **celle de l'ancienne équipe : `abs_y` 2150,
`bust_y` 2246**. Le torse y penche légèrement en avant, mais elle tient. C'est le
défaut du script, et les poses du floss sont jouées sans décalage.

Un réglage **3° plus en arrière** (`abs_y` 2133, `bust_y` 2229) a tenu un temps, puis
a fait tomber le robot. Pour régler le torse en cours de spectacle, les ordres
`arriere N` et `avant N` restent disponibles, avec les mains sur le robot.

**`bust_y` chauffe : 50 °C pendant un maintien prolongé**, ce qui déclenche l'alerte
chaleur du script (`TEMP_ALERTE`). Il s'arrête à 60 °C. Éviter les longues tenues.

Ces valeurs confirment l'en-tête de la calibration : un torse droit demande `abs_y` et
`bust_y` nettement au-dessus de 2048. Elles ne fixent pas pour autant le zéro du torse :
une pose debout n'est pas la pose zéro de la simulation.

### Pièges

- **Couché, la pièce entre `abs_y` et `bust_y` peut basculer**, et `bust_y` sort alors
  de ses limites EEPROM. Avec la marge par défaut (2,6°), le script refuse de prendre
  ce moteur. `--marge-deg 12` le ramène sur sa limite à la place, sous couple
  limité et à vitesse lente.
- **Le robot ne garde pas l'équilibre seul.** Les moteurs tiennent la pose, rien ne
  corrige l'équilibre. Il est tombé quand le torse a bougé. Les mains restent sur lui
  pendant chaque ordre, en particulier `arriere`, `avant` et `floss`.

## 6. Reste à faire avant un premier rejeu de politique

1. Vérifier les zéros restants : les hanches x et z (photo prise du dessus, robot
   couché), et le torse si le robot penche au premier rejeu.
2. Repasser sur les sens dont la confiance est « moyenne » si un zéro surprend.
3. Écrire le pont observation → action : lecture des positions, conversion par
   `load_calibration`, consigne. Il faut aussi définir ce qu'on donne à la politique
   pour les observations qu'aucun capteur ne fournit (voir
   `docs/ARCHITECTURE.md`).
4. Premier rejeu robot suspendu, couple bridé.
