# Architecture

Comment c'est construit, et où se trouve quoi. À lire avant de toucher à
`src/environments/poppy_humanoid_env.py`.

Les diagrammes suivent le modèle **C4** : on zoome progressivement, du contexte
général jusqu'aux modules. Ils sont en Mermaid — GitHub les rend directement, il n'y a
aucun outil à installer et aucune image à régénérer quand le code bouge.

---

## Niveau 1 — Contexte

Qui parle à quoi. Le dépôt produit une politique ; le robot la consomme.

```mermaid
graph LR
    equipe["Équipe<br/>6 personnes"]
    depot["Poppy Simulation<br/><i>ce dépôt</i>"]
    machine["Machine d'entraînement<br/>CPU nombreux, GPU souhaitable"]
    pont["rosbridge<br/><i>websocket :9090</i>"]
    robot["Poppy physique<br/>25 moteurs Dynamixel<br/>83 cm, 3,5 kg"]

    equipe -->|"écrit, lance, regarde"| depot
    depot -->|"entraîne 10M pas<br/>plusieurs heures"| machine
    machine -->|"politique + normalisation<br/><i>models/</i>"| depot
    depot -->|"positions articulaires<br/>JSON, 25 angles en rad"| pont
    pont -->|"consignes moteur"| robot

    style depot fill:#1f6f8b,stroke:#0d3d4d,color:#fff
    style robot fill:#8b1f3f,stroke:#4d0d20,color:#fff
```

**La flèche qui n'existe pas encore** : rien ne remonte du robot vers le dépôt. La
boucle est ouverte — on envoie des consignes, on ne lit aucun état réel. Et ce pont
n'a jamais parlé à un vrai robot, seulement au faux `rosbridge` de la pile compose.

---

## Niveau 2 — Conteneurs

Quatre images Docker, un socle commun, plus deux outils qui tournent en natif.

```mermaid
graph TB
    subgraph docker["Docker"]
        base["<b>base</b><br/>python 3.11 + torch + SB3"]
        train["<b>train</b><br/>+ MuJoCo osmesa, TensorBoard<br/><i>profils: train, eval, dev</i>"]
        bridge["<b>bridge</b><br/>+ roslibpy<br/><i>profil: robot</i>"]
        vision["<b>vision</b><br/>+ transformers<br/><i>ne démarre pas</i>"]
        ros["<b>rosbridge</b><br/>ROS 2 Humble<br/><i>faux robot</i>"]
        tb["<b>tensorboard</b><br/>:6006"]

        base --> train
        base --> bridge
        base --> vision
    end

    subgraph natif["En natif, hors Docker"]
        viewer["scripts/viewer.py<br/><i>fenêtre OpenGL</i>"]
        visu["scripts/visualize.py<br/><i>fenêtre OpenGL</i>"]
    end

    logs[("logs/<br/><i>sorties</i>")]
    models[("models/<br/><i>politiques</i>")]

    train -->|écrit| logs
    tb -->|lit| logs
    models -->|lu par| bridge
    models -->|lu par| visu
    bridge -->|":9090"| ros
    vision -.->|":8000"| bridge

    style train fill:#1f6f8b,stroke:#0d3d4d,color:#fff
    style vision fill:#5a5a5a,stroke:#333,color:#fff,stroke-dasharray: 5 5
```

`vision` est en pointillés : l'image se construit, mais le serveur s'arrête à l'import
(`depth_anything_3` absent de PyPI).

`viewer.py` et `visualize.py` sont hors Docker délibérément : faire sortir une fenêtre
OpenGL d'un conteneur sous Windows demande un serveur X pour zéro bénéfice.

Le détail de chaque image — cibles, arguments de build, volumes — est dans
[`DOCKER.md`](DOCKER.md).

---

## Niveau 3 — Composants

Les modules Python et **les imports réels** entre eux. C'est ce diagramme qui répond à
« la vision, c'est où ? ».

```mermaid
graph TB
    subgraph entrees["scripts/ — les cinq points d'entrée"]
        tp["train_poppy.py"]
        ev["evaluate.py"]
        vz["visualize.py"]
        vw["viewer.py"]
        rr["run_robot.py"]
    end

    subgraph coeur["src/ — la bibliothèque"]
        env["environments/<b>poppy_humanoid_env.py</b><br/>physique, récompense, DR"]
        fac["environments/env_factory.py<br/>vectorisation + normalisation"]
        cfg["config/<br/>loaders.py, settings.py"]
        rob["robot/<br/>simulation_adapter, ros_publisher"]
    end

    subgraph pause["src/sensors/ — en pause, rien ne l'importe"]
        dr["depth_rays.py<br/><i>rayons dans MuJoCo</i>"]
        ds["depth_server.py<br/><i>ne démarre pas</i>"]
        dc["depth_client.py"]
        da["depthmap_analysis.py"]
    end

    mujoco["MuJoCo"]
    sb3["Stable-Baselines3"]
    yaml[("configs/*.yaml")]
    assets[("assets/<br/>MJCF + 52 STL")]

    tp --> fac
    tp --> cfg
    ev --> fac
    ev --> cfg
    vz --> env
    rr --> rob
    rob --> env
    fac --> env
    fac --> cfg
    cfg --> yaml
    env --> assets
    env --> mujoco
    vw --> mujoco
    fac --> sb3
    tp --> sb3

    dc --> da
    dr -.->|"cherche un corps<br/><b>torso</b> inexistant<br/>dans Poppy"| mujoco
    dc -.->|websocket| ds

    style env fill:#1f6f8b,stroke:#0d3d4d,color:#fff
    style pause fill:#2a2a2a,stroke:#666,color:#aaa
    style ds fill:#5a5a5a,stroke:#333,color:#fff,stroke-dasharray: 5 5
```

**Ce qu'il faut retenir du diagramme :**

- `poppy_humanoid_env.py` est le seul module que tout traverse. Le modifier touche
  l'entraînement, l'évaluation, la visualisation **et** le robot.
- `src/sensors/` est un **îlot** : aucune flèche ne part de `scripts/` ni de `src/`
  vers lui. La vision n'est branchée sur rien. Son état détaillé est dans
  [`../src/sensors/README.md`](../src/sensors/README.md).
- `viewer.py` ne passe pas par `src/` du tout : il charge le MJCF avec `mujoco` brut.
  C'est voulu — il doit marcher même si `src/` est cassé.

---

## Le contrat observation → action

**C'est le contrat à connaître avant toute modification.** Le changer invalide tous
les modèles entraînés : une politique chargée avec un espace d'observation différent
échoue sur `spaces must have the same shape`.

### Observation — 63 dimensions, `float64`

| Indices | Contenu | Unité |
|---|---|---|
| `0` | Hauteur du bassin (z) | m |
| `1:5` | Quaternion de la racine (w, x, y, z) | — |
| `5:30` | Position des 25 articulations | rad |
| `30:36` | Vitesse de la racine (3 linéaires + 3 angulaires) | m/s, rad/s |
| `36:61` | Vitesse des 25 articulations | rad/s |
| `61:63` | Force de contact, pied droit puis pied gauche | N |

Soit `qpos[2:]` (30 valeurs — les positions globales x et y sont volontairement
exclues, pour que la politique ne dépende pas de l'endroit où le robot se trouve),
puis `qvel` (31), puis les deux contacts.

### Action — 25 dimensions, une par articulation

Une action est une **position cible**, pas un couple. Le trajet complet :

```
action ∈ [-1, 1]
   │  action = 0  → pose debout par défaut (init_qpos)
   │  action = +1 → limite mécanique haute
   │  action = -1 → limite mécanique basse
   ▼
position cible  (rad)
   │  PD :  τ = kp·(cible − q) − kd·q̇      kp = 8 Nm/rad, kd = 0,5 Nm·s/rad
   ▼
couple  (Nm)
   │  bornage sur actuator_ctrlrange (max ≈ 3,1 Nm)
   ▼
MuJoCo, frame_skip = 5 pas de 2 ms → 10 ms par pas de politique
```

Le contrôle en **position** plutôt qu'en couple est un choix de conception, pas un
détail : c'est ce que fait un servomoteur Dynamixel réel, qui a sa propre boucle PID
interne. Une politique entraînée en couple ne se transférerait pas.

⚠️ `action_space` **n'est pas** `[-1, 1]` en réalité — voir
[`TECHNIQUE.md` §8](TECHNIQUE.md#8-ce-qui-est-cassé-et-connu).

### Récompense — huit termes

| Terme | Signe | Ce que ça encourage |
|---|:---:|---|
| `forward_reward` | + | Avancer. Plafonné à 0,5 m/s pour éviter les foulées absurdes. |
| `healthy_reward` | + | Rester dans la plage de hauteur (ne pas tomber) |
| `upright_reward` | + | Garder le buste vertical |
| `gait_reward` | + | Alterner les appuis : 0,3 si un seul pied touche, 0,1 si les deux |
| `lateral_cost` | − | Dériver sur le côté |
| `ctrl_cost` | − | Actions de grande amplitude |
| `action_rate_cost` | − | Changements brusques entre deux actions |
| `joint_vel_cost` | − | Articulations qui tournent vite |

`healthy` et `upright` rapportent chacun jusqu'à 1000 par épisode complet, quand
`forward` en rapporte quelques dizaines. **Ce déséquilibre a une conséquence
mesurée** : la politique la mieux notée du dépôt reste debout sans marcher. Voir
[`../models/README.md`](../models/README.md).

### Randomisation de domaine

Ré-échantillonnée à chaque `reset()`, uniquement quand `floor_noise=True` :
friction du sol, masses des corps (±15 %), orientation initiale (±15°), et poussées
externes aléatoires sur le bassin (~2 % des pas). Elle sert le transfert vers le réel :
une politique qui ne tient que sur un sol parfait ne tiendra pas sur un vrai plancher.

La restitution est censée être randomisée aussi, mais ne l'est pas — bug documenté
dans [`TECHNIQUE.md` §8](TECHNIQUE.md#8-ce-qui-est-cassé-et-connu).

---

## La frontière simulation / réel

```
SIMULATION                          │  RÉEL
                                    │
politique  ──►  action [-1,1]       │
                    │               │
                    ▼               │
            PD → couple → MuJoCo    │
                    │               │
                    ▼               │
         positions articulaires ────┼──►  JSON {motor_ids, angles_rad}
                  (rad)             │      25 noms, 25 angles
                                    │              │
                                    │              ▼
                                    │     rosbridge :9090
                                    │              │
                                    │              ▼
                                    │     servos Dynamixel
                                    │
                    ◄───────────────┼───  (rien ne revient)
```

`SimulationAdapter` déroule la simulation et publie, après chaque pas, les 25
positions articulaires. Ce qu'il envoie est **ce que la simulation a obtenu**, pas ce
que la politique a demandé — le PD et la physique se sont interposés.

Trois choses à savoir avant de brancher un vrai robot :

1. **Boucle ouverte.** Aucun état réel n'est relu. Si un moteur bloque, rien ne le sait.
2. **Période de contrôle par défaut : 5 secondes** (`POPPY_CONTROL_PERIOD_S`), alors
   que la simulation avance de 10 ms par pas. C'est un réglage de sécurité, pas un
   régime de marche.
3. **La cible par défaut est le faux robot**, jamais une adresse réelle. Viser un vrai
   robot demande de définir `POPPY_ROSBRIDGE_HOST` explicitement. C'est une règle, pas
   un défaut de configuration — voir [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
