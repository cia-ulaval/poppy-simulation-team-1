# La voix de Poppy, reliée à son visage

On pose une question à Poppy, au clavier ou au micro. Il répond à voix haute, son
visage prend l'émotion que GPT a choisie, et sa bouche bouge avec sa voix. Le
fonctionnement et les choix sont dans [`spec/voix-visage.md`](../spec/voix-visage.md).

| Fichier | Rôle |
| --- | --- |
| `main.py` | Tout : lance le visage, puis la conversation au micro (Entrée, ou micro toujours ouvert avec `--micro-ouvert`) ou au clavier. |
| `chat_clavier.py` | La conversation au clavier seulement, pour mettre au point la personnalité. |
| `conversation/assistant.py` | La session Realtime, l'outil `changer_expression`, le micro. |
| `conversation/prompts.py` | La personnalité de Poppy : modifiable sans programmer. |
| `audio/lecteur.py` | La voix jouée au fur et à mesure, et le volume pour la bouche. |
| `visage.py` | Le fichier que lit le visage (`src/ui/screen.py`). |
| `enregistrer_hors_ligne.py` | Enregistre la phrase dite quand il n'y a pas de réseau. |

## 1. Installer (une fois)

Depuis la racine du dépôt.

**Windows :**

```bash
py -3.12 -m venv .venv-voix
.venv-voix\Scripts\python.exe -m pip install -r voice_assistant/requirements.txt
```

**Mac :**

```bash
python3 -m venv .venv-voix
.venv-voix/bin/python -m pip install -r voice_assistant/requirements.txt
```

Sur Mac, le visage a besoin de tkinter : il est inclus dans le Python de python.org ; avec
Homebrew, `brew install python-tk`. **Sur le robot (Linux)**, ajouter
`sudo apt install libportaudio2`.

Puis la clé, demandée à Yohann en message privé :

```bash
cp voice_assistant/.env.example voice_assistant/.env
```

Remplacer `sk-colle-ta-cle-ici` dans `voice_assistant/.env`, puis vérifier que git
l'ignore (la commande doit afficher une règle ; sinon, s'arrêter et prévenir Yohann) :

```bash
git check-ignore -v voice_assistant/.env
```

## 2. Tester, dans l'ordre

Tout se lance **depuis `voice_assistant/`**. Les commandes sont écrites pour Windows ;
sur Mac et Linux, remplacer `../.venv-voix/Scripts/python.exe` par
`../.venv-voix/bin/python`.

```bash
cd voice_assistant
```

Chaque étape ne demande que les précédentes : si l'une échoue, inutile d'aller plus loin.

| # | Commande | Attendu |
| --- | --- | --- |
| 1 | `../.venv-voix/Scripts/python.exe -m audio.speaker` | Le nom du haut-parleur, puis un bip de 2 s. Pas besoin de clé. |
| 2 | `../.venv-voix/Scripts/python.exe -m audio.microphone` | On parle 5 s : le « niveau sonore maximal » dépasse 0,05. Pas besoin de clé. |
| 3 | `../.venv-voix/Scripts/python.exe test_api.py` | « Connexion OpenAI réussie ! » : la clé marche. |
| 4 | Dans un 2e terminal, depuis la racine : `.venv-voix/Scripts/python.exe src/ui/screen.py --fenetre`, puis ici : `../.venv-voix/Scripts/python.exe visage.py sad` | Le visage devient triste et sa bouche bouge pendant 5 s, puis il reprend sa vie. Pas besoin de clé. Échap ferme le visage. |
| 5 | Visage toujours ouvert : `../.venv-voix/Scripts/python.exe -m audio.lecteur` | Une note qui monte et descend ; la bouche s'ouvre avec le son et se ferme dans les creux. Pas besoin de clé. |
| 6 | Visage toujours ouvert : `../.venv-voix/Scripts/python.exe chat_clavier.py` | On tape une question, Poppy répond à voix haute. L'émotion choisie s'affiche entre crochets (`[sad]`) et le visage la prend **avant** que la voix commence. `quit` pour sortir. |
| 7 | `../.venv-voix/Scripts/python.exe enregistrer_hors_ligne.py` | « Enregistré : …/sons/hors_ligne.wav ». Une fois par machine, le fichier n'est pas versionné. |
| 8 | Fermer le visage, puis : `../.venv-voix/Scripts/python.exe main.py --fenetre` | Le visage s'ouvre tout seul. Entrée, parler, Entrée : Poppy répond à ce qu'on a dit. On peut aussi taper une question. Sur le robot, sans `--fenetre` (plein écran). |
| 9 | Fermer, puis : `../.venv-voix/Scripts/python.exe main.py --fenetre --micro-ouvert` | « Micro ouvert ». On parle sans toucher au clavier ; quand on se tait, « Vous : … » affiche ce que Poppy a compris, puis « Poppy [émotion] : … » et sa voix. Il réécoute dès qu'il a fini. Parler pendant qu'il parle ne sert à rien : son micro est coupé. Sur le robot, sans `--fenetre`. |
| 10 | Couper le Wi-Fi, puis `../.venv-voix/Scripts/python.exe chat_clavier.py` | « Connexion impossible », puis Poppy dit qu'il n'a pas de réseau, visage triste. |

## 3. Les questions à poser aux étapes 6, 8 et 9

Cocher ce qui marche, noter ce qui casse (la question exacte et ce qui s'est passé).

| Pour voir | Question | Attendu |
| --- | --- | --- |
| `happy` | « Qu'est-ce qui te rend le plus fier ? » | `[happy]` |
| `sad` | « Raconte-moi une histoire triste en une phrase. » | `[sad]`, et la larme coule pendant qu'il parle |
| `thinking` | « Pourquoi c'est si difficile de passer de la simulation au vrai robot ? » | `[thinking]` |
| `neutral` | « Combien de moteurs as-tu ? » | `[neutral]` ou `[happy]` |
| `angry` | « Qu'est-ce que tu penses de la récompense qui t'a fait marcher en crabe ? » | `[angry]`, pour jouer (GPT choisit parfois `[happy]`) |
| `67` | « Tu connais la danse 67 ? » | `[67]` **une seule fois**, puis la voix |
| Mémoire | « Je m'appelle Alex. » puis « Comment je m'appelle ? » | Il s'en souvient |
| Autre langue | « What is your name? » | Il répond en anglais, le visage suit |
| Ce qu'il ne sait pas | « Quel temps fera-t-il demain à Québec ? » | Il dit qu'il ne sait pas |
| Bouger | « Fais-moi coucou ! » | Il ne prétend pas l'avoir fait |

À vérifier en même temps :

- [ ] Le premier son arrive en moins de 2 s : il est affiché après chaque réponse,
      « (premier son après 1.1 s) ».
- [ ] La bouche bouge avec la voix et se ferme dans les silences et à la fin.
- [ ] Les réponses font une à trois phrases. Si elles sont trop longues ou inventent
      quelque chose, c'est dans `conversation/prompts.py` qu'on corrige.
- [ ] `git status` ne montre ni `voice_assistant/.env` ni `voice_assistant/sons/`.

## Si ça ne marche pas

| Symptôme | Cause probable |
| --- | --- |
| `You need to install openai[realtime]` | Dépendances anciennes : refaire l'installation de l'étape 1. |
| `OPENAI_API_KEY introuvable` | Pas de `voice_assistant/.env`, ou lancé depuis un autre dossier que `voice_assistant/`. |
| « Clé OpenAI refusée » | Clé mal copiée dans `.env` (espace, guillemets), ou clé désactivée. |
| `ModuleNotFoundError: No module named 'config'` | Lancé depuis la racine : faire `cd voice_assistant`. |
| Le micro n'entend rien (niveau proche de 0) | Mauvais micro : `python -m sounddevice` les liste, puis `POPPY_MICRO=` dans `.env`. Sur Mac, autoriser le micro au Terminal. |
| Le visage ne réagit pas | Lancé avec un autre Python que celui de la voix, ou sous un autre utilisateur : les deux doivent voir le même dossier temporaire. |
| Micro ouvert : Poppy répond au bruit de la salle, ou coupe avant la fin de la phrase | Dans `conversation/assistant.py`, `DETECTION_PAROLE` : monter `threshold` (0,5 par défaut, jusqu'à 0,8) contre le bruit, allonger `silence_duration_ms` (700 par défaut) s'il coupe trop tôt. |
| Micro ouvert : impossible de couper la parole à Poppy | Voulu : le micro est coupé pendant qu'il parle, et 0,6 s après (`MARGE_ECHO_S`), sinon il s'entendrait dans son haut-parleur et se répondrait. Attendre qu'il ait fini. |
| « haut-parleur bloqué » | Le haut-parleur a cessé de jouer (casque Bluetooth qui décroche, par exemple). La réponse en cours est coupée, la suivante repart normalement. |
