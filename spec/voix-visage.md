# Poppy parle avec son visage

> Pour Seynabou et Seilamoney. Suite de la PR #18 (expressions faciales) et de
> [`voix-conversation.md`](voix-conversation.md).

## Objectif

On pose une question à Poppy. Il répond avec sa voix, son visage prend l'émotion que
GPT a choisie pour sa réponse, et sa bouche bouge au rythme de ce qu'il dit.

## À savoir avant de commencer

- **Ce qui marche déjà :**
  - `src/ui/screen.py` : le visage, avec `set_emotion(nom)` pour l'émotion et
    `set_talking(True/False)` pour la parole, qui garde l'émotion ;
  - `voice_assistant/test_voice.py` : une question écrite dans le code est envoyée à
    GPT Realtime (`gpt-realtime-2.1-mini`, voix `marin`), et la réponse sort au
    haut-parleur.
- **Ce que vous allez construire :** aujourd'hui, Poppy sait répondre à voix haute,
  mais seulement à une question écrite d'avance dans le code, et le programme s'arrête
  après sa réponse. Il reste à pouvoir lui poser ses propres questions et enchaîner,
  au clavier puis au micro, et à relier sa voix à son visage. `main.py` et
  `conversation/assistant.py` sont vides : c'est là que ça peut aller.
- **La personnalité de Poppy** est dans `voice_assistant/conversation/prompts.py` :
  qui l'a construit, le club, les deux équipes, les anecdotes, le style. Toute
  l'équipe peut la modifier sans programmer.
- **Ça ne touche pas aux moteurs.** Seulement l'écran et le son.
- **Branche :** `feat/voix-visage`, créée depuis `main` à jour
  (`git switch -c feat/voix-visage origin/main`). Faites des commits petits et
  séparés (un par étape), et ouvrez la PR en brouillon dès le premier commit.

## Étape 0 — Installer et brancher la clé

### La clé API

1. **Demander la clé à Yohann, en message privé.** Elle est liée au compte du club et
   à son budget. Ne la partagez jamais dans un canal Discord public, une capture
   d'écran ou un commit.
2. Copier le modèle et coller la clé :

   ```bash
   cp voice_assistant/.env.example voice_assistant/.env
   ```

   Puis ouvrir `voice_assistant/.env` et remplacer `sk-colle-ta-cle-ici` par la vraie
   clé.
3. Vérifier que git ignore bien le fichier. La commande doit afficher une règle
   `.gitignore` ; si elle n'affiche rien, **arrêtez-vous** et prévenez Yohann :

   ```bash
   git check-ignore -v voice_assistant/.env
   ```

`config/settings.py` lit la clé dans ce fichier au démarrage. Sans clé, il s'arrête
avec « OPENAI_API_KEY introuvable ».

### Les dépendances

```bash
pip install -r voice_assistant/requirements.txt
```

Sous Windows, `python` tout court peut ouvrir le Microsoft Store au lieu de Python :
utilisez `py`, ou le chemin complet de votre environnement.

### Tester : en ligne de commande ou à la voix

Tout se lance **depuis `voice_assistant/`**, et les modules de `audio/` avec
`python -m`, sinon leurs imports échouent :

```bash
cd voice_assistant
```

**Le micro et le haut-parleur** sont ceux par défaut du système, c'est-à-dire ceux qui
sont actifs dans vos réglages son. Pour en choisir d'autres, `python -m sounddevice`
les liste (le défaut est marqué `>` pour le micro et `<` pour le haut-parleur). Il
suffit ensuite de décommenter `POPPY_MICRO` ou `POPPY_HAUT_PARLEUR` dans le `.env`,
avec le numéro ou un morceau du nom. Sur le robot, c'est `1` et `2`.

**En ligne de commande**, en texte seulement, sans son :

| Commande | Attendu |
| --- | --- |
| `python test_api.py` | « Connexion OpenAI réussie ! » : la clé marche. |
| `python test_realtime.py` | Poppy se présente par écrit, avec sa personnalité. |

**À la voix :**

| Commande | Attendu |
| --- | --- |
| `python -m audio.speaker` | Le nom du haut-parleur utilisé, puis un bip de 2 s. Pas besoin de clé. |
| `python -m audio.microphone` | Le nom du micro, puis on parle 5 s : le « niveau sonore maximal » doit dépasser 0,05. S'il reste proche de 0, ce n'est pas le bon micro, ou il est coupé. Pas besoin de clé. |
| `python test_voice.py` | Poppy se présente à voix haute, et le texte s'affiche en même temps. |

Ces scripts posent une question écrite dans le code. Pour poser vos propres
questions, c'est l'étape 2 au clavier, puis l'étape 5 au micro.

## Où regarder

| Fichier | Ce qu'il contient |
| --- | --- |
| `voice_assistant/test_voice.py` | Connexion Realtime et configuration de la session : la base de départ. |
| `voice_assistant/config/peripheriques.py` | Le micro et le haut-parleur (ceux du système, ou ceux du `.env`), leur fréquence, et `reechantillonner()`. |
| `voice_assistant/audio/speaker.py` | La lecture, de 24 kHz vers la fréquence du haut-parleur. Pour l'instant, elle attend d'avoir tout l'audio avant de jouer. |
| `voice_assistant/audio/microphone.py` | L'enregistrement, à la fréquence du micro, pour une durée fixe. |
| `voice_assistant/conversation/prompts.py` | `POPPY_SYSTEM_PROMPT` (la personnalité) et `VISAGE_PROMPT` (comment se servir du visage). |
| `src/ui/screen.py` | Le visage : `EMOTIONS`, `set_emotion()`, `set_talking()`, `animate_talking_mouth()`. |

## Ce qui relie la voix au visage

Ce sont deux programmes séparés. La voix écrit un petit fichier, et le visage le
relit toutes les 40 ms :

    {"emotion": "happy", "parle": true, "bouche": 0.6}

- **Emplacement :** `/dev/shm/poppy_visage` si `/dev/shm` existe (Linux, le robot),
  sinon `poppy_visage` dans `tempfile.gettempdir()` (Windows, Mac). Les deux
  programmes appliquent la même règle. `/dev/shm` est en mémoire vive : sur la carte
  SD du robot, écrire 25 fois par seconde l'userait.
- **Toujours écrire dans un fichier temporaire, puis faire `os.replace`**, pour que le
  visage ne lise jamais un fichier à moitié écrit.
- **Fichier absent, illisible ou vieux de plus de 3 s :** le visage vit sa vie comme
  aujourd'hui.

Ce contrat permet de travailler à deux en même temps : l'une fait le côté visage
(étapes 1 et 4b), l'autre le côté voix (étapes 2, 3 et 4a), et chacune teste son côté
en écrivant ou en lisant le fichier à la main.

## À faire

Ce qui suit est **un chemin proposé, pas une recette imposée**. Vous êtes libres de
vous y prendre autrement : autre découpage, autre bibliothèque, autre réglage de la
détection de voix, autre voix que `marin`... Deux limites seulement : la section
« Ne pas faire », et le contrat entre la voix et le visage (si vous le changez,
changez les deux côtés et dites-le dans la PR). Chaque étape se teste seule, avant de
passer à la suivante.

1. **Le visage obéit au fichier.** Dans `screen.py`, une boucle `root.after(40, ...)`
   lit le fichier et appelle `set_emotion()` **seulement quand l'émotion change**
   (sinon les animations repartent à zéro), et `set_talking()` quand `parle` change.
   Une émotion inconnue est ignorée. Test sur le robot :
   `echo '{"emotion":"sad","parle":true,"bouche":0.5}' > /dev/shm/poppy_visage`

2. **Tester la voix au clavier.** C'est l'option 1, celle qui sert tous les jours.
   Écrire `voice_assistant/chat_clavier.py` : une **seule** session Realtime ouverte,
   puis une boucle « on tape une question → Poppy répond à voix haute et le texte de
   sa réponse s'affiche » (`response.output_audio_transcript.delta`). Comme la session
   reste ouverte, Poppy se souvient de la conversation. Taper `quit` pour sortir.
   C'est l'outil pour mettre au point la personnalité sans micro : posez-lui les
   questions (voir « Questions pour explorer ») et corrigez `prompts.py` jusqu'à ce que les réponses soient
   bonnes.

3. **GPT choisit l'émotion.** Dans le `session.update` :

   ```python
   "instructions": POPPY_SYSTEM_PROMPT + VISAGE_PROMPT,
   "tools": [{
       "type": "function",
       "name": "changer_expression",
       "description": "Change l'expression du visage de Poppy sur son écran.",
       "parameters": {
           "type": "object",
           "properties": {"emotion": {"type": "string",
                                      "enum": ["happy", "sad", "thinking", "neutral", "angry"]}},
           "required": ["emotion"],
       },
   }],
   "tool_choice": "auto",
   ```

   L'`enum` reprend **exactement** les noms de `EMOTIONS` : GPT ne peut choisir qu'un
   visage qui existe. Ajoutez `"67"` quand le visage 67 sera fusionné. Dans la boucle
   d'événements, quand arrive `response.output_item.done` avec
   `event.item.type == "function_call"` :
   - lire l'émotion dans `json.loads(event.item.arguments)` et l'écrire dans le
     fichier ;
   - renvoyer `conversation.item.create` avec
     `{"type": "function_call_output", "call_id": event.item.call_id, "output": "ok"}` ;
   - à la fin de la réponse (`response.done`), relancer avec `response.create()`.
     **Sans cette relance, Poppy change de visage puis se tait.**

   Test au clavier : « Raconte-moi une histoire triste » doit donner `sad`, puis la
   voix.

4. **La bouche suit la voix.**
   - **4a, côté voix :** jouer l'audio au fur et à mesure qu'il arrive, au lieu de tout
     attendre comme `test_voice.py`. Utiliser un `sounddevice.OutputStream` avec une
     file d'attente : chaque morceau reçu va dans la file, et le callback du
     haut-parleur la vide. Le volume de chaque bloc (RMS, ramené entre 0 et 1) est
     calculé **dans le callback**, au moment où le son sort, pas quand il arrive : GPT
     envoie plus vite qu'on ne joue. On écrit `parle: true` au premier son, `bouche` à
     chaque bloc, et `parle: false` quand la file est vide après `response.done`.
   - **4b, côté visage :** `animate_talking_mouth()` choisit sa forme selon `bouche`
     au lieu du hasard : `close` sous 0,1, `e` ou `i` pour un volume moyen, `a` ou `o`
     pour un volume fort. Sans fichier récent, il garde le hasard actuel.

5. **Parler au micro.** C'est l'option 2, à faire une fois le clavier au point. Elle
   fonctionne en appuyant pour parler, ce qui est plus fiable qu'un mot de réveil dans
   une salle bruyante :
   - désactiver la détection automatique de fin de phrase :
     `"audio": {"input": {"turn_detection": None, ...}}` ;
   - tant que la touche est enfoncée, lire le micro par petits blocs
     (`sounddevice.InputStream`) et les envoyer avec `input_audio_buffer.append`, en
     PCM16 mono à 24 kHz : `reechantillonner(bloc, micro.sample_rate, GPT_SAMPLE_RATE)` ;
   - au relâchement, faire `input_audio_buffer.commit` puis `response.create()`, et
     mettre l'émotion `thinking` jusqu'au premier son.

   Pistes d'amélioration pour plus tard : la détection automatique de fin de phrase
   (`turn_detection` de type `server_vad`), pour parler sans touche ; un mot de réveil
   (« Hé Poppy ») ; et l'interruption, quand Poppy se tait si on lui coupe la parole.

6. **Sans Internet.** Si la connexion échoue, jouer une phrase préenregistrée (« Je
   n'ai pas de réseau, mais je peux vous faire coucou ! ») avec le visage `sad`, au
   lieu de rester muet.

7. **Tout rassembler dans `main.py`.** On lance le visage, puis la voix, et on parle.

## Questions pour explorer

Il n'y a pas encore de questions de démo : c'est à vous de les trouver. Voici des
points de départ pour faire apparaître chaque visage et pousser les fonctionnalités
dans leurs retranchements. Inventez-en d'autres, notez celles qui marchent bien et
celles qui cassent quelque chose, et gardez les meilleures pour la démo.

| Pour voir | Exemple de question |
| --- | --- |
| `happy` | « Qu'est-ce qui te rend le plus fier depuis ta construction ? » |
| `sad` | « Ça te fait quoi de marcher en simulation, mais pas encore en vrai ? » |
| `thinking` | « Pourquoi c'est si difficile de passer de la simulation au vrai robot ? » |
| `neutral` | « Combien de moteurs as-tu, et à quoi sert le Raspberry Pi dans ta tête ? » |
| `angry`, pour jouer | « Qu'est-ce que tu penses de la récompense qui t'a fait marcher en crabe pendant des mois ? » |
| `67`, une fois la PR #19 fusionnée et `"67"` ajouté à l'`enum` | « Tu connais la danse 67 ? » |

Et pour tester les limites :

- **Plusieurs émotions dans une même réponse.** « Raconte ton parcours, de ta
  première présentation à aujourd'hui. » Est-ce que Poppy change de visage en cours de
  route, ou une seule fois au début ? C'est à découvrir, et à régler dans
  `VISAGE_PROMPT` si besoin.
- **Une autre langue.** Posez une question en anglais : il doit répondre en anglais,
  et le visage doit suivre quand même.
- **Ce qu'il ne sait pas.** « Quel temps fera-t-il demain à Québec ? » Il doit le
  dire au lieu d'inventer.
- **Une demande de bouger.** « Fais-moi coucou ! » Il ne doit pas prétendre l'avoir
  fait : il n'a aucun outil pour les moteurs.
- **L'interruption.** Au micro, coupez-lui la parole. Que se passe-t-il ? La
  bouche s'arrête-t-elle avec la voix ?
- **La latence.** Mesurez le temps entre la fin de la question et le premier son.

## Ressources

- Guide de l'API Realtime d'OpenAI : <https://developers.openai.com/api/docs/guides/realtime>
- Les conversations Realtime, avec l'audio et l'appel de fonctions :
  <https://developers.openai.com/api/docs/guides/realtime-conversations>
- La liste de tous les événements, ceux qu'on envoie et ceux qu'on reçoit :
  <https://developers.openai.com/api/reference/resources/realtime/client-events> et
  <https://developers.openai.com/api/reference/resources/realtime/server-events>
- Les exemples Realtime du SDK Python, dont un avec le micro en continu :
  <https://github.com/openai/openai-python/tree/main/examples/realtime>
- `sounddevice` : lecture et enregistrement en continu (`InputStream`,
  `OutputStream`, callbacks) : <https://python-sounddevice.readthedocs.io/>
- `tkinter`, pour la boucle `after()` du visage :
  <https://docs.python.org/3/library/tkinter.html>
- Dans le dépôt : [`voix-conversation.md`](voix-conversation.md) (pourquoi ces choix,
  et l'alternative ElevenLabs), et la PR #18 (comment le visage est construit).
- En cas de blocage : Yohann, ou un assistant de code (Claude, ChatGPT) auquel on
  donne ce fichier et le code concerné.

## Ne pas faire

- **Committer la clé ou `voice_assistant/.env`.** Vérifiez `git status` avant chaque
  commit, et indexez les fichiers un par un, jamais `git add .`.
- **Enregistrer l'audio** des conversations. Rien n'est stocké.
- **Donner à GPT un outil qui fait bouger les moteurs** (coucou, danse). C'est une
  décision d'équipe, et les règles de sécurité robot de `CONTRIBUTING.md`
  s'appliquent.
- **Changer `talking` en émotion,** ou mélanger la bouche et l'émotion : la bouche
  appartient à la parole, le reste à l'émotion.

## Terminé quand

- [ ] Dans `chat_clavier.py`, Poppy répond à voix haute à vos questions, avec sa
      personnalité, et chaque émotion est apparue au moins une fois.
- [ ] Une liste de questions de démo choisies parmi celles qui ont bien marché.
- [ ] L'émotion change au début de chaque réponse, pas après.
- [ ] La bouche bouge avec la voix et se ferme dans les silences.
- [ ] Poppy peut parler en étant triste : la larme coule pendant qu'il parle.
- [ ] Au micro : moins de 2 s entre le relâchement de la touche et le début de la
      réponse.
- [ ] Sans Internet, il le dit au lieu de rester muet.
- [ ] Aucune clé dans l'historique git.
